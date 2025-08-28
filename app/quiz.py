import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

from openai import OpenAI


def _pack_context(
    docs: List[str],
    metas: List[Dict],
    char_budget: int = 12000,
    seed: int | None = None,
) -> List[Dict]:
    if seed is not None:
        random.seed(seed)

    grouped = defaultdict(list)
    for text, meta in zip(docs, metas):
        t = (text or "").strip()
        if not t:
            continue
        grouped[meta.get("source", "unknown")].append((t, meta))

    for src in grouped:
        random.shuffle(grouped[src])

    packed, used = [], 0
    sources = list(grouped.keys())
    idxs = {src: 0 for src in sources}

    while used < char_budget and sources:
        for src in list(sources):
            if idxs[src] >= len(grouped[src]):
                sources.remove(src)
                continue
            t, meta = grouped[src][idxs[src]]
            idxs[src] += 1
            extra = len(t) + 100
            if used + extra > char_budget:
                return packed
            packed.append(
                {"text": t, "source": meta.get("source"), "chunk": meta.get("chunk")}
            )
            used += extra
    return packed


def _difficulty_from_level(level: int) -> str:
    if level <= 5:
        return "easy"
    if level <= 7:
        return "medium"
    return "hard"


# ---- UPDATED SCHEMA (adds answer_short_question and grading) ----
QUIZ_JSON_SCHEMA = """\
Return STRICT JSON with this shape:

{
  "title": string,
  "questions": [
    {
      "id": string,
      "type": "true_false" | "mcq_single" | "mcq_multi" | "answer_short_question",
      "level": number,
      "difficulty": "easy" | "medium" | "hard",
      "question": string,
      "options": [string, ...],
      "correctAnswers": [string, ...],   // for short-answer: MUST be []
      "explanation": string,
      "citations": [ { "source": string, "chunk": number } ],
      "grading": {                      // REQUIRED for answer_short_question; OPTIONAL otherwise
        "rubric": string,               // clear criteria grounded in context
        "keywords": [string, ...],      // expected key terms/phrases present in the context
        "maxChars": number              // OPTIONAL cap for learner's response (e.g. 300)
      }
    }
  ]
}
Rules:
- id is unique per question, e.g. "q1", "q2", etc.
- type is one of: "true_false", "mcq_single", "mcq_multi", "answer_short_question".
- level is an integer from 1 to 10, where 1 is easiest and 10 is hardest.
- For mcq_single: correctAnswers length MUST be 1 (one option TEXT).
- For mcq_multi: correctAnswers length MUST be between 2 and 4 (inclusive).
- For true_false: options MUST be ["True","False"]; correctAnswers MUST be ["True"] or ["False"].
- For answer_short_question: options MUST be [], and correctAnswers MUST be [] (answers are NOT revealed). Include a concise grading rubric and keywords grounded in the provided context.
- Each question MUST include level 1-10. Map difficulty: 1-5 easy, 6-7 medium, 8-10 hard.
- All content MUST be grounded ONLY in context and include at least one citation per question.
"""

SYSTEM_PROMPT = (
    "You create high-quality quizzes grounded ONLY in the provided context.\n"
    "Read the whole context carefully; there might be examples and code snippets.\n"
    "Use all sources of data and follow the JSON schema exactly. Do not invent facts.\n"
    "For 'answer_short_question' items, DO NOT reveal the answer; include a precise grading rubric and keywords."
)


def _build_user_prompt(
    topic_hint: str | None, packed: List[Dict], num_questions: int, types: List[str]
) -> str:
    header = f"Generate {num_questions} questions. Allowed types: {', '.join(types)}."
    if topic_hint:
        header += f" Focus on: {topic_hint}."
    header += (
        "\n\nContext chunks (with citations):\n"
        "— Cite using the [source | chunk] provided below.\n"
    )
    body = []
    for blk in packed:
        body.append(
            f"[source: {blk['source']} | chunk: {blk['chunk']}]\n{blk['text']}\n"
        )
    tail = f"\nOutput format:\n{QUIZ_JSON_SCHEMA}"
    return header + "\n".join(body) + tail


def _to_string_answers(qtype: str, options: List[str], candidate: Any) -> List[str]:
    """
    Normalize various incoming shapes (indices, text, bool) into a
    deduped, in-range list of option TEXTS.
    """

    def idx_to_text(idx: int) -> Optional[str]:
        return options[idx] if 0 <= idx < len(options) else None

    if qtype == "answer_short_question":
        # We never surface the answer in the question JSON.
        return []

    # Already a list of strings?
    if isinstance(candidate, list) and all(isinstance(x, str) for x in candidate):
        return [opt for opt in options if opt in candidate]

    # List of indices?
    if isinstance(candidate, list) and all(isinstance(x, int) for x in candidate):
        return [t for x in candidate if (t := idx_to_text(x)) is not None]

    # Single string?
    if isinstance(candidate, str):
        return [candidate] if candidate in options else []

    # Single bool? (only meaningful for true/false)
    if isinstance(candidate, bool):
        if qtype == "true_false":
            return ["True"] if candidate is True else ["False"]
        return []

    return []


def _normalize_quiz(data: Dict[str, Any]) -> Dict[str, Any]:
    """Post-process to enforce invariants for the frontend. correctAnswers are TEXTS."""
    title = data.get("title") or "Quiz"
    questions = data.get("questions") or []
    out_qs = []

    for i, q in enumerate(questions, 1):
        qid = q.get("id") or f"q{i}"
        qtype = q.get("type")
        level = int(q.get("level", 5))
        if level < 1:
            level = 1
        if level > 10:
            level = 10
        difficulty = _difficulty_from_level(level)
        question = (q.get("question") or "").strip()

        # Options defaulting by type
        options = q.get("options") or (
            ["True", "False"] if qtype == "true_false" else []
        )

        # Enforce options length by type
        if qtype == "mcq_single":
            if len(options) < 4:
                needed = 4 - len(options)
                options = options + [f"Option {i}" for i in range(1, needed + 1)]
            else:
                options = options[:4]

        if qtype == "mcq_multi":
            if len(options) < 3:
                needed = 3 - len(options)
                options = options + [f"Option {i}" for i in range(1, needed + 1)]
            elif len(options) > 7:
                options = options[:7]

        if qtype == "true_false":
            options = ["True", "False"]

        if qtype == "answer_short_question":
            options = []  # no options

        # Normalize correctAnswers to TEXTS
        candidate = q.get("correctAnswers")
        if candidate is None:
            candidate = q.get("answer")

        correct_texts = _to_string_answers(qtype, options, candidate)

        # Per-type enforcement
        if qtype == "mcq_single":
            if len(correct_texts) == 0 and options:
                correct_texts = [options[0]]
            else:
                correct_texts = correct_texts[:1]

        elif qtype == "mcq_multi":
            present = [opt for opt in options if opt in correct_texts]
            if len(present) < 2:
                for opt in options:
                    if opt not in present:
                        present.append(opt)
                    if len(present) >= 2:
                        break
            if len(present) > 4:
                present = present[:4]
            correct_texts = present

        elif qtype == "true_false":
            if correct_texts == ["True"]:
                correct_texts = ["True"]
            elif correct_texts == ["False"]:
                correct_texts = ["False"]
            else:
                correct_texts = ["False"]

        elif qtype == "answer_short_question":
            # MUST be empty — we never reveal the answer.
            correct_texts = []

        explanation = (q.get("explanation") or "").strip()

        # Normalize citations
        citations = q.get("citations") or []
        if not isinstance(citations, list):
            citations = []
        citations = [
            {"source": str(c.get("source")), "chunk": int(c.get("chunk", 0))}
            for c in citations
            if isinstance(c, dict) and "source" in c
        ]

        # Normalize grading
        grading = q.get("grading") or {}
        if qtype == "answer_short_question":
            # enforce presence of rubric/keywords
            rubric = (grading.get("rubric") or "").strip()
            keywords = [
                str(x).strip()
                for x in (grading.get("keywords") or [])
                if str(x).strip()
            ]
            max_chars = grading.get("maxChars")
            if not rubric:
                rubric = "Answer concisely using facts from the cited context. Mention the key terms listed in keywords; do not invent facts."
            if not keywords:
                keywords = []
            grading = {"rubric": rubric, "keywords": keywords}
            if isinstance(max_chars, int) and max_chars > 0:
                grading["maxChars"] = max_chars
        else:
            grading = grading if isinstance(grading, dict) else {}

        out_qs.append(
            {
                "id": qid,
                "type": qtype,
                "level": level,
                "difficulty": difficulty,
                "question": question,
                "options": options,
                "correctAnswers": correct_texts,
                "explanation": explanation,
                "citations": citations,
                "grading": grading,
            }
        )
    return {"title": title, "questions": out_qs}


def generate_quiz_from_chunks(
    docs: List[str],
    metas: List[Dict],
    num_questions: int = 12,
    types: List[str] = [
        "mcq_single",
        "mcq_multi",
        "true_false",
        "answer_short_question",
    ],
    topic_hint: str | None = None,
    model: str | None = None,
) -> Dict[str, Any]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    packed = _pack_context(docs, metas, char_budget=12000)
    if not packed:
        return {"title": "Quiz", "questions": []}

    prompt = _build_user_prompt(topic_hint, packed, num_questions, types)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.9,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    try:
        data = json.loads(raw)
    except Exception:
        data = {"title": "Quiz", "questions": []}

    return _normalize_quiz(data)


# ------------------ NEW: short-answer evaluator ------------------ #

EVAL_SYSTEM_PROMPT = (
    "You are a strict grader. Decide whether the learner's short answer is correct and also give a correct short answer"
    "In the answer you should provide the correct answer in 3 or 4 words"
    "using ONLY the provided context chunks and the question's grading rubric/keywords. "
    "Be concise and avoid revealing hidden answers or unrelated facts."
)

EVAL_JSON_SCHEMA = """\
Return STRICT JSON:

{
  "result": "correct" | "incorrect",
  "answer": string,           // provide correct answer 
  "score": number,              // 0.0 to 1.0, where 0.7+ is usually correct
  "feedback": string,           // one short sentence
  "citations": [ { "source": string, "chunk": number } ]
}
Rules:
- Cite only from provided context chunks.
- Consider rubric and keywords. If essential keywords are missing or contradicted, prefer 'incorrect'.
- If ambiguous or unsupported by context, prefer 'incorrect'.
"""


def evaluate_short_answer(
    user_answer: str,
    question: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Grade a learner's response for an `answer_short_question` using citations from question.

    Args:
        user_answer: The student's answer text
        question: Question object containing grading criteria and citations

    Returns:
        {
            "result": "correct"|"incorrect",
            "score": float,                # 0..1
            "feedback": str,               # short note
            "citations": [{"source": str, "chunk": int}]
        }
    """
    if question.get("type") != "answer_short_question":
        raise ValueError(
            "evaluate_short_answer requires a question of type 'answer_short_question'."
        )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    # Build prompt using question citations
    citations = question.get("citations", [])
    grading = question.get("grading", {})

    header = (
        "Evaluate the learner's short answer to the question below.\n\n"
        f"Question: {question.get('question', '').strip()}\n"
        f"Rubric: {grading.get('rubric', '').strip()}\n"
        f"Keywords: {', '.join(grading.get('keywords', []))}\n"
        f"Learner Answer: {user_answer.strip()}\n\n"
        "Context from citations:\n"
    )

    # Use citations from the question object
    prompt = header + "\n".join(
        [
            f"[source: {citation['source']} | chunk: {citation['chunk']}]"
            for citation in citations
        ]
    )

    prompt += f"\n\nOutput format:\n{EVAL_JSON_SCHEMA}"

    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    try:
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        print("error")
        return {
            "result": "incorrect",
            "answer": "",
            "score": 0.0,
            "feedback": "Error evaluating answer.",
            "citations": citations,
        }
    return {
        "result": "correct" if data.get("result") == "correct" else "incorrect",
        "score": max(0.0, min(1.0, float(data.get("score", 0.0)))),
        "answer": (data.get("answer") or "").strip(),
        "feedback": (data.get("feedback") or "").strip(),
        "citations": citations,
    }
