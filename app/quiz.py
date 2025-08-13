import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List

from openai import OpenAI


def _pack_context(
    docs: List[str],
    metas: List[Dict],
    char_budget: int = 12000,
    seed: int | None = None,
) -> List[Dict]:
    if seed is not None:
        random.seed(seed)

    # Group by source
    grouped = defaultdict(list)
    for text, meta in zip(docs, metas):
        t = (text or "").strip()
        if not t:
            continue
        grouped[meta.get("source", "unknown")].append((t, meta))

    # Shuffle within each source
    for src in grouped:
        random.shuffle(grouped[src])

    # Round-robin selection
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


QUIZ_JSON_SCHEMA = """\
Return STRICT JSON with this shape:

{
  "title": string,
  "questions": [
    {
      "id": string,
      "type": "true_false" | "mcq_single" | "mcq_multi",
      "level": number,
      "difficulty": "easy" | "medium" | "hard",
      "question": string,
      "options": [string, ...],
      "correctAnswers": [string, ...],   // answer TEXTS from options
      "explanation": string,
      "citations": [ { "source": string, "chunk": number } ]
    }
  ]
}
Rules:
- id is unique per question, e.g. "q1", "q2", etc.
- type is one of: "true_false", "mcq_single", "mcq_multi".
- level is an integer from 1 to 10, where 1 is easiest and 10 is hardest.
- For mcq_single: correctAnswers length MUST be 1 (one option TEXT).
- For mcq_multi: options MUST be between 3 and 7 (prefer 5-6), and correctAnswers options could be in range of 2-4 options out of the 3 to 7.
- For true_false: options MUST be ["True","False"]; correctAnswers MUST be ["True"] or ["False"].
- Each question MUST include level 1-10. Map difficulty: 1-5 easy, 6-7 medium, 8-10 hard.
- All content MUST be grounded ONLY in context and include at least one citation per question.
"""

SYSTEM_PROMPT = (
    "You create high-quality quizzes grounded ONLY in the provided context.\n"
    "Read the whole context carefully there might be some example and code snippets.\n"
    "you should smartly gather information from the context and generate questions.\n"
    "Make sure you use all sources of data.\n"
    "Follow the JSON schema exactly. Do not add extra fields. Do not invent facts."
)


def _build_user_prompt(
    topic_hint: str | None, packed: List[Dict], num_questions: int, types: List[str]
) -> str:
    header = f"Generate {num_questions} questions. Allowed types: {', '.join(types)}."
    if topic_hint:
        header += f" Focus on: {topic_hint}."
    header += "\n\nContext chunks (with citations):\n"
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

    # Helper maps
    def idx_to_text(idx: int) -> str | None:
        return options[idx] if 0 <= idx < len(options) else None

    # Already a list of strings?
    if isinstance(candidate, list) and all(isinstance(x, str) for x in candidate):
        return [opt for opt in options if opt in candidate]  # keep order by options

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
            # Exactly 4 options
            if len(options) < 4:
                needed = 4 - len(options)
                options = options + [f"Option {i}" for i in range(1, needed + 1)]
            else:
                options = options[:4]

        if qtype == "mcq_multi":
            # 3–7 options
            if len(options) < 3:
                needed = 3 - len(options)
                options = options + [f"Option {i}" for i in range(1, needed + 1)]
            elif len(options) > 7:
                options = options[:7]

        if qtype == "true_false":
            options = ["True", "False"]

        # Normalize correctAnswers to TEXTS
        candidate = q.get("correctAnswers")
        if candidate is None:
            # fallback legacy fields
            candidate = q.get(
                "answer"
            )  # could be str | list[str] | int | list[int] | bool

        correct_texts = _to_string_answers(qtype, options, candidate)

        # Per-type enforcement
        if qtype == "mcq_single":
            # exactly one text; deterministic fallback to first option
            if len(correct_texts) == 0 and options:
                correct_texts = [options[0]]
            else:
                correct_texts = correct_texts[:1]

        elif qtype == "mcq_multi":
            # Must be 2–4 texts; deterministically pad/trim using options order
            # De-duplicate while preserving options order
            present = [opt for opt in options if opt in correct_texts]
            if len(present) < 2:
                # pad with earliest options not already included
                for opt in options:
                    if opt not in present:
                        present.append(opt)
                    if len(present) >= 2:
                        break
            # Trim to at most 4
            if len(present) > 4:
                present = present[:4]
            correct_texts = present

        elif qtype == "true_false":
            # Must be ["True"] or ["False"]; default to ["False"] if unclear
            if correct_texts == ["True"]:
                correct_texts = ["True"]
            elif correct_texts == ["False"]:
                correct_texts = ["False"]
            else:
                correct_texts = ["False"]

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

        out_qs.append(
            {
                "id": qid,
                "type": qtype,
                "level": level,
                "difficulty": difficulty,
                "question": question,
                "options": options,
                "correctAnswers": correct_texts,  # TEXT answers
                "explanation": explanation,
                "citations": citations,
            }
        )
    return {"title": title, "questions": out_qs}


def generate_quiz_from_chunks(
    docs: List[str],
    metas: List[Dict],
    num_questions: int = 12,
    types: List[str] = ["mcq_single", "mcq_multi", "true_false"],
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

    # Normalize to guarantee frontend invariants
    return _normalize_quiz(data)
