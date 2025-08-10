import os, json
from typing import List, Dict, Any
from openai import OpenAI

def _pack_context(docs: List[str], metas: List[Dict], char_budget: int = 12000) -> List[Dict]:
    packed, used = [], 0
    for text, meta in zip(docs, metas):
        t = (text or "").strip()
        if not t: 
            continue
        block = {"text": t, "source": meta.get("source"), "chunk": meta.get("chunk")}
        extra = len(t) + 100
        if used + extra > char_budget:
            break
        packed.append(block); used += extra
    return packed

def _difficulty_from_level(level: int) -> str:
    if level <= 5: return "easy"
    if level <= 7: return "medium"
    return "hard"

QUIZ_JSON_SCHEMA = """\
Return STRICT JSON with this shape:

{
  "title": string,
  "questions": [
    {
      "id": string,                         // e.g., "q1"
      "type": "true_false" | "mcq_single" | "mcq_multi",
      "level": number,                      // integer 1-10
      "difficulty": "easy" | "medium" | "hard",
      "question": string,
      "options": [string, ...],             // for mcq_* (exactly 4 options); for true_false use ["True","False"]
      "correctAnswers": [number],           // indices into options (0-based). For true_false, use [0] or [1].
      "explanation": string,
      "citations": [ { "source": string, "chunk": number } ]
    }
  ]
}
Rules:
- Types allowed: true_false, mcq_single, mcq_multi.
- For mcq_single: correctAnswers length MUST be 1.
- For mcq_multi: correctAnswers length MUST be 2-3 (not all 4).
- For true_false: options MUST be ["True","False"]; correctAnswers MUST be [0] or [1].
- Each question MUST include level 1-10. Map difficulty: 1-5 easy, 6-7 medium, 8-10 hard.
- All content MUST be grounded ONLY in context and include at least one citation per question.
"""

SYSTEM_PROMPT = (
  "You create high-quality quizzes grounded ONLY in the provided context.\n"
  "Follow the JSON schema exactly. Do not add extra fields. Do not invent facts."
)

def _build_user_prompt(topic_hint: str | None, packed: List[Dict], num_questions: int, types: List[str]) -> str:
    header = f"Generate {num_questions} questions. Allowed types: {', '.join(types)}."
    if topic_hint:
        header += f" Focus on: {topic_hint}."
    header += "\n\nContext chunks (with citations):\n"
    body = []
    for blk in packed:
        body.append(f"[source: {blk['source']} | chunk: {blk['chunk']}]\n{blk['text']}\n")
    tail = f"\nOutput format:\n{QUIZ_JSON_SCHEMA}"
    return header + "\n".join(body) + tail

def _normalize_quiz(data: Dict[str, Any]) -> Dict[str, Any]:
    """Post-process to enforce invariants for the frontend."""
    title = data.get("title") or "Quiz"
    questions = data.get("questions") or []
    out_qs = []
    for i, q in enumerate(questions, 1):
        qid = q.get("id") or f"q{i}"
        qtype = q.get("type")
        level = int(q.get("level", 5))
        if level < 1: level = 1
        if level > 10: level = 10
        difficulty = _difficulty_from_level(level)
        question = (q.get("question") or "").strip()
        options = q.get("options") or (["True","False"] if qtype == "true_false" else [])
        # Ensure options length for MCQ
        if qtype in ("mcq_single", "mcq_multi"):
            # pad/trim to exactly 4
            options = (options + ["Option"]*4)[:4]
        if qtype == "true_false":
            options = ["True", "False"]
        # Correct answers as list of indices
        ca = q.get("correctAnswers")
        if isinstance(ca, list) and all(isinstance(x, int) for x in ca):
            correct = ca
        else:
            # fallback: try to map from answer text if provided
            ans = q.get("answer")
            if isinstance(ans, list):
                correct = [options.index(a) for a in ans if a in options]
            elif isinstance(ans, str):
                correct = [options.index(ans)] if ans in options else []
            elif isinstance(ans, bool):
                correct = [0] if ans is True else [1]
            else:
                correct = []
        # Enforce per-type rules
        if qtype == "mcq_single":
            correct = correct[:1]
        if qtype == "mcq_multi":
            correct = [c for c in correct if 0 <= c < len(options)]
            # keep 2-3 answers if possible
            if len(correct) < 2 and len(options) >= 2:
                correct = list({0,1})  # minimal fallback
            if len(correct) > 3:
                correct = correct[:3]
        if qtype == "true_false":
            correct = [0] if correct == [0] else [1] if correct == [1] else [1]  # default False if unclear

        explanation = (q.get("explanation") or "").strip()
        citations = q.get("citations") or []
        if not isinstance(citations, list): citations = []
        # normalize citation items
        citations = [
            {"source": str(c.get("source")), "chunk": int(c.get("chunk", 0))}
            for c in citations if isinstance(c, dict)
        ]
        out_qs.append({
            "id": qid,
            "type": qtype,
            "level": level,
            "difficulty": difficulty,
            "question": question,
            "options": options,
            "correctAnswers": correct,
            "explanation": explanation,
            "citations": citations,
        })
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
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    packed = _pack_context(docs, metas, char_budget=12000)
    if not packed:
        return {"title": "Quiz", "questions": []}

    prompt = _build_user_prompt(topic_hint, packed, num_questions, types)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
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
