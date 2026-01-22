import json
import os
from typing import Any, Dict, List

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_hands_one_shot(
    preweg: Dict[str, Any],
    model: str = "gpt-4o-mini",
    dry_run: bool = False,
) -> List[int]:
    """
    Returns a list of ints (0/1/2), one per step.
    If dry_run=True or no API key: simple heuristic.
    """
    steps = preweg.get("steps", []) or []
    steps_sorted = sorted(
        [s for s in steps if isinstance(s, dict)],
        key=lambda s: s.get("step_index", 0),
    )
    if not steps_sorted:
        return []

    # Heuristic fallback (no API): if mentions "hold" / "press" / "support" → 2 else 1 if any action else 0
    if dry_run or not os.getenv("OPENAI_API_KEY"):
        out: List[int] = []
        for st in steps_sorted:
            text = (st.get("full_description") or "").lower()
            if not text.strip():
                out.append(0)
            elif any(k in text for k in ["hold", "support", "steady", "while", "both hands", "two hands"]):
                out.append(2)
            else:
                out.append(1)
        return out

    lines: List[str] = []
    lines.append("You are an assistant that estimates the NUMBER OF HANDS a person uses in each repair step.")
    lines.append("")
    lines.append("Return ONLY integers 0, 1, or 2 for each step:")
    lines.append("- 0 = no hands used (pure info/warning, no action)")
    lines.append("- 1 = one hand likely sufficient (press, pull small part, use single tool)")
    lines.append("- 2 = two hands likely needed (hold + pull, stabilize + unscrew, pry while holding, lift large part, align with both hands)")
    lines.append("")
    lines.append("Rules:")
    lines.append("- Use the step text ONLY.")
    lines.append("- If uncertain, choose the SMALLER number (1 over 2).")
    lines.append("- Safety notes alone → 0.")
    lines.append("")
    lines.append("Return ONLY valid JSON: an array of integers, one per step.")
    lines.append("Example: [1, 2, 0, 1]")
    lines.append("")
    lines.append("Now here are the steps:")

    for idx, step in enumerate(steps_sorted, start=1):
        text = step.get("full_description") or ""
        lines.append(f"\nStep {idx}:")
        lines.append('"""')
        lines.append(text)
        lines.append('"""')

    prompt = "\n".join(lines)

    resp = client.responses.create(model=model, input=prompt)
    raw = resp.output_text.strip()

    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        data = json.loads(raw)
    except Exception as e:
        print(f"[WARN] hands: failed to parse JSON: {e}")
        print(raw)
        return []

    if not isinstance(data, list):
        return []

    out: List[int] = []
    for x in data:
        try:
            v = int(x)
        except Exception:
            v = 1
        if v < 0:
            v = 0
        if v > 2:
            v = 2
        out.append(v)

    return out
