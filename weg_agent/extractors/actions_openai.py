import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_actions_one_shot(
    preweg: Dict[str, Any],
    model: str = "gpt-4o-mini",
    dry_run: bool = False,
) -> List[List[str]]:
    """
    Returns list-of-lists (per step).
    If dry_run=True or no API key: heuristic actions from bullet lines.
    """
    steps = preweg.get("steps", []) or []
    steps_sorted = sorted(
        [s for s in steps if isinstance(s, dict)],
        key=lambda s: s.get("step_index", 0),
    )
    if not steps_sorted:
        return []

    if dry_run or not os.getenv("OPENAI_API_KEY"):
        out: List[List[str]] = []
        for st in steps_sorted:
            text = (st.get("full_description") or "").splitlines()
            # keep bullet-like lines
            bullets = []
            for ln in text:
                ln = ln.strip()
                if ln.startswith("- "):
                    bullets.append(ln[2:].strip())
            out.append(bullets[:8])
        return out

    lines: List[str] = []
    lines.append("You are an assistant that converts appliance repair guide steps into ATOMIC ACTIONS.")
    lines.append("")
    lines.append("Definition of an ACTION:")
    lines.append("- A single operational instruction a person can perform.")
    lines.append("- Usually starts with an imperative verb.")
    lines.append("- One concrete operation per action.")
    lines.append("")
    lines.append("IMPORTANT — Do NOT output as actions:")
    lines.append("- Safety notes, warnings, tips, or commentary.")
    lines.append("- Pure background context WITHOUT an operation.")
    lines.append("- Pure lists of items/components without an operation.")
    lines.append("- Duplicate actions.")
    lines.append("")
    lines.append("IMPORTANT — Conditional actions:")
    lines.append("- If conditional BUT clearly an operation, INCLUDE it and keep the condition.")
    lines.append("")
    lines.append("Return ONLY valid JSON: a JSON array of arrays. i-th array corresponds to Step i.")
    lines.append('Example: [["Unplug the device."], [], ["Remove the screws.", "Lift off the cover."]]')
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
        print(f"[WARN] actions: failed to parse JSON: {e}")
        print(raw)
        return []

    if not isinstance(data, list):
        return []

    cleaned: List[List[str]] = []
    for entry in data:
        if isinstance(entry, list):
            cleaned.append([str(x).strip() for x in entry if str(x).strip()])
        else:
            cleaned.append([])
    return cleaned
