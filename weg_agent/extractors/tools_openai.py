import json
import os
from typing import Any, Dict, List

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_tools_one_shot(
    preweg: Dict[str, Any],
    model: str = "gpt-4o-mini",
    dry_run: bool = False,
) -> List[List[str]]:
    """
    Returns list-of-lists (per step).
    If dry_run=True or no API key: return empty lists.
    """
    steps = preweg.get("steps", []) or []
    steps_sorted = sorted(
        [s for s in steps if isinstance(s, dict)],
        key=lambda s: s.get("step_index", 0),
    )
    if not steps_sorted:
        return []

    if dry_run or not os.getenv("OPENAI_API_KEY"):
        return [[] for _ in steps_sorted]

    lines: List[str] = []
    lines.append("You are an assistant that extracts ONLY the tools used by a human in each step of an appliance repair guide.")
    lines.append("")
    lines.append("Definition of a 'tool': something the person HOLDS, OPERATES, APPLIES, or EMPLOYS.")
    lines.append("Examples: screwdrivers, wrenches, pliers, pry tool, multimeter, drill, cloth, towel, tape, brush, bucket.")
    lines.append("")
    lines.append("Do NOT include:")
    lines.append("- Objects being removed/replaced (screws, panels, hoses, fans, connectors, etc).")
    lines.append("- Appliance components/targets being acted on.")
    lines.append("- Body parts.")
    lines.append("")
    lines.append("If no tools are used, return [].")
    lines.append("")
    lines.append("Return ONLY valid JSON: array of arrays, i-th array is Step i tools.")
    lines.append('Example: [["Phillips screwdriver"], [], ["cloth", "tape"]]')
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
        print(f"[WARN] tools: failed to parse JSON: {e}")
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
