from typing import Any, Dict, List, Optional

def _pick_one(items: List[str]) -> Optional[str]:
    items = [x.strip() for x in items if isinstance(x, str) and x.strip()]
    return items[0] if items else None


def build_weg(
    preweg: Dict[str, Any],
    actions_per_step: List[List[str]],
    tools_per_step: List[List[str]],
    hands_per_step: List[int],
) -> Dict[str, Any]:
    steps = preweg.get("steps", []) or []
    steps_sorted = sorted(
        [s for s in steps if isinstance(s, dict)],
        key=lambda s: s.get("step_index", 0),
    )

    n = len(steps_sorted)

    # pad/truncate to match number of steps
    if len(actions_per_step) < n:
        actions_per_step = actions_per_step + ([[]] * (n - len(actions_per_step)))
    else:
        actions_per_step = actions_per_step[:n]

    if len(tools_per_step) < n:
        tools_per_step = tools_per_step + ([[]] * (n - len(tools_per_step)))
    else:
        tools_per_step = tools_per_step[:n]

    if len(hands_per_step) < n:
        hands_per_step = hands_per_step + ([None] * (n - len(hands_per_step)))
    else:
        hands_per_step = hands_per_step[:n]

    # --- header exactly like your format ---
    weg: Dict[str, Any] = {
        "header": {
            "title": preweg.get("title") or "",
            "description": preweg.get("summary") or "",
            "toolbox": preweg.get("toolbox", []) or [],
        },
        "steps": [],
    }

    for i, st in enumerate(steps_sorted):
        step_id = int(st.get("step_index", i + 1))
        description = st.get("full_description", "") or ""

        actions = actions_per_step[i]
        tool = _pick_one(tools_per_step[i])

        # part: future extractor (leave None for now)
        part = None

        # hands: int 0/1/2 (or None if missing)
        hands = hands_per_step[i]
        if hands is not None:
            try:
                hands = int(hands)
            except Exception:
                hands = None
            if hands is not None:
                hands = max(0, min(2, hands))

        step_obj: Dict[str, Any] = {
            "step_id": step_id,
            "description": description,
            "actions": actions,
            "tool": tool,
            "part": part,
            "geometric_location": None,
            "hands": hands,
        }

        weg["steps"].append(step_obj)

    return weg
