import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def find_guides(root: Path) -> List[Path]:
    """
    Recursively find pre-WEG guide json files inside .../guides/*.json
    Avoid cache outputs like *_actions.json, *_tools.json, *_WEG.json
    """
    root = Path(root)
    candidates = list(root.rglob("guides/*.json"))
    out: List[Path] = []
    for p in candidates:
        name = p.name
        if name.endswith("_actions.json") or name.endswith("_tools.json") or name.endswith("_WEG.json"):
            continue
        out.append(p)
    return sorted(out)
