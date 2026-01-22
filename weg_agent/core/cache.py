from pathlib import Path
from typing import List, Optional
from .io_utils import load_json, save_json


def cache_path(guide_path: Path, suffix: str) -> Path:
    return guide_path.with_name(guide_path.stem + suffix)


def load_list_of_lists(path: Path, key: str) -> Optional[List[List[str]]]:
    if not path.exists():
        return None
    try:
        d = load_json(path)
        val = d.get(key)
        if not isinstance(val, list):
            return None

        out: List[List[str]] = []
        for x in val:
            if isinstance(x, list):
                out.append([str(t).strip() for t in x if str(t).strip()])
            else:
                out.append([])
        return out
    except Exception:
        return None


def save_list_of_lists(path: Path, guide_path: Path, key: str, value: List[List[str]]) -> None:
    save_json(path, {"guide_path": str(guide_path), key: value})
from typing import List, Optional

def load_int_list(path: Path, key: str) -> Optional[List[int]]:
    if not path.exists():
        return None
    try:
        d = load_json(path)
        val = d.get(key)
        if not isinstance(val, list):
            return None
        out: List[int] = []
        for x in val:
            try:
                out.append(int(x))
            except Exception:
                out.append(1)
        return out
    except Exception:
        return None


def save_int_list(path: Path, guide_path: Path, key: str, value: List[int]) -> None:
    save_json(path, {"guide_path": str(guide_path), key: value})

