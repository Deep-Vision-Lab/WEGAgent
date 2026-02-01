"""
I/O Utilities for loading and saving JSON files.
"""
import json
from pathlib import Path
from typing import Any


def load_json(path: Path | str) -> dict[str, Any]:
    """Load JSON file and return as dictionary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path | str, data: Any, indent: int = 2) -> Path:
    """Save data to JSON file. Creates parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    return path


def find_preweg_guides(root: Path | str, exclude_outputs: bool = True) -> list[Path]:
    """
    Find all pre-WEG guide JSON files under a root directory.
    
    Args:
        root: Root directory to search
        exclude_outputs: If True, exclude *_WEG.json, *_actions.json, etc.
    
    Returns:
        Sorted list of paths to guide JSON files
    """
    root = Path(root)
    if not root.exists():
        return []

    # Look for guides in guides/ subdirectories
    candidates = list(root.rglob("guides/*.json"))

    if not exclude_outputs:
        return sorted(candidates)

    # Filter out generated outputs
    output_suffixes = ("_WEG.json", "_actions.json", "_tools.json", "_hands.json", "_parts.json")
    filtered = [p for p in candidates if not any(p.name.endswith(s) for s in output_suffixes)]

    return sorted(filtered)


def get_output_path(guide_path: Path | str, suffix: str) -> Path:
    """
    Generate output path for intermediate/final results.
    
    Example:
        guide_path = "data/preweg/.../167672_guide.json"
        suffix = "_WEG.json"
        returns: "data/preweg/.../{suffix}.json"
    """
    path = Path(guide_path)
    return path.parent / suffix
