import logging
from pathlib import Path
from typing import Any, Dict, List

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from ..core.io_utils import load_json, save_json
from ..extractors.actions_openai import extract_actions_one_shot
from ..extractors.tools_openai import extract_tools_one_shot
from ..extractors.hands_openai import extract_hands_one_shot
from ..core.weg_builder import build_weg

# Optional: reduce spam
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langgraph").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)


def _actions_path(guide_path: str | Path) -> Path:
    p = Path(guide_path)
    return p.with_name(p.stem + "_actions.json")


def _tools_path(guide_path: str | Path) -> Path:
    p = Path(guide_path)
    return p.with_name(p.stem + "_tools.json")


def _hands_path(guide_path: str | Path) -> Path:
    p = Path(guide_path)
    return p.with_name(p.stem + "_hands.json")


def _weg_path(guide_path: str | Path) -> Path:
    p = Path(guide_path)
    return p.with_name(p.stem + "_WEG.json")


@tool
def load_preweg(guide_path: str) -> str:
    """
    Validate that the guide exists and return the normalized path string.
    (We don't return the whole JSON to avoid huge logs.)
    """
    p = Path(guide_path)
    if not p.exists():
        raise FileNotFoundError(f"Guide not found: {guide_path}")
    return str(p)


@tool
def extract_actions_to_file(guide_path: str, model: str = "gpt-4o-mini") -> str:
    """
    Extract actions_per_step and save next to the guide as *_actions.json.
    Returns the saved actions file path.
    """
    p = Path(guide_path)
    preweg = load_json(p)
    actions = extract_actions_one_shot(preweg, model=model)

    out = _actions_path(p)
    save_json(out, {"guide_path": str(p), "actions_per_step": actions})
    return str(out)


@tool
def extract_tools_to_file(guide_path: str, model: str = "gpt-4o-mini") -> str:
    """
    Extract tools_per_step and save next to the guide as *_tools.json.
    Returns the saved tools file path.
    """
    p = Path(guide_path)
    preweg = load_json(p)
    tools = extract_tools_one_shot(preweg, model=model)

    out = _tools_path(p)
    save_json(out, {"guide_path": str(p), "tools_per_step": tools})
    return str(out)


@tool
def extract_hands_to_file(guide_path: str, model: str = "gpt-4o-mini", dry_run: bool = False) -> str:
    """
    Estimate hands_per_step (0/1/2) and save next to the guide as *_hands.json.
    Returns the saved hands file path.
    """
    p = Path(guide_path)
    preweg = load_json(p)
    hands = extract_hands_one_shot(preweg, model=model, dry_run=dry_run)

    out = _hands_path(p)
    save_json(out, {"guide_path": str(p), "hands_per_step": hands})
    return str(out)


@tool
def build_weg_from_files(guide_path: str) -> str:
    """
    Build and save the final *_WEG.json by reading:
      - preWEG guide JSON
      - *_actions.json
      - *_tools.json
      - *_hands.json
    Returns the saved WEG path.
    """
    p = Path(guide_path)
    preweg = load_json(p)

    ap = _actions_path(p)
    tp = _tools_path(p)
    hp = _hands_path(p)

    if not ap.exists():
        raise FileNotFoundError(f"Missing actions file: {ap}")
    if not tp.exists():
        raise FileNotFoundError(f"Missing tools file: {tp}")
    if not hp.exists():
        raise FileNotFoundError(f"Missing hands file: {hp}")

    actions = load_json(ap).get("actions_per_step", [])
    tools = load_json(tp).get("tools_per_step", [])
    hands = load_json(hp).get("hands_per_step", [])

    weg_obj = build_weg(preweg, actions, tools, hands)

    out = _weg_path(p)
    save_json(out, weg_obj)
    return str(out)


def run_langchain_prompt_agent_single_guide(
    guide_path: str,
    model: str = "gpt-4o-mini",
    dry_run: bool = False,
) -> str:
    """
    LangChain multi-tool 'prompt agent' for ONE guide.

    The agent should:
      1) load_preweg
      2) extract_actions_to_file
      3) extract_tools_to_file
      4) extract_hands_to_file
      5) build_weg_from_files

    Returns the saved *_WEG.json path.
    """
    guide_path = str(Path(guide_path))
    out_path = str(_weg_path(guide_path))

    llm = ChatOpenAI(model=model, temperature=0)

    graph = create_agent(
        model=llm,
        tools=[
            load_preweg,
            extract_actions_to_file,
            extract_tools_to_file,
            extract_hands_to_file,
            build_weg_from_files,
        ],
        system_prompt=(
            "You are an AI agent that converts a pre-WEG guide into a WEG JSON using tool calls.\n"
            "Tools:\n"
            "- load_preweg(guide_path) -> normalized path\n"
            "- extract_actions_to_file(guide_path, model) -> *_actions.json\n"
            "- extract_tools_to_file(guide_path, model) -> *_tools.json\n"
            "- extract_hands_to_file(guide_path, model, dry_run) -> *_hands.json\n"
            "- build_weg_from_files(guide_path) -> *_WEG.json\n"
            "Rules:\n"
            "- Call load_preweg first.\n"
            "- Then call extract_actions_to_file, extract_tools_to_file, extract_hands_to_file.\n"
            "- Then call build_weg_from_files.\n"
            "- After build_weg_from_files succeeds, respond exactly with: DONE\n"
            "Never invent paths; use the guide_path provided by the user.\n"
        ),
        debug=True,
    )

    user_msg = (
        f"guide_path: {guide_path}\n"
        f"model: {model}\n"
        f"dry_run: {dry_run}\n"
        "Use tool calls to produce the WEG JSON and save it. Finish by saying DONE."
    )

    try:
        graph.invoke({"messages": [{"role": "user", "content": user_msg}]})
    except Exception:
        graph.invoke({"messages": [("user", user_msg)]})

    return out_path
