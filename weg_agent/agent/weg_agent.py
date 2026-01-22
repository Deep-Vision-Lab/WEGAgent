import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..core.io_utils import load_json, save_json
from ..core.cache import cache_path, load_list_of_lists, save_list_of_lists
from ..core.weg_builder import build_weg
from ..extractors.actions_openai import extract_actions_one_shot
from ..extractors.tools_openai import extract_tools_one_shot
from ..core.cache import load_int_list, save_int_list
from ..extractors.hands_openai import extract_hands_one_shot



@dataclass
class WegAgentConfig:
    model: str = "gpt-4o-mini"
    resume: bool = True
    dry_run: bool = False


class WegAgent:
    def __init__(self, cfg: WegAgentConfig):
        self.cfg = cfg

    def run_one(self, guide_path: Path) -> Path:
        guide_path = Path(guide_path)
        preweg = load_json(guide_path)

        # cache files
        actions_cache = cache_path(guide_path, "_actions.json")
        tools_cache = cache_path(guide_path, "_tools.json")
        weg_out = cache_path(guide_path, "_WEG.json")
        hands_cache = cache_path(guide_path, "_hands.json")


        # 1) actions
        actions = None
        if self.cfg.resume:
            actions = load_list_of_lists(actions_cache, "actions_per_step")
        if actions is None:
            actions = extract_actions_one_shot(preweg, model=self.cfg.model, dry_run=self.cfg.dry_run)
            save_list_of_lists(actions_cache, guide_path, "actions_per_step", actions)

        # 2) tools
        tools = None
        if self.cfg.resume:
            tools = load_list_of_lists(tools_cache, "tools_per_step")
        if tools is None:
            tools = extract_tools_one_shot(preweg, model=self.cfg.model, dry_run=self.cfg.dry_run)
            save_list_of_lists(tools_cache, guide_path, "tools_per_step", tools)
        # 3) hands
        hands = None
        if self.cfg.resume:
            hands = load_int_list(hands_cache, "hands_per_step")
        if hands is None:
            hands = extract_hands_one_shot(preweg, model=self.cfg.model, dry_run=self.cfg.dry_run)
            save_int_list(hands_cache, guide_path, "hands_per_step", hands)


        # 3) build weg
        weg = build_weg(preweg, actions, tools,hands)
        save_json(weg_out, weg)

        print(f"[OK] WEG saved: {weg_out}")
        return weg_out
