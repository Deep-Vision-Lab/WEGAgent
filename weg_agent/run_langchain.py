import argparse
from pathlib import Path
from typing import List, Optional

from weg_agent.agent.langchain_prompt_agent import run_langchain_prompt_agent_single_guide


def find_preweg_guides(root: Path) -> List[Path]:
    """Find original pre-WEG guide JSONs under **/guides/*.json, excluding agent outputs."""
    candidates = list(root.rglob("guides/*.json"))
    out: List[Path] = []
    for p in candidates:
        name = p.name
        if (
            name.endswith("_WEG.json")
            or name.endswith("_actions.json")
            or name.endswith("_tools.json")
            or name.endswith("_hands.json")
        ):
            continue
        out.append(p)
    return sorted(out)


def choose_guide_interactively(guides: List[Path]) -> Optional[Path]:
    if not guides:
        print("[WARN] No guides found under the given root.")
        return None

    print("\nAvailable guides:\n")
    for i, g in enumerate(guides, start=1):
        print(f"{i:>3}. {g.as_posix()}")

    while True:
        s = input("\nEnter guide number (or 'q' to quit): ").strip().lower()
        if s in {"q", "quit", "exit"}:
            return None
        try:
            idx = int(s)
            if 1 <= idx <= len(guides):
                return guides[idx - 1]
        except ValueError:
            pass
        print(f"Invalid input. Enter a number 1-{len(guides)} or 'q'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LangChain prompt-agent on a selected guide.")
    parser.add_argument("--root", type=str, default="preweg_data/appliances", help="Root to scan for guides/")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--dry-run", action="store_true", help="Do not call OpenAI for hands (uses fallback if implemented)")
    parser.add_argument("--guide", type=str, default=None, help="Run directly on this guide path (skip interactive)")
    args = parser.parse_args()

    if args.guide:
        guide_path = Path(args.guide)
    else:
        guides = find_preweg_guides(Path(args.root))
        guide_path = choose_guide_interactively(guides)
        if guide_path is None:
            print("[INFO] Quit.")
            return

    print(f"\n[INFO] Running LangChain prompt-agent on:\n  {guide_path.as_posix()}\n")
    saved = run_langchain_prompt_agent_single_guide(
        guide_path=str(guide_path),
        model=args.model,
        dry_run=args.dry_run,
    )
    print(f"\n[OK] WEG saved:\n  {saved}\n")


if __name__ == "__main__":
    main()
