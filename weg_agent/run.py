import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

from .core.io_utils import find_guides
from .agent.weg_agent import WegAgent, WegAgentConfig


def main():
    load_dotenv()

    p = argparse.ArgumentParser()
    p.add_argument("--guide", type=str, help="Path to one pre-WEG guide json")
    p.add_argument("--root", type=str, help="Root folder to scan (e.g. preweg_data/appliances)")
    p.add_argument("--resume", action="store_true", help="Use cached *_actions/_tools if exists")
    p.add_argument("--no-resume", action="store_true", help="Ignore cache and recompute")
    p.add_argument("--dry-run", action="store_true", help="No API calls (heuristics only)")
    p.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    args = p.parse_args()

    resume = True
    if args.no_resume:
        resume = False
    if args.resume:
        resume = True

    cfg = WegAgentConfig(model=args.model, resume=resume, dry_run=args.dry_run)
    agent = WegAgent(cfg)

    if args.guide:
        agent.run_one(Path(args.guide))
        return

    if args.root:
        guides = find_guides(Path(args.root))
        print(f"[INFO] Found {len(guides)} guides")
        for gp in guides:
            print(f"[INFO] Processing: {gp}")
            agent.run_one(gp)
        return

    raise SystemExit("Provide --guide OR --root")


if __name__ == "__main__":
    main()
