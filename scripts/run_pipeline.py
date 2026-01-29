#!/usr/bin/env python3
"""
CLI script to run the full WEG extraction pipeline.
"""
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from weg_pipeline.graph import run_pipeline
from weg_pipeline.utils import find_preweg_guides
from weg_pipeline.validators import validate_weg

app = typer.Typer(help="Run the WEG extraction pipeline")
console = Console()


@app.command()
def run(
    guide: str = typer.Option(None, "--guide", "-g", help="Path to specific guide JSON"),
    root: str = typer.Option("data/preweg", "--root", "-r", help="Root directory to scan for guides"),
    llm_model: str = typer.Option("claude-3-5-haiku-20241022", "--llm", help="LLM model for text agents"),
    vlm_model: str = typer.Option("claude-3-5-haiku-20241022", "--vlm", help="VLM model for vision agents"),
    llm_provider: str = typer.Option(None, "--llm-provider", help="LLM provider (openai/anthropic/google). Auto-detected if not set."),
    vlm_provider: str = typer.Option(None, "--vlm-provider", help="VLM provider (openai/anthropic/google). Auto-detected if not set."),
    sequential: bool = typer.Option(False, "--sequential", "-s", help="Run agents sequentially (for debugging)"),
    no_review: bool = typer.Option(False, "--no-review", help="Skip the reviewer agent"),
    validate: bool = typer.Option(True, "--validate/--no-validate", help="Validate output WEG"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
):
    """Run the WEG pipeline on one or more guides."""
    
    # Auto-detect provider from model name if not specified
    def detect_provider(model: str) -> str:
        if "claude" in model.lower():
            return "anthropic"
        elif "gpt" in model.lower() or "o1" in model.lower():
            return "openai"
        elif "gemini" in model.lower():
            return "google"
        return "openai"  # default
    
    config = {
        "llm_model": llm_model,
        "llm_provider": llm_provider or detect_provider(llm_model),
        "vlm_model": vlm_model,
        "vlm_provider": vlm_provider or detect_provider(vlm_model),
        "verbose": not quiet,
    }
    
    if not quiet:
        console.print(f"[dim]LLM: {config['llm_model']} ({config['llm_provider']})[/dim]")
        console.print(f"[dim]VLM: {config['vlm_model']} ({config['vlm_provider']})[/dim]")
    
    # Determine which guides to process
    if guide:
        guides = [Path(guide)]
    else:
        guides = find_preweg_guides(root)
        if not guides:
            console.print(f"[yellow]No guides found in {root}[/yellow]")
            console.print("Use --guide to specify a guide, or crawl some first with scripts/crawl.py")
            raise typer.Exit(1)
    
    console.print(f"\n[bold cyan]Processing {len(guides)} guide(s)...[/bold cyan]\n")
    
    for guide_path in guides:
        console.print(f"\n{'='*60}")
        console.print(f"[bold]Guide: {guide_path}[/bold]")
        console.print('='*60)
        
        weg_path = str(guide_path).split('.')[0] + "_WEG.json"
        if Path(weg_path).exists():
            console.print(f"[yellow]WEG already exists at {weg_path}, skipping.[/yellow]")
            continue 

        try:
            state = run_pipeline(
                guide_path=guide_path,
                config=config,
                parallel=not sequential,
                enable_reviewer=not no_review,
            )
            
            # Validate if requested
            if validate and state.get("weg"):
                console.print("\n[bold]Validating WEG...[/bold]")
                result = validate_weg(state["weg"])
                console.print(str(result))
                
        except Exception as e:
            console.print(f"[bold red]Pipeline failed: {e}[/bold red]")
            if not quiet:
                import traceback
                traceback.print_exc()


@app.command()
def list_guides(
    root: str = typer.Option("data/preweg", "--root", "-r", help="Root directory to scan"),
):
    """List available pre-WEG guides."""
    guides = find_preweg_guides(root)
    
    if not guides:
        console.print(f"[yellow]No guides found in {root}[/yellow]")
        return
    
    console.print(f"\n[bold]Found {len(guides)} guides:[/bold]\n")
    for i, g in enumerate(guides, 1):
        console.print(f"  {i:3}. {g}")
    console.print()


if __name__ == "__main__":
    app()
