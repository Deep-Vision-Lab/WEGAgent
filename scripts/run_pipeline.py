#!/usr/bin/env python3
"""
CLI script to run the full WEG extraction pipeline.

Supports both V1 and V2 pipelines:
- V1: Action Agent extracts all quadruple fields at once
- V2: Specialized agents (each focuses on one task), combiner builds quadruples
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
from weg_pipeline.graph.pipeline_v2 import run_pipeline_v2
from weg_pipeline.utils import find_preweg_guides
from weg_pipeline.validators import validate_weg

app = typer.Typer(help="Run the WEG extraction pipeline")
console = Console()


@app.command()
def run(
    guide: str = typer.Option(None, "--guide", "-g", help="Path to specific guide JSON"),
    root: str = typer.Option("data/preweg", "--root", "-r", help="Root directory to scan for guides"),
    llm_model: str = typer.Option("claude-sonnet-4-5-20250929", "--llm", help="LLM model for text agents"),
    vlm_model: str = typer.Option("claude-sonnet-4-5-20250929", "--vlm", help="VLM model for vision agents"),
    llm_provider: str = typer.Option(None, "--llm-provider", help="LLM provider (openai/anthropic/google/ollama). Auto-detected if not set."),
    vlm_provider: str = typer.Option(None, "--vlm-provider", help="VLM provider (openai/anthropic/google/ollama). Auto-detected if not set."),
    sequential: bool = typer.Option(False, "--sequential", "-s", help="Run agents sequentially (for debugging)"),
    no_review: bool = typer.Option(False, "--no-review", help="Skip the reviewer agent (V1 only)"),
    validate: bool = typer.Option(True, "--validate/--no-validate", help="Validate output WEG"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
    v2: bool = typer.Option(False, "--v2", help="Use V2 pipeline (specialized agents)"),
    use_ollama: bool = typer.Option(False, "--ollama", help="Use Ollama local models (auto-sets provider to ollama)"),
):
    """
    Run the WEG pipeline on one or more guides.
    
    V1 Pipeline (default): Action Agent extracts all quadruple fields
    V2 Pipeline (--v2): Specialized agents, combiner builds quadruples
    
    Examples:
        # Use cloud models (default)
        python run_pipeline.py --guide path/to/guide.json
        
        # Use Ollama local models
        python run_pipeline.py --ollama --llm llama3.2:3b --vlm llama3.2-vision:11b
    """
    
    # Auto-detect provider from model name if not specified
    def detect_provider(model: str) -> str:
        model_lower = model.lower()
        if "claude" in model_lower:
            return "anthropic"
        elif "gpt" in model_lower or "o1" in model_lower:
            return "openai"
        elif "gemini" in model_lower:
            return "google"
        elif any(x in model_lower for x in ["llama", "mistral", "qwen", "phi", "llava", "minicpm", "moondream"]):
            return "ollama"
        return "openai"  # default
    
    # If --ollama flag is set, override providers
    if use_ollama:
        llm_provider = "ollama"
        vlm_provider = "ollama"
        # Use default ollama models if cloud models were specified
        if "claude" in llm_model.lower() or "gpt" in llm_model.lower() or "gemini" in llm_model.lower():
            llm_model = "llama3.2:3b"
        if "claude" in vlm_model.lower() or "gpt" in vlm_model.lower() or "gemini" in vlm_model.lower():
            vlm_model = "llama3.2-vision:11b"
    
    config = {
        "llm_model": llm_model,
        "llm_provider": llm_provider or detect_provider(llm_model),
        "vlm_model": vlm_model,
        "vlm_provider": vlm_provider or detect_provider(vlm_model),
        "verbose": not quiet,
    }
    
    if not quiet:
        pipeline_name = "V2 (Specialized Agents)" if v2 else "V1 (Standard)"
        provider_info = "[green](LOCAL)[/green]" if config['llm_provider'] == "ollama" else "[blue](CLOUD)[/blue]"
        console.print(f"[bold cyan]Pipeline: {pipeline_name}[/bold cyan] {provider_info}")
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
        
        # Check for existing WEG — filename is {guide_id}_WEG.json (or _WEG_v2.json)
        guide_id  = guide_path.stem.split("_")[0]
        suffix    = "_WEG_v2.json" if v2 else "_WEG.json"
        weg_path  = guide_path.parent / f"{guide_id}{suffix}"
        if weg_path.exists():
            console.print(f"[yellow]WEG already exists at {weg_path}, skipping.[/yellow]")
            continue

        try:
            if v2:
                # V2 Pipeline: Specialized agents
                state = run_pipeline_v2(
                    guide_path=guide_path,
                    config=config,
                    parallel=not sequential,
                    suffix=suffix.split(".")[0],
                )
            else:
                # V1 Pipeline: Standard
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
