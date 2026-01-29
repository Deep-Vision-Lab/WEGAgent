#!/usr/bin/env python3
"""
CLI script to test individual agents.
"""
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich import print_json
import json

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from weg_pipeline.agents import (
    ActionAgent,
    ToolAgent,
    HandsAgent,
    PartTextAgent,
    PartVisionAgent,
)
from weg_pipeline.models.preweg import PreWEGGuide
from weg_pipeline.utils import load_json

app = typer.Typer(help="Test individual extraction agents")
console = Console()

AGENTS = {
    "action": ActionAgent,
    "tool": ToolAgent,
    "hands": HandsAgent,
    "part_text": PartTextAgent,
    "part_vision": PartVisionAgent,
}


@app.command()
def run(
    agent: str = typer.Argument(..., help=f"Agent to run: {list(AGENTS.keys())}"),
    guide: str = typer.Option(..., "--guide", "-g", help="Path to pre-WEG guide JSON"),
    model: str = typer.Option(None, "--model", "-m", help="Override model"),
    provider: str = typer.Option(None, "--provider", "-p", help="Override provider"),
    step: int = typer.Option(None, "--step", "-s", help="Process only this step (1-based)"),
    output: str = typer.Option(None, "--output", "-o", help="Save result to file"),
):
    """Run a single agent on a guide."""
    
    if agent not in AGENTS:
        console.print(f"[red]Unknown agent: {agent}[/red]")
        console.print(f"Available: {list(AGENTS.keys())}")
        raise typer.Exit(1)
    
    # Load guide
    guide_path = Path(guide)
    console.print(f"\n[bold cyan]Loading guide: {guide_path}[/bold cyan]")
    
    guide_data = load_json(guide_path)
    guide_obj = PreWEGGuide(**guide_data)
    
    console.print(f"  Title: {guide_obj.title}")
    console.print(f"  Steps: {guide_obj.num_steps}")
    
    # Create agent
    AgentClass = AGENTS[agent]
    kwargs = {"verbose": True}
    if model:
        kwargs["model"] = model
    if provider:
        kwargs["provider"] = provider
    
    agent_instance = AgentClass(**kwargs)
    
    # Run agent
    console.print(f"\n[bold magenta]Running {agent} agent...[/bold magenta]\n")
    
    try:
        if agent == "part_vision":
            # Special handling for vision agent
            result = agent_instance.run_all_steps(guide_obj)
        elif step:
            # Single step mode
            result = agent_instance.run(guide_obj, step_index=step)
        else:
            # Full guide mode
            result = agent_instance.run(guide_obj)
        
        # Display result
        console.print("\n[bold green]Result:[/bold green]\n")
        result_dict = result.model_dump() if hasattr(result, "model_dump") else result
        print_json(json.dumps(result_dict, indent=2, default=str))
        
        # Save if requested
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result_dict, indent=2, default=str))
            console.print(f"\n[green]Saved to: {output_path}[/green]")
            
    except Exception as e:
        console.print(f"[bold red]Agent failed: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def list_agents():
    """List available agents and their descriptions."""
    console.print("\n[bold]Available Agents:[/bold]\n")
    
    descriptions = {
        "action": "Extracts atomic actions from step descriptions (VLM)",
        "tool": "Identifies tools used in each step (LLM)",
        "hands": "Estimates number of hands needed (LLM)",
        "part_text": "Extracts part names from text (LLM)",
        "part_vision": "Localizes parts with bounding boxes (VLM)",
    }
    
    for name, desc in descriptions.items():
        console.print(f"  [cyan]{name:12}[/cyan] - {desc}")
    console.print()


if __name__ == "__main__":
    app()
