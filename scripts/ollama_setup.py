#!/usr/bin/env python3
"""
Ollama Setup Helper for WEGv2 Pipeline

This script helps you set up and manage local Ollama models for the WEG pipeline.

Usage:
    python scripts/ollama_setup.py check      # Check Ollama status and installed models
    python scripts/ollama_setup.py install    # Install recommended models
    python scripts/ollama_setup.py test       # Test LLM and VLM functionality
"""

import subprocess
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Ollama setup and management for WEGv2")
console = Console()


# ============================================
# RECOMMENDED MODELS FOR WEG PIPELINE
# ============================================

RECOMMENDED_MODELS = {
    "llm": {
        # Best balance of speed/quality for text tasks
        "llama3.2:3b": {
            "description": "Fast, efficient text model - RECOMMENDED for LLM tasks",
            "size": "~2GB",
            "use_case": "Tool extraction, hands estimation, part text analysis",
            "priority": 1,
        },
        "llama3.1:8b": {
            "description": "Higher quality text model, slower",
            "size": "~4.7GB",
            "use_case": "Better reasoning for complex instructions",
            "priority": 2,
        },
        "mistral:7b": {
            "description": "Alternative fast text model",
            "size": "~4.1GB", 
            "use_case": "Good instruction following",
            "priority": 3,
        },
        "qwen2.5:7b": {
            "description": "Excellent for structured outputs",
            "size": "~4.4GB",
            "use_case": "JSON extraction tasks",
            "priority": 4,
        },
    },
    "vlm": {
        # Vision-language models for image analysis
        "llama3.2-vision:11b": {
            "description": "Best open-source VLM - RECOMMENDED for vision tasks",
            "size": "~7.9GB",
            "use_case": "Action extraction with images, part localization",
            "priority": 1,
        },
        "llava:7b": {
            "description": "Lighter vision model, good for basic tasks",
            "size": "~4.5GB",
            "use_case": "Simple image understanding",
            "priority": 2,
        },
        "llava:13b": {
            "description": "Better quality vision, more VRAM needed",
            "size": "~8GB",
            "use_case": "Higher accuracy image analysis",
            "priority": 3,
        },
        "minicpm-v": {
            "description": "Compact but capable vision model",
            "size": "~5GB",
            "use_case": "Resource-constrained environments",
            "priority": 4,
        },
    },
}


def check_ollama_running() -> bool:
    """Check if Ollama service is running."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def get_installed_models() -> list[str]:
    """Get list of installed Ollama models."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []


def pull_model(model_name: str) -> bool:
    """Pull a model using ollama CLI."""
    try:
        console.print(f"[cyan]Pulling {model_name}...[/cyan] (this may take a while)")
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=False,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        console.print("[red]Error: ollama CLI not found. Install from https://ollama.ai[/red]")
        return False


@app.command()
def check():
    """Check Ollama status and show installed models."""
    console.print("\n[bold cyan]🔍 Checking Ollama Setup[/bold cyan]\n")
    
    # Check if Ollama is running
    if check_ollama_running():
        console.print("[green]✓[/green] Ollama service is running")
    else:
        console.print("[red]✗[/red] Ollama service is NOT running")
        console.print("  [dim]Start with: ollama serve[/dim]")
        console.print("  [dim]Install from: https://ollama.ai[/dim]")
        return
    
    # Get installed models
    installed = get_installed_models()
    
    if installed:
        console.print(f"\n[bold]Installed Models ({len(installed)}):[/bold]")
        for model in sorted(installed):
            # Check if it's a recommended model
            is_llm = any(model.startswith(m.split(":")[0]) for m in RECOMMENDED_MODELS["llm"])
            is_vlm = any(model.startswith(m.split(":")[0]) for m in RECOMMENDED_MODELS["vlm"])
            
            if is_vlm:
                console.print(f"  [magenta]👁 {model}[/magenta] (VLM)")
            elif is_llm:
                console.print(f"  [blue]📝 {model}[/blue] (LLM)")
            else:
                console.print(f"  [dim]  {model}[/dim]")
    else:
        console.print("\n[yellow]No models installed yet[/yellow]")
    
    # Show recommendations
    console.print("\n[bold]Recommended Models for WEG Pipeline:[/bold]")
    
    table = Table(show_header=True)
    table.add_column("Type", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Status")
    table.add_column("Description")
    
    for model_type in ["llm", "vlm"]:
        for model_name, info in RECOMMENDED_MODELS[model_type].items():
            if info["priority"] <= 2:  # Show top 2 per type
                is_installed = any(model_name.split(":")[0] in m for m in installed)
                status = "[green]✓ Installed[/green]" if is_installed else "[yellow]Not installed[/yellow]"
                table.add_row(
                    model_type.upper(),
                    model_name,
                    status,
                    info["description"],
                )
    
    console.print(table)


@app.command()
def install(
    llm_only: bool = typer.Option(False, "--llm-only", help="Install only LLM models"),
    vlm_only: bool = typer.Option(False, "--vlm-only", help="Install only VLM models"),
    minimal: bool = typer.Option(False, "--minimal", help="Install only priority-1 models"),
):
    """Install recommended Ollama models for the WEG pipeline."""
    console.print("\n[bold cyan]📦 Installing Recommended Models[/bold cyan]\n")
    
    if not check_ollama_running():
        console.print("[red]Error: Ollama is not running. Start it with: ollama serve[/red]")
        raise typer.Exit(1)
    
    installed = get_installed_models()
    to_install = []
    
    # Determine which models to install
    for model_type in ["llm", "vlm"]:
        if llm_only and model_type == "vlm":
            continue
        if vlm_only and model_type == "llm":
            continue
            
        for model_name, info in RECOMMENDED_MODELS[model_type].items():
            if minimal and info["priority"] > 1:
                continue
            if info["priority"] <= 2:  # Install top 2 priority models
                # Check if already installed
                if not any(model_name.split(":")[0] in m for m in installed):
                    to_install.append((model_type, model_name, info))
    
    if not to_install:
        console.print("[green]All recommended models are already installed![/green]")
        return
    
    console.print(f"Models to install ({len(to_install)}):")
    for model_type, model_name, info in to_install:
        console.print(f"  • {model_name} ({info['size']}) - {info['description']}")
    
    console.print()
    if not typer.confirm("Continue with installation?"):
        raise typer.Exit(0)
    
    # Install models
    for model_type, model_name, info in to_install:
        console.print(f"\n[bold]Installing {model_name}...[/bold]")
        if pull_model(model_name):
            console.print(f"[green]✓ {model_name} installed successfully[/green]")
        else:
            console.print(f"[red]✗ Failed to install {model_name}[/red]")
    
    console.print("\n[bold green]Installation complete![/bold green]")
    console.print("\nUsage example:")
    console.print("  [dim]python scripts/run_pipeline.py --ollama --guide path/to/guide.json[/dim]")


@app.command()
def test(
    llm_model: str = typer.Option("llama3.2:3b", "--llm", help="LLM model to test"),
    vlm_model: str = typer.Option("llama3.2-vision:11b", "--vlm", help="VLM model to test"),
):
    """Test LLM and VLM functionality with Ollama."""
    console.print("\n[bold cyan]🧪 Testing Ollama Models[/bold cyan]\n")
    
    if not check_ollama_running():
        console.print("[red]Error: Ollama is not running[/red]")
        raise typer.Exit(1)
    
    # Add project to path
    sys.path.insert(0, str(__file__).rsplit("/scripts", 1)[0])
    
    from weg_pipeline.utils.llm_utils import OllamaClient
    
    # Test LLM
    console.print(f"[bold]Testing LLM: {llm_model}[/bold]")
    try:
        llm_client = OllamaClient(model=llm_model)
        
        if not llm_client.is_model_available():
            console.print(f"[yellow]Model {llm_model} not found. Trying to pull...[/yellow]")
            if not pull_model(llm_model):
                raise Exception("Failed to pull model")
        
        response = llm_client.complete(
            "What tools would you need to remove a refrigerator door? List 3 tools briefly.",
            system_prompt="You are a helpful assistant. Be concise."
        )
        console.print(f"[green]✓ LLM Response:[/green]\n{response[:500]}")
    except Exception as e:
        console.print(f"[red]✗ LLM test failed: {e}[/red]")
    
    console.print()
    
    # Test VLM
    console.print(f"[bold]Testing VLM: {vlm_model}[/bold]")
    try:
        vlm_client = OllamaClient(model=vlm_model)
        
        if not vlm_client.is_model_available():
            console.print(f"[yellow]Model {vlm_model} not found. Trying to pull...[/yellow]")
            if not pull_model(vlm_model):
                raise Exception("Failed to pull model")
        
        # Check if it's a vision model
        if not vlm_client._is_vision_model:
            console.print(f"[yellow]Warning: {vlm_model} may not support vision[/yellow]")
        else:
            console.print(f"[green]✓ VLM {vlm_model} supports vision capability[/green]")
            console.print("[dim]Note: Full image test requires an actual image file[/dim]")
            
    except Exception as e:
        console.print(f"[red]✗ VLM test failed: {e}[/red]")
    
    console.print("\n[bold]Test Summary:[/bold]")
    console.print("  Run full pipeline with: python scripts/run_pipeline.py --ollama --guide <guide.json>")


@app.command()
def list_models():
    """List all recommended models with details."""
    console.print("\n[bold cyan]📋 All Recommended Ollama Models[/bold cyan]\n")
    
    installed = get_installed_models() if check_ollama_running() else []
    
    for model_type in ["llm", "vlm"]:
        console.print(f"\n[bold]{model_type.upper()} Models:[/bold]")
        
        table = Table(show_header=True)
        table.add_column("Model", style="green")
        table.add_column("Size")
        table.add_column("Use Case")
        table.add_column("Priority")
        table.add_column("Status")
        
        for model_name, info in sorted(
            RECOMMENDED_MODELS[model_type].items(),
            key=lambda x: x[1]["priority"]
        ):
            is_installed = any(model_name.split(":")[0] in m for m in installed)
            status = "✓" if is_installed else "—"
            priority = "⭐" * (4 - info["priority"])
            
            table.add_row(
                model_name,
                info["size"],
                info["use_case"],
                priority,
                status,
            )
        
        console.print(table)
    
    console.print("\n[dim]Install with: python scripts/ollama_setup.py install[/dim]")


if __name__ == "__main__":
    app()
