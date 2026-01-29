#!/usr/bin/env python3
"""
CLI script to crawl guides from iFixit.
"""
import sys
from pathlib import Path

import typer
from rich.console import Console

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from weg_pipeline.crawler import crawl_guides, crawl_guide, DEVICE_KEYWORDS

app = typer.Typer(help="Crawl repair guides from iFixit")
console = Console()


@app.command()
def search(
    device: str = typer.Option(..., "--device", "-d", help=f"Device category: {list(DEVICE_KEYWORDS.keys())}"),
    keyword: str = typer.Option(None, "--keyword", "-k", help="Custom search keyword"),
    limit: int = typer.Option(5, "--limit", "-l", help="Max guides to crawl"),
    output: str = typer.Option("data/preweg", "--output", "-o", help="Output directory"),
):
    """Search and crawl guides by device type."""
    output_path = Path(output)
    
    console.print(f"\n[bold cyan]Crawling {device} guides...[/bold cyan]\n")
    
    try:
        paths = crawl_guides(
            output_root=output_path,
            device_key=device,
            keyword=keyword,
            limit=limit,
        )
        
        console.print(f"\n[bold green]Successfully crawled {len(paths)} guides:[/bold green]")
        for p in paths:
            console.print(f"  - {p}")
            
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def single(
    guide_id: int = typer.Argument(..., help="iFixit guide ID"),
    device: str = typer.Option("appliances", "--device", "-d", help="Device category"),
    output: str = typer.Option("data/preweg", "--output", "-o", help="Output directory"),
):
    """Crawl a single guide by ID."""
    output_path = Path(output)
    
    console.print(f"\n[bold cyan]Crawling guide {guide_id}...[/bold cyan]\n")
    
    try:
        path = crawl_guide(
            guide_id=guide_id,
            output_root=output_path,
            device_key=device,
        )
        console.print(f"\n[bold green]Saved to: {path}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def devices():
    """List supported device categories."""
    console.print("\n[bold]Supported device categories:[/bold]\n")
    for key, default_keyword in DEVICE_KEYWORDS.items():
        console.print(f"  [cyan]{key}[/cyan] → default search: '{default_keyword}'")
    console.print()


if __name__ == "__main__":
    app()
