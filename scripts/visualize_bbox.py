#!/usr/bin/env python3
"""
Visualize bounding boxes from WEG output on images.

This script reads a WEG JSON file and draws bounding boxes around 
detected parts on the corresponding images.
"""
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console
from rich.progress import track

app = typer.Typer(help="Visualize bounding boxes from WEG files")
console = Console()

# Color palette for different parts (RGBA)
COLORS = [
    (255, 0, 0, 200),      # Red
    (0, 255, 0, 200),      # Green
    (0, 0, 255, 200),      # Blue
    (255, 255, 0, 200),    # Yellow
    (255, 0, 255, 200),    # Magenta
    (0, 255, 255, 200),    # Cyan
    (255, 128, 0, 200),    # Orange
    (128, 0, 255, 200),    # Purple
    (0, 255, 128, 200),    # Spring Green
    (255, 128, 128, 200),  # Light Red
]


def load_weg(weg_path: Path) -> dict:
    """Load WEG JSON file."""
    with open(weg_path, "r") as f:
        return json.load(f)


def get_font(size: int = 12):
    """Get a font for labels. Falls back to default if custom fonts unavailable."""
    try:
        # Try to load a nicer font
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except (IOError, OSError):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except (IOError, OSError):
            # Fall back to default
            return ImageFont.load_default()


def draw_bbox_on_image(
    image: Image.Image,
    parts: list[dict],
    line_width: int = 3,
    font_size: int = 14,
) -> Image.Image:
    """
    Draw bounding boxes on an image.
    
    Args:
        image: PIL Image to draw on
        parts: List of part dicts with 'name', 'bbox', 'confidence'
        line_width: Width of bounding box lines
        font_size: Font size for labels
    
    Returns:
        Image with bounding boxes drawn
    """
    # Create a copy to avoid modifying original
    img = image.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    font = get_font(font_size)
    
    for i, part in enumerate(parts):
        bbox = part.get("bbox", {})
        name = part.get("name", "unknown")
        confidence = part.get("confidence", 0.0)
        
        # Skip if no valid bbox
        if not all(k in bbox for k in ["x1", "y1", "x2", "y2"]):
            continue
        
        x1, y1 = bbox["x1"], bbox["y1"]
        x2, y2 = bbox["x2"], bbox["y2"]
        
        # Get color for this part (cycle through palette)
        color = COLORS[i % len(COLORS)]
        
        # Draw bounding box
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=color[:3],
            width=line_width,
        )
        
        # Create label text
        label = f"{name} ({confidence:.0%})"
        
        # Get text bounding box for background
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Draw label background
        padding = 2
        label_y = y1 - text_height - padding * 2
        if label_y < 0:
            label_y = y2 + padding  # Put below if no space above
        
        draw.rectangle(
            [(x1, label_y), (x1 + text_width + padding * 2, label_y + text_height + padding * 2)],
            fill=color[:3] + (180,),  # Semi-transparent background
        )
        
        # Draw label text
        draw.text(
            (x1 + padding, label_y + padding),
            label,
            fill=(255, 255, 255),
            font=font,
        )
    
    return img


def visualize_step(
    step: dict,
    weg_base_path: Path,
    output_dir: Path,
    show_primary: bool = True,
    show_all_parts: bool = True,
) -> list[Path]:
    """
    Visualize bounding boxes for a single step.
    
    Args:
        step: Step dict from WEG
        weg_base_path: Base path for resolving relative image paths
        output_dir: Directory to save output images
        show_primary: Draw primary part bbox
        show_all_parts: Draw all parts_all bboxes
    
    Returns:
        List of saved image paths
    """
    step_id = step.get("step_id", 0)
    saved_paths = []
    
    # Collect all parts to draw for each image
    images_parts = {}  # image_path -> list of parts
    
    # Add primary part
    if show_primary:
        primary_part = step.get("primary_part", {})
        if primary_part and primary_part.get("image_path"):
            img_path = primary_part["image_path"]
            if img_path not in images_parts:
                images_parts[img_path] = []
            # Mark primary part specially
            part_copy = primary_part.copy()
            part_copy["name"] = f"[PRIMARY] {part_copy.get('name', 'unknown')}"
            images_parts[img_path].append(part_copy)
    
    # Add all parts
    if show_all_parts:
        for part in step.get("parts_all", []):
            img_path = part.get("image_path")
            if img_path:
                if img_path not in images_parts:
                    images_parts[img_path] = []
                # Avoid duplicate with primary
                if not any(p.get("name", "").startswith("[PRIMARY]") and 
                          p.get("bbox") == part.get("bbox") 
                          for p in images_parts[img_path]):
                    images_parts[img_path].append(part)
    
    # Process each image
    for img_rel_path, parts in images_parts.items():
        # Resolve image path relative to WEG file location or workspace
        img_path = weg_base_path / img_rel_path
        if not img_path.exists():
            # Try relative to current directory
            img_path = Path(img_rel_path)
        
        if not img_path.exists():
            console.print(f"[yellow]Image not found: {img_rel_path}[/yellow]")
            continue
        
        try:
            img = Image.open(img_path)
            
            # Draw bounding boxes
            annotated = draw_bbox_on_image(img, parts)
            
            # Save output
            output_name = f"step_{step_id:02d}_{img_path.stem}_annotated.jpg"
            output_path = output_dir / output_name
            annotated.save(output_path, quality=95)
            saved_paths.append(output_path)
            
        except Exception as e:
            console.print(f"[red]Error processing {img_path}: {e}[/red]")
    
    return saved_paths


@app.command()
def visualize(
    weg_path: str = typer.Argument(..., help="Path to WEG JSON file"),
    output_dir: str = typer.Option(None, "--output", "-o", help="Output directory (default: <weg_dir>/visualized)"),
    steps: str = typer.Option(None, "--steps", "-s", help="Comma-separated list of step IDs to visualize (default: all)"),
    primary_only: bool = typer.Option(False, "--primary-only", "-p", help="Only show primary part bbox"),
    all_only: bool = typer.Option(False, "--all-only", "-a", help="Only show parts_all bboxes"),
):
    """
    Visualize bounding boxes from a WEG file on images.
    
    Draws colored bounding boxes around detected parts with labels 
    showing part name and confidence score.
    """
    weg_file = Path(weg_path)
    if not weg_file.exists():
        console.print(f"[red]WEG file not found: {weg_path}[/red]")
        raise typer.Exit(1)
    
    # Determine output directory
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = weg_file.parent / "visualized"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Output directory: {out_dir}[/bold]")
    
    # Load WEG
    console.print(f"[bold cyan]Loading WEG: {weg_path}[/bold cyan]")
    weg = load_weg(weg_file)
    
    # Base path for resolving relative image paths
    # Go up from WEG file to find workspace root
    base_path = weg_file.parent
    while base_path.name not in ["WEGv2", "DIYCrawler"] and base_path.parent != base_path:
        base_path = base_path.parent
    
    # Filter steps if specified
    weg_steps = weg.get("steps", [])
    if steps:
        step_ids = set(int(s.strip()) for s in steps.split(","))
        weg_steps = [s for s in weg_steps if s.get("step_id") in step_ids]
    
    console.print(f"[bold]Processing {len(weg_steps)} steps...[/bold]")
    
    # Determine what to show
    show_primary = not all_only
    show_all = not primary_only
    
    total_images = 0
    for step in track(weg_steps, description="Visualizing..."):
        saved = visualize_step(
            step,
            base_path,
            out_dir,
            show_primary=show_primary,
            show_all_parts=show_all,
        )
        total_images += len(saved)
    
    console.print(f"\n[bold green]✓ Created {total_images} annotated images in {out_dir}[/bold green]")


@app.command()
def summary(
    weg_path: str = typer.Argument(..., help="Path to WEG JSON file"),
):
    """
    Print a summary of parts detected in each step.
    """
    weg_file = Path(weg_path)
    if not weg_file.exists():
        console.print(f"[red]WEG file not found: {weg_path}[/red]")
        raise typer.Exit(1)
    
    weg = load_weg(weg_file)
    
    console.print(f"\n[bold cyan]WEG Summary: {weg.get('title', 'Unknown')}[/bold cyan]")
    console.print(f"Guide ID: {weg.get('guide_id')}")
    console.print(f"Total Steps: {len(weg.get('steps', []))}")
    console.print()
    
    for step in weg.get("steps", []):
        step_id = step.get("step_id")
        task_name = step.get("task_name", "")
        primary = step.get("primary_part", {})
        all_parts = step.get("parts_all", [])
        quadruples = step.get("action_quadruples", [])
        
        console.print(f"[bold]Step {step_id}: {task_name}[/bold]")
        
        if primary:
            console.print(f"  Primary part: [id={primary.get('part_id')}] {primary.get('name')} ({primary.get('confidence', 0):.0%})")
        
        if all_parts:
            console.print(f"  All parts ({len(all_parts)}):")
            for p in all_parts:
                bbox = p.get("bbox", {})
                bbox_str = f"({bbox.get('x1')},{bbox.get('y1')})-({bbox.get('x2')},{bbox.get('y2')})"
                console.print(f"    - [id={p.get('part_id')}] {p.get('name')}: {p.get('confidence', 0):.0%} @ {bbox_str}")
        
        if quadruples:
            console.print(f"  Action quadruples ({len(quadruples)}):")
            for q in quadruples:
                part_ref = f"→ part_id={q.get('part_id')}" if q.get('part_id') else "(no visual)"
                console.print(f"    - {q.get('action')} {q.get('component')} {part_ref}")
        
        console.print()


if __name__ == "__main__":
    app()
