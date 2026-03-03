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
    
    # Use CWD as workspace root (image paths in WEG are relative to project root)
    base_path = Path.cwd()
    
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


@app.command()
def online_vis(
    weg_path: str = typer.Argument(..., help="Path to WEG JSON file"),
    steps: str = typer.Option(None, "--steps", "-s", help="Comma-separated list of step IDs to visualize (default: all)"),
    primary_only: bool = typer.Option(False, "--primary-only", "-p", help="Only show primary part bbox"),
    all_only: bool = typer.Option(False, "--all-only", "-a", help="Only show parts_all bboxes"),
):
    """
    Interactively view bounding boxes from a WEG file using OpenCV.

    Navigate with ENTER (next) and P (prev). Press Q to quit.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        console.print("[red]OpenCV not installed. Run: pip install opencv-python numpy[/red]")
        raise typer.Exit(1)

    weg_file = Path(weg_path)
    if not weg_file.exists():
        console.print(f"[red]WEG file not found: {weg_path}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold cyan]Loading WEG: {weg_path}[/bold cyan]")
    weg = load_weg(weg_file)

    # Use CWD as workspace root (image paths in WEG are relative to project root)
    base_path = Path.cwd()

    weg_steps = weg.get("steps", [])
    if steps:
        step_ids = set(int(s.strip()) for s in steps.split(","))
        weg_steps = [s for s in weg_steps if s.get("step_id") in step_ids]

    show_primary = not all_only
    show_all = not primary_only

    # Each frame: dict with cv_img, pil_img, label, part_ref (direct ref into weg dict)
    frames = []

    def render_cv(pil_img, part_ref):
        """Render PIL image with current bbox to an OpenCV BGR array."""
        annotated = draw_bbox_on_image(pil_img, [part_ref])
        return cv2.cvtColor(np.array(annotated.convert("RGB")), cv2.COLOR_RGB2BGR)

    console.print("[bold]Preparing images...[/bold]")
    for step in track(weg_steps, description="Loading..."):
        step_id = step.get("step_id", 0)
        task_name = step.get("task_name", "")

        # Collect all candidate parts per image
        images_parts: dict[str, list[dict]] = {}

        if show_primary:
            primary_part = step.get("primary_part", {})
            if primary_part and primary_part.get("image_path"):
                img_key = primary_part["image_path"]
                images_parts.setdefault(img_key, []).append(primary_part)

        if show_all:
            for part in step.get("parts_all", []):
                img_key = part.get("image_path")
                if img_key:
                    images_parts.setdefault(img_key, []).append(part)

        # Keep only the single best-confidence part per image
        best_parts = {
            img_key: max(parts, key=lambda p: p.get("confidence", 0))
            for img_key, parts in images_parts.items()
            if parts
        }

        for img_rel_path, part_ref in best_parts.items():
            img_path = base_path / img_rel_path
            if not img_path.exists():
                img_path = Path(img_rel_path)
            if not img_path.exists():
                console.print(f"[yellow]Image not found: {img_rel_path}[/yellow]")
                continue

            try:
                pil_img = Image.open(img_path).convert("RGB")
                part_name = part_ref.get("name", "unknown")
                confidence = part_ref.get("confidence", 0)
                label = f"Step {step_id}: {task_name}  |  {part_name} ({confidence:.0%})  |  {img_path.name}"
                frames.append({
                    "cv_img": render_cv(pil_img, part_ref),
                    "pil_img": pil_img,
                    "label": label,
                    "part_ref": part_ref,   # direct reference into weg dict
                    "step_id": step_id,
                })
            except Exception as e:
                console.print(f"[red]Error processing {img_path}: {e}[/red]")

    if not frames:
        console.print("[red]No images to display.[/red]")
        raise typer.Exit(1)

    console.print(f"[bold green]Loaded {len(frames)} image(s). ENTER: next  P: prev  E: annotate  Q: quit[/bold green]")

    strip_h = 46
    idx = 0
    window_name = "WEG Visualizer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 520)

    def make_strip(w, hint, label, accent=(0, 180, 160)):
        strip = np.zeros((strip_h, w, 3), dtype=np.uint8)
        strip[:] = (18, 18, 18)
        strip[0:2, :] = accent
        cv2.putText(strip, hint, (10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (110, 110, 110), 1, cv2.LINE_AA)
        cv2.putText(strip, label, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)
        return strip

    while True:
        frame = frames[idx]
        cv_img = frame["cv_img"]
        h, w = cv_img.shape[:2]

        strip = make_strip(
            w,
            hint="ENTER: next   P: prev   E: annotate bbox   Q: quit",
            label=f"[{idx + 1}/{len(frames)}]  {frame['label']}",
        )
        cv2.imshow(window_name, np.vstack([cv_img, strip]))

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == 13:                 # Enter → next
            idx = min(idx + 1, len(frames) - 1)
        elif key == ord('p'):           # P → prev
            idx = max(idx - 1, 0)
        elif key == ord('e'):           # E → annotate
            part_ref = frame["part_ref"]
            pil_img  = frame["pil_img"]

            # Annotation state shared with mouse callback
            ann = {"drawing": False, "x0": 0, "y0": 0, "x1": 0, "y1": 0, "done": False}

            def mouse_cb(event, x, y, flags, param):
                yc = min(y, h - 1)  # clamp to image area, ignore strip
                if event == cv2.EVENT_LBUTTONDOWN:
                    ann.update(drawing=True, x0=x, y0=yc, x1=x, y1=yc, done=False)
                elif event == cv2.EVENT_MOUSEMOVE and ann["drawing"]:
                    ann["x1"], ann["y1"] = x, yc
                elif event == cv2.EVENT_LBUTTONUP and ann["drawing"]:
                    ann["drawing"] = False
                    ann["x1"], ann["y1"] = x, yc
                    ann["done"] = True

            cv2.setMouseCallback(window_name, mouse_cb)

            while True:
                preview = np.array(pil_img)[:, :, ::-1].copy()  # PIL RGB → BGR
                if ann["x0"] != ann["x1"] or ann["y0"] != ann["y1"]:
                    cv2.rectangle(
                        preview,
                        (min(ann["x0"], ann["x1"]), min(ann["y0"], ann["y1"])),
                        (max(ann["x0"], ann["x1"]), max(ann["y0"], ann["y1"])),
                        (0, 220, 100), 2,
                    )
                ann_strip = make_strip(
                    w,
                    hint="Click and drag to draw bbox   ESC to cancel",
                    label=f"[ANNOTATE]  {frame['label']}",
                    accent=(0, 140, 255),  # orange accent in edit mode
                )
                cv2.imshow(window_name, np.vstack([preview, ann_strip]))

                k = cv2.waitKey(20) & 0xFF
                if k == 27:             # ESC → cancel
                    break
                if ann["done"]:
                    x0 = min(ann["x0"], ann["x1"])
                    y0 = min(ann["y0"], ann["y1"])
                    x1 = max(ann["x0"], ann["x1"])
                    y1 = max(ann["y0"], ann["y1"])
                    if x1 > x0 and y1 > y0:
                        # Update the bbox directly in the weg dict (part_ref is a live reference)
                        part_ref["bbox"] = {"x1": x0, "y1": y0, "x2": x1, "y2": y1}
                        # Re-render the frame with the new bbox
                        frame["cv_img"] = render_cv(pil_img, part_ref)
                        # Save weg back to disk
                        with open(weg_file, "w") as f:
                            json.dump(weg, f, indent=2)
                        console.print(
                            f"[bold green]Saved new bbox for '{part_ref.get('name')}' "
                            f"in step {frame['step_id']} → {weg_file.name}[/bold green]"
                        )
                    break

            # Remove mouse callback when done
            cv2.setMouseCallback(window_name, lambda *a: None)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    app()
