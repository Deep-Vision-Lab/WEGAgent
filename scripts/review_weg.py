#!/usr/bin/env python3
"""
Interactive WEG reviewer with edit and delete capabilities.

Split-panel OpenCV window: left = annotated image, right = step metadata.

Views (toggle with V):
  PARTS view   — parts list, bboxes, confidence
  ACTIONS view — task name, actions, action quadruples (hands, tool, component)

Keys (PARTS view):  D=delete part  E=edit bbox
Keys (ACTIONS view): T=edit task name  H=toggle hands  A=edit action text
Shared keys: V=toggle view  F=flag step  K=mark OK  O=open URL  ENTER/N/P=navigate  Q=quit

Usage:
    python scripts/review_weg.py data/preweg/.../guides/_WEG.json
"""
import json
import textwrap
import webbrowser
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app     = typer.Typer(help="Interactive WEG step reviewer with edit/delete")
console = Console()

# ── Layout ─────────────────────────────────────────────────────────────────────
PANEL_W = 400
STRIP_H = 46
WIN_W   = 980
WIN_H   = 580

# RGBA palette for bboxes
COLORS = [
    (255,  80,  80, 220),
    ( 80, 220,  80, 220),
    ( 80, 130, 255, 220),
    (255, 210,  60, 220),
    (220,  80, 220, 220),
    ( 60, 220, 220, 220),
    (255, 145,  30, 220),
    (160,  80, 255, 220),
    ( 60, 210, 140, 220),
    (255, 160, 130, 220),
]

# Interaction modes
MODE_NORMAL      = "normal"
MODE_DELETE      = "delete"
MODE_EDIT_SELECT = "edit_select"
MODE_EDIT_DRAW   = "edit_draw"
MODE_HANDS_EDIT  = "hands_edit"    # pick which quadruple
MODE_HANDS_SET   = "hands_set"     # pick the hands count
MODE_ACTION_EDIT = "action_edit"

# Info panel views
VIEW_PARTS   = "parts"
VIEW_ACTIONS = "actions"


# ── Font helper ────────────────────────────────────────────────────────────────

def _get_font(size: int = 11, bold: bool = False):
    from PIL import ImageFont
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


# ── Bottom strip ───────────────────────────────────────────────────────────────

def make_strip(w: int, hint: str, label: str, accent=(0, 180, 160)):
    import cv2
    import numpy as np
    s = np.zeros((STRIP_H, w, 3), dtype=np.uint8)
    s[:] = (18, 18, 18)
    s[0:2, :] = accent
    cv2.putText(s, hint,  (10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (110, 110, 110), 1, cv2.LINE_AA)
    cv2.putText(s, label, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)
    return s


# ── Info panel ─────────────────────────────────────────────────────────────────

def make_info_panel(
    height: int,
    step: dict,
    header: dict,
    step_idx: int,
    total_steps: int,
    mode: str,
    view: str = VIEW_PARTS,
    edit_quad_i: int = -1,
) -> "np.ndarray":
    """Render the right info panel as a numpy BGR array."""
    import numpy as np
    from PIL import Image as PILImage, ImageDraw

    BG   = (22, 22, 28)
    FG   = (220, 220, 220)
    DIM  = (110, 110, 120)
    TEAL = (0, 180, 160)
    RED  = (220, 90,  90)
    YEL  = (210, 190, 60)
    GRN  = (80,  200, 100)
    ORG  = (255, 160, 40)
    SEP  = (50,  50,  60)
    FLG  = (200, 80,  80)
    VER  = (80,  200, 120)

    img  = PILImage.new("RGB", (PANEL_W, height), BG)
    draw = ImageDraw.Draw(img)

    f_sm = _get_font(10)
    f_md = _get_font(12)
    f_lg = _get_font(14, bold=True)

    pad = 10
    y   = 10

    def wt(txt, x, yy, font, color, max_chars=44, max_lines=5):
        """Wrapped text — returns new y."""
        for line in (textwrap.wrap(txt, width=max_chars) or [txt])[:max_lines]:
            draw.text((x, yy), line, fill=color, font=font)
            yy += (font.size if hasattr(font, "size") else 13) + 2
        return yy

    def hsep(yy):
        draw.line([(pad, yy), (PANEL_W - pad, yy)], fill=SEP, width=1)
        return yy + 8

    # ── Guide header (shared) ─────────────────────────────────────────────────
    title = header.get("title", "Unknown Guide")
    y = wt(title, pad, y, f_lg, TEAL, max_chars=36, max_lines=2)
    url = header.get("source_url", "")
    if url:
        y = wt((url if len(url) <= 46 else url[:43] + "..."), pad, y, f_sm, DIM, max_lines=1)
    y = hsep(y + 2)

    # ── Step counter + task name (shared) ─────────────────────────────────────
    draw.text((pad, y), f"Step {step_idx + 1} / {total_steps}", fill=DIM, font=f_sm)
    y += 13

    # Verified / flagged badge
    if step.get("verified"):
        draw.text((PANEL_W - 70, y - 13), "✓ OK", fill=VER, font=f_md)
    elif step.get("flagged"):
        draw.text((PANEL_W - 80, y - 13), "⚑ FLAGGED", fill=FLG, font=f_md)

    task = step.get("task_name", "")
    y = wt(task, pad, y, f_lg, FG, max_chars=34, max_lines=2)
    y += 2
    y = hsep(y)

    # ══════════════════════════════════════════════════════════════════════════
    #  PARTS VIEW
    # ══════════════════════════════════════════════════════════════════════════
    if view == VIEW_PARTS:
        parts   = step.get("parts_all", [])
        primary = step.get("primary_part") or {}
        pp_id, pp_name = primary.get("part_id"), primary.get("name")

        if mode in (MODE_DELETE, MODE_EDIT_SELECT):
            prompt = "DELETE which part?" if mode == MODE_DELETE else "EDIT BBOX — select part"
            draw.text((pad, y), prompt, fill=(RED if mode == MODE_DELETE else ORG), font=f_md)
            y += 18
            draw.text((pad, y), "Press 1–9   ESC to cancel", fill=DIM, font=f_sm)
            y += 16
            for i, part in enumerate(parts):
                draw.text((pad + 4, y),
                          f"{i + 1}.  {part.get('name', '?')}  ({part.get('confidence', 0):.0%})",
                          fill=COLORS[i % len(COLORS)][:3], font=f_md)
                y += 18
        else:
            draw.text((pad, y), f"Parts  ({len(parts)})", fill=TEAL, font=f_sm)
            y += 14
            for i, part in enumerate(parts):
                conf  = part.get("confidence", 0.0)
                name  = part.get("name", "unknown")
                star  = " ★" if (part.get("part_id") == pp_id and name == pp_name) else ""
                ccol  = GRN if conf >= 0.90 else YEL if conf >= 0.75 else RED
                draw.text((pad,      y),      f"{i + 1}.", fill=COLORS[i % len(COLORS)][:3], font=f_md)
                draw.text((pad + 24, y),      f"{name}{star}", fill=FG,   font=f_md)
                draw.text((pad + 24, y + 14), f"{conf:.0%}",  fill=ccol,  font=f_sm)
                y += 32
            y += 4

        # Controls
        ctrl_y = height - 115
        if ctrl_y > y:
            y = ctrl_y
        y = hsep(y)
        for k, d in [("ENTER/N", "Next"), ("P", "Prev"), ("V", "Switch to Actions view"),
                     ("D", "Delete part"), ("E", "Edit bbox"),
                     ("F", "Flag step"), ("K", "Mark OK"), ("O", "Open URL"), ("Q", "Quit")]:
            draw.text((pad,      y), k, fill=TEAL, font=f_sm)
            draw.text((pad + 70, y), d, fill=DIM,  font=f_sm)
            y += 14

    # ══════════════════════════════════════════════════════════════════════════
    #  ACTIONS VIEW
    # ══════════════════════════════════════════════════════════════════════════
    else:
        quads = step.get("action_quadruples", [])
        acts  = step.get("actions", [])

        # ── Actions list ──────────────────────────────────────────────────────
        if acts:
            draw.text((pad, y), "Actions", fill=TEAL, font=f_sm)
            y += 13
            for i, act in enumerate(acts):
                y = wt(f"{i + 1}. {act}", pad + 4, y, f_sm, (160, 160, 160), max_chars=42, max_lines=2)
            y += 4
            y = hsep(y)

        # ── Action quadruples ─────────────────────────────────────────────────
        if mode in (MODE_HANDS_EDIT, MODE_HANDS_SET, MODE_ACTION_EDIT):
            if mode == MODE_HANDS_EDIT:
                prompt = "SET HANDS — select quadruple (1–9)"
            elif mode == MODE_HANDS_SET:
                prompt = "SET HANDS — press digit for count"
            else:
                prompt = "EDIT ACTION TEXT — press number"
            draw.text((pad, y), prompt, fill=ORG, font=f_md)
            y += 18
            draw.text((pad, y), "ESC to cancel", fill=DIM, font=f_sm)
            y += 16
            for i, q in enumerate(quads):
                hands   = q.get("hands", "?")
                hcol    = GRN if hands == 1 else YEL if hands == 2 else RED
                is_sel  = (mode == MODE_HANDS_SET and i == edit_quad_i)
                num_col = (255, 255, 100) if is_sel else ORG
                draw.text((pad,      y), f"{i + 1}.", fill=num_col, font=f_md)
                draw.text((pad + 24, y), f"{q.get('action', '?')} {q.get('component', '')}",
                          fill=FG, font=f_md)
                draw.text((pad + 24, y + 14), f"hands: {hands}", fill=hcol, font=f_sm)
                y += 32
        else:
            if quads:
                draw.text((pad, y), f"Action Quadruples  ({len(quads)})", fill=TEAL, font=f_sm)
                y += 14
                for i, q in enumerate(quads):
                    action    = q.get("action", "?")
                    component = q.get("component", "?")
                    hands     = q.get("hands", "?")
                    tool      = q.get("tool", "")
                    hcol      = GRN if hands == 1 else YEL if hands == 2 else RED

                    draw.text((pad,      y),      f"{i + 1}. [{action}]", fill=ORG, font=f_md)
                    y = wt(component, pad + 24, y + 14, f_sm, FG, max_chars=38, max_lines=1)
                    y -= 2
                    draw.text((pad + 24, y), f"hands: {hands}", fill=hcol, font=f_sm)
                    if tool:
                        draw.text((pad + 90, y), f"tool: {tool[:22]}", fill=DIM, font=f_sm)
                    y += 16
                y += 4
            else:
                draw.text((pad, y), "No action quadruples", fill=DIM, font=f_sm)
                y += 16

        # Controls
        ctrl_y = height - 130
        if ctrl_y > y:
            y = ctrl_y
        y = hsep(y)
        for k, d in [("ENTER/N", "Next"), ("P", "Prev"), ("V", "Switch to Parts view"),
                     ("T", "Edit task name"), ("H", "Toggle hands in quadruple"),
                     ("A", "Edit action text"), ("F", "Flag step"), ("K", "Mark OK"),
                     ("O", "Open URL"), ("Q", "Quit")]:
            draw.text((pad,      y), k, fill=TEAL, font=f_sm)
            draw.text((pad + 70, y), d, fill=DIM,  font=f_sm)
            y += 14

    return __import__("numpy").array(img)[:, :, ::-1]   # RGB → BGR


# ── Main command ───────────────────────────────────────────────────────────────

@app.command()
def review(
    weg_path: str = typer.Argument(None, help="Path to WEG JSON file (omit to pick from list)"),
    steps: str = typer.Option(None, "--steps", "-s", help="Comma-separated step IDs (default: all)"),
    start: int = typer.Option(0, "--start", help="Start at this 0-based step index"),
):
    """
    Interactively review a WEG file step by step.

    Run without arguments to pick from a list of available WEGs.
    Reviewed output is saved as {name}_reviewed.json — original is never modified.

    Left  : annotated image — all parts numbered with colored bboxes.
    Right : step metadata — toggle between PARTS and ACTIONS views with V.

    PARTS view  : D=delete part  E=edit bbox
    ACTIONS view: T=edit task name  H=toggle hands  A=edit action text
    Shared      : V=toggle view  F=flag  K=mark OK  O=open URL  ENTER/N/P=navigate  Q=quit
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        console.print("[red]OpenCV not installed. Run: pip install opencv-python numpy[/red]")
        raise typer.Exit(1)

    from PIL import Image as PILImage

    # ── Pick WEG from list if no path given ───────────────────────────────────
    if weg_path is None:
        search_root = Path("data/preweg")
        if not search_root.exists():
            search_root = Path.cwd()
        all_wegs = sorted(
            p for p in search_root.rglob("*_WEG.json")
            if "_reviewed" not in p.name
        )
        if not all_wegs:
            console.print(f"[red]No *_WEG.json files found under {search_root}[/red]")
            console.print("Pass a path directly:  python scripts/review_weg.py path/to/guide_WEG.json")
            raise typer.Exit(1)

        console.print("\n[bold cyan]Available WEGs[/bold cyan]\n")
        for i, p in enumerate(all_wegs, 1):
            # Build a readable label from path parts
            parts = p.parts
            try:
                # .../appliances/<category>/<brand>/<model>/guides/<file>
                app_idx = list(parts).index("appliances")
                label   = " / ".join(parts[app_idx + 1: app_idx + 4])
            except (ValueError, IndexError):
                label = str(p.parent)
            reviewed = p.with_name(p.stem + "_reviewed.json").exists()
            tick = "[green]✓[/green] " if reviewed else "  "
            console.print(f"  {tick}[cyan]{i:3}.[/cyan]  {label}  [dim]{p.name}[/dim]")

        console.print()
        raw = console.input("[yellow]Select WEG number: [/yellow]").strip()
        try:
            n = int(raw)
            if not 1 <= n <= len(all_wegs):
                raise ValueError
        except ValueError:
            console.print("[red]Invalid selection.[/red]")
            raise typer.Exit(1)
        weg_path = str(all_wegs[n - 1])

    # ── Load ──────────────────────────────────────────────────────────────────
    weg_file = Path(weg_path)
    if not weg_file.exists():
        console.print(f"[red]WEG file not found: {weg_path}[/red]")
        raise typer.Exit(1)

    # Reviewed copy goes next to the original, never overwrites it
    save_file = weg_file.with_name(weg_file.stem + "_reviewed.json")

    console.print(f"\n[bold cyan]Loading:[/bold cyan]  {weg_file}")
    console.print(f"[bold cyan]Saving to:[/bold cyan] {save_file}\n")

    # Start from existing reviewed file if it exists, so edits accumulate
    load_file = save_file if save_file.exists() else weg_file
    with open(load_file, "r", encoding="utf-8") as f:
        weg = json.load(f)

    header    = weg.get("header", {})
    all_steps = weg.get("steps", [])

    if steps:
        ids       = set(int(s.strip()) for s in steps.split(","))
        all_steps = [s for s in all_steps if s.get("step_id") in ids]

    if not all_steps:
        console.print("[red]No steps found.[/red]")
        raise typer.Exit(1)

    base_path = Path.cwd()

    # ── Image cache ───────────────────────────────────────────────────────────
    _cache: dict[str, PILImage.Image] = {}

    def load_pil(rel: str) -> Optional[PILImage.Image]:
        if rel in _cache:
            return _cache[rel]
        for cand in [base_path / rel, Path(rel)]:
            if cand.exists():
                img = PILImage.open(cand).convert("RGB")
                _cache[rel] = img
                return img
        return None

    # ── Annotated image renderer ──────────────────────────────────────────────
    def render_annotated(pil_img: PILImage.Image, step: dict,
                         area_w: int, area_h: int) -> np.ndarray:
        from PIL import ImageDraw
        primary = step.get("primary_part") or {}
        pp_id, pp_name = primary.get("part_id"), primary.get("name")

        img  = pil_img.copy()
        draw = ImageDraw.Draw(img, "RGBA")
        font = _get_font(11)

        for i, part in enumerate(step.get("parts_all", [])):
            bbox = part.get("bbox", {})
            if not all(k in bbox for k in ("x1", "y1", "x2", "y2")):
                continue
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            color = COLORS[i % len(COLORS)]
            is_primary = (part.get("part_id") == pp_id and part.get("name") == pp_name)
            draw.rectangle([(x1, y1), (x2, y2)], outline=color[:3], width=(4 if is_primary else 2))

            label = f"{i + 1}. {part.get('name', '?')} {part.get('confidence', 0):.0%}"
            try:
                tb = draw.textbbox((x1, y1), label, font=font)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
            except AttributeError:
                tw, th = len(label) * 7, 14
            ly = y1 - th - 4 if y1 - th - 4 >= 0 else y2 + 2
            draw.rectangle([(x1, ly), (x1 + tw + 4, ly + th + 4)], fill=color[:3] + (190,))
            draw.text((x1 + 2, ly + 2), label, fill=(255, 255, 255), font=font)

        iw, ih = img.size
        scale  = min(area_w / iw, area_h / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        img    = img.resize((nw, nh), PILImage.LANCZOS)
        canvas = PILImage.new("RGB", (area_w, area_h), (15, 15, 15))
        canvas.paste(img, ((area_w - nw) // 2, (area_h - nh) // 2))
        return np.array(canvas)[:, :, ::-1]

    # ── State ─────────────────────────────────────────────────────────────────
    idx         = max(0, min(start, len(all_steps) - 1))
    mode        = MODE_NORMAL
    view        = VIEW_PARTS
    ann         = {}
    edit_part_i = -1
    edit_quad_i = -1

    WIN_NAME = "WEG Reviewer"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, WIN_W, WIN_H)

    # ── Coordinate mapping ────────────────────────────────────────────────────
    def display_to_img(dx, dy, pil_img):
        area_w = WIN_W - PANEL_W
        area_h = WIN_H - STRIP_H
        iw, ih = pil_img.size
        scale  = min(area_w / iw, area_h / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        ox = int((dx - (area_w - nw) // 2) / scale)
        oy = int((dy - (area_h - nh) // 2) / scale)
        return max(0, min(ox, iw - 1)), max(0, min(oy, ih - 1))

    # ── Mouse callback ────────────────────────────────────────────────────────
    def mouse_cb(event, x, y, flags, param):
        if mode != MODE_EDIT_DRAW:
            return
        area_w = WIN_W - PANEL_W
        area_h = WIN_H - STRIP_H
        xc, yc = max(0, min(x, area_w - 1)), max(0, min(y, area_h - 1))
        if event == cv2.EVENT_LBUTTONDOWN:
            ann.update(drawing=True, x0=xc, y0=yc, x1=xc, y1=yc, done=False)
        elif event == cv2.EVENT_MOUSEMOVE and ann.get("drawing"):
            ann["x1"], ann["y1"] = xc, yc
        elif event == cv2.EVENT_LBUTTONUP and ann.get("drawing"):
            ann["drawing"] = False
            ann["x1"], ann["y1"] = xc, yc
            ann["done"] = True

    cv2.setMouseCallback(WIN_NAME, mouse_cb)

    # ── build_display ─────────────────────────────────────────────────────────
    def build_display(preview_rect=False) -> np.ndarray:
        step   = all_steps[idx]
        area_w = WIN_W - PANEL_W
        area_h = WIN_H - STRIP_H

        primary = step.get("primary_part") or {}
        img_rel = (primary.get("image_path") or
                   next((p.get("image_path") for p in step.get("parts_all", [])
                         if p.get("image_path")), None))
        pil_img = load_pil(img_rel) if img_rel else None

        # Left image
        if pil_img:
            if preview_rect:
                iw, ih = pil_img.size
                scale  = min(area_w / iw, area_h / ih)
                nw, nh = int(iw * scale), int(ih * scale)
                tmp    = pil_img.resize((nw, nh), PILImage.LANCZOS)
                canvas = PILImage.new("RGB", (area_w, area_h), (15, 15, 15))
                canvas.paste(tmp, ((area_w - nw) // 2, (area_h - nh) // 2))
                img_bgr = np.array(canvas)[:, :, ::-1].copy()
                if ann.get("x0") is not None and (ann["x0"] != ann.get("x1", ann["x0"]) or
                                                   ann["y0"] != ann.get("y1", ann["y0"])):
                    cv2.rectangle(img_bgr,
                                  (min(ann["x0"], ann["x1"]), min(ann["y0"], ann["y1"])),
                                  (max(ann["x0"], ann["x1"]), max(ann["y0"], ann["y1"])),
                                  (0, 220, 80), 2)
            else:
                img_bgr = render_annotated(pil_img, step, area_w, area_h)
        else:
            img_bgr = np.full((area_h, area_w, 3), 35, dtype=np.uint8)
            cv2.putText(img_bgr, "Image not found", (20, area_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 60, 60), 2)

        # Strip
        hint_map = {
            MODE_NORMAL:      ("V=toggle view  ENTER/N=next  P=prev  F=flag  K=ok  O=url  Q=quit  "
                               + ("D=delete  E=edit bbox" if view == VIEW_PARTS else "T=task  H=hands  A=action")),
            MODE_DELETE:      "Press 1–9 to delete part   ESC: cancel",
            MODE_EDIT_SELECT: "Press 1–9 to pick part to edit   ESC: cancel",
            MODE_EDIT_DRAW:   "Click+drag to draw new bbox   ESC: cancel",
            MODE_HANDS_EDIT:  "Press 1–9 to select quadruple   ESC: cancel",
            MODE_HANDS_SET:   "Press 1–9 to set number of hands   ESC: cancel",
            MODE_ACTION_EDIT: "Press 1–9 to select quadruple, then type in terminal   ESC: cancel",
        }
        accent_map = {
            MODE_NORMAL:      (0, 180, 160),
            MODE_DELETE:      (40,  60, 200),
            MODE_EDIT_SELECT: (0,  160, 255),
            MODE_EDIT_DRAW:   (0,  160, 255),
            MODE_HANDS_EDIT:  (0,  160, 255),
            MODE_HANDS_SET:   (0,  160, 255),
            MODE_ACTION_EDIT: (0,  160, 255),
        }
        task    = step.get("task_name", "")
        fname   = Path(img_rel).name if img_rel else "—"
        flag_lbl = " [FLAGGED]" if step.get("flagged") else (" [OK]" if step.get("verified") else "")
        strip = make_strip(
            area_w,
            hint   = hint_map[mode],
            label  = f"[{idx + 1}/{len(all_steps)}]  {task}{flag_lbl}  |  {fname}",
            accent = accent_map[mode],
        )
        left = np.vstack([img_bgr, strip])

        right = make_info_panel(
            height      = left.shape[0],
            step        = step,
            header      = header,
            step_idx    = idx,
            total_steps = len(all_steps),
            mode        = mode,
            view        = view,
            edit_quad_i = edit_quad_i,
        )

        lh, rh = left.shape[0], right.shape[0]
        if lh > rh:
            right = np.vstack([right, np.zeros((lh - rh, PANEL_W, 3), dtype=np.uint8)])
        elif rh > lh:
            left  = np.vstack([left,  np.zeros((rh - lh, area_w,  3), dtype=np.uint8)])

        return np.hstack([left, right])

    # ── Save ──────────────────────────────────────────────────────────────────
    def save(msg=""):
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(weg, f, indent=2)
        console.print(f"[bold green]Saved → {save_file.name}{' — ' + msg if msg else ''}[/bold green]")

    # ── Main loop ─────────────────────────────────────────────────────────────
    console.print(
        f"[bold green]Loaded {len(all_steps)} step(s). "
        "V=toggle view  ENTER/N=next  P=prev  D=delete  E=edit  T=task  H=hands  A=action  F=flag  K=ok  O=url  Q=quit"
        "[/bold green]"
    )

    while True:
        cv2.imshow(WIN_NAME, build_display(preview_rect=(mode == MODE_EDIT_DRAW)))
        wait_ms = 20 if mode == MODE_EDIT_DRAW else 0
        key     = cv2.waitKey(wait_ms) & 0xFF

        # ── Finalise bbox draw ────────────────────────────────────────────────
        if mode == MODE_EDIT_DRAW and ann.get("done"):
            x0d = min(ann["x0"], ann["x1"]);  y0d = min(ann["y0"], ann["y1"])
            x1d = max(ann["x0"], ann["x1"]);  y1d = max(ann["y0"], ann["y1"])
            if x1d <= x0d or y1d <= y0d:
                console.print("[yellow]Rectangle too small — cancelled. Try again with E.[/yellow]")
            else:
                step    = all_steps[idx]
                img_rel = ((step.get("primary_part") or {}).get("image_path") or
                           next((p.get("image_path") for p in step.get("parts_all", [])
                                 if p.get("image_path")), None))
                pil_img = load_pil(img_rel) if img_rel else None
                if pil_img is None:
                    console.print("[red]Could not load image — bbox not saved.[/red]")
                else:
                    ox0, oy0 = display_to_img(x0d, y0d, pil_img)
                    ox1, oy1 = display_to_img(x1d, y1d, pil_img)
                    part = step["parts_all"][edit_part_i]
                    part["bbox"] = {"x1": ox0, "y1": oy0, "x2": ox1, "y2": oy1}
                    pp = step.get("primary_part") or {}
                    if pp.get("part_id") == part.get("part_id") and pp.get("name") == part.get("name"):
                        pp["bbox"] = part["bbox"]
                    save(f"bbox updated for '{part.get('name')}' → ({ox0},{oy0})–({ox1},{oy1})")
            ann.clear()
            mode = MODE_NORMAL
            continue

        if key == 0xFF:
            continue

        # ESC cancels current mode
        if key == 27:
            if mode != MODE_NORMAL:
                ann.clear()
                edit_quad_i = -1
                mode = MODE_NORMAL
            continue

        if key == ord('q'):
            break

        # ── Shared keys (any mode / view) ─────────────────────────────────────
        if mode == MODE_NORMAL:

            # Navigate
            if key in (13, ord('n')):
                idx = min(idx + 1, len(all_steps) - 1)
            elif key == ord('p'):
                idx = max(idx - 1, 0)

            # Toggle view
            elif key == ord('v'):
                view = VIEW_ACTIONS if view == VIEW_PARTS else VIEW_PARTS
                mode = MODE_NORMAL   # reset any sub-mode

            # Open URL
            elif key == ord('o'):
                url = header.get("source_url", "")
                if url:
                    webbrowser.open(url)
                    console.print(f"[cyan]Opened: {url}[/cyan]")
                else:
                    console.print("[yellow]No source URL.[/yellow]")

            # Flag / verify
            elif key == ord('f'):
                step = all_steps[idx]
                step["flagged"]  = not step.get("flagged", False)
                step["verified"] = False
                save(f"step {step.get('step_id')} {'flagged' if step['flagged'] else 'unflagged'}")

            elif key == ord('k'):
                step = all_steps[idx]
                step["verified"] = not step.get("verified", False)
                step["flagged"]  = False
                save(f"step {step.get('step_id')} {'verified' if step['verified'] else 'unverified'}")

            # ── E key works in both views ─────────────────────────────────────
            elif key == ord('e'):
                parts = all_steps[idx].get("parts_all", [])
                if not parts:
                    console.print("[yellow]No parts to edit.[/yellow]")
                elif len(parts) == 1:
                    edit_part_i = 0
                    ann.clear()
                    mode = MODE_EDIT_DRAW
                    console.print("[cyan]Draw mode: click and drag on the image to draw new bbox. ESC to cancel.[/cyan]")
                else:
                    mode = MODE_EDIT_SELECT
                    console.print("[cyan]Select part to edit: press 1–9. ESC to cancel.[/cyan]")

            # ── PARTS view keys ───────────────────────────────────────────────
            elif view == VIEW_PARTS:
                if key == ord('d'):
                    if all_steps[idx].get("parts_all"):
                        mode = MODE_DELETE
                    else:
                        console.print("[yellow]No parts to delete.[/yellow]")

            # ── ACTIONS view keys ─────────────────────────────────────────────
            elif view == VIEW_ACTIONS:
                # T — edit task name via terminal
                if key == ord('t'):
                    step = all_steps[idx]
                    current = step.get("task_name", "")
                    console.print(f"[yellow]Current task name:[/yellow] {current}")
                    new_val = console.input("[yellow]New task name (Enter to keep): [/yellow]").strip()
                    if new_val:
                        step["task_name"] = new_val
                        save(f"task_name updated")

                # H — toggle hands in a quadruple
                elif key == ord('h'):
                    if all_steps[idx].get("action_quadruples"):
                        mode = MODE_HANDS_EDIT
                    else:
                        console.print("[yellow]No action quadruples.[/yellow]")

                # A — edit action text in a quadruple
                elif key == ord('a'):
                    if all_steps[idx].get("action_quadruples"):
                        mode = MODE_ACTION_EDIT
                    else:
                        console.print("[yellow]No action quadruples.[/yellow]")

        # ── DELETE mode ───────────────────────────────────────────────────────
        elif mode == MODE_DELETE:
            if ord('1') <= key <= ord('9'):
                n = key - ord('0')
                step  = all_steps[idx]
                parts = step.get("parts_all", [])
                if 1 <= n <= len(parts):
                    removed = parts.pop(n - 1)
                    pp = step.get("primary_part") or {}
                    if (pp.get("part_id") == removed.get("part_id") and
                            pp.get("name") == removed.get("name")):
                        step["primary_part"] = parts[0] if parts else None
                    save(f"deleted '{removed.get('name')}'")
                    mode = MODE_NORMAL

        # ── EDIT SELECT mode ──────────────────────────────────────────────────
        elif mode == MODE_EDIT_SELECT:
            if ord('1') <= key <= ord('9'):
                n = key - ord('0')
                parts = all_steps[idx].get("parts_all", [])
                if 1 <= n <= len(parts):
                    edit_part_i = n - 1
                    ann.clear()
                    mode = MODE_EDIT_DRAW
                    console.print(f"[cyan]Editing bbox for '{parts[edit_part_i].get('name')}'. Click and drag to draw. ESC to cancel.[/cyan]")

        # ── HANDS EDIT mode — select which quadruple ──────────────────────────
        elif mode == MODE_HANDS_EDIT:
            if ord('1') <= key <= ord('9'):
                n     = key - ord('0')
                quads = all_steps[idx].get("action_quadruples", [])
                if 1 <= n <= len(quads):
                    edit_quad_i = n - 1
                    mode        = MODE_HANDS_SET
                    q = quads[edit_quad_i]
                    console.print(
                        f"[cyan]Quadruple {n}: [{q.get('action')}] {q.get('component')}  "
                        f"current hands={q.get('hands', '?')}. Press 1–9 to set new count.[/cyan]"
                    )

        # ── HANDS SET mode — press digit to set count ─────────────────────────
        elif mode == MODE_HANDS_SET:
            if ord('1') <= key <= ord('9'):
                n     = key - ord('0')
                step  = all_steps[idx]
                quads = step.get("action_quadruples", [])
                if 0 <= edit_quad_i < len(quads):
                    quads[edit_quad_i]["hands"] = n
                    save(f"hands set to {n} for quadruple {edit_quad_i + 1}")
                edit_quad_i = -1
                mode        = MODE_NORMAL

        # ── ACTION EDIT mode ──────────────────────────────────────────────────
        elif mode == MODE_ACTION_EDIT:
            if ord('1') <= key <= ord('9'):
                n     = key - ord('0')
                step  = all_steps[idx]
                quads = step.get("action_quadruples", [])
                if 1 <= n <= len(quads):
                    q = quads[n - 1]
                    console.print(f"[yellow]Quadruple {n}:[/yellow] action=[{q.get('action')}]  "
                                  f"component=[{q.get('component')}]")
                    field = console.input("[yellow]Edit field (action/component/tool/precise_action): [/yellow]").strip()
                    if field in q or field in ("action", "component", "tool", "precise_action"):
                        current = q.get(field, "")
                        console.print(f"[yellow]Current '{field}':[/yellow] {current}")
                        new_val = console.input(f"[yellow]New value (Enter to keep): [/yellow]").strip()
                        if new_val:
                            q[field] = new_val
                            save(f"'{field}' updated in quadruple {n}")
                    mode = MODE_NORMAL

    cv2.destroyAllWindows()


if __name__ == "__main__":
    app()
