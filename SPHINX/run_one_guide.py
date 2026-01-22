import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image, ImageDraw, ImageFont

from SPHINX import SPHINXModel


# ============================================================
# Your helpers (kept, only minor adjustments)
# ============================================================

def clean_model_text(text: str) -> str:
    # Fix invalid JSON like confidence: 0.0-1.0
    text = re.sub(r'("confidence"\s*:\s*)0\.0\s*-\s*1\.0', r"\g<1>0.5", text)
    return text


def find_first_balanced_block(text: str, open_ch: str, close_ch: str):
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    return None


def extract_first_json(text: str):
    text = clean_model_text(str(text)).strip()

    # direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    obj = find_first_balanced_block(text, "{", "}")
    if obj:
        try:
            return json.loads(clean_model_text(obj))
        except Exception:
            pass

    arr = find_first_balanced_block(text, "[", "]")
    if arr:
        try:
            return json.loads(clean_model_text(arr))
        except Exception:
            pass

    raise ValueError(f"Could not parse JSON from model output:\n{text[:1000]}")


def normalize_parts_payload(data):
    if isinstance(data, list):
        return {"parts": data}
    if isinstance(data, dict):
        if "parts" in data and isinstance(data["parts"], list):
            return data
        if "name" in data and "bbox_xyxy" in data:
            return {"parts": [data]}
        return {"parts": []}
    return {"parts": []}


def maybe_normalized_to_pixels(bbox, w, h):
    # If bbox is [0..1] normalized, convert to pixels
    if (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(v, (int, float)) for v in bbox)
        and all(0.0 <= float(v) <= 1.0 for v in bbox)
    ):
        x1, y1, x2, y2 = map(float, bbox)
        return [x1 * w, y1 * h, x2 * w, y2 * h]
    return bbox


def clamp_bbox_xyxy(b, w, h):
    x1, y1, x2, y2 = b
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w - 1))
    y2 = max(0, min(int(round(y2)), h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def draw_bboxes_and_save(image_path: Path, parts: List[Dict[str, Any]], out_path: Path, line_width: int = 4):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for p in parts:
        name = str(p.get("name", "")).strip()
        bbox = p.get("bbox_xyxy", None)
        conf = p.get("confidence", None)
        if not name or not isinstance(bbox, list) or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=line_width)

        label = name
        if isinstance(conf, (int, float)):
            label = f"{name} ({conf:.2f})"

        pad = 2
        if font is not None:
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
            tw, th = right - left, bottom - top
        else:
            tw, th = (8 * len(label), 14)

        lx1, ly1 = x1, max(0, y1 - th - 2 * pad)
        lx2, ly2 = x1 + tw + 2 * pad, ly1 + th + 2 * pad
        draw.rectangle([lx1, ly1, lx2, ly2], fill="red")
        draw.text((lx1 + pad, ly1 + pad), label, fill="white", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def build_prompt(step_text: str, force_one_part: bool):
    part_count_rule = "Return EXACTLY 1 part." if force_one_part else "Return 1 to 4 parts maximum."
    return f"""
Return VALID JSON ONLY. No extra text. No markdown. No explanations.

Task: Identify the device PART being worked on in this repair step and give a bounding box for it in the image.

Rules:
- Parts are physical components (NOT tools, NOT actions, NOT generic words like "device").
- Only include parts of the device we work on in this step.
- {part_count_rule}
- bbox_xyxy MUST be 4 INTEGERS in PIXELS: [x1, y1, x2, y2]
- confidence MUST be ONE FLOAT between 0 and 1
- DO NOT output placeholders like [PART], x1, y1, x2, y2, [confidence score]. Fill real values.

Return JSON in exactly this shape (example with REAL values):
{{"parts":[{{"name":PART,"bbox_xyxy":[x1,y1,x2,y2],"confidence":confidence}}]}}

Step text:
{step_text}
""".strip()


def ask_model_for_json(model, img, prompt: str, temperature: float):
    qas = [[prompt, None]]
    return model.generate_response(
        qas,
        img,
        max_gen_len=512,
        temperature=temperature,
        top_p=0.9,
        seed=0,
    )


def run_one_step(model, step_text: str, image_path: Path, force_one_part: bool = True, debug_dir: Optional[Path] = None):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    prompt = build_prompt(step_text, force_one_part)

    raw = ask_model_for_json(model, img, prompt, temperature=0.2)

    if debug_dir is None:
        debug_dir = Path("outputs/sphinx_debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    raw_path = debug_dir / f"{image_path.stem}_raw.txt"
    raw_path.write_text(str(raw), encoding="utf-8")

    try:
        parsed = extract_first_json(raw)
    except Exception:
        strict_prompt = "ONLY OUTPUT JSON. " + prompt
        raw2 = ask_model_for_json(model, img, strict_prompt, temperature=0.0)
        (debug_dir / f"{image_path.stem}_raw_retry.txt").write_text(str(raw2), encoding="utf-8")
        parsed = extract_first_json(raw2)
        raw = raw2  # keep last raw

    data = normalize_parts_payload(parsed)

    parts_out = []
    for p in data.get("parts", []):
        if not isinstance(p, dict):
            continue

        name = str(p.get("name", "")).strip()
        bbox = p.get("bbox_xyxy", None)
        conf = p.get("confidence", 0.0)

        if not name or not isinstance(bbox, list) or len(bbox) != 4:
            continue

        # Guard common placeholder failure
        if name in {"[PART]", "PART", "part"}:
            continue

        bbox = maybe_normalized_to_pixels(bbox, w, h)
        bbox = clamp_bbox_xyxy(bbox, w, h)

        try:
            conf = float(conf)
        except Exception:
            conf = 0.0

        parts_out.append({"name": name, "bbox_xyxy": bbox, "confidence": conf})

    return {
        "image_path": str(image_path),
        "image_size": [w, h],
        "parts": parts_out,
        "raw_model_output": str(raw),
        "raw_saved_to": str(raw_path),
    }


# ============================================================
# NEW: Run Sphinx on an entire pre-WEG guide JSON
# ============================================================

def load_preweg(preweg_path: Union[str, Path]) -> Dict[str, Any]:
    preweg_path = Path(preweg_path)
    return json.loads(preweg_path.read_text(encoding="utf-8"))


def process_preweg_guide(
    preweg_json_path: Union[str, Path],
    output_json_path: Union[str, Path],
    *,
    sphinx_ckpt: Union[str, Path] = "checkpoints/sphinx_tiny",
    force_one_part: bool = True,
    annotate_images: bool = False,
    resume: bool = True,
):
    preweg_json_path = Path(preweg_json_path)
    output_json_path = Path(output_json_path)

    preweg = load_preweg(preweg_json_path)

    # Load model ONCE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SPHINXModel.from_pretrained(pretrained_path=str(sphinx_ckpt), with_visual=True).to(device)
    model.eval()

    guide_id = preweg.get("guide_id")
    title = preweg.get("title")
    steps = preweg.get("steps", []) or []

    out_dir = output_json_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # resume
    if resume and output_json_path.exists():
        out = json.loads(output_json_path.read_text(encoding="utf-8"))
        done = {(r.get("step_index"), r.get("image_index")) for r in out.get("predictions", [])}
    else:
        out = {
            "guide_id": guide_id,
            "title": title,
            "source_preweg_json": str(preweg_json_path),
            "sphinx_checkpoint": str(sphinx_ckpt),
            "force_one_part": force_one_part,
            "predictions": [],  # flat list (easier for later processing)
        }
        done = set()

    debug_dir = out_dir / "debug_raw"
    ann_dir = out_dir / "annotated"  # optional

    for step in steps:
        step_index = step.get("step_index")
        step_id_ifixit = step.get("step_id_ifixit")
        step_text = step.get("full_description", "") or ""
        images = step.get("images", []) or []

        for img_i, img_info in enumerate(images, start=1):
            saved_path = img_info.get("saved_path")
            src_url = img_info.get("src_url")

            key = (step_index, img_i)
            if key in done:
                continue

            record = {
                "step_index": step_index,
                "step_id_ifixit": step_id_ifixit,
                "full_description": step_text,
                "image_index": img_i,
                "saved_path": saved_path,
                "src_url": src_url,
                "parts": [],
                "image_size": None,
                "raw_saved_to": None,
                "error": None,
                "annotated_path": None,
            }

            try:
                if not saved_path:
                    raise ValueError("Missing saved_path in pre-WEG step image entry.")
                img_path = Path(saved_path)
                if not img_path.exists():
                    raise FileNotFoundError(f"Image file not found: {img_path}")

                # Run sphinx for this step+image
                res = run_one_step(
                    model,
                    step_text=step_text,
                    image_path=img_path,
                    force_one_part=force_one_part,
                    debug_dir=debug_dir,
                )

                record["parts"] = res["parts"]
                record["image_size"] = res["image_size"]
                record["raw_saved_to"] = res["raw_saved_to"]

                # optionally save annotated image
                if annotate_images:
                    ann_path = ann_dir / f"guide_{guide_id}_step_{step_index}_img_{img_i}_annotated.jpg"
                    draw_bboxes_and_save(img_path, res["parts"], ann_path)
                    record["annotated_path"] = str(ann_path)

            except Exception as e:
                record["error"] = str(e)

            out["predictions"].append(record)

            # Save after each image (safe for cluster)
            output_json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    return output_json_path


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SPHINX on all steps+images in a pre-WEG guide JSON.")
    parser.add_argument("--preweg", type=str, required=True, help="Path to pre-WEG guide JSON.")
    parser.add_argument("--out", type=str, required=True, help="Path to output JSON file.")
    parser.add_argument("--ckpt", type=str, default="checkpoints/sphinx_tiny", help="SPHINX checkpoint dir.")
    parser.add_argument("--force_one_part", action="store_true", help="Force exactly 1 part per image.")
    parser.add_argument("--multi_part", action="store_true", help="Allow up to 4 parts per image (overrides force_one_part).")
    parser.add_argument("--annotate", action="store_true", help="Save annotated images.")
    parser.add_argument("--no_resume", action="store_true", help="Disable resume; overwrite output JSON.")
    args = parser.parse_args()

    force_one = True
    if args.multi_part:
        force_one = False
    elif args.force_one_part:
        force_one = True

    out_path = Path(args.out)
    if args.no_resume and out_path.exists():
        out_path.unlink()

    saved = process_preweg_guide(
        args.preweg,
        out_path,
        sphinx_ckpt=args.ckpt,
        force_one_part=force_one,
        annotate_images=args.annotate,
        resume=(not args.no_resume),
    )
    print(f"[OK] Saved results to: {saved}")
