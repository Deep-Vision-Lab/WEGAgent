import json
import re
from pathlib import Path

from SPHINX import SPHINXModel
from PIL import Image, ImageDraw, ImageFont
import torch


def clean_model_text(text: str) -> str:
    # Fix invalid JSON like confidence: 0.0-1.0
    text = re.sub(r'("confidence"\s*:\s*)0\.0\s*-\s*1\.0', r"\g<1>0.5", text)
    return text


def find_first_balanced_block(text: str, open_ch: str, close_ch: str):
    """
    Return the first balanced {...} or [...] block, or None.
    """
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
                return text[start : i + 1]
    return None


def extract_first_json(text: str):
    """
    Extract first JSON object or array from text.
    """
    text = clean_model_text(text).strip()

    # Direct parse first
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

    raise ValueError(f"Could not parse JSON from model output:\n{text}")


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


def draw_bboxes_and_save(image_path: Path, parts: list[dict], out_path: Path, line_width: int = 4):
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


def run_one_step(model, step_text: str, image_path: Path, force_one_part: bool = True):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    prompt = build_prompt(step_text, force_one_part)

    # 1st try
    raw = ask_model_for_json(model, img, prompt, temperature=0.2)

    # Always save raw output for debugging
    out_dir = Path("outputs/sphinx_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{image_path.stem}_raw.txt"
    raw_path.write_text(str(raw), encoding="utf-8")

    # Parse with retry
    try:
        parsed = extract_first_json(raw)
    except Exception:
        # 2nd try: force strictness
        strict_prompt = "ONLY OUTPUT JSON. " + prompt
        raw2 = ask_model_for_json(model, img, strict_prompt, temperature=0.0)
        (out_dir / f"{image_path.stem}_raw_retry.txt").write_text(str(raw2), encoding="utf-8")
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
        "raw_model_output": raw,
        "raw_saved_to": str(raw_path),
    }


if __name__ == "__main__":
    pretrained_path = Path("checkpoints/sphinx_tiny")
    image_path = Path("img1.jpg")
    step_text = "- Plug in the iron box and electric cord and bottom surface if the iron heats up or not."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SPHINXModel.from_pretrained(pretrained_path=str(pretrained_path), with_visual=True).to(device)
    model.eval()

    result = run_one_step(model, step_text, image_path, force_one_part=True)
    print(json.dumps(result, indent=2))

    # Save JSON
    out_dir = Path("outputs/sphinx_test")
    json_path = out_dir / f"{image_path.stem}_parts.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save annotated image
    img_out_path = out_dir / f"{image_path.stem}_annotated.jpg"
    draw_bboxes_and_save(image_path, result["parts"], img_out_path)

    print("Saved raw model output to:", Path(result["raw_saved_to"]).resolve())
    print("Saved JSON to:", json_path.resolve())
    print("Saved annotated image to:", img_out_path.resolve())
