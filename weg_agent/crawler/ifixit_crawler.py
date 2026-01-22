import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import requests

BASE_URL = "https://www.ifixit.com/api/2.0"

# These are the allowed --device values in CLI.
# The value is the default keyword passed to /suggest if user doesn't override.
DEVICE_KEYWORDS = {
    "refrigerators": "Refrigerator",
    "washing_machines": "Washer",
    "dishwashers": "Dishwasher",
    "dryers": "Dryer",
    "clothes_iron": "Clothes iron",
}


def fetch_json(url: str, params: Optional[Dict] = None) -> Dict:
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def fetch_guide(guide_id: int) -> Dict:
    url = f"{BASE_URL}/guides/{guide_id}"
    return fetch_json(url)


def search_guides(keyword: str, limit: int) -> List[Dict]:
    """
    Use /suggest with doctypes=guide.
    Returns list of dicts: {guideid, title, category, url}.
    """
    url = f"{BASE_URL}/suggest/{keyword}"
    params = {"doctypes": "guide"}
    data = fetch_json(url, params=params)

    results = []
    for r in data.get("results", []):
        if r.get("dataType") != "guide":
            continue
        results.append(
            {
                "guideid": r["guideid"],
                "title": r["title"],
                "category": r.get("category"),
                "url": r.get("url"),
            }
        )
        if len(results) >= limit:
            break
    return results


def slugify(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^\w\-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unknown"


def sanitize_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = s.strip()
    return s or None


def build_device_brand_type_paths(
    preweg_root: Path,
    device_key: str,
    category_name: str,
) -> Tuple[Path, Path]:
    """
    Creates:
      preweg_root/appliances/<device_key>/<brand>/<type>/guides
      preweg_root/appliances/<device_key>/<brand>/<type>/imgs
    """
    brand = category_name.split()[0] if category_name else "unknown"
    brand_slug = slugify(brand)
    type_slug = slugify(category_name) if category_name else f"{brand_slug}_type"

    base = preweg_root / "appliances" / device_key / brand_slug / type_slug
    guides_dir = base / "guides"
    imgs_dir = base / "imgs"
    guides_dir.mkdir(parents=True, exist_ok=True)
    imgs_dir.mkdir(parents=True, exist_ok=True)
    return guides_dir, imgs_dir


def normalize_media_list(value) -> List[Dict]:
    """
    Normalize any media field which might be:
      - dict
      - list[dict]
      - dict-of-dicts
      - dict with {"type": "...", "data": [...]}
    into list[dict].
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [m for m in value if isinstance(m, dict)]
    if isinstance(value, dict):
        # style: {"type": "image", "data": [ {standard:...}, ... ]}
        if isinstance(value.get("data"), list):
            out: List[Dict] = []
            base = {k: v for k, v in value.items() if k != "data"}
            for img in value["data"]:
                if not isinstance(img, dict):
                    continue
                m = dict(base)
                m["image"] = img
                out.append(m)
            return out

        # single media dict
        if "type" in value or "image" in value or "id" in value:
            return [value]

        # dict-of-dicts
        return [v for v in value.values() if isinstance(v, dict)]
    return []


def pick_image_url(media: Dict) -> Optional[str]:
    """
    Extract best image URL from different formats.
    """
    if not isinstance(media, dict):
        return None

    img = media.get("image")
    if isinstance(img, dict):
        url = img.get("standard") or img.get("medium") or img.get("original") or img.get("thumbnail")
        if url:
            return url

    return (
        media.get("standard")
        or media.get("medium")
        or media.get("original")
        or media.get("thumbnail")
        or media.get("url")
        or media.get("src")
        or media.get("href")
    )


def download_image(url: str, dest_path: Path) -> None:
    """
    Download an image to dest_path if it doesn't exist yet.
    """
    if dest_path.exists():
        return
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARN] Failed to download image {url}: {e}")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(resp.content)


def extract_relevant_urls_from_guide(guide: Dict) -> List[str]:
    """
    Collect URLs of videos / external links from guide-level and step media.
    """
    urls = set()

    top_media = normalize_media_list(guide.get("media"))
    for m in top_media:
        mtype = (m.get("type") or "").lower()
        if "video" in mtype or "youtube" in mtype or "external" in mtype:
            url = m.get("url") or m.get("src") or m.get("href")
            if url:
                urls.add(url)

    for step in guide.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        media_items = normalize_media_list(step.get("media"))
        for m in media_items:
            mtype = (m.get("type") or "").lower()
            if "video" in mtype or "youtube" in mtype or "external" in mtype:
                url = m.get("url") or m.get("src") or m.get("href")
                if url:
                    urls.add(url)

    return list(urls)


def convert_guide_to_preweg(
    guide: Dict,
    guides_dir: Path,
    imgs_dir: Path,
) -> Path:
    """
    Build the final pre-WEG JSON structure for one guide.
    Saves the JSON and returns its path.
    """
    guide_id = guide.get("guideid") or guide.get("guide_id")
    source_url = guide.get("url")
    title = sanitize_text(guide.get("title"))
    device = sanitize_text(guide.get("device") or guide.get("subject"))
    summary = sanitize_text(guide.get("summary") or guide.get("introduction"))

    toolbox: List[str] = []
    for t in guide.get("tools", []) or []:
        if not isinstance(t, dict):
            continue
        text = t.get("text") or (t.get("tool") or {}).get("name")
        if text:
            toolbox.append(text)

    parts: List[str] = []
    for p in guide.get("parts", []) or []:
        if not isinstance(p, dict):
            continue
        name = p.get("text") or (p.get("part") or {}).get("name")
        if name:
            parts.append(name)

    relevant_urls = extract_relevant_urls_from_guide(guide)

    steps_out: List[Dict] = []
    global_images: List[Dict] = []

    steps = guide.get("steps", []) or []
    for idx, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue

        step_id_ifixit = step.get("stepid")

        # Build full_description from "lines"
        lines = step.get("lines", []) or []
        text_chunks: List[str] = []
        for line in lines:
            if not isinstance(line, dict):
                continue
            txt = line.get("text_raw") or line.get("text") or ""
            if not txt:
                continue
            txt = txt.strip()
            is_bullet = bool(line.get("bullet")) or (str(line.get("style", "")).lower() == "bullet")
            text_chunks.append(f"- {txt}" if is_bullet else txt)

        full_description = "\n".join(text_chunks).strip()

        # Step images from step["media"]
        step_images: List[Dict] = []
        media_items = normalize_media_list(step.get("media"))

        image_index = 1
        for media in media_items:
            mtype = (media.get("type") or "").lower()
            if not ("image" in mtype or "photo" in mtype or "thumbnail" in mtype):
                continue

            src_url = pick_image_url(media)
            if not src_url:
                continue

            img_filename = f"guide_{guide_id}_step_{idx}_{image_index}.jpg"
            img_path = imgs_dir / img_filename
            download_image(src_url, img_path)
            saved_path = img_path.as_posix()

            step_images.append({"src_url": src_url, "saved_path": saved_path})

            caption = None
            if isinstance(media.get("image"), dict):
                caption = media["image"].get("caption")
            if not caption:
                caption = media.get("caption")

            global_images.append({"src_url": src_url, "saved_path": saved_path, "caption": caption or None})

            image_index += 1

        steps_out.append(
            {
                "step_index": idx,
                "step_id_ifixit": step_id_ifixit,
                "full_description": full_description,
                "images": step_images,
            }
        )

    preweg = {
        "guide_id": guide_id,
        "source_url": source_url,
        "title": title,
        "device": device,
        "summary": summary,
        "toolbox": toolbox,
        "parts": parts,
        "relevant_urls": relevant_urls,
        "steps": steps_out,
        "images": global_images,
    }

    title_for_filename = title or f"guide_{guide_id}"
    json_name = f"{guide_id}_{slugify(title_for_filename)}.json"
    json_path = guides_dir / json_name
    json_path.write_text(json.dumps(preweg, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Saved pre-WEG JSON: {json_path.as_posix()}")
    return json_path


def crawl(
    preweg_root: Path,
    device_key: str,
    keyword: str,
    limit: int,
) -> List[Path]:
    """
    Search and download up to `limit` guides, save pre-WEG JSONs.
    Returns list of saved guide json paths.
    """
    preweg_root = Path(preweg_root)
    device_key = device_key.strip().lower()

    if device_key not in DEVICE_KEYWORDS:
        raise ValueError(
            f"Invalid device_key={device_key!r}. Choose one of: {list(DEVICE_KEYWORDS.keys())}"
        )

    print(f"[INFO] Searching guides for keyword={keyword!r}, device={device_key!r}, limit={limit}")
    metas = search_guides(keyword, limit)
    print(f"[INFO] Found {len(metas)} guide candidates")

    saved_paths: List[Path] = []
    for meta in metas:
        guide_id = meta["guideid"]
        print(f"[INFO] Fetching guide {guide_id} ...")
        guide = fetch_guide(guide_id)

        category = (guide.get("category") or meta.get("category") or f"guide_{guide_id}").strip()
        guides_dir, imgs_dir = build_device_brand_type_paths(preweg_root, device_key, category)

        saved = convert_guide_to_preweg(guide, guides_dir, imgs_dir)
        saved_paths.append(saved)

    return saved_paths
