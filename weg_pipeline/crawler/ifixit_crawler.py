"""
iFixit Crawler - Fetches repair guides and converts to pre-WEG format.

This module is adapted from WEGAgent with improvements for WEGv2.
"""
import json
import re
from pathlib import Path
from typing import Any, Optional

import requests
from rich.console import Console

console = Console()

BASE_URL = "https://www.ifixit.com/api/2.0"

# Supported device categories
DEVICE_KEYWORDS = {
    "refrigerators": "Refrigerator",
    "washing_machines": "Washer",
    "dishwashers": "Dishwasher",
    "dryers": "Dryer",
    "clothes_iron": "Clothes iron",
    "microwave": "Microwave",
    "oven": "Oven",
    "vacuum": "Vacuum",
}


def fetch_json(url: str, params: Optional[dict] = None, timeout: int = 20) -> dict:
    """Fetch JSON from URL with error handling."""
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_guide(guide_id: int) -> dict:
    """Fetch a single guide by ID."""
    url = f"{BASE_URL}/guides/{guide_id}"
    return fetch_json(url)


def search_guides(keyword: str, limit: int = 10) -> list[dict]:
    """
    Search for guides using iFixit's suggest API.

    Returns:
        List of dicts with: guideid, title, category, url
    """
    url = f"{BASE_URL}/suggest/{keyword}"
    params = {"doctypes": "guide"}
    data = fetch_json(url, params=params)

    results = []
    for r in data.get("results", []):
        if r.get("dataType") != "guide":
            continue
        results.append({
            "guideid": r["guideid"],
            "title": r["title"],
            "category": r.get("category"),
            "url": r.get("url"),
        })
        if len(results) >= limit:
            break
    return results


def search_guides_paginated(keyword: str, offset: int = 0, page_size: int = 20) -> list[dict]:
    """
    Search for guides using iFixit's paginated /search/ endpoint.
    Used when suggest results are exhausted.

    Returns:
        List of dicts with: guideid, title, category, url
    """
    url = f"{BASE_URL}/search/{keyword}"
    params = {"doctypes": "guide", "limit": page_size, "offset": offset}
    try:
        data = fetch_json(url, params=params)
    except Exception:
        return []

    results = []
    for r in data.get("results", []):
        if r.get("dataType") != "guide":
            continue
        guide_id = r.get("guideid")
        if guide_id is None:
            continue
        results.append({
            "guideid": guide_id,
            "title": r.get("title", ""),
            "category": r.get("category"),
            "url": r.get("url"),
        })
    return results


def slugify(name: str) -> str:
    """Convert string to filesystem-safe slug."""
    name = name.strip().lower()
    name = re.sub(r"[^\w\-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unknown"


def sanitize_text(s: Optional[str]) -> Optional[str]:
    """Clean and validate text."""
    if s is None:
        return None
    s = s.strip()
    return s or None


def normalize_media_list(value: Any) -> list[dict]:
    """
    Normalize various media field formats to list of dicts.
    
    Handles:
    - list[dict]
    - dict with "data" key
    - single dict
    - dict of dicts
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [m for m in value if isinstance(m, dict)]
    if isinstance(value, dict):
        if isinstance(value.get("data"), list):
            out = []
            base = {k: v for k, v in value.items() if k != "data"}
            for img in value["data"]:
                if isinstance(img, dict):
                    m = dict(base)
                    m["image"] = img
                    out.append(m)
            return out
        if "type" in value or "image" in value or "id" in value:
            return [value]
        return [v for v in value.values() if isinstance(v, dict)]
    return []


def pick_image_url(media: dict) -> Optional[str]:
    """Extract best image URL from media dict."""
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


def download_image(url: str, dest_path: Path, timeout: int = 30) -> bool:
    """
    Download image to dest_path if it doesn't exist.
    
    Returns:
        True if downloaded or already exists, False on failure
    """
    if dest_path.exists():
        return True
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(resp.content)
        return True
    except Exception as e:
        console.print(f"[yellow][WARN] Failed to download image {url}: {e}[/yellow]")
        return False


def build_output_paths(
    output_root: Path,
    device_key: str,
    category_name: str,
) -> tuple[Path, Path]:
    """
    Create output directory structure.
    
    Structure: output_root/appliances/<device>/<brand>/<type>/guides|imgs
    
    Returns:
        (guides_dir, imgs_dir)
    """
    brand = category_name.split()[0] if category_name else "unknown"
    brand_slug = slugify(brand)
    type_slug = slugify(category_name) if category_name else f"{brand_slug}_type"

    base = output_root / "appliances" / device_key / brand_slug / type_slug
    guides_dir = base / "guides"
    imgs_dir = base / "imgs"
    guides_dir.mkdir(parents=True, exist_ok=True)
    imgs_dir.mkdir(parents=True, exist_ok=True)
    return guides_dir, imgs_dir


def convert_guide_to_preweg(
    guide: dict,
    guides_dir: Path,
    imgs_dir: Path,
) -> Path:
    """
    Convert iFixit guide to pre-WEG JSON format.
    
    Returns:
        Path to saved JSON file
    """
    guide_id = guide.get("guideid") or guide.get("guide_id")
    source_url = guide.get("url")
    title = sanitize_text(guide.get("title"))
    device = sanitize_text(guide.get("device") or guide.get("subject"))
    summary = sanitize_text(guide.get("summary") or guide.get("introduction"))

    # Extract toolbox
    toolbox = []
    for t in guide.get("tools", []) or []:
        if isinstance(t, dict):
            text = t.get("text") or (t.get("tool") or {}).get("name")
            if text:
                toolbox.append(text)

    # Extract parts
    parts = []
    for p in guide.get("parts", []) or []:
        if isinstance(p, dict):
            name = p.get("text") or (p.get("part") or {}).get("name")
            if name:
                parts.append(name)

    # Process steps
    steps_out = []
    global_images = []

    for idx, step in enumerate(guide.get("steps", []) or [], start=1):
        if not isinstance(step, dict):
            continue

        step_id_ifixit = step.get("stepid")

        # Build description from lines
        text_chunks = []
        for line in step.get("lines", []) or []:
            if not isinstance(line, dict):
                continue
            txt = line.get("text_raw") or line.get("text") or ""
            if not txt:
                continue
            txt = txt.strip()
            is_bullet = bool(line.get("bullet")) or str(line.get("style", "")).lower() == "bullet"
            text_chunks.append(f"- {txt}" if is_bullet else txt)

        full_description = "\n".join(text_chunks).strip()

        # Process step images
        step_images = []
        media_items = normalize_media_list(step.get("media"))

        for img_idx, media in enumerate(media_items, start=1):
            mtype = (media.get("type") or "").lower()
            if not any(t in mtype for t in ("image", "photo", "thumbnail")):
                continue

            src_url = pick_image_url(media)
            if not src_url:
                continue

            img_filename = f"guide_{guide_id}_step_{idx}_{img_idx}.jpg"
            img_path = imgs_dir / img_filename

            if download_image(src_url, img_path):
                saved_path = img_path.as_posix()
                step_images.append({"src_url": src_url, "saved_path": saved_path})

                caption = None
                if isinstance(media.get("image"), dict):
                    caption = media["image"].get("caption")
                if not caption:
                    caption = media.get("caption")

                global_images.append({
                    "src_url": src_url,
                    "saved_path": saved_path,
                    "caption": caption,
                })

        steps_out.append({
            "step_index": idx,
            "step_id_ifixit": step_id_ifixit,
            "full_description": full_description,
            "images": step_images,
        })

    # Build pre-WEG structure
    preweg = {
        "guide_id": guide_id,
        "source_url": source_url,
        "title": title,
        "device": device,
        "summary": summary,
        "toolbox": toolbox,
        "parts": parts,
        "relevant_urls": [],
        "steps": steps_out,
        "images": global_images,
    }

    # Save JSON
    title_slug = slugify(title) if title else f"guide_{guide_id}"
    json_name = f"{guide_id}_{title_slug}.json"
    json_path = guides_dir / json_name
    json_path.write_text(json.dumps(preweg, indent=2, ensure_ascii=False), encoding="utf-8")

    console.print(f"[green][OK] Saved pre-WEG JSON: {json_path.as_posix()}[/green]")
    return json_path


def crawl_guide(
    guide_id: int,
    output_root: Path,
    device_key: str = "appliances",
) -> Path:
    """
    Crawl a single guide by ID.
    
    Returns:
        Path to saved pre-WEG JSON
    """
    console.print(f"[cyan]Fetching guide {guide_id}...[/cyan]")
    guide = fetch_guide(guide_id)

    category = guide.get("category", f"guide_{guide_id}").strip()
    guides_dir, imgs_dir = build_output_paths(output_root, device_key, category)

    return convert_guide_to_preweg(guide, guides_dir, imgs_dir)


def get_existing_guide_ids(output_root: Path) -> set[int]:
    """
    Scan output_root for all already-crawled guide IDs.
    Guide JSON filenames start with the guide ID (e.g. 113759_title.json).
    """
    existing = set()
    for json_file in output_root.rglob("*.json"):
        try:
            guide_id = int(json_file.stem.split("_")[0])
            existing.add(guide_id)
        except (ValueError, IndexError):
            pass
    return existing


def crawl_guides(
    output_root: Path,
    device_key: str,
    keyword: Optional[str] = None,
    limit: int = 5,
) -> list[Path]:
    """
    Search and crawl multiple guides.

    Skips any guide whose ID already exists locally and fetches
    additional candidates until the requested limit is reached.

    Args:
        output_root: Root directory for output
        device_key: Device category (from DEVICE_KEYWORDS)
        keyword: Search keyword (uses default for device if not provided)
        limit: Maximum number of NEW guides to crawl

    Returns:
        List of paths to saved pre-WEG JSONs
    """
    device_key = device_key.strip().lower()

    if device_key not in DEVICE_KEYWORDS:
        raise ValueError(
            f"Invalid device_key={device_key!r}. "
            f"Choose from: {list(DEVICE_KEYWORDS.keys())}"
        )

    keyword = keyword or DEVICE_KEYWORDS[device_key]

    existing_ids = get_existing_guide_ids(output_root)
    if existing_ids:
        console.print(f"[cyan]Found {len(existing_ids)} already-crawled guide(s) locally — will skip them[/cyan]")

    console.print(f"[cyan]Searching guides: keyword={keyword!r}, device={device_key}, limit={limit}[/cyan]")

    # Phase 1: suggest endpoint (fast, no pagination)
    suggest_results = search_guides(keyword, limit + len(existing_ids) + 10)
    seen_ids: set[int] = set()
    candidate_queue: list[dict] = []
    for m in suggest_results:
        gid = m["guideid"]
        if gid not in seen_ids:
            seen_ids.add(gid)
            candidate_queue.append(m)

    saved_paths: list[Path] = []
    page_offset = 0
    page_size   = 20
    exhausted   = False

    while len(saved_paths) < limit and not exhausted:
        # Refill queue from paginated search when suggest results run out
        if not candidate_queue:
            console.print(f"[cyan]Fetching more candidates (offset={page_offset})…[/cyan]")
            page = search_guides_paginated(keyword, offset=page_offset, page_size=page_size)
            page_offset += page_size
            if not page:
                exhausted = True
                break
            for m in page:
                gid = m["guideid"]
                if gid not in seen_ids:
                    seen_ids.add(gid)
                    candidate_queue.append(m)

        if not candidate_queue:
            exhausted = True
            break

        meta     = candidate_queue.pop(0)
        guide_id = meta["guideid"]

        if guide_id in existing_ids:
            console.print(f"[yellow][SKIP] Guide {guide_id} already exists locally[/yellow]")
            continue

        console.print(f"[cyan]Fetching guide {guide_id}: {meta.get('title', 'Unknown')}[/cyan]")
        try:
            guide = fetch_guide(guide_id)
            category = guide.get("category") or meta.get("category") or f"guide_{guide_id}"
            guides_dir, imgs_dir = build_output_paths(output_root, device_key, category.strip())
            saved = convert_guide_to_preweg(guide, guides_dir, imgs_dir)
            saved_paths.append(saved)
            existing_ids.add(guide_id)
        except Exception as e:
            console.print(f"[red][ERROR] Failed to crawl guide {guide_id}: {e}[/red]")

    if len(saved_paths) < limit:
        console.print(
            f"[yellow][WARN] Only found {len(saved_paths)}/{limit} new guides "
            f"(others were already crawled or unavailable)[/yellow]"
        )

    return saved_paths
