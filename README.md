# WEGAgent — Work & Equipment Guide Pipeline

A multi-agent AI pipeline that crawls iFixit repair guides and extracts structured **WEG** (Work & Equipment Guide) files containing step-by-step actions, part bounding boxes, tool lists, and action quadruples.

---

## Installation

**Requirements:** Python 3.10+

```bash
# 1. Clone and enter the project
cd WEGAgent

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API keys
#    Create a .env file in the project root with:
#    ANTHROPIC_API_KEY=sk-ant-...
#    OPENAI_API_KEY=sk-...       (optional)
#    GOOGLE_API_KEY=...          (optional)
```

---

## Workflow Overview

```
1. Crawl guides from iFixit  →  data/preweg/.../guides/{id}_preWEG.json
2. Run the pipeline          →  data/preweg/.../guides/{id}_WEG.json
3. Review & annotate         →  review_weg.py  (interactive UI)
```

---

## Step 1 — Crawl Guides

### List available device categories
```bash
python scripts/crawl.py devices
```

Supported categories: `refrigerators`, `washing_machines`, `dishwashers`, `dryers`, `clothes_iron`, `microwave`, `oven`, `vacuum`

### Crawl by device type
```bash
# Crawl 2 guides per category (adjust --limit as needed)
python scripts/crawl.py search --device refrigerators    --limit 2
python scripts/crawl.py search --device washing_machines --limit 2
python scripts/crawl.py search --device dishwashers      --limit 2
python scripts/crawl.py search --device dryers           --limit 2
python scripts/crawl.py search --device microwave        --limit 2
```

### Crawl a specific guide by iFixit ID
```bash
python scripts/crawl.py single 113759 --device microwave
```

Crawled guides are saved to `data/preweg/appliances/<category>/<brand>/<model>/guides/`.

---

## Step 2 — Generate WEGs

### Process all crawled guides
```bash
python scripts/run_pipeline.py run
```

### Process a single guide
```bash
python scripts/run_pipeline.py run --guide data/preweg/appliances/microwave/ge/ge_profile_microwave_oven/guides/113759_preWEG.json
```

### Pipeline options

| Flag | Description | Default |
|---|---|---|
| `--v2` | Use V2 pipeline (specialized agents per field) | V1 |
| `--llm` | LLM model for text agents | `claude-sonnet-4-5-20250929` |
| `--vlm` | VLM model for vision agents | `claude-sonnet-4-5-20250929` |
| `--sequential` | Run agents one at a time (useful for debugging) | parallel |
| `--no-review` | Skip the reviewer agent (V1 only) | reviewer on |
| `--no-validate` | Skip output validation | validation on |
| `--ollama` | Use local Ollama models instead of cloud | cloud |

### Examples
```bash
# V2 pipeline with Claude
python scripts/run_pipeline.py run --v2

# Use local Ollama models
python scripts/run_pipeline.py run --ollama --llm llama3.2:3b --vlm llama3.2-vision:11b

# Debug a single guide sequentially
python scripts/run_pipeline.py run --guide path/to/guide.json --sequential
```

Output WEGs are saved alongside the preWEG file as `{guide_id}_WEG.json` (V1) or `{guide_id}_..._WEG_v2.json` (V2).

### List all available guides
```bash
python scripts/run_pipeline.py list-guides
```

---

## Step 3 — Review & Annotate WEGs

The interactive reviewer lets you visually verify and correct every field produced by the agents.

### Launch

**Pick from a list (recommended):**
```bash
python scripts/review_weg.py
```
This scans `data/preweg/` and shows a numbered menu. WEGs that have already been reviewed are marked with `✓`.

**Open a specific file directly:**
```bash
python scripts/review_weg.py data/preweg/appliances/microwave/ge/ge_profile_microwave_oven/guides/113759_WEG.json
```

### Output file

The original `{guide_id}_WEG.json` is **never modified**. All edits are written to:
```
{guide_id}_WEG_reviewed.json   ← same folder as the original
```
If you re-open the same WEG later, the reviewer automatically loads your previous edits from the `_reviewed` file so work accumulates across sessions.

### Split-panel UI

- **Left panel** — annotated image with all parts numbered and colour-coded bboxes
- **Right panel** — step metadata; toggle between two views with `V`
  - **Parts view** — lists every bbox with confidence score
  - **Actions view** — shows task name and all action quadruples (action, component, tool, hands count)

### What you can verify and edit

| Field | How |
|---|---|
| Bounding boxes | `E` to redraw (click+drag on image) |
| Delete wrong parts | `D` then part number |
| Task name | `V` → Actions view → `T` → type in terminal |
| Action verb / component / tool | `V` → Actions view → `A` → pick quadruple number → type in terminal |
| Hands count | `V` → Actions view → `H` → pick quadruple number → press digit (1–9) |
| Flag step for later | `F` — marks step with a red FLAGGED badge |
| Mark step as verified | `K` — marks step with a green OK badge |
| Check source guide | `O` — opens the original iFixit URL in your browser |

### Keyboard reference

#### Navigation (always active)
| Key | Action |
|---|---|
| `Enter` / `N` | Next step |
| `P` | Previous step |
| `V` | Toggle Parts / Actions view |
| `O` | Open source URL in browser |
| `F` | Flag step |
| `K` | Mark step as verified / OK |
| `Q` | Quit |
| `ESC` | Cancel current mode |

#### Parts view
| Key | Action |
|---|---|
| `E` | Edit bbox — select part if multiple, then click+drag to draw new box |
| `D` | Delete a part — press part number to confirm |

#### Actions view
| Key | Action |
|---|---|
| `T` | Edit task name (prompt appears in terminal) |
| `H` | Set hands count — press quadruple number, then press the digit (e.g. `1` or `2`) |
| `A` | Edit action/component/tool/precise_action — press quadruple number, then type in terminal |

All edits save immediately to `{guide_id}_WEG_reviewed.json`.

---

## WEG File Format

WEGs are saved as `{guide_id}_WEG.json` in the guide's `guides/` directory.

```json
{
  "header": {
    "title": "GE Profile Microwave Oven Touchpad Repair",
    "source_url": "https://www.ifixit.com/Guide/...",
    "toolbox": ["Philips Screwdriver (medium size)"],
    "guide_id": 113759
  },
  "steps": [
    {
      "step_id": 1,
      "task_name": "Remove exhaust screws",
      "description": "...",
      "actions": ["Remove the two philips screws..."],
      "action_quadruples": [
        {
          "action": "remove",
          "precise_action": "unscrew",
          "tool": "Philips Screwdriver (medium size)",
          "component": "exhaust fan cover mounting screws",
          "hands": 1
        }
      ],
      "hints": ["..."],
      "primary_part": {
        "name": "exhaust fan cover mounting screws",
        "bbox": { "x1": 165, "y1": 98, "x2": 185, "y2": 116 },
        "confidence": 0.95,
        "image_path": "data\\preweg\\...\\imgs\\step_2_1.jpg"
      },
      "parts_all": [ ... ]
    }
  ]
}
```

---

## Project Structure

```
WEGAgent/
├── scripts/
│   ├── crawl.py              # Crawl iFixit guides
│   ├── run_pipeline.py       # Run WEG extraction pipeline
│   ├── review_weg.py         # Interactive WEG reviewer / annotator
│   └── visualize_bbox.py     # Visualize bboxes from a WEG file
├── weg_pipeline/
│   ├── agents/               # Extraction agents (actions, tools, parts, hands)
│   ├── crawler/              # iFixit API crawler
│   ├── graph/                # LangGraph pipeline (V1 and V2)
│   ├── models/               # PreWEG and WEG data models
│   ├── utils/                # IO helpers, path utilities
│   └── validators/           # WEG quality validators
└── data/
    └── preweg/
        └── appliances/
            └── <category>/<brand>/<model>/
                ├── guides/   # preWEG and WEG JSON files
                └── imgs/     # Step images
```

---

## License

MIT
