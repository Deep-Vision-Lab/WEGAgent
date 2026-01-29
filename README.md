# WEGv2 - Agentic Pipeline for DIY Manual → WEG Conversion

An intelligent multi-agent pipeline that converts DIY repair guides (from iFixit) into structured **WEG (Workflow Execution Guidance)** JSON files for AR guidance systems.

Built with **LangGraph** for orchestration and powered by multiple specialized AI agents.

## 🏗️ Architecture

```
                                    ┌──────────────┐
                                    │  Pre-WEG     │
                                    │  (Crawled)   │
                                    └──────┬───────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │      Load Guide        │
                              └────────────┬───────────┘
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    ▼                      ▼                      ▼
        ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
        │   ACTION AGENT    │  │    TOOL AGENT     │  │   HANDS AGENT     │
        │   (Gemini VLM)    │  │   (GPT-4o-mini)   │  │   (GPT-4o-mini)   │
        │   • Actions       │  │   • Tools/step    │  │   • Hands (0-2)   │
        │   • Quadruples    │  │   • Toolbox       │  └─────────┬─────────┘
        │   • Hints         │  └─────────┬─────────┘            │
        └─────────┬─────────┘            │                      │
                  │           ┌──────────┴──────────┐           │
                  │           │   PART TEXT AGENT   │           │
                  │           │   (GPT-4o-mini)     │           │
                  │           │   • Part names      │           │
                  │           └──────────┬──────────┘           │
                  │                      ▼                      │
                  │           ┌───────────────────┐             │
                  │           │ PART VISION AGENT │             │
                  │           │ (Gemini 2.5 VLM)  │             │
                  │           │ • Bounding boxes  │             │
                  │           │ • Red circle det. │             │
                  │           └─────────┬─────────┘             │
                  │                     │                       │
                  └─────────────────────┼───────────────────────┘
                                        ▼
                              ┌───────────────────┐
                              │       SYNC        │
                              └─────────┬─────────┘
                                        ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │                        REVIEWER AGENT                              │
    │                       (Claude 3.5 Haiku)                           │
    │  • Validates all outputs    • Component reasoning check            │
    │  • Triggers refinement loop • Sends feedback to Action Agent       │
    └───────────────────────────────────┬───────────────────────────────┘
                                        ▼
                              ┌───────────────────┐
                              │     COMBINER      │
                              │  (Merge outputs)  │
                              └─────────┬─────────┘
                                        ▼
                              ┌───────────────────┐
                              │    WEG JSON       │
                              │   (Final Output)  │
                              └───────────────────┘
```

> 📄 See [docs/pipeline_diagram.md](docs/pipeline_diagram.md) for the complete detailed diagram.

## 🤖 Agents Overview

| Agent | Model | Provider | Task |
|-------|-------|----------|------|
| **Action Agent** | Gemini 1.5 Flash | Google | Extract atomic actions, quadruples, hints |
| **Tool Agent** | GPT-4o-mini | OpenAI | Identify tools per step |
| **Hands Agent** | GPT-4o-mini | OpenAI | Estimate hands needed (0/1/2) |
| **Part Text Agent** | GPT-4o-mini | OpenAI | Extract part names from text |
| **Part Vision Agent** | Gemini 2.5 Flash | Google | Localize parts with bounding boxes |
| **Reviewer Agent** | Claude 3.5 Haiku | Anthropic | Validate & trigger refinement |

### Action Quadruples

Each action is extracted with a structured quadruple:

```json
{
  "action": "remove",
  "precise_action": "unscrew",
  "tool": "Phillips screwdriver",
  "component": "mounting screw",
  "hands": 1,
  "full_action": "Remove the mounting screws with a Phillips screwdriver"
}
```

### Reviewer Agent & Refinement Loop

The **Reviewer Agent** validates all extraction results and triggers refinement when issues are found:

1. **Validates** component reasoning (is "refrigerator" the part touched, or the "power plug"?)
2. **Checks** action-tool-hands consistency
3. **Identifies** missing or incomplete extractions
4. **Sends feedback** to Action Agent for refinement
5. **Merges** refined results back into the pipeline state

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env with your API keys (OpenAI, Google, Anthropic)

# 4. Crawl a guide
python scripts/crawl.py --device refrigerators --limit 1

# 5. Run the pipeline
python scripts/run_pipeline.py run --guide data/preweg/path/to/guide.json
```

## 📁 Project Structure

```
WEGv2/
├── config/                 # Settings and configuration
├── docs/                   # Documentation & diagrams
├── weg_pipeline/
│   ├── agents/            # 6 specialized extraction agents
│   │   ├── action_agent.py      # VLM - actions & quadruples
│   │   ├── tool_agent.py        # LLM - tools
│   │   ├── hands_agent.py       # LLM - hands estimation
│   │   ├── part_text_agent.py   # LLM - part names
│   │   ├── part_vision_agent.py # VLM - bounding boxes
│   │   └── reviewer_agent.py    # LLM - validation & refinement
│   ├── graph/             # LangGraph orchestration
│   │   ├── pipeline.py    # Pipeline construction
│   │   ├── nodes.py       # Graph node functions
│   │   └── state.py       # Pipeline state schema
│   ├── combiner/          # Result merging into WEG
│   ├── crawler/           # iFixit guide fetching
│   ├── models/            # Pydantic schemas (pre-WEG, intermediate, WEG)
│   ├── validators/        # Output validation
│   └── utils/             # Shared utilities (LLM, image, logging)
├── data/
│   ├── preweg/            # Crawled raw guides
│   └── weg_output/        # Generated WEG files
├── scripts/               # CLI entry points
│   ├── crawl.py           # Crawl guides from iFixit
│   ├── run_pipeline.py    # Run the full pipeline
│   ├── run_single_agent.py # Debug individual agents
│   └── visualize_bbox.py  # Visualize bounding boxes
└── tests/                 # Test suite
```

## 🔑 API Keys Required

| Provider | Model | Used By |
|----------|-------|---------|
| **OpenAI** | GPT-4o-mini | Tool, Hands, Part Text Agents |
| **Google** | Gemini 1.5/2.5 Flash | Action, Part Vision Agents |
| **Anthropic** | Claude 3.5 Haiku | Reviewer Agent |

Set in `.env`:
```bash
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AI...
ANTHROPIC_API_KEY=sk-ant-...
```

## 📊 WEG Output Format

```json
{
  "header": {
    "title": "LG Refrigerator Evaporator Fan Replacement",
    "description": "Replace a broken evaporator fan...",
    "toolbox": ["Phillips screwdriver", "Flathead screwdriver"],
    "parts_list": ["Evaporator fan", "Wire harness"],
    "guide_id": 167672
  },
  "steps": [
    {
      "step_id": 1,
      "task_name": "Disconnect power",
      "description": "Unplug the refrigerator from the wall outlet.",
      "actions": ["Unplug the refrigerator"],
      "action_quadruples": [
        {
          "action": "unplug",
          "precise_action": "disconnect",
          "tool": null,
          "component": "power cord plug",
          "hands": 1,
          "full_action": "Unplug the refrigerator from the wall outlet"
        }
      ],
      "hints": ["Ensure the refrigerator is not running"],
      "tool": null,
      "part": {
        "name": "power cord plug",
        "bbox": {"x1": 0.45, "y1": 0.6, "x2": 0.55, "y2": 0.7},
        "confidence": 0.92
      },
      "images": ["step1_img1.jpg"]
    }
  ]
}
```

## 🛠️ Development

```bash
# Run the full pipeline
python scripts/run_pipeline.py run --guide data/preweg/path/to/guide.json

# Run with logging
python scripts/run_pipeline.py run --guide path/to/guide.json --save-logs

# Run in sequential mode (for debugging)
python scripts/run_pipeline.py run --guide path/to/guide.json --sequential

# Run without reviewer
python scripts/run_pipeline.py run --guide path/to/guide.json --no-reviewer

# Run single agent for debugging
python scripts/run_single_agent.py --agent action --guide path/to/guide.json

# Visualize bounding boxes
python scripts/visualize_bbox.py --weg data/preweg/path/to/guide_WEG.json

# Run tests
pytest tests/
```

## 📈 Pipeline Execution Modes

| Mode | Command | Description |
|------|---------|-------------|
| **Parallel** (default) | `run` | Agents run concurrently, ~2-3x faster |
| **Sequential** | `--sequential` | Agents run one by one (debugging) |
| **No Reviewer** | `--no-reviewer` | Skip validation & refinement |

## 🔄 Recent Updates

- ✅ **Reviewer Agent** with refinement loop for quality validation
- ✅ **Action Quadruples** with precise_action field for technical verbs
- ✅ **Part Vision Agent** with red circle/annotation detection
- ✅ **LangGraph orchestration** with parallel execution
- ✅ **Comprehensive logging** with pipeline logs
- ✅ **Bounding box visualization** tool
