# WEGv2 - Multi-Agent Pipeline

A multi-agent pipeline that converts DIY repair guides into structured **WEG (Workflow Execution Guidance)** JSON files for AR guidance systems.

Built with **LangGraph** and powered by specialized AI agents (OpenAI, Google Gemini, Anthropic).

---

## Overview

WEGv2 uses 6 specialized agents working in parallel to extract:
- **Actions** - Step-by-step atomic actions with quadruples (action, tool, component, hands)
- **Tools** - Required tools per step and overall toolbox
- **Parts** - Component names with bounding box localization
- **Validation** - Reviewer agent for quality control and refinement

---

## Getting Started

### 1. Clone & Setup

```bash
git clone https://github.com/Deep-Vision-Lab/WEGAgent.git
cd WEGAgent/WEGv2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the `WEGv2` directory:

```
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
ANTHROPIC_API_KEY=your_anthropic_key
```

### 3. Run the Pipeline

**Crawl a guide from iFixit:**
```bash
python scripts/crawl.py --guide-id 167672
```

**Run the full extraction pipeline:**
```bash
python scripts/run_pipeline.py --guide-id 167672
```

**Debug a single agent:**
```bash
python scripts/run_single_agent.py --agent action --guide-id 167672
```

---

## Project Structure

```
WEGv2/
├── config/           # Settings and configuration
├── weg_pipeline/
│   ├── agents/       # 6 specialized extraction agents
│   ├── graph/        # LangGraph orchestration
│   ├── combiner/     # Result merging into WEG
│   ├── crawler/      # iFixit guide fetching
│   ├── models/       # Pydantic schemas
│   └── utils/        # Shared utilities
├── data/
│   ├── preweg/       # Crawled raw guides
│   └── weg_output/   # Generated WEG files
└── scripts/          # CLI entry points
```

---

## API Keys Required

| Provider | Used For |
|----------|----------|
| **OpenAI** | Tool, Hands, Part Text Agents |
| **Google** | Action, Part Vision Agents |
| **Anthropic** | Reviewer Agent |
