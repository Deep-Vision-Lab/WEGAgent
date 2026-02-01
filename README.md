# WEGv2 - Web Extraction Guides Pipeline v2

A multi-agent AI pipeline for extracting and structuring repair guide information from iFixit using vision and language models.

## Features

- **Multi-Agent Architecture**: Specialized agents for actions, tools, parts, hands detection, and review
- **Vision + Language**: Combines vision models (GPT-4V, Gemini) with language models (Claude, GPT-4, Llama)
- **Graph-based Pipeline**: LangGraph-powered workflow for structured data extraction
- **Quality Validation**: Built-in validators for extraction quality assurance

## Installation

1. Clone the repository and navigate to WEGv2:
```bash
cd WEGv2
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### Run the full pipeline:
```bash
python scripts/run_pipeline.py
```

### Run single agent:
```bash
python scripts/run_single_agent.py
```

### Crawl guides:
```bash
python scripts/crawl.py
```

## Project Structure

```
WEGv2/
├── weg_pipeline/          # Core pipeline components
│   ├── agents/           # Specialized extraction agents
│   ├── crawler/          # iFixit crawler
│   ├── graph/            # LangGraph pipeline definitions
│   ├── models/           # Data models (PreWEG, WEG)
│   └── validators/       # Quality validation
├── scripts/              # Execution scripts
├── data/                 # Input/output data
├── config/               # Configuration settings
└── tests/                # Unit tests
```

## Requirements

- Python 3.10+
- OpenAI API key (GPT-4, GPT-4V)
- Anthropic API key (Claude)
- Google API key (Gemini)

## License

MIT
