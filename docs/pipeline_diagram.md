# WEGv2 Pipeline Architecture

## Pipeline Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    WEGv2 AGENTIC PIPELINE                                        │
│                              DIY Repair Guide → WEG JSON Conversion                              │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

                                          ┌──────────────┐
                                          │    START     │
                                          └──────┬───────┘
                                                 │
                                                 ▼
                                    ┌────────────────────────┐
                                    │      Load Guide        │
                                    │    (pre-WEG JSON)      │
                                    └────────────┬───────────┘
                                                 │
                    ┌────────────────────────────┼────────────────────────────┐
                    │                            │                            │
                    ▼                            ▼                            ▼
    ┌───────────────────────────┐  ┌───────────────────────────┐  ┌───────────────────────────┐
    │      ACTION AGENT         │  │       TOOL AGENT          │  │      HANDS AGENT          │
    │       (VLM/LLM)           │  │        (LLM)              │  │        (LLM)              │
    │  ┌─────────────────────┐  │  │  ┌─────────────────────┐  │  │  ┌─────────────────────┐  │
    │  │ Extracts:           │  │  │  │ Extracts:           │  │  │  │ Extracts:           │  │
    │  │ • Atomic actions    │  │  │  │ • Tools per step    │  │  │  │ • Hands count (0-2) │  │
    │  │ • Task names        │  │  │  │ • Primary tool      │  │  │  │ • Reasoning         │  │
    │  │ • Hints/warnings    │  │  │  │ • Global toolbox    │  │  │  └─────────────────────┘  │
    │  │ • Action Quadruples:│  │  │  └─────────────────────┘  │  │                           │
    │  │   <action, tool,    │  │  │                           │  │  Model: GPT-4o-mini       │
    │  │    component, hands>│  │  │  Model: GPT-4o-mini       │  │  Provider: OpenAI         │
    │  └─────────────────────┘  │  │  Provider: OpenAI         │  └───────────────┬───────────┘
    │                           │  └───────────────┬───────────┘                  │
    │  Model: Gemini 1.5 Flash  │                  │                              │
    │  Provider: Google         │                  │                              │
    └───────────────┬───────────┘                  │                              │
                    │                              │                              │
                    │              ┌───────────────────────────┐                  │
                    │              │     PART TEXT AGENT       │                  │
                    │              │        (LLM)              │                  │
                    │              │  ┌─────────────────────┐  │                  │
                    │              │  │ Extracts:           │  │                  │
                    │              │  │ • Part names        │  │                  │
                    │              │  │ • Primary part      │  │                  │
                    │              │  │ • Secondary parts   │  │                  │
                    │              │  └─────────────────────┘  │                  │
                    │              │                           │                  │
                    │              │  Model: GPT-4o-mini       │                  │
                    │              │  Provider: OpenAI         │                  │
                    │              └───────────────┬───────────┘                  │
                    │                              │                              │
                    │                              ▼                              │
                    │              ┌───────────────────────────┐                  │
                    │              │    PART VISION AGENT      │                  │
                    │              │        (VLM)              │                  │
                    │              │  ┌─────────────────────┐  │                  │
                    │              │  │ Extracts:           │  │                  │
                    │              │  │ • Part bounding box │  │                  │
                    │              │  │ • Detection conf.   │  │                  │
                    │              │  │ • Red circle detect │  │                  │
                    │              │  │ • Visual annotation │  │                  │
                    │              │  └─────────────────────┘  │                  │
                    │              │                           │                  │
                    │              │  Model: Gemini 2.5 Flash  │                  │
                    │              │  Provider: Google         │                  │
                    │              └───────────────┬───────────┘                  │
                    │                              │                              │
                    └──────────────────────────────┼──────────────────────────────┘
                                                   │
                                                   ▼
                                    ┌────────────────────────┐
                                    │    SYNC BARRIER        │
                                    │  (Wait for all agents) │
                                    └────────────┬───────────┘
                                                 │
                                                 ▼
    ┌────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                     REVIEWER AGENT                                          │
    │                                      (LLM - Claude)                                         │
    │  ┌──────────────────────────────────────────────────────────────────────────────────────┐  │
    │  │ Validates all agent outputs:                                                          │  │
    │  │ • Checks action correctness against descriptions                                      │  │
    │  │ • Validates COMPONENT reasoning (is it the physical part touched?)                    │  │
    │  │ • Checks tool-action consistency                                                      │  │
    │  │ • Validates hands estimations                                                         │  │
    │  │ • Identifies missing/incomplete extractions                                           │  │
    │  └──────────────────────────────────────────────────────────────────────────────────────┘  │
    │                                                                                             │
    │  Model: Claude 3.5 Haiku                                                                    │
    │  Provider: Anthropic                                                                        │
    │                                                                                             │
    │  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
    │  │                           REFINEMENT LOOP                                            │   │
    │  │  ┌─────────────────┐    ┌─────────────────────────────────────────────────────┐     │   │
    │  │  │ Issues Found?   │───▶│ Send feedback to ACTION AGENT for refinement       │     │   │
    │  │  │                 │    │ • Component reasoning errors                         │     │   │
    │  │  │ - Component     │    │ • Missing actions                                    │     │   │
    │  │  │   errors        │    │ • Incomplete extractions                             │     │   │
    │  │  │ - Missing steps │    └──────────────────────────┬──────────────────────────┘     │   │
    │  │  │ - Invalid data  │                               │                                │   │
    │  │  └─────────────────┘                               ▼                                │   │
    │  │                                    ┌───────────────────────────┐                     │   │
    │  │                                    │   Refined Results         │                     │   │
    │  │                                    │   Merged Back to State    │                     │   │
    │  │                                    └───────────────────────────┘                     │   │
    │  └─────────────────────────────────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                  │
                                                  ▼
                                    ┌────────────────────────┐
                                    │       COMBINER         │
                                    │  ┌──────────────────┐  │
                                    │  │ Merges:          │  │
                                    │  │ • Actions        │  │
                                    │  │ • Tools          │  │
                                    │  │ • Parts + BBoxes │  │
                                    │  │ • Hints          │  │
                                    │  │ • Quadruples     │  │
                                    │  └──────────────────┘  │
                                    └────────────┬───────────┘
                                                 │
                                                 ▼
                                    ┌────────────────────────┐
                                    │      WEG JSON          │
                                    │   (Final Output)       │
                                    └────────────┬───────────┘
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │     END      │
                                          └──────────────┘
```

## Agent Details

### 1. Action Agent (VLM)
| Attribute | Value |
|-----------|-------|
| **Model** | Gemini 1.5 Flash |
| **Provider** | Google |
| **Input** | Step descriptions + images |
| **Output** | Atomic actions, task names, hints, action quadruples |

**Action Quadruple Format:**
```json
{
  "action": "remove",
  "precise_action": "unscrew",
  "tool": "Phillips screwdriver",
  "component": "mounting screw",
  "hands": 1,
  "full_action": "Remove the mounting screws using a Phillips screwdriver"
}
```

### 2. Tool Agent (LLM)
| Attribute | Value |
|-----------|-------|
| **Model** | GPT-4o-mini |
| **Provider** | OpenAI |
| **Input** | Step descriptions, global toolbox |
| **Output** | Tools per step, primary tool, global toolbox |

### 3. Hands Agent (LLM)
| Attribute | Value |
|-----------|-------|
| **Model** | GPT-4o-mini |
| **Provider** | OpenAI |
| **Input** | Step descriptions |
| **Output** | Hands count (0/1/2) per step with reasoning |

**Hands Values:**
- `0` = No hands (information only)
- `1` = One hand sufficient
- `2` = Two hands required

### 4. Part Text Agent (LLM)
| Attribute | Value |
|-----------|-------|
| **Model** | GPT-4o-mini |
| **Provider** | OpenAI |
| **Input** | Step descriptions |
| **Output** | Part names, primary part per step |

### 5. Part Vision Agent (VLM)
| Attribute | Value |
|-----------|-------|
| **Model** | Gemini 2.5 Flash |
| **Provider** | Google |
| **Input** | Step images + part names from text agent |
| **Output** | Bounding boxes, confidence scores |

**Features:**
- Red circle/annotation detection
- Normalized bounding box coordinates [0.0-1.0]
- Cross-references with text-extracted part names

### 6. Reviewer Agent (LLM)
| Attribute | Value |
|-----------|-------|
| **Model** | Claude 3.5 Haiku |
| **Provider** | Anthropic |
| **Input** | All agent outputs + original guide |
| **Output** | Validation results, refinement requests |

**Validation Checks:**
- ✅ Action correctness vs descriptions
- ✅ Component reasoning (physical part touched)
- ✅ Tool-action consistency
- ✅ Hands estimation reasonability
- ✅ Missing/incomplete extractions

**Refinement Loop:**
- Identifies issues per step
- Sends specific feedback to Action Agent
- Merges refined results back into state
- Max iterations configurable (default: 2)

## Pipeline Execution Modes

### Parallel Mode (Default)
```
load_guide → [action_agent, tool_agent, hands_agent, part_text_agent] → part_vision_agent → sync → reviewer → combiner
```
- Agents run concurrently where possible
- Part Vision waits for Part Text (dependency)
- ~2-3x faster than sequential

### Sequential Mode (Debug)
```
load_guide → action_agent → tool_agent → hands_agent → part_text_agent → part_vision_agent → sync → reviewer → combiner
```
- Easier to debug individual agents
- Full logging per step

## Data Flow

```
Pre-WEG JSON (Crawled Guide)
         │
         ▼
┌────────────────────┐
│ guide_id: 167672   │
│ title: "..."       │
│ steps: [...]       │
│ toolbox: [...]     │
│ images: [...]      │
└────────────────────┘
         │
         ▼ (Pipeline Processing)
         │
         ▼
┌────────────────────┐
│ WEG JSON Output    │
│ ─────────────────  │
│ header:            │
│   title, toolbox   │
│ steps: [           │
│   step_id          │
│   task_name        │
│   actions: [...]   │
│   action_quadruples│
│   hints: [...]     │
│   tool             │
│   part: {          │
│     name, bbox     │
│   }                │
│ ]                  │
└────────────────────┘
```
