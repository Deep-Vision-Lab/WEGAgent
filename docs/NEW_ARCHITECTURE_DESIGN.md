# New Agent Architecture Design

## Overview

This document outlines the redesigned agent architecture where **each agent focuses on a single task**, and the **combiner is responsible for building the WEG** (including action quadruples) from all agent outputs.

---

## Current Architecture (Problems)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ACTION AGENT (overloaded)                         │
│                                                                             │
│  Extracts:                                                                  │
│  • Actions (atomic action sentences)                                        │
│  • Tools (for each action)                                                  │
│  • Components (for each action)                                             │
│  • Hands (for each action)                                                  │
│  • Task name, hints                                                         │
│  • Builds full quadruples: <action, tool, component, hands>                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              COMBINER                                        │
│  Just merges outputs - doesn't build anything                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Problems:**
1. Action Agent prompt is too complex (>500 lines)
2. Single point of failure - if one aspect fails, all fails
3. Hard to improve individual extraction tasks
4. Components are often confused with device names
5. Tool extraction in Action Agent lacks toolbox context
6. Hands estimation lacks action-specific reasoning

---

## New Architecture (Proposed)

```
                              ┌──────────────┐
                              │  Load Guide  │
                              │  (+ device   │
                              │    type)     │
                              └──────┬───────┘
                                     │
                                     ▼
                        ╔═══════════════════╗
                        ║   ACTION AGENT    ║
                        ║ (FIRST - Leader)  ║
                        ╠═══════════════════╣
                        ║ Extracts:         ║
                        ║ • Actions list    ║
                        ║ • Atomic breakdown║
                        ║ • Task name       ║
                        ║ • Hints           ║
                        ╚═════════┬═════════╝
                                  │
      ┌───────────────────────────┼───────────────────────────┐
      │                           │                           │
      │    Passes action context to all:                      │
      │                                                       │
      ▼                           ▼                           ▼
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│ PART TEXT     │         │   TOOL AGENT  │         │  HANDS AGENT  │
│ AGENT         │         │               │         │               │
├───────────────┤         ├───────────────┤         ├───────────────┤
│ • Parts in    │         │ • Tool per    │         │ • Hands per   │
│   step        │         │   action      │         │   action      │
│ • Component   │         │ • Uses        │         │               │
│   per action  │         │   toolbox     │         │               │
│ • Uses device │         │               │         │               │
│   context     │         │               │         │               │
└───────┬───────┘         └───────┬───────┘         └───────┬───────┘
        │                         │                         │
        ▼                         │                         │
┌───────────────┐                 │                         │
│ PART VISION   │                 │                         │
│ AGENT         │                 │                         │
├───────────────┤                 │                         │
│ • Localize    │                 │                         │
│   parts &     │                 │                         │
│   components  │                 │                         │
│   from Part   │                 │                         │
│   Text Agent  │                 │                         │
└───────┬───────┘                 │                         │
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  │
                                  ▼
                        ╔═════════════════════════════════════════╗
                        ║              SMART COMBINER              ║
                        ╠═════════════════════════════════════════╣
                        ║  Responsibilities:                       ║
                        ║  • Build action quadruples from inputs  ║
                        ║  • Match components → parts (linking)   ║
                        ║  • Validate consistency                 ║
                        ║  • Generate final WEG structure         ║
                        ╚═════════════════════════════════════════╝
```

---

## Agent Responsibilities (New)

> **Note:** Reasoning is logged to file, NOT included in output models.

### 1. Action Agent (Leader Agent - Runs FIRST)

**Single Focus:** Extract and structure actions from step descriptions

**Input:**
- Step text description
- Step images (for context)

**Output:** (reuses existing `ActionExtractionResult`)
```python
class ActionExtractionResult:
    step_index: int
    task_name: str                    # e.g., "Remove ice maker"
    actions: list[str]                # Full action sentences
    action_quadruples: list[ActionQuadruple]  # Simplified - only action, precise_action, full_action
    hints: list[str]                  # Tips, warnings, notes
```

**Key Changes:**
- Does NOT extract tools (tool=None in quadruple)
- Does NOT estimate hands (hands=0 in quadruple)  
- Does NOT identify physical components (component=target_description placeholder)
- Just breaks down what actions exist and their sequence

---

### 2. Part Text Agent (Enhanced - Handles Parts AND Components)

**Single Focus:** Extract parts mentioned AND identify physical components being touched per action

**Input:**
- Action list from Action Agent (action_quadruples)
- Step description (context)
- **Device type** (e.g., "refrigerator", "microwave", "dishwasher") - helps reasoning

**Output:**
```python
class PartTextResult:
    step_index: int
    device_type: str                  # e.g., "refrigerator"
    parts: list[str]                  # All parts mentioned in step
    primary_part: str | None          # Main part being worked on
    components_per_action: list[ComponentPerAction]  # Physical component per action

class ComponentPerAction:
    action_id: int                    # Index into action_quadruples (0-based)
    component: str                    # "power cord plug", "mounting screw"
```

**Key Responsibility:**
- Extract all parts mentioned in the step
- For EACH action, identify the **physical component being touched**
- **Critical reasoning**: "What does my hand/tool PHYSICALLY TOUCH?"
- Example: "Unplug the refrigerator" → component is "power cord plug", NOT "refrigerator"
- Uses device context to avoid confusing device with component

---

### 3. Tool Agent (Enhanced)

**Single Focus:** Identify tools used for each action

**Input:**
- Action list from Action Agent
- Step description
- Global toolbox (important context!)

**Output:** (enhanced)
```python
class ToolExtractionResult:
    step_index: int
    tools: list[str]                  # All tools in step (existing)
    primary_tool: str | None          # Main tool (existing)
    tools_per_action: list[ToolPerAction]  # NEW: per-action mapping

class ToolPerAction:
    action_id: int                    # Index into action_quadruples (0-based)
    tool: str | None                  # Tool name or None (bare hands)
```

**Key Changes:**
- Now receives action list as context
- Must match to global toolbox when possible
- Infers tools from actions (e.g., "unscrew" → screwdriver)

---

### 4. Hands Agent (Enhanced)

**Single Focus:** Estimate hands required per action

**Input:**
- Action list from Action Agent
- Step description

**Output:** (enhanced)
```python
class HandsEstimationResult:
    step_index: int
    hands: int                        # Step-level (existing, for backward compat)
    hands_per_action: list[HandsPerAction]  # NEW: per-action estimation

class HandsPerAction:
    action_id: int                    # Index into action_quadruples (0-based)
    hands: int                        # 0, 1, or 2
```

**Key Changes:**
- Now estimates per-action, not just per-step
- Receives action context for better reasoning

---

### 5. Part Vision Agent (Enhanced)

**Single Focus:** Localize parts AND components in images with bounding boxes

**Input:**
- Parts list from Part Text Agent
- Components list from Part Text Agent (per action)
- Step images
- Hints (may indicate locations)

**Output:** (existing model)
```python
class PartVisionResult:
    step_index: int
    image_index: int
    parts: list[PartBBoxInfo]         # Includes both parts and components
```

**Key Changes:**
- Receives BOTH parts and components to find (from Part Text Agent)
- Component names from `components_per_action` are added to detection list
- More focused detection task

---

### 6. Reviewer Agent (Integrated Throughout)

**Single Focus:** Validate each agent's output and trigger refinement when needed

**Access:** Shared pipeline state (all agent outputs)

**Review Points:**
1. **After Action Agent** → Validate actions are properly extracted
2. **After Part Text Agent** → Validate components are PHYSICAL parts touched (not device names)
3. **After Tool Agent** → Validate tools match description and toolbox
4. **After Hands Agent** → Validate hands estimates are reasonable

**Input:** (per review)
- Agent output being reviewed
- Original step descriptions
- Shared state (other agent outputs for context)

**Output:**
```python
class StepReviewResult:
    step_index: int
    is_valid: bool
    issues: list[str]
    feedback_for_agent: str           # Specific feedback for the agent
    confidence: float

class AgentReviewResult:
    agent_name: str                   # Which agent was reviewed
    steps_needing_refinement: list[int]
    refinement_requests: list[RefinementRequest]
```

**Refinement Loop:**
```
Agent completes → Reviewer validates → Issues found?
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │ YES                                       │ NO
                    ▼                                           ▼
            Send feedback to agent                         Continue to
            Agent refines output                           next phase
            (max N retries)
```

**What Reviewer Checks Per Agent:**

| Agent | Reviewer Checks |
|-------|----------------|
| **Action Agent** | Actions extracted correctly? Action verbs valid? Task name appropriate? Hints vs actions separated? |
| **Part Text Agent** | Components are PHYSICAL parts (not device names)? Parts mentioned found? Device context used? |
| **Tool Agent** | Tools match description? Tools exist in toolbox? Tools match action verbs? |
| **Hands Agent** | Hands reasonable for action? 0 for info-only? 2 for heavy lifting? |

**Example Reviewer Feedback:**
```
"Step 1: The component 'refrigerator' is the DEVICE, not what you touch. 
When unplugging, your hand grabs the PLUG. 
Change component to 'power cord plug' or 'electrical plug'."
```

---

### 7. Smart Combiner (Final Phase)

**Responsibilities:**

1. **Build Action Quadruples**
   ```python
   for each step:
       for each action in actions:
           quadruple = {
               "action": action.action_verb,
               "precise_action": action.precise_action,
               "tool": tools[action_id].tool,
               "component": components[action_id].component,
               "hands": hands[action_id].hands,
               "full_action": action.full_action,
           }
   ```

2. **Link Components to Part Detections**
   - Match component names to detected bounding boxes
   - Assign part_ids for AR linking

3. **Validate Consistency**
   - Check if tool matches action (screwdriver for "unscrew")
   - Check if hands make sense for action type
   - Flag suspicious extractions

4. **Generate Final WEG**
   - Combine all pieces into structured format
   - Add metadata, toolbox, etc.

---

## Pipeline Flow (New)

```
Phase 1: Load Guide (extract device type from title/category)
          │
          ▼
Phase 2: ACTION AGENT ──────────────────┐
          │                              │
          ▼                              ▼
        [Output] ──────────────► REVIEWER (validates actions)
          │                              │
          │◄─────── Refinement ◄─────────┘ (if issues)
          │
          ├──────────────────────────────────────┐
          │                                      │
          ▼                                      ▼
Phase 3: PART TEXT AGENT ────┐          TOOL AGENT ────────────┐
          │                  │               │                  │
          ▼                  ▼               ▼                  ▼
        [Output] ────► REVIEWER          [Output] ────► REVIEWER
          │                │                 │                │
          │◄── Refinement ◄┘                 │◄── Refinement ◄┘
          │                                  │
          ▼                                  │
Phase 4: PART VISION AGENT                   │
          │                                  │
          ▼                                  ▼
                                      HANDS AGENT ─────────────┐
                                           │                   │
                                           ▼                   ▼
                                        [Output] ────► REVIEWER
                                           │                │
                                           │◄── Refinement ◄┘
                                           │
          └────────────────────────────────┼───────────────────┘
                                           │
                                           ▼
Phase 5: SYNC (wait for all)
          │
          ▼
Phase 6: SMART COMBINER
          │
          ▼
        WEG JSON
```

### Reviewer Integration Details

Each agent follows this pattern:
```python
def agent_node_with_review(state):
    # 1. Run agent
    result = agent.run(...)
    
    # 2. Review output
    review = reviewer.review_agent_output(
        agent_name="PartTextAgent",
        output=result,
        guide=guide,
        state=state  # Access to all other outputs
    )
    
    # 3. Refinement loop (max 2 retries)
    retries = 0
    while review.needs_refinement and retries < 2:
        result = agent.refine(feedback=review.feedback)
        review = reviewer.review_agent_output(...)
        retries += 1
    
    return {"part_text": result}
```

---

## Data Flow Example

**Device:** Refrigerator  
**Step Text:** "Use a Phillips screwdriver to remove the two mounting screws securing the ice maker."

### Action Agent Output (actions only):
```json
{
  "step_index": 3,
  "task_name": "Remove ice maker mounting screws",
  "actions": ["Use a Phillips screwdriver to remove the two mounting screws securing the ice maker"],
  "action_quadruples": [
    {
      "action": "remove",
      "precise_action": "unscrew",
      "tool": null,
      "component": "two mounting screws securing the ice maker",
      "hands": 0,
      "full_action": "Use a Phillips screwdriver to remove the two mounting screws securing the ice maker"
    }
  ],
  "hints": []
}
```

### Part Text Agent Output (parts + components):
```json
{
  "step_index": 3,
  "device_type": "refrigerator",
  "parts": ["mounting screw", "ice maker"],
  "primary_part": "mounting screw",
  "components_per_action": [
    {
      "action_id": 0,
      "component": "mounting screw"
    }
  ]
}
```

### Tool Agent Output:
```json
{
  "step_index": 3,
  "tools": ["Phillips screwdriver"],
  "primary_tool": "Phillips screwdriver",
  "tools_per_action": [
    {
      "action_id": 0,
      "tool": "Phillips screwdriver"
    }
  ]
}
```

### Hands Agent Output:
```json
{
  "step_index": 3,
  "hands": 1,
  "hands_per_action": [
    {
      "action_id": 0,
      "hands": 1
    }
  ]
}
```

### Part Vision Agent Output:
```json
{
  "step_index": 3,
  "image_index": 1,
  "parts": [
    {
      "name": "mounting screw",
      "bbox": [0.45, 0.32, 0.52, 0.41],
      "confidence": 0.92,
      "marked_by_annotation": true
    },
    {
      "name": "ice maker",
      "bbox": [0.20, 0.15, 0.80, 0.70],
      "confidence": 0.88,
      "marked_by_annotation": false
    }
  ]
}
```

### Combiner Builds Final Quadruple:
```json
{
  "action": "remove",
  "precise_action": "unscrew",
  "tool": "Phillips screwdriver",
  "component": "mounting screw",
  "hands": 1,
  "full_action": "Use a Phillips screwdriver to remove the two mounting screws securing the ice maker",
  "part_id": 1
}
```

> **Note:** All reasoning is logged to file during agent execution, not included in outputs.

---

## Benefits of New Architecture

| Aspect | Current | New |
|--------|---------|-----|
| **Prompt Complexity** | 500+ lines | ~100 lines each |
| **Single Agent Failure** | Breaks everything | Only that aspect fails |
| **Component Reasoning** | Often confused | Dedicated agent for reasoning |
| **Tool Matching** | No toolbox context | Full toolbox awareness |
| **Hands Estimation** | Per-step | Per-action |
| **Debugging** | Hard | Easy to isolate |
| **Model Tuning** | One prompt to tune | Tune individual agents |
| **Cost Optimization** | All use VLM | Most can use cheaper LLM |

---

## Implementation Notes

### Branch: `feature/specialized-agents`

### Files to Create/Modify:
1. `agents/action_agent.py` - Add `run_actions_only()` method (simplified extraction)
2. `agents/part_text_agent.py` - Enhance to extract components per action + device context
3. `agents/tool_agent.py` - Add `run_for_actions()` method (per-action extraction)
4. `agents/hands_agent.py` - Add `run_for_actions()` method (per-action estimation)
5. `agents/part_vision_agent.py` - Update to receive both parts and components list
6. `agents/reviewer_agent.py` - Add per-agent review methods (`review_action_output()`, `review_part_text_output()`, etc.)
7. `models/intermediate.py` - Add `ComponentPerAction`, `ToolPerAction`, `HandsPerAction`
8. `models/preweg.py` - Add `device_type` extraction from guide title/category
9. `combiner/weg_combiner.py` - Add `combine_to_weg_v2()` (smart combination logic)
10. `graph/pipeline_v2.py` - NEW pipeline with new flow + reviewer integration
11. `graph/nodes.py` - Add new node functions with reviewer hooks
12. `scripts/run_pipeline.py` - Add flag to choose v1 or v2 pipeline

### Key Principle: Backward Compatibility
- Existing `run()` methods remain unchanged
- New `run_actions_only()` / `run_for_actions()` methods added
- `pipeline.py` (v1) untouched, `pipeline_v2.py` (v2) created
- Both pipelines can coexist

### Device Type Extraction
- Parse from guide title: "Samsung Refrigerator Ice Maker Replacement" → "refrigerator"
- Or from category field if available
- Fallback to "appliance" if not detected
- Used by Part Text Agent to avoid component confusion

### Reviewer Integration Pattern
Each agent node follows this pattern:
```python
def run_with_review(agent, reviewer, state, max_retries=2):
    # Run agent
    result = agent.run(...)
    
    # Review
    review = reviewer.review_agent_output(agent.AGENT_NAME, result, state)
    
    # Refinement loop
    retries = 0
    while review.needs_refinement and retries < max_retries:
        result = agent.refine(feedback=review.feedback, ...)
        review = reviewer.review_agent_output(...)
        retries += 1
    
    return result
```

### Migration Strategy:
1. Create new branch
2. Add per-action models to intermediate.py
3. Add device_type to PreWEG model
4. Update Action Agent (add actions-only method)
5. Enhance Part Text Agent (add components per action)
6. Update Tool/Hands agents (per-action methods)
7. Add per-agent review methods to ReviewerAgent
8. Create pipeline_v2.py with reviewer integration
9. Add combine_to_weg_v2()
10. Test end-to-end
11. Compare output quality with v1

---

## Questions to Resolve

1. **Device type detection strategy?**
   - Parse from guide title (e.g., "Samsung Refrigerator..." → "refrigerator")
   - Use category field if available
   - Manual override option?
   - Decision: Parse from title, fallback to category, then "appliance"

2. **Parallel vs Sequential for dependent agents?**
   - Action Agent must run first (+ review)
   - Part Text, Tool, Hands can run in parallel (all depend on Action)
   - Each runs then gets reviewed before continuing
   - Part Vision depends on Part Text

3. **How to handle action splitting?**
   - If text says "remove and clean the filter"
   - Action Agent should split into 2 actions
   - Each gets its own component, tool, hands mapping

4. **Reviewer Agent scope?**
   - **Decision: Review EACH agent's output immediately after completion**
   - Reviewer has access to shared state for context
   - Triggers refinement per-agent (max 2 retries each)
   - This catches issues early before they propagate

5. **Max refinement retries?**
   - Default: 2 retries per agent
   - Configurable via pipeline config
   - After max retries, continue with best effort result

---

## Implementation Status

### Completed ✅

1. **Design Document** - This file
2. **Feature Branch** - `feature/specialized-agents`
3. **V2 Models** (`weg_pipeline/models/intermediate.py`):
   - `ComponentPerAction` - component per action_id
   - `ToolPerAction` - tool per action_id
   - `HandsPerAction` - hands per action_id
   - `PartTextResultV2`, `PartsTextPerGuideV2`
   - `ToolExtractionResultV2`, `ToolsPerGuideV2`
   - `HandsEstimationResultV2`, `HandsPerGuideV2`

4. **Device Type Extraction** (`weg_pipeline/models/preweg.py`):
   - `extract_device_type()` function
   - `DEVICE_KEYWORDS` list (refrigerator, dishwasher, etc.)
   - `device_type` property on `PreWEGGuide`

5. **Action Agent V2** (`weg_pipeline/agents/action_agent.py`):
   - `get_system_prompt_v2()` - actions-only prompt
   - `build_prompt_v2()` - simplified prompt
   - `run_actions_only()` - extract actions without tool/component/hands
   - `refine_actions_only()` - refinement for actions only

6. **Part Text Agent V2** (`weg_pipeline/agents/part_text_agent.py`):
   - `get_system_prompt_v2()` - parts + components focus
   - `build_prompt_v2()` - includes device type context
   - `run_with_actions()` - extract parts AND components per action
   - `refine_with_actions()` - refinement with feedback

7. **Tool Agent V2** (`weg_pipeline/agents/tool_agent.py`):
   - `get_system_prompt_v2()` - per-action tool extraction
   - `build_prompt_v2()` - toolbox-aware prompting
   - `run_for_actions()` - extract tools per action
   - `refine_for_actions()` - refinement with feedback

8. **Hands Agent V2** (`weg_pipeline/agents/hands_agent.py`):
   - `get_system_prompt_v2()` - per-action hands estimation
   - `build_prompt_v2()` - action-specific reasoning
   - `run_for_actions()` - estimate hands per action
   - `refine_for_actions()` - refinement with feedback

9. **Reviewer Agent V2** (`weg_pipeline/agents/reviewer_agent.py`):
   - `review_actions_v2()` - validate Action Agent output
   - `review_parts_text_v2()` - validate Part Text Agent (critical: component != device)
   - `review_tools_v2()` - validate Tool Agent output
   - `review_hands_v2()` - validate Hands Agent output

10. **V2 Pipeline State** (`weg_pipeline/graph/state_v2.py`):
    - `PipelineStateV2` - new state with V2 agent outputs and per-agent reviews

11. **V2 Pipeline Nodes** (`weg_pipeline/graph/nodes_v2.py`):
    - All V2 agent nodes with integrated reviewer calls
    - `action_agent_node_v2`, `review_actions_node_v2`
    - `part_text_agent_node_v2`, `review_parts_text_node_v2`
    - `tool_agent_node_v2`, `review_tools_node_v2`
    - `hands_agent_node_v2`, `review_hands_node_v2`
    - `part_vision_agent_node_v2`
    - `sync_node_v2`, `combiner_node_v2`

12. **V2 Pipeline** (`weg_pipeline/graph/pipeline_v2.py`):
    - `create_pipeline_v2()` - builds LangGraph with V2 flow
    - `run_pipeline_v2()` - runs V2 pipeline

13. **V2 Combiner** (`weg_pipeline/combiner/weg_combiner.py`):
    - `combine_to_weg_v2()` - builds quadruples from all agent outputs

14. **CLI Update** (`scripts/run_pipeline.py`):
    - `--v2` flag to use V2 pipeline
    - Auto-detects WEG suffix (_WEG_v2.json)

### V2 Pipeline Flow

```
load_guide
    │
    ▼
action_agent_v2 ─────────► review_actions_v2
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
part_text_agent_v2         tool_agent_v2             hands_agent_v2
        │                          │                          │
        ▼                          ▼                          ▼
review_parts_text_v2       review_tools_v2           review_hands_v2
        │                          │                          │
        ▼                          │                          │
part_vision_agent_v2              │                          │
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   │
                                   ▼
                              sync_v2
                                   │
                                   ▼
                            combiner_v2 ─────► _WEG_v2.json
```

### Usage

```bash
# V1 Pipeline (default)
python scripts/run_pipeline.py run --guide data/preweg/167672_preWEG.json

# V2 Pipeline (specialized agents)
python scripts/run_pipeline.py run --guide data/preweg/167672_preWEG.json --v2

# V2 with sequential execution (debugging)
python scripts/run_pipeline.py run --guide data/preweg/167672_preWEG.json --v2 --sequential
```

### Backward Compatibility

- V1 methods preserved in all agents (`run()`, `refine_step()`)
- V2 methods added alongside (`run_actions_only()`, `run_with_actions()`, etc.)
- Both pipelines can run on same guides
- Outputs go to different files (`_WEG.json` vs `_WEG_v2.json`)

