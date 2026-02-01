"""
Part Text Agent - Extracts part names from step descriptions.

Uses LLM to identify parts being manipulated in each step based on text analysis.
V2: Also extracts the physical component being touched for each action.
"""
from typing import Optional

from ..models.preweg import PreWEGGuide
from ..models.intermediate import (
    PartTextResult, 
    PartsTextPerGuide,
    PartTextResultV2,
    PartsTextPerGuideV2,
    ComponentPerAction,
    ActionsPerGuide,
)
from ..utils.llm_utils import extract_json_from_response
from .base_agent import BaseAgent


class PartTextAgent(BaseAgent):
    """
    Extracts part names from repair step text.
    
    Identifies:
    - Parts being removed/replaced
    - Parts being manipulated
    - Parts being connected/disconnected
    """

    AGENT_NAME = "PartTextAgent"
    REQUIRES_VISION = False
    DEFAULT_PROVIDER = "openai"
    DEFAULT_MODEL = "gpt-4o-mini"

    def get_system_prompt(self) -> str:
        return """You are an expert at identifying appliance parts mentioned in repair guides.

Your task is to extract the PARTS being manipulated in each repair step.

Definition of a PART:
- A physical component of the appliance
- Something being removed, replaced, connected, disconnected, or manipulated
- Examples: fan, motor, bracket, connector, wire harness, screw, panel, cover, filter

DO NOT include:
- Tools (screwdriver, wrench, pliers)
- Generic terms ("device", "appliance", "unit")
- Actions or verbs
- Materials (unless they're part names like "gasket")

Identify the PRIMARY part (the main focus of the step) and any SECONDARY parts.

Return ONLY valid JSON. No explanations, no markdown."""

    def build_prompt(self, guide: PreWEGGuide, **kwargs) -> str:
        """Build prompt for part extraction."""
        # Include parts list from guide for context
        parts_text = ""
        if guide.parts:
            parts_text = f"\nParts mentioned in guide:\n{guide.parts}\n"

        steps_text = []
        for step in sorted(guide.steps, key=lambda s: s.step_index):
            steps_text.append(f"""
Step {step.step_index}:
\"\"\"
{step.full_description}
\"\"\"
""")

        return f"""Extract parts being manipulated in each step.
{parts_text}
Guide: {guide.title}
Device: {guide.device or "Unknown"}

{chr(10).join(steps_text)}

Return JSON array:
[
  {{
    "step_index": 1,
    "parts": ["part1", "part2"],
    "primary_part": "part1",
    "reasoning": "optional: why this is the primary part"
  }},
  ...
]

Rules:
- "parts": ALL parts mentioned/manipulated in the step
- "primary_part": The MAIN part being worked on (the focus of this step)
- Be specific: "evaporator fan" not just "fan"
- Use consistent naming across steps
- Return exactly {guide.num_steps} entries"""

    def parse_response(
        self,
        response: str,
        guide: PreWEGGuide,
        **kwargs,
    ) -> PartsTextPerGuide:
        """Parse LLM response into structured parts."""
        try:
            data = extract_json_from_response(response)
        except ValueError as e:
            self.log_error(f"Failed to parse JSON: {e}")
            return PartsTextPerGuide(guide_id=guide.guide_id, steps=[])

        if not isinstance(data, list):
            self.log_error(f"Expected list, got {type(data)}")
            return PartsTextPerGuide(guide_id=guide.guide_id, steps=[])

        steps = []
        for item in data:
            if not isinstance(item, dict):
                continue

            step_index = item.get("step_index", 0)
            parts = item.get("parts", [])
            primary = item.get("primary_part")
            reasoning = item.get("reasoning")

            if isinstance(parts, list):
                parts = [str(p).strip() for p in parts if str(p).strip()]
            else:
                parts = []

            steps.append(PartTextResult(
                step_index=step_index,
                parts=parts,
                primary_part=primary,
                reasoning=reasoning,
            ))

        return PartsTextPerGuide(guide_id=guide.guide_id, steps=steps)

    # ==================== V2 Pipeline Methods ====================
    
    def get_system_prompt_v2(self) -> str:
        """System prompt for V2 pipeline - components AND their containing parts per action."""
        return """You are an expert at identifying appliance components and parts in repair guides.

Your task is to:
1. Extract all PARTS mentioned in the step (for context)
2. For EACH ACTION, identify:
   - COMPONENT: What the hand/tool PHYSICALLY TOUCHES
   - PART: The larger part that CONTAINS the component

=== COMPONENT (What you TOUCH) ===
The specific physical object that your hand or tool makes direct contact with.
- This is the thing you grab, push, pull, turn, or apply tool to
- Be specific: "power cord plug" not "cord", "mounting screw" not "screw"

=== PART (What CONTAINS the component) ===
The larger assembly or section that contains the component.
- This provides context for WHERE the component is located
- Can be null if the component IS the main part (e.g., lifting out a motor)

CRITICAL EXAMPLES:

1. "Unplug the refrigerator"
   - component: "power cord plug" (what your hand grabs)
   - part: "power cord" or null (the plug is part of the cord)

2. "Remove the screws securing the panel"
   - component: "screw" or "mounting screw" (what the screwdriver touches)
   - part: "panel" (the screws are securing the panel)

3. "Disconnect the wire harness from the motor"
   - component: "wire harness connector" (the plug you grab)
   - part: "motor" (where the connector plugs into)

4. "Lift out the fan assembly"
   - component: "fan assembly" (what your hands grab)
   - part: null (the fan IS the main part being removed)

5. "Press the release tab on the bracket"
   - component: "release tab" (what your finger presses)
   - part: "bracket" (the tab is part of the bracket)

6. "Open the freezer door"
   - component: "door handle" or "freezer door" (what you grab)
   - part: null (the door is the main part)

Return ONLY valid JSON. No explanations, no markdown."""

    def build_prompt_v2(
        self, 
        guide: PreWEGGuide, 
        actions: ActionsPerGuide,
        **kwargs,
    ) -> str:
        """Build prompt for V2 pipeline - components and their containing parts per action."""
        device_type = guide.device_type
        
        # Build per-step data with actions
        steps_data = []
        for step in sorted(guide.steps, key=lambda s: s.step_index):
            # Find actions for this step
            step_actions = None
            for a in actions.steps:
                if a.step_index == step.step_index:
                    step_actions = a
                    break
            
            action_list = []
            if step_actions:
                for i, q in enumerate(step_actions.action_quadruples):
                    action_list.append({
                        "action_id": i,
                        "action": q.action,
                        "full_action": q.full_action,
                    })
            
            steps_data.append(f"""
Step {step.step_index}:
Description: \"\"\"{step.full_description}\"\"\"
Actions to analyze:
{action_list}
""")

        return f"""For each action, identify the COMPONENT (what you touch) and PART (what contains it).

Guide: {guide.title}
Device Type: {device_type.upper()} (DO NOT use "{device_type}" as a component!)

{chr(10).join(steps_data)}

REMEMBER:
- component = what your hand/tool PHYSICALLY TOUCHES
- part = the larger part that CONTAINS the component (or null if component IS the main part)

Return JSON array:
[
  {{
    "step_index": 1,
    "parts": ["all parts mentioned in step"],
    "primary_part": "main part being worked on",
    "components_per_action": [
      {{"action_id": 0, "component": "what you touch", "part": "what contains it or null"}},
      {{"action_id": 1, "component": "what you touch", "part": "what contains it or null"}}
    ]
  }},
  ...
]

Return exactly {guide.num_steps} entries."""

    def parse_response_v2(
        self,
        response: str,
        guide: PreWEGGuide,
        **kwargs,
    ) -> PartsTextPerGuideV2:
        """Parse V2 LLM response into structured parts with components and parts per action."""
        try:
            data = extract_json_from_response(response)
        except ValueError as e:
            self.log_error(f"Failed to parse JSON: {e}")
            return PartsTextPerGuideV2(guide_id=guide.guide_id, device_type=guide.device_type, steps=[])

        if not isinstance(data, list):
            self.log_error(f"Expected list, got {type(data)}")
            return PartsTextPerGuideV2(guide_id=guide.guide_id, device_type=guide.device_type, steps=[])

        steps = []
        for item in data:
            if not isinstance(item, dict):
                continue

            step_index = item.get("step_index", 0)
            parts = item.get("parts", [])
            primary = item.get("primary_part")

            if isinstance(parts, list):
                parts = [str(p).strip() for p in parts if str(p).strip()]
            else:
                parts = []

            # Parse components per action (now includes part field)
            components_per_action = []
            raw_components = item.get("components_per_action", [])
            if isinstance(raw_components, list):
                for comp_data in raw_components:
                    if isinstance(comp_data, dict):
                        action_id = comp_data.get("action_id", 0)
                        component = comp_data.get("component", "")
                        part = comp_data.get("part")  # Can be None
                        if component:
                            components_per_action.append(ComponentPerAction(
                                action_id=action_id,
                                component=component.strip(),
                                part=part.strip() if part else None,
                            ))

            steps.append(PartTextResultV2(
                step_index=step_index,
                device_type=guide.device_type,
                parts=parts,
                primary_part=primary,
                components_per_action=components_per_action,
            ))

        return PartsTextPerGuideV2(
            guide_id=guide.guide_id, 
            device_type=guide.device_type,
            steps=steps,
        )

    def run_with_actions(
        self, 
        guide: PreWEGGuide, 
        actions: ActionsPerGuide,
        **kwargs,
    ) -> PartsTextPerGuideV2:
        """
        V2 Pipeline: Extract parts and components per action.
        
        Args:
            guide: The PreWEG guide
            actions: Action extraction results from Action Agent
        
        Returns:
            Parts and components for each step/action
        """
        self.log(f"[V2] Extracting parts and components for guide: {guide.title}")
        self.log(f"[V2] Device type: {guide.device_type}")
        
        prompt = self.build_prompt_v2(guide, actions, **kwargs)
        system_prompt = self.get_system_prompt_v2()
        
        response = self.client.complete(prompt, system_prompt=system_prompt)
        result = self.parse_response_v2(response, guide, **kwargs)
        
        total_components = sum(len(s.components_per_action) for s in result.steps)
        self.log(f"[V2] Extracted {total_components} components from {len(result.steps)} steps")
        
        return result

    def refine_with_actions(
        self,
        guide: PreWEGGuide,
        actions: ActionsPerGuide,
        step_index: int,
        feedback: str,
    ) -> PartTextResultV2:
        """V2 Pipeline: Refine component/part extraction based on reviewer feedback."""
        self.log(f"[V2] Refining step {step_index} based on feedback")
        
        step = guide.get_step(step_index)
        if not step:
            raise ValueError(f"Step {step_index} not found")
        
        # Find actions for this step
        step_actions = None
        for a in actions.steps:
            if a.step_index == step_index:
                step_actions = a
                break
        
        action_list = []
        if step_actions:
            for i, q in enumerate(step_actions.action_quadruples):
                action_list.append({
                    "action_id": i,
                    "action": q.action,
                    "full_action": q.full_action,
                })

        prompt = f"""For each action, identify the COMPONENT (what you touch) and PART (what contains it).

REVIEWER FEEDBACK - PLEASE ADDRESS:
{feedback}

Device Type: {guide.device_type.upper()} (DO NOT use "{guide.device_type}" as a component!)

Step {step_index}:
Description: \"\"\"{step.full_description}\"\"\"
Actions to analyze:
{action_list}

REMEMBER:
- component = what your hand/tool PHYSICALLY TOUCHES
- part = the larger part that CONTAINS the component (or null if component IS the main part)

Return JSON:
{{
  "step_index": {step_index},
  "parts": ["all parts mentioned"],
  "primary_part": "main part",
  "components_per_action": [
    {{"action_id": 0, "component": "what you touch", "part": "what contains it or null"}}
  ]
}}"""

        system_prompt = self.get_system_prompt_v2()
        response = self.client.complete(prompt, system_prompt=system_prompt)
        
        try:
            data = extract_json_from_response(response)
            if not isinstance(data, dict):
                return PartTextResultV2(step_index=step_index, device_type=guide.device_type)
            
            parts = data.get("parts", [])
            if isinstance(parts, list):
                parts = [str(p).strip() for p in parts if str(p).strip()]
            else:
                parts = []
            
            components_per_action = []
            raw_components = data.get("components_per_action", [])
            if isinstance(raw_components, list):
                for comp_data in raw_components:
                    if isinstance(comp_data, dict):
                        part = comp_data.get("part")
                        components_per_action.append(ComponentPerAction(
                            action_id=comp_data.get("action_id", 0),
                            component=comp_data.get("component", "").strip(),
                            part=part.strip() if part else None,
                        ))
            
            return PartTextResultV2(
                step_index=step_index,
                device_type=guide.device_type,
                parts=parts,
                primary_part=data.get("primary_part"),
                components_per_action=components_per_action,
            )
        except ValueError as e:
            self.log_error(f"Failed to parse refinement response: {e}")
            return PartTextResultV2(step_index=step_index, device_type=guide.device_type)
