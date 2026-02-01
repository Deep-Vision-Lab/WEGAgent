"""
Action Extraction Agent - Extracts atomic actions from repair steps.

Uses VLM (Vision-Language Model) to analyze both text and images,
identifying specific actionable instructions with structured quadruples.
"""
from pathlib import Path
from typing import Any, Optional

from ..models.preweg import PreWEGGuide
from ..models.intermediate import ActionExtractionResult, ActionsPerGuide, ActionQuadruple
from ..utils.llm_utils import extract_json_from_response
from ..utils.logger import get_logger, LogLevel
from .base_agent import BaseAgent


class ActionAgent(BaseAgent):
    """
    Extracts atomic actions from repair guide steps.
    
    An action is a single, concrete instruction that can be performed,
    typically starting with an imperative verb (e.g., "Remove", "Disconnect", "Pull").
    
    Also extracts:
    - Task name for each step
    - Hints/tips/warnings (non-action information)
    - Action quadruples: <Action, Tool, Component, Hands>
    
    IMPORTANT: This agent must REASON about components, not just extract text.
    """

    AGENT_NAME = "ActionAgent"
    REQUIRES_VISION = True  # Can use images for better context
    DEFAULT_PROVIDER = "google"
    DEFAULT_MODEL = "gemini-1.5-flash"

    def get_system_prompt(self) -> str:
        return """You are an expert at analyzing appliance repair guides and extracting structured action information.

Your task is to convert repair step descriptions into:
1. A TASK NAME - a short title describing what the step accomplishes
2. ATOMIC ACTIONS - each with a structured quadruple
3. HINTS - tips, warnings, notes that are not actions

=== TASK NAME ===
A brief 2-5 word title summarizing the step (e.g., "Remove drawer", "Disconnect power", "Reassembly notes")

=== ATOMIC ACTION ===
A single, concrete instruction that a person can perform:
- Starts with an imperative verb (Remove, Disconnect, Pull, Press, Rotate, etc.)
- One specific operation per action
- Should be executable without further breakdown

For EACH action, provide a QUADRUPLE:
{
  "action": "the action verb extracted from THIS SPECIFIC action sentence",
  "precise_action": "a more specific/technical verb if applicable (e.g., 'unscrew' instead of 'remove')",
  "tool": "tool name or null if bare hands",
  "component": "THE ACTUAL PHYSICAL COMPONENT BEING TOUCHED/MANIPULATED",
  "hands": number of hands needed (0=no hands/info only, 1=one hand, 2=two hands),
  "full_action": "the complete action sentence"
}

=== CRITICAL: ACTION VERB EXTRACTION ===
The "action" field MUST be extracted from each individual action sentence (full_action), NOT from the step description!

IMPORTANT: Each action sentence has its OWN verb. Extract the verb from THAT sentence specifically.

EXAMPLE - Step description: "Rinse thoroughly and reattach the spray arm to the dishwasher."
This becomes MULTIPLE actions:
1. "Rinse thoroughly..." → action: "rinse" (from this sentence)
2. "Attach the longer spray arm to the bottom" → action: "attach" (from THIS sentence, NOT "reattach" from description!)
3. "Attach the shorter spray arm to the top" → action: "attach" (from THIS sentence)

WRONG: Using "reattach" for action #2 because the description said "reattach"
CORRECT: Using "attach" for action #2 because THAT sentence starts with "Attach"

=== PRECISE ACTION REASONING ===
The "precise_action" field captures the MORE SPECIFIC technical action when applicable.

REASONING EXAMPLES:

EXAMPLE 1: "Use a Phillips screwdriver to remove the screws securing the drawer"
- action: "remove" (what the sentence says)
- precise_action: "unscrew" (more precise - you're turning/unscrewing, not just removing)
- Reasoning: When using a screwdriver on screws, you're UNSCREWING them

EXAMPLE 2: "Pull open the freezer door"
- action: "pull" (the motion described)
- precise_action: "open" (the actual goal/result)
- Reasoning: The purpose is to OPEN the door, pulling is just the method

EXAMPLE 3: "Remove the bolts from the bracket"
- action: "remove"
- precise_action: "unbolt" (technical term for removing bolts)

EXAMPLE 4: "Disconnect the wire harness connector"
- action: "disconnect" (already precise)
- precise_action: null (no more precise term needed)

EXAMPLE 5: "Lift and remove the upper drawer"
- This should be TWO separate actions:
  1. action: "lift", precise_action: null
  2. action: "remove", precise_action: null

WHEN TO USE precise_action:
- "remove screws" → precise_action: "unscrew"
- "remove bolts" → precise_action: "unbolt"
- "remove nuts" → precise_action: "unthread" or "loosen"
- "remove clips" → precise_action: "unclip"
- "pull to open" → precise_action: "open"
- "push to close" → precise_action: "close"
- "turn to loosen" → precise_action: "loosen"
- If already precise (unplug, disconnect, squeeze), set precise_action to null

=== CRITICAL: COMPONENT REASONING ===
The "component" field must be the ACTUAL PHYSICAL PART your hands touch, NOT the device name!

You MUST REASON about what is physically being manipulated:

EXAMPLE 1: "Unplug your refrigerator"
- WRONG component: "refrigerator" (you don't touch the whole refrigerator)
- CORRECT component: "power cord plug" or "electrical plug" (this is what your hand grabs)
- Reasoning: When you unplug something, your hand grabs the PLUG, not the appliance

EXAMPLE 2: "Remove the screw securing the drawer"  
- WRONG component: "drawer" (you're not touching the drawer, you're removing the screw)
- CORRECT component: "drawer mounting screw" (this is what the tool touches)

EXAMPLE 3: "Open the refrigerator door"
- CORRECT component: "refrigerator door" or "door handle" (you physically touch the door/handle)

EXAMPLE 4: "Disconnect the wire harness connector"
- WRONG component: "wire harness" (too general)
- CORRECT component: "wire harness connector" (the specific thing you grab and pull)

ASK YOURSELF: "What does my hand (or tool) PHYSICALLY TOUCH during this action?"

=== HINTS ===
Non-action information such as:
- Safety warnings (e.g., "Don't try to remove fully as it's still connected")
- Tips (e.g., "The covers can be tricky to reinstall")
- Notes for later (e.g., "Make a note of the connections for re-installation")
- Background information (e.g., "If you are testing X, you may also need...")

=== HANDS ESTIMATION ===
Estimate hands required based on:
- 0 hands: Pure information/warning with no physical action
- 1 hand: Simple operations (turning screws, pressing buttons, plugging/unplugging connectors)
- 2 hands: Lifting heavy objects, holding + manipulating, stabilizing + pulling

Return ONLY valid JSON. No explanations, no markdown."""

    def build_prompt(self, guide: PreWEGGuide, **kwargs) -> str:
        """Build prompt for full guide or single step."""
        step_index = kwargs.get("step_index")
        feedback = kwargs.get("feedback")  # Reviewer feedback for refinement
        toolbox = guide.toolbox if guide.toolbox else []
        toolbox_str = ", ".join(toolbox) if toolbox else "None specified"

        if step_index is not None:
            # Single step mode
            step = guide.get_step(step_index)
            if not step:
                raise ValueError(f"Step {step_index} not found in guide")

            feedback_section = ""
            if feedback:
                feedback_section = f"""
=== REVIEWER FEEDBACK - PLEASE CORRECT ===
The reviewer found issues with previous extraction:
{feedback}

Please carefully re-analyze and provide corrected extraction.
=======================================
"""

            return f"""Extract structured action information from this repair step.

Available Tools: {toolbox_str}
{feedback_section}
Step {step_index}:
\"\"\"
{step.full_description}
\"\"\"

REMEMBER: The "component" must be what you PHYSICALLY TOUCH, not the device name!
- "Unplug the refrigerator" → component is "power cord plug", NOT "refrigerator"
- "Remove the screw" → component is "screw", NOT what the screw holds

CRITICAL: Extract "action" from EACH individual action sentence, not the description!
- "Attach the spray arm" → action: "attach" (from THIS sentence)
- Don't use verbs from the general description for other action sentences

REMEMBER: Extract BOTH action verbs:
- "action": the verb from THIS SPECIFIC action sentence (full_action)
- "precise_action": a more technical/specific verb if applicable, or null

Return JSON:
{{
  "step_index": {step_index},
  "task_name": "short task title",
  "actions": [
    {{
      "action": "verb extracted from THIS action sentence (full_action)",
      "precise_action": "more specific verb or null",
      "tool": "tool name or null",
      "component": "the PHYSICAL part being touched/manipulated",
      "hands": 1,
      "full_action": "full action sentence",
      "reasoning": "brief explanation of why this component (what does the hand touch?)"
    }}
  ],
  "hints": ["hint1", "hint2"]
}}"""

        # Full guide mode
        steps_text = []
        for step in sorted(guide.steps, key=lambda s: s.step_index):
            steps_text.append(f"""
Step {step.step_index}:
\"\"\"
{step.full_description}
\"\"\"
""")

        return f"""Extract structured action information from ALL steps of this repair guide.

Guide: {guide.title}
Available Tools: {toolbox_str}

{chr(10).join(steps_text)}

CRITICAL REMINDER - COMPONENT REASONING:
The "component" must be what you PHYSICALLY TOUCH during the action!
- "Unplug the refrigerator" → component: "power cord plug" (what your hand grabs)
- "Remove the drawer screws" → component: "drawer screws" (what the tool touches)
- "Open the freezer door" → component: "freezer door" or "door handle"

CRITICAL REMINDER - ACTION VERB EXTRACTION:
The "action" field MUST come from EACH individual action sentence, NOT the step description!

EXAMPLE - If description says "reattach the spray arm" but you split into multiple actions:
- "Rinse the spray arm" → action: "rinse" ✓
- "Attach the longer spray arm to bottom" → action: "attach" ✓ (NOT "reattach"!)
- "Attach the shorter spray arm to top" → action: "attach" ✓ (NOT "reattach"!)

Extract the verb from each full_action sentence independently!

Also extract "precise_action" when a more technical verb applies:
- "remove the screws" → action: "remove", precise_action: "unscrew"
- "pull open the door" → action: "pull", precise_action: "open"

Return JSON array with one object per step:
[
  {{
    "step_index": 1,
    "task_name": "short task title",
    "actions": [
      {{
        "action": "verb extracted from THIS action's full_action sentence",
        "precise_action": "more specific verb or null",
        "tool": "tool name or null", 
        "component": "the PHYSICAL part being touched",
        "hands": 1,
        "full_action": "full action sentence",
        "reasoning": "brief explanation of component choice"
      }}
    ],
    "hints": ["hint1", "hint2"]
  }},
  ...
]

IMPORTANT: 
- Return exactly {guide.num_steps} objects, one per step
- Empty arrays for steps with no actions/hints
- Use tools from the toolbox when mentioned in the text
- "component" should be the PHYSICAL OBJECT being touched (not the device/appliance)
- "action" must match the verb in the full_action sentence"""

    def parse_response(
        self,
        response: str,
        guide: PreWEGGuide,
        **kwargs,
    ) -> ActionsPerGuide | ActionExtractionResult:
        """Parse LLM response into structured actions."""
        step_index = kwargs.get("step_index")

        try:
            data = extract_json_from_response(response)
            # Log if we got partial data (less steps than expected)
            if isinstance(data, list) and len(data) < guide.num_steps:
                self.log(f"Warning: Only parsed {len(data)} of {guide.num_steps} steps (response may have been truncated)")
        except ValueError as e:
            self.log_error(f"Failed to parse JSON: {e}")
            # Return empty result on failure
            if step_index is not None:
                return ActionExtractionResult(step_index=step_index, actions=[])
            return ActionsPerGuide(guide_id=guide.guide_id, steps=[])

        if step_index is not None:
            # Single step mode
            return self._parse_single_step(data, step_index)

        # Full guide mode - expect list of objects
        if not isinstance(data, list):
            self.log_error(f"Expected list, got {type(data)}")
            return ActionsPerGuide(guide_id=guide.guide_id, steps=[])

        steps = []
        for step_data in data:
            if isinstance(step_data, dict):
                idx = step_data.get("step_index", len(steps) + 1)
                steps.append(self._parse_single_step(step_data, idx))
            else:
                # Fallback for old format (list of action strings)
                idx = len(steps) + 1
                if isinstance(step_data, list):
                    actions = [str(a).strip() for a in step_data if str(a).strip()]
                    steps.append(ActionExtractionResult(step_index=idx, actions=actions))
                else:
                    steps.append(ActionExtractionResult(step_index=idx, actions=[]))

        return ActionsPerGuide(guide_id=guide.guide_id, steps=steps)

    def _parse_single_step(self, data: dict | list, step_index: int) -> ActionExtractionResult:
        """Parse a single step's data into ActionExtractionResult."""
        if isinstance(data, list):
            # Old format: just list of action strings
            actions = [str(a).strip() for a in data if str(a).strip()]
            return ActionExtractionResult(step_index=step_index, actions=actions)

        if not isinstance(data, dict):
            return ActionExtractionResult(step_index=step_index, actions=[])

        # New format with quadruples
        task_name = data.get("task_name")
        hints = data.get("hints", [])
        if not isinstance(hints, list):
            hints = []
        hints = [str(h).strip() for h in hints if str(h).strip()]

        raw_actions = data.get("actions", [])
        if not isinstance(raw_actions, list):
            raw_actions = []

        actions = []
        quadruples = []

        for action_data in raw_actions:
            if isinstance(action_data, str):
                # Old format: just string
                actions.append(action_data.strip())
            elif isinstance(action_data, dict):
                # New format: quadruple object
                full_action = action_data.get("full_action", "")
                if full_action:
                    actions.append(full_action.strip())

                # Build quadruple
                action_verb = action_data.get("action", "").lower().strip()
                tool = action_data.get("tool")
                if tool and isinstance(tool, str):
                    tool = tool.strip() if tool.strip() else None
                component = action_data.get("component", "unknown")
                hands = action_data.get("hands", 1)
                
                # Validate hands
                if not isinstance(hands, int) or hands < 0:
                    hands = 1
                elif hands > 2:
                    hands = 2

                # Extract precise_action if available
                precise_action = action_data.get("precise_action")
                if precise_action and isinstance(precise_action, str):
                    precise_action = precise_action.lower().strip() if precise_action.strip() else None
                else:
                    precise_action = None

                if action_verb and component:
                    quadruples.append(ActionQuadruple(
                        action=action_verb,
                        precise_action=precise_action,
                        tool=tool,
                        component=component,
                        hands=hands,
                        full_action=full_action or f"{action_verb} {component}",
                    ))
                    
                    # Log reasoning if available
                    logger = get_logger()
                    if logger and action_data.get("reasoning"):
                        precise_info = f" (precise: {precise_action})" if precise_action else ""
                        logger.agent_reasoning(
                            self.AGENT_NAME,
                            step_index,
                            full_action,
                            action_data.get("reasoning", ""),
                            f"{action_verb}{precise_info} {component}",
                        )

        return ActionExtractionResult(
            step_index=step_index,
            task_name=task_name,
            actions=actions,
            action_quadruples=quadruples,
            hints=hints,
        )

    def refine_step(
        self,
        guide: PreWEGGuide,
        step_index: int,
        feedback: str,
        original_result: Optional[ActionExtractionResult] = None,
    ) -> ActionExtractionResult:
        """
        Refine extraction for a single step based on reviewer feedback.
        
        This method is called by the reviewer when issues are detected.
        
        Args:
            guide: The PreWEG guide
            step_index: Which step to refine
            feedback: Specific feedback from reviewer on what to fix
            original_result: The original extraction that needs refinement
        
        Returns:
            Refined ActionExtractionResult
        """
        logger = get_logger()
        if logger:
            logger.log(
                LogLevel.AGENT_ACTION,
                self.AGENT_NAME,
                f"Refining step based on reviewer feedback",
                step_index=step_index,
                details={"feedback": feedback},
            )
        
        self.log(f"Refining step {step_index} based on feedback: {feedback[:100]}...")
        
        # Build prompt with feedback
        prompt = self.build_prompt(guide, step_index=step_index, feedback=feedback)
        system_prompt = self.get_system_prompt()
        
        response = self.client.complete(prompt, system_prompt=system_prompt)
        refined_result = self.parse_response(response, guide, step_index=step_index)
        
        # Log the refinement
        if logger and original_result:
            logger.agent_response_to_feedback(
                self.AGENT_NAME,
                step_index,
                original_value={
                    "actions": original_result.actions,
                    "quadruples": [q.component for q in original_result.action_quadruples],
                },
                corrected_value={
                    "actions": refined_result.actions,
                    "quadruples": [q.component for q in refined_result.action_quadruples],
                },
                explanation=f"Refined based on feedback: {feedback[:100]}",
            )
        
        return refined_result

    def run_with_images(
        self,
        guide: PreWEGGuide,
        step_index: int,
        image_paths: list[Path | str],
    ) -> ActionExtractionResult:
        """
        Run action extraction on a single step with its images.
        
        This provides more context from visual information.
        """
        step = guide.get_step(step_index)
        if not step:
            raise ValueError(f"Step {step_index} not found")

        if not image_paths:
            # Fall back to text-only
            return self.run(guide, step_index=step_index)

        toolbox = guide.toolbox if guide.toolbox else []
        toolbox_str = ", ".join(toolbox) if toolbox else "None specified"

        # Use first image for now (can be extended to multiple)
        prompt = f"""Extract structured action information from this repair step. Use BOTH the text description AND the image to understand what actions are performed.

Available Tools: {toolbox_str}

Step {step_index}:
\"\"\"
{step.full_description}
\"\"\"

The image shows the current state or action being performed in this step.

Return JSON:
{{
  "step_index": {step_index},
  "task_name": "short task title",
  "actions": [
    {{
      "action": "verb",
      "tool": "tool name or null",
      "component": "component name",
      "hands": 1,
      "full_action": "full action sentence"
    }}
  ],
  "hints": ["hint1", "hint2"]
}}"""

        system_prompt = self.get_system_prompt()

        self.log(f"Processing step {step_index} with image")
        response = self.client.complete_with_image(
            prompt,
            image_paths[0],
            system_prompt=system_prompt,
        )

        return self.parse_response(response, guide, step_index=step_index)

    # ==================== V2 Pipeline Methods ====================
    
    def get_system_prompt_v2(self) -> str:
        """System prompt for V2 pipeline - actions only, no tool/component/hands."""
        return """You are an expert at analyzing appliance repair guides and extracting action information.

Your task is to extract ONLY the actions from repair step descriptions:
1. A TASK NAME - a short title describing what the step accomplishes
2. ATOMIC ACTIONS - individual actionable instructions
3. HINTS - tips, warnings, notes that are not actions

=== TASK NAME ===
A brief 2-5 word title summarizing the step (e.g., "Remove drawer", "Disconnect power")

=== ATOMIC ACTION ===
A single, concrete instruction that a person can perform:
- Starts with an imperative verb (Remove, Disconnect, Pull, Press, Rotate, etc.)
- One specific operation per action
- Should be executable without further breakdown

For EACH action, extract:
{
  "action": "the main action verb (remove, pull, disconnect, etc.)",
  "precise_action": "a more specific/technical verb if applicable, or null",
  "full_action": "the complete action sentence",
  "target_description": "what the action targets (raw text from description)"
}

=== PRECISE ACTION ===
The "precise_action" field captures the MORE SPECIFIC technical action:
- "remove screws" → precise_action: "unscrew"
- "remove bolts" → precise_action: "unbolt"
- "pull to open" → precise_action: "open"
- If already precise (unplug, disconnect), set to null

=== HINTS ===
Non-action information such as:
- Safety warnings
- Tips for easier work
- Notes for later steps
- Background information

=== IMPORTANT ===
In V2 mode, you do NOT extract:
- Tools (another agent handles this)
- Components (another agent handles this)
- Hands count (another agent handles this)

Return ONLY valid JSON. No explanations, no markdown."""

    def build_prompt_v2(self, guide: PreWEGGuide, **kwargs) -> str:
        """Build prompt for V2 pipeline - actions only."""
        step_index = kwargs.get("step_index")

        if step_index is not None:
            # Single step mode
            step = guide.get_step(step_index)
            if not step:
                raise ValueError(f"Step {step_index} not found in guide")

            return f"""Extract action information from this repair step.

Step {step_index}:
\"\"\"
{step.full_description}
\"\"\"

Return JSON:
{{
  "step_index": {step_index},
  "task_name": "short task title",
  "actions": [
    {{
      "action": "main verb",
      "precise_action": "specific verb or null",
      "full_action": "full action sentence",
      "target_description": "what the action targets"
    }}
  ],
  "hints": ["hint1", "hint2"]
}}"""

        # Full guide mode
        steps_text = []
        for step in sorted(guide.steps, key=lambda s: s.step_index):
            steps_text.append(f"""
Step {step.step_index}:
\"\"\"
{step.full_description}
\"\"\"
""")

        return f"""Extract action information from ALL steps of this repair guide.

Guide: {guide.title}
Device Type: {guide.device_type}

{chr(10).join(steps_text)}

Return JSON array with one object per step:
[
  {{
    "step_index": 1,
    "task_name": "short task title",
    "actions": [
      {{
        "action": "main verb",
        "precise_action": "specific verb or null",
        "full_action": "full action sentence",
        "target_description": "what the action targets"
      }}
    ],
    "hints": ["hint1", "hint2"]
  }},
  ...
]

Return exactly {guide.num_steps} objects, one per step."""

    def parse_response_v2(
        self,
        response: str,
        guide: PreWEGGuide,
        **kwargs,
    ) -> ActionsPerGuide | ActionExtractionResult:
        """Parse V2 LLM response into structured actions (no tool/component/hands)."""
        step_index = kwargs.get("step_index")

        try:
            data = extract_json_from_response(response)
        except ValueError as e:
            self.log_error(f"Failed to parse JSON: {e}")
            if step_index is not None:
                return ActionExtractionResult(step_index=step_index, actions=[])
            return ActionsPerGuide(guide_id=guide.guide_id, steps=[])

        if step_index is not None:
            return self._parse_single_step_v2(data, step_index)

        # Full guide mode
        if not isinstance(data, list):
            self.log_error(f"Expected list, got {type(data)}")
            return ActionsPerGuide(guide_id=guide.guide_id, steps=[])

        steps = []
        for step_data in data:
            if isinstance(step_data, dict):
                idx = step_data.get("step_index", len(steps) + 1)
                steps.append(self._parse_single_step_v2(step_data, idx))

        return ActionsPerGuide(guide_id=guide.guide_id, steps=steps)

    def _parse_single_step_v2(self, data: dict, step_index: int) -> ActionExtractionResult:
        """Parse a single step's V2 data (actions only, placeholder for tool/component/hands)."""
        if not isinstance(data, dict):
            return ActionExtractionResult(step_index=step_index, actions=[])

        task_name = data.get("task_name")
        hints = data.get("hints", [])
        if not isinstance(hints, list):
            hints = []
        hints = [str(h).strip() for h in hints if str(h).strip()]

        raw_actions = data.get("actions", [])
        if not isinstance(raw_actions, list):
            raw_actions = []

        actions = []
        quadruples = []

        for action_data in raw_actions:
            if isinstance(action_data, str):
                actions.append(action_data.strip())
                # Create placeholder quadruple
                quadruples.append(ActionQuadruple(
                    action=action_data.split()[0].lower() if action_data.split() else "action",
                    precise_action=None,
                    tool=None,  # V2: filled by Tool Agent
                    component=action_data,  # V2: filled by Part Text Agent
                    hands=0,  # V2: filled by Hands Agent
                    full_action=action_data,
                ))
            elif isinstance(action_data, dict):
                full_action = action_data.get("full_action", "")
                if full_action:
                    actions.append(full_action.strip())

                action_verb = action_data.get("action", "").lower().strip()
                precise_action = action_data.get("precise_action")
                if precise_action and isinstance(precise_action, str):
                    precise_action = precise_action.lower().strip() if precise_action.strip() else None
                else:
                    precise_action = None

                target_desc = action_data.get("target_description", full_action)

                if action_verb:
                    quadruples.append(ActionQuadruple(
                        action=action_verb,
                        precise_action=precise_action,
                        tool=None,  # V2: filled by Tool Agent
                        component=target_desc,  # V2: placeholder, filled by Part Text Agent
                        hands=0,  # V2: filled by Hands Agent
                        full_action=full_action or f"{action_verb} {target_desc}",
                    ))

        return ActionExtractionResult(
            step_index=step_index,
            task_name=task_name,
            actions=actions,
            action_quadruples=quadruples,
            hints=hints,
        )

    def run_actions_only(self, guide: PreWEGGuide, **kwargs) -> ActionsPerGuide:
        """
        V2 Pipeline: Extract actions only (no tool/component/hands).
        
        Other agents will fill in tool, component, and hands separately.
        """
        self.log(f"[V2] Extracting actions only for guide: {guide.title}")
        
        prompt = self.build_prompt_v2(guide, **kwargs)
        system_prompt = self.get_system_prompt_v2()
        
        response = self.client.complete(prompt, system_prompt=system_prompt)
        result = self.parse_response_v2(response, guide, **kwargs)
        
        if isinstance(result, ActionsPerGuide):
            total_actions = sum(len(s.actions) for s in result.steps)
            self.log(f"[V2] Extracted {total_actions} actions from {len(result.steps)} steps")
        
        return result

    def refine_actions_only(
        self,
        guide: PreWEGGuide,
        step_index: int,
        feedback: str,
    ) -> ActionExtractionResult:
        """V2 Pipeline: Refine action extraction based on reviewer feedback."""
        self.log(f"[V2] Refining step {step_index} based on feedback")
        
        step = guide.get_step(step_index)
        if not step:
            raise ValueError(f"Step {step_index} not found")

        prompt = f"""Extract action information from this repair step.

REVIEWER FEEDBACK - PLEASE ADDRESS:
{feedback}

Step {step_index}:
\"\"\"
{step.full_description}
\"\"\"

Return JSON:
{{
  "step_index": {step_index},
  "task_name": "short task title",
  "actions": [
    {{
      "action": "main verb",
      "precise_action": "specific verb or null",
      "full_action": "full action sentence",
      "target_description": "what the action targets"
    }}
  ],
  "hints": ["hint1", "hint2"]
}}"""

        system_prompt = self.get_system_prompt_v2()
        response = self.client.complete(prompt, system_prompt=system_prompt)
        return self.parse_response_v2(response, guide, step_index=step_index)
