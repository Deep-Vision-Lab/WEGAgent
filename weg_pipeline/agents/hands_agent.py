"""
Hands Estimation Agent - Estimates the number of hands needed for each step.

Uses LLM to analyze actions and context to determine if 0, 1, or 2 hands are needed.
V2: Estimates hands per action.
"""
from typing import Optional

from ..models.preweg import PreWEGGuide
from ..models.intermediate import (
    HandsEstimationResult, 
    HandsPerGuide,
    HandsEstimationResultV2,
    HandsPerGuideV2,
    HandsPerAction,
    ActionsPerGuide,
)
from ..utils.llm_utils import extract_json_from_response
from .base_agent import BaseAgent


class HandsAgent(BaseAgent):
    """
    Estimates the number of hands required for each step.
    
    Values:
    - 0: No hands needed (information only, safety warning)
    - 1: One hand sufficient (simple actions like pressing, pulling small parts)
    - 2: Two hands needed (holding + manipulating, lifting heavy parts, stabilizing)
    """

    AGENT_NAME = "HandsAgent"
    REQUIRES_VISION = False
    DEFAULT_PROVIDER = "openai"
    DEFAULT_MODEL = "gpt-4o-mini"

    def get_system_prompt(self) -> str:
        return """You are an expert at analyzing repair tasks and estimating the physical effort required.

Your task is to estimate the NUMBER OF HANDS a person needs for each repair step.

Return values:
- 0 = No hands needed (pure information, safety warning, or no physical action)
- 1 = One hand sufficient (press button, pull small connector, use single tool)
- 2 = Two hands needed (hold + pull, stabilize + unscrew, lift large part, align components)

Guidelines for deciding:
- "Hold X while doing Y" → 2 hands
- "Support X and then..." → 2 hands  
- "Use both hands to..." → 2 hands
- "Lift the [heavy component]" → likely 2 hands
- "Press the button" → 1 hand
- "Unplug the cord" → 1 hand
- "Disconnect the connector" → usually 1 hand (unless mentioned to hold)
- "Remove screws" → 1 hand (tool in one hand)
- Safety notes alone → 0 hands

When uncertain, choose the LOWER number (bias toward 1 over 2).

Return ONLY valid JSON. No explanations, no markdown."""

    def build_prompt(self, guide: PreWEGGuide, **kwargs) -> str:
        """Build prompt for hands estimation."""
        steps_text = []
        for step in sorted(guide.steps, key=lambda s: s.step_index):
            steps_text.append(f"""
Step {step.step_index}:
\"\"\"
{step.full_description}
\"\"\"
""")

        return f"""Estimate the number of hands needed for each step.

Guide: {guide.title}

{chr(10).join(steps_text)}

Return JSON array of objects:
[
  {{"step_index": 1, "hands": 1, "reasoning": "brief reason"}},
  {{"step_index": 2, "hands": 2, "reasoning": "hold part while removing screws"}},
  ...
]

Return exactly {guide.num_steps} entries. Reasoning is optional but helpful."""

    def parse_response(
        self,
        response: str,
        guide: PreWEGGuide,
        **kwargs,
    ) -> HandsPerGuide:
        """Parse LLM response into structured hands estimation."""
        try:
            data = extract_json_from_response(response)
        except ValueError as e:
            self.log_error(f"Failed to parse JSON: {e}")
            return HandsPerGuide(guide_id=guide.guide_id, steps=[])

        if not isinstance(data, list):
            self.log_error(f"Expected list, got {type(data)}")
            return HandsPerGuide(guide_id=guide.guide_id, steps=[])

        steps = []
        for item in data:
            if not isinstance(item, dict):
                continue

            step_index = item.get("step_index", 0)
            hands = item.get("hands", 1)
            reasoning = item.get("reasoning")

            # Validate hands value
            try:
                hands = int(hands)
                hands = max(0, min(2, hands))  # Clamp to 0-2
            except (ValueError, TypeError):
                hands = 1  # Default

            steps.append(HandsEstimationResult(
                step_index=step_index,
                hands=hands,
                reasoning=reasoning,
            ))

        return HandsPerGuide(guide_id=guide.guide_id, steps=steps)

    # ==================== V2 Pipeline Methods ====================
    
    def get_system_prompt_v2(self) -> str:
        """System prompt for V2 pipeline - hands per action."""
        return """You are an expert at analyzing repair tasks and estimating the physical effort required.

Your task is to estimate the NUMBER OF HANDS needed for EACH specific action.

=== Return Values ===
- 0 = No hands needed (pure information, safety note, no physical action)
- 1 = One hand sufficient (single tool operation, press button, pull connector)
- 2 = Two hands needed (hold + manipulate, lift heavy part, stabilize + work)

=== Guidelines ===
ONE HAND (1):
- Using a screwdriver to remove screws
- Unplugging a cord (grab plug, pull)
- Pressing a button or tab
- Disconnecting a small connector
- Pulling out a small component

TWO HANDS (2):
- "Hold X while doing Y" explicitly mentioned
- Lifting large/heavy components (doors, panels, motors)
- Stabilizing + manipulating (hold part steady while unscrewing)
- Aligning components during installation
- Supporting weight while disconnecting

NO HANDS (0):
- Pure safety warnings
- Information-only steps
- "Note:" or "Warning:" statements

=== Bias ===
When uncertain, bias toward 1 hand (simpler is usually correct).

Return ONLY valid JSON. No explanations, no markdown."""

    def build_prompt_v2(
        self, 
        guide: PreWEGGuide, 
        actions: ActionsPerGuide,
        **kwargs,
    ) -> str:
        """Build prompt for V2 pipeline - hands per action."""
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
Actions:
{action_list}
""")

        return f"""Estimate the number of hands needed for each action.

Guide: {guide.title}

{chr(10).join(steps_data)}

Return JSON array:
[
  {{
    "step_index": 1,
    "hands": 1,
    "hands_per_action": [
      {{"action_id": 0, "hands": 1}},
      {{"action_id": 1, "hands": 2}}
    ]
  }},
  ...
]

Rules:
- "hands" at step level = maximum hands needed for any action in that step
- "hands_per_action" = hands for each specific action
- Values: 0, 1, or 2 only

Return exactly {guide.num_steps} entries."""

    def parse_response_v2(
        self,
        response: str,
        guide: PreWEGGuide,
        **kwargs,
    ) -> HandsPerGuideV2:
        """Parse V2 LLM response into structured hands per action."""
        try:
            data = extract_json_from_response(response)
        except ValueError as e:
            self.log_error(f"Failed to parse JSON: {e}")
            return HandsPerGuideV2(guide_id=guide.guide_id, steps=[])

        if not isinstance(data, list):
            self.log_error(f"Expected list, got {type(data)}")
            return HandsPerGuideV2(guide_id=guide.guide_id, steps=[])

        steps = []
        for item in data:
            if not isinstance(item, dict):
                continue

            step_index = item.get("step_index", 0)
            hands = item.get("hands", 1)

            # Validate step-level hands value
            try:
                hands = int(hands)
                hands = max(0, min(2, hands))
            except (ValueError, TypeError):
                hands = 1

            # Parse hands per action
            hands_per_action = []
            raw_hands = item.get("hands_per_action", [])
            if isinstance(raw_hands, list):
                for hands_data in raw_hands:
                    if isinstance(hands_data, dict):
                        action_id = hands_data.get("action_id", 0)
                        action_hands = hands_data.get("hands", 1)
                        try:
                            action_hands = int(action_hands)
                            action_hands = max(0, min(2, action_hands))
                        except (ValueError, TypeError):
                            action_hands = 1
                        hands_per_action.append(HandsPerAction(
                            action_id=action_id,
                            hands=action_hands,
                        ))

            steps.append(HandsEstimationResultV2(
                step_index=step_index,
                hands=hands,
                hands_per_action=hands_per_action,
            ))

        return HandsPerGuideV2(guide_id=guide.guide_id, steps=steps)

    def run_for_actions(
        self, 
        guide: PreWEGGuide, 
        actions: ActionsPerGuide,
        **kwargs,
    ) -> HandsPerGuideV2:
        """
        V2 Pipeline: Estimate hands per action.
        
        Args:
            guide: The PreWEG guide
            actions: Action extraction results from Action Agent
        
        Returns:
            Hands estimation for each step and action
        """
        self.log(f"[V2] Estimating hands per action for guide: {guide.title}")
        
        prompt = self.build_prompt_v2(guide, actions, **kwargs)
        system_prompt = self.get_system_prompt_v2()
        
        response = self.client.complete(prompt, system_prompt=system_prompt)
        result = self.parse_response_v2(response, guide, **kwargs)
        
        total_estimates = sum(len(s.hands_per_action) for s in result.steps)
        self.log(f"[V2] Estimated hands for {total_estimates} actions from {len(result.steps)} steps")
        
        return result

    def refine_for_actions(
        self,
        guide: PreWEGGuide,
        actions: ActionsPerGuide,
        step_index: int,
        feedback: str,
    ) -> HandsEstimationResultV2:
        """V2 Pipeline: Refine hands estimation based on reviewer feedback."""
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

        prompt = f"""Estimate the number of hands needed for each action.

REVIEWER FEEDBACK - PLEASE ADDRESS:
{feedback}

Step {step_index}:
Description: \"\"\"{step.full_description}\"\"\"
Actions:
{action_list}

Return JSON:
{{
  "step_index": {step_index},
  "hands": 1,
  "hands_per_action": [
    {{"action_id": 0, "hands": 1}}
  ]
}}"""

        system_prompt = self.get_system_prompt_v2()
        response = self.client.complete(prompt, system_prompt=system_prompt)
        
        try:
            data = extract_json_from_response(response)
            if not isinstance(data, dict):
                return HandsEstimationResultV2(step_index=step_index, hands=1)
            
            hands = data.get("hands", 1)
            try:
                hands = int(hands)
                hands = max(0, min(2, hands))
            except (ValueError, TypeError):
                hands = 1
            
            hands_per_action = []
            raw_hands = data.get("hands_per_action", [])
            if isinstance(raw_hands, list):
                for hands_data in raw_hands:
                    if isinstance(hands_data, dict):
                        action_hands = hands_data.get("hands", 1)
                        try:
                            action_hands = int(action_hands)
                            action_hands = max(0, min(2, action_hands))
                        except (ValueError, TypeError):
                            action_hands = 1
                        hands_per_action.append(HandsPerAction(
                            action_id=hands_data.get("action_id", 0),
                            hands=action_hands,
                        ))
            
            return HandsEstimationResultV2(
                step_index=step_index,
                hands=hands,
                hands_per_action=hands_per_action,
            )
        except ValueError as e:
            self.log_error(f"Failed to parse refinement response: {e}")
            return HandsEstimationResultV2(step_index=step_index, hands=1)
