"""
Hands Estimation Agent - Estimates the number of hands needed for each step.

Uses LLM to analyze actions and context to determine if 0, 1, or 2 hands are needed.
"""
from typing import Optional

from ..models.preweg import PreWEGGuide
from ..models.intermediate import HandsEstimationResult, HandsPerGuide
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
