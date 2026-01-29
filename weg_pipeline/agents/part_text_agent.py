"""
Part Text Agent - Extracts part names from step descriptions.

Uses LLM to identify parts being manipulated in each step based on text analysis.
"""
from typing import Optional

from ..models.preweg import PreWEGGuide
from ..models.intermediate import PartTextResult, PartsTextPerGuide
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
