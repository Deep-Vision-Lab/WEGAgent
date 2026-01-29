"""
Tool Extraction Agent - Identifies tools used in each repair step.

Uses LLM to analyze text and match against global toolbox.
"""
from typing import Optional

from ..models.preweg import PreWEGGuide
from ..models.intermediate import ToolExtractionResult, ToolsPerGuide
from ..utils.llm_utils import extract_json_from_response
from .base_agent import BaseAgent


class ToolAgent(BaseAgent):
    """
    Extracts tools used in each step of a repair guide.
    
    Strategy:
    1. Extract explicitly mentioned tools from step text
    2. Infer tools from actions (e.g., "unscrew" → screwdriver)
    3. Match against global toolbox for consistency
    """

    AGENT_NAME = "ToolAgent"
    REQUIRES_VISION = False
    DEFAULT_PROVIDER = "openai"
    DEFAULT_MODEL = "gpt-4o-mini"

    def get_system_prompt(self) -> str:
        return """You are an expert at identifying tools used in appliance repair tasks.

Your task is to extract ONLY the tools that a human uses/holds/operates in each step.

Definition of a TOOL:
- Something a person HOLDS, OPERATES, or APPLIES
- Examples: screwdriver, wrench, pliers, pry tool, multimeter, drill, cloth, tape, brush

DO NOT include:
- Parts being removed/replaced (screws, panels, connectors, fans, hoses)
- Appliance components being worked on
- Body parts (hands, fingers)
- Generic terms like "tool" without specifics

If multiple tools are used, list all of them.
If no tools are explicitly needed or mentioned, return an empty list.

Return ONLY valid JSON. No explanations, no markdown."""

    def build_prompt(self, guide: PreWEGGuide, **kwargs) -> str:
        """Build prompt for tool extraction."""
        # Include global toolbox for context
        toolbox_text = ""
        if guide.toolbox:
            toolbox_text = f"\nGlobal Toolbox (tools available for this repair):\n{guide.toolbox}\n"

        steps_text = []
        for step in sorted(guide.steps, key=lambda s: s.step_index):
            steps_text.append(f"""
Step {step.step_index}:
\"\"\"
{step.full_description}
\"\"\"
""")

        return f"""Extract tools used in each step of this repair guide.
{toolbox_text}
Guide: {guide.title}

{chr(10).join(steps_text)}

Return JSON with this structure:
{{
  "global_toolbox": ["tool1", "tool2"],  // Confirmed tools from header
  "steps": [
    {{"step_index": 1, "tools": ["tool1"], "primary_tool": "tool1"}},
    {{"step_index": 2, "tools": [], "primary_tool": null}},
    ...
  ]
}}

Rules:
- "tools": ALL tools used in that step
- "primary_tool": the MAIN tool for that step (most important), or null
- Match tool names to the global toolbox when possible (use exact names)
- Infer tools from actions if not explicitly stated (e.g., "unscrew" → likely a screwdriver)
- Return exactly {guide.num_steps} step entries"""

    def parse_response(
        self,
        response: str,
        guide: PreWEGGuide,
        **kwargs,
    ) -> ToolsPerGuide:
        """Parse LLM response into structured tools."""
        try:
            data = extract_json_from_response(response)
        except ValueError as e:
            self.log_error(f"Failed to parse JSON: {e}")
            return ToolsPerGuide(
                guide_id=guide.guide_id,
                global_toolbox=guide.toolbox,
                steps=[],
            )

        if not isinstance(data, dict):
            self.log_error(f"Expected dict, got {type(data)}")
            return ToolsPerGuide(
                guide_id=guide.guide_id,
                global_toolbox=guide.toolbox,
                steps=[],
            )

        global_toolbox = data.get("global_toolbox", guide.toolbox) or []
        steps_data = data.get("steps", [])

        steps = []
        for item in steps_data:
            if not isinstance(item, dict):
                continue

            step_index = item.get("step_index", 0)
            tools = item.get("tools", [])
            primary = item.get("primary_tool")

            if isinstance(tools, list):
                tools = [str(t).strip() for t in tools if str(t).strip()]
            else:
                tools = []

            steps.append(ToolExtractionResult(
                step_index=step_index,
                tools=tools,
                primary_tool=primary,
            ))

        return ToolsPerGuide(
            guide_id=guide.guide_id,
            global_toolbox=global_toolbox,
            steps=steps,
        )
