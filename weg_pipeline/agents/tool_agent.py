"""
Tool Extraction Agent - Identifies tools used in each repair step.

Uses LLM to analyze text and match against global toolbox.
V2: Identifies tools per action.
"""
from typing import Optional

from ..models.preweg import PreWEGGuide
from ..models.intermediate import (
    ToolExtractionResult, 
    ToolsPerGuide,
    ToolExtractionResultV2,
    ToolsPerGuideV2,
    ToolPerAction,
    ActionsPerGuide,
)
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

    # ==================== V2 Pipeline Methods ====================
    
    def get_system_prompt_v2(self) -> str:
        """System prompt for V2 pipeline - tools per action."""
        return """You are an expert at identifying tools used in appliance repair tasks.

Your task is to identify the TOOL used for EACH specific action.

=== Definition of a TOOL ===
- Something a person HOLDS, OPERATES, or APPLIES
- Examples: screwdriver, wrench, pliers, pry tool, multimeter, drill, cloth, tape, brush

=== NOT a Tool ===
- Parts being removed/replaced (screws, panels, connectors)
- Appliance components
- Body parts (hands, fingers)
- Generic terms like "tool" without specifics

=== Per-Action Tool Assignment ===
For each action, determine:
1. Is a tool explicitly mentioned? Use that tool.
2. Can a tool be inferred from the action? (e.g., "unscrew" → screwdriver)
3. Does it match a tool in the global toolbox? Use the exact toolbox name.
4. Is it bare hands? Use null.

=== Inference Rules ===
- "unscrew", "remove screws" → screwdriver (Phillips/flathead based on context)
- "pry", "pop off" → pry tool, spudger, or flathead screwdriver
- "disconnect connector" → usually bare hands (null)
- "wipe", "clean" → cloth or towel
- "cut" → knife, scissors, or wire cutters
- "measure", "test" → multimeter

Return ONLY valid JSON. No explanations, no markdown."""

    def build_prompt_v2(
        self, 
        guide: PreWEGGuide, 
        actions: ActionsPerGuide,
        **kwargs,
    ) -> str:
        """Build prompt for V2 pipeline - tools per action."""
        toolbox_str = ", ".join(guide.toolbox) if guide.toolbox else "None specified"
        
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
                        "precise_action": q.precise_action,
                        "full_action": q.full_action,
                    })
            
            steps_data.append(f"""
Step {step.step_index}:
Description: \"\"\"{step.full_description}\"\"\"
Actions:
{action_list}
""")

        return f"""Identify the tool used for each action.

Guide: {guide.title}
Global Toolbox: [{toolbox_str}]

{chr(10).join(steps_data)}

IMPORTANT:
- Match tools to the global toolbox when possible (use exact names)
- Use null for actions that don't require tools (bare hands)
- Infer tools from action verbs if not explicitly stated

Return JSON array:
[
  {{
    "step_index": 1,
    "tools": ["tool1", "tool2"],
    "primary_tool": "main tool or null",
    "tools_per_action": [
      {{"action_id": 0, "tool": "Phillips screwdriver"}},
      {{"action_id": 1, "tool": null}}
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
    ) -> ToolsPerGuideV2:
        """Parse V2 LLM response into structured tools per action."""
        try:
            data = extract_json_from_response(response)
        except ValueError as e:
            self.log_error(f"Failed to parse JSON: {e}")
            return ToolsPerGuideV2(
                guide_id=guide.guide_id,
                global_toolbox=guide.toolbox,
                steps=[],
            )

        if not isinstance(data, list):
            self.log_error(f"Expected list, got {type(data)}")
            return ToolsPerGuideV2(
                guide_id=guide.guide_id,
                global_toolbox=guide.toolbox,
                steps=[],
            )

        steps = []
        for item in data:
            if not isinstance(item, dict):
                continue

            step_index = item.get("step_index", 0)
            tools = item.get("tools", [])
            primary = item.get("primary_tool")

            if isinstance(tools, list):
                tools = [str(t).strip() for t in tools if str(t).strip()]
            else:
                tools = []

            # Parse tools per action
            tools_per_action = []
            raw_tools = item.get("tools_per_action", [])
            if isinstance(raw_tools, list):
                for tool_data in raw_tools:
                    if isinstance(tool_data, dict):
                        action_id = tool_data.get("action_id", 0)
                        tool = tool_data.get("tool")
                        if tool and isinstance(tool, str):
                            tool = tool.strip() if tool.strip() else None
                        tools_per_action.append(ToolPerAction(
                            action_id=action_id,
                            tool=tool,
                        ))

            steps.append(ToolExtractionResultV2(
                step_index=step_index,
                tools=tools,
                primary_tool=primary,
                tools_per_action=tools_per_action,
            ))

        return ToolsPerGuideV2(
            guide_id=guide.guide_id,
            global_toolbox=guide.toolbox,
            steps=steps,
        )

    def run_for_actions(
        self, 
        guide: PreWEGGuide, 
        actions: ActionsPerGuide,
        **kwargs,
    ) -> ToolsPerGuideV2:
        """
        V2 Pipeline: Identify tools per action.
        
        Args:
            guide: The PreWEG guide
            actions: Action extraction results from Action Agent
        
        Returns:
            Tools for each step and action
        """
        self.log(f"[V2] Identifying tools per action for guide: {guide.title}")
        
        prompt = self.build_prompt_v2(guide, actions, **kwargs)
        system_prompt = self.get_system_prompt_v2()
        
        response = self.client.complete(prompt, system_prompt=system_prompt)
        result = self.parse_response_v2(response, guide, **kwargs)
        
        total_tools = sum(len(s.tools_per_action) for s in result.steps)
        self.log(f"[V2] Identified tools for {total_tools} actions from {len(result.steps)} steps")
        
        return result

    def refine_for_actions(
        self,
        guide: PreWEGGuide,
        actions: ActionsPerGuide,
        step_index: int,
        feedback: str,
    ) -> ToolExtractionResultV2:
        """V2 Pipeline: Refine tool extraction based on reviewer feedback."""
        self.log(f"[V2] Refining step {step_index} based on feedback")
        
        step = guide.get_step(step_index)
        if not step:
            raise ValueError(f"Step {step_index} not found")
        
        toolbox_str = ", ".join(guide.toolbox) if guide.toolbox else "None specified"
        
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
                    "precise_action": q.precise_action,
                    "full_action": q.full_action,
                })

        prompt = f"""Identify the tool used for each action.

REVIEWER FEEDBACK - PLEASE ADDRESS:
{feedback}

Global Toolbox: [{toolbox_str}]

Step {step_index}:
Description: \"\"\"{step.full_description}\"\"\"
Actions:
{action_list}

Return JSON:
{{
  "step_index": {step_index},
  "tools": ["tool1"],
  "primary_tool": "main tool or null",
  "tools_per_action": [
    {{"action_id": 0, "tool": "tool name or null"}}
  ]
}}"""

        system_prompt = self.get_system_prompt_v2()
        response = self.client.complete(prompt, system_prompt=system_prompt)
        
        try:
            data = extract_json_from_response(response)
            if not isinstance(data, dict):
                return ToolExtractionResultV2(step_index=step_index)
            
            tools = data.get("tools", [])
            if isinstance(tools, list):
                tools = [str(t).strip() for t in tools if str(t).strip()]
            else:
                tools = []
            
            tools_per_action = []
            raw_tools = data.get("tools_per_action", [])
            if isinstance(raw_tools, list):
                for tool_data in raw_tools:
                    if isinstance(tool_data, dict):
                        tool = tool_data.get("tool")
                        if tool and isinstance(tool, str):
                            tool = tool.strip() if tool.strip() else None
                        tools_per_action.append(ToolPerAction(
                            action_id=tool_data.get("action_id", 0),
                            tool=tool,
                        ))
            
            return ToolExtractionResultV2(
                step_index=step_index,
                tools=tools,
                primary_tool=data.get("primary_tool"),
                tools_per_action=tools_per_action,
            )
        except ValueError as e:
            self.log_error(f"Failed to parse refinement response: {e}")
            return ToolExtractionResultV2(step_index=step_index)
