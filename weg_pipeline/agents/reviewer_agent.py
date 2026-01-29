"""
Reviewer Agent - Validates and reviews outputs from other agents.

This agent checks the quality and correctness of extracted data by
reasoning about each step's description against the extracted information.
When issues are found, it provides specific feedback to agents for refinement.
"""
from typing import Any, Optional
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ..models.preweg import PreWEGGuide
from ..models.intermediate import (
    ActionsPerGuide,
    ActionExtractionResult,
    ToolsPerGuide,
    HandsPerGuide,
    PartsTextPerGuide,
)
from ..utils.llm_utils import extract_json_from_response
from ..utils.logger import get_logger, LogLevel
from .base_agent import BaseAgent


class StepReviewResult(BaseModel):
    """Review result for a single step."""
    step_index: int
    is_valid: bool = True
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    feedback_for_action_agent: str = ""  # Specific feedback for ActionAgent
    feedback_for_tool_agent: str = ""     # Specific feedback for ToolAgent
    feedback_for_hands_agent: str = ""    # Specific feedback for HandsAgent
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ReviewResult(BaseModel):
    """Complete review result for all agents' outputs."""
    guide_id: int
    overall_valid: bool = True
    steps: list[StepReviewResult] = Field(default_factory=list)
    summary: str = ""
    missing_steps: list[int] = Field(default_factory=list)
    steps_needing_refinement: list[int] = Field(default_factory=list)


@dataclass
class RefinementRequest:
    """Request for an agent to refine its output."""
    step_index: int
    agent_name: str
    feedback: str
    original_issues: list[str] = field(default_factory=list)


class ReviewerAgent(BaseAgent):
    """
    Reviews and validates outputs from extraction agents.
    
    Key responsibilities:
    1. Validate extracted data against original descriptions
    2. Check component reasoning (is this the PHYSICAL part touched?)
    3. Provide specific feedback for agents to refine their work
    4. Trigger refinement loop when issues are found
    
    Checks:
    - Are actions properly extracted from descriptions?
    - Are COMPONENTS correctly identified (not devices, but actual parts)?
    - Do tools match what's mentioned in the text?
    - Are hands estimations reasonable?
    - Are parts correctly identified?
    - Are there any missing or incomplete extractions?
    """

    AGENT_NAME = "ReviewerAgent"
    REQUIRES_VISION = False
    DEFAULT_PROVIDER = "anthropic"
    DEFAULT_MODEL = "claude-3-5-haiku-20241022"

    def get_system_prompt(self) -> str:
        return """You are an expert quality assurance reviewer for appliance repair guide data extraction.

Your task is to validate that extracted data correctly represents the original step descriptions.

=== CRITICAL VALIDATION: COMPONENT REASONING ===

The most important check: Is the "component" field the ACTUAL PHYSICAL PART being touched?

COMMON ERRORS TO FLAG:
- "Unplug the refrigerator" with component="refrigerator" 
  → WRONG! Should be "power cord plug" or "electrical plug"
  → The hand touches the PLUG, not the whole refrigerator
  
- "Open the freezer door" with component="freezer"
  → WRONG! Should be "freezer door" or "door handle"
  → The hand touches the DOOR, not the freezer
  
- "Remove the screw from the panel" with component="panel"
  → WRONG! Should be "screw" or "panel screw"
  → The tool touches the SCREW, not the panel

ASK: "What does the person's hand (or tool) PHYSICALLY TOUCH during this action?"

=== OTHER VALIDATION CRITERIA ===

1. ACTIONS:
   - Each action should be an imperative verb phrase from the description
   - Action quadruples should have: action verb, tool (if used), component, hands estimate
   - "action" should be a single verb (remove, pull, lift, etc.)
   - Hands: 0 = info only, 1 = one-handed task, 2 = two-handed task

2. TOOLS:
   - Tools should only be listed if explicitly mentioned in the description
   - Tool names should match the toolbox when possible

3. HANDS:
   - 0: No physical action (just information/warnings)
   - 1: Simple one-handed operations (turning screws, pressing buttons)
   - 2: Actions requiring both hands (lifting, holding + manipulating)

4. HINTS:
   - Non-action information: warnings, tips, notes
   - Should NOT contain action items

=== FEEDBACK FORMAT ===

When you find issues, provide SPECIFIC FEEDBACK that the agent can use to correct:

Good feedback: "Step 1: The component 'refrigerator' is the DEVICE, not what you touch. 
When unplugging, your hand grabs the PLUG. Change component to 'power cord plug' or 'electrical plug'."

Bad feedback: "Wrong component" (too vague)

Return ONLY valid JSON."""

    def build_prompt(
        self,
        guide: PreWEGGuide,
        actions: Optional[ActionsPerGuide] = None,
        tools: Optional[ToolsPerGuide] = None,
        hands: Optional[HandsPerGuide] = None,
        parts_text: Optional[PartsTextPerGuide] = None,
        **kwargs,
    ) -> str:
        """Build prompt to review extracted data."""
        
        # Build a consolidated view of all extractions per step
        steps_review = []
        
        for step in sorted(guide.steps, key=lambda s: s.step_index):
            idx = step.step_index
            
            # Get extracted data for this step
            step_actions = None
            step_tools = None
            step_hands = None
            step_parts = None
            
            if actions:
                for a in actions.steps:
                    if a.step_index == idx:
                        step_actions = a
                        break
            
            if tools:
                for t in tools.steps:
                    if t.step_index == idx:
                        step_tools = t
                        break
            
            if hands:
                for h in hands.steps:
                    if h.step_index == idx:
                        step_hands = h
                        break
            
            if parts_text:
                for p in parts_text.steps:
                    if p.step_index == idx:
                        step_parts = p
                        break
            
            step_data = {
                "step_index": idx,
                "description": step.full_description,
                "extracted": {
                    "task_name": step_actions.task_name if step_actions else None,
                    "actions": step_actions.actions if step_actions else [],
                    "action_quadruples": [
                        {
                            "action": q.action,
                            "tool": q.tool,
                            "component": q.component,
                            "hands": q.hands,
                        }
                        for q in (step_actions.action_quadruples if step_actions else [])
                    ],
                    "hints": step_actions.hints if step_actions else [],
                    "tools": step_tools.tools if step_tools else [],
                    "hands": step_hands.hands if step_hands else None,
                    "parts": step_parts.parts if step_parts else [],
                }
            }
            steps_review.append(step_data)
        
        import json
        steps_json = json.dumps(steps_review, indent=2)
        
        return f"""Review the following extracted data from a repair guide.

Guide: {guide.title}
Total Steps: {guide.num_steps}
Toolbox: {guide.toolbox}

=== EXTRACTED DATA PER STEP ===
{steps_json}

=== TASK ===
Review each step and validate:

1. **COMPONENT CHECK (MOST IMPORTANT)**:
   - Is the "component" the PHYSICAL PART being touched?
   - NOT the device/appliance name
   - Example: "Unplug refrigerator" → component should be "power cord plug", NOT "refrigerator"

2. Are actions correctly extracted from the description?
3. Is the task_name appropriate?
4. Is the hands estimate reasonable for the task?
5. Are hints properly separated from actions?
6. Are any steps missing data that should be there?

Return JSON:
{{
  "overall_valid": true/false,
  "missing_steps": [list of step indices with missing/empty actions that should have them],
  "steps_needing_refinement": [list of step indices that need agent refinement],
  "steps": [
    {{
      "step_index": 1,
      "is_valid": true/false,
      "issues": ["issue1", "issue2"],
      "suggestions": ["suggestion1"],
      "feedback_for_action_agent": "Specific feedback for ActionAgent to fix issues (or empty if no issues)",
      "feedback_for_tool_agent": "Specific feedback for ToolAgent (or empty)",
      "feedback_for_hands_agent": "Specific feedback for HandsAgent (or empty)",
      "confidence": 0.9
    }}
  ],
  "summary": "Brief overall assessment"
}}

Only flag steps with ACTUAL issues. Provide SPECIFIC, ACTIONABLE feedback."""

    def parse_response(
        self,
        response: str,
        guide: PreWEGGuide,
        **kwargs,
    ) -> ReviewResult:
        """Parse reviewer response into structured result."""
        try:
            data = extract_json_from_response(response)
        except ValueError as e:
            self.log_error(f"Failed to parse JSON: {e}")
            return ReviewResult(
                guide_id=guide.guide_id,
                overall_valid=False,
                summary=f"Failed to parse review response: {e}",
            )

        if not isinstance(data, dict):
            return ReviewResult(
                guide_id=guide.guide_id,
                overall_valid=False,
                summary="Invalid response format",
            )

        steps = []
        for step_data in data.get("steps", []):
            if isinstance(step_data, dict):
                steps.append(StepReviewResult(
                    step_index=step_data.get("step_index", 0),
                    is_valid=step_data.get("is_valid", True),
                    issues=step_data.get("issues", []),
                    suggestions=step_data.get("suggestions", []),
                    feedback_for_action_agent=step_data.get("feedback_for_action_agent", ""),
                    feedback_for_tool_agent=step_data.get("feedback_for_tool_agent", ""),
                    feedback_for_hands_agent=step_data.get("feedback_for_hands_agent", ""),
                    confidence=step_data.get("confidence", 1.0),
                ))

        return ReviewResult(
            guide_id=guide.guide_id,
            overall_valid=data.get("overall_valid", True),
            steps=steps,
            summary=data.get("summary", ""),
            missing_steps=data.get("missing_steps", []),
            steps_needing_refinement=data.get("steps_needing_refinement", []),
        )

    def review(
        self,
        guide: PreWEGGuide,
        actions: Optional[ActionsPerGuide] = None,
        tools: Optional[ToolsPerGuide] = None,
        hands: Optional[HandsPerGuide] = None,
        parts_text: Optional[PartsTextPerGuide] = None,
    ) -> ReviewResult:
        """
        Review all agent outputs for a guide.
        
        Args:
            guide: Original pre-WEG guide
            actions: ActionAgent output
            tools: ToolAgent output
            hands: HandsAgent output
            parts_text: PartTextAgent output
        
        Returns:
            ReviewResult with validation status, issues, and feedback
        """
        logger = get_logger()
        
        # First, check for obviously missing data
        missing_steps = []
        if actions:
            extracted_indices = {s.step_index for s in actions.steps}
            for step in guide.steps:
                if step.step_index not in extracted_indices:
                    missing_steps.append(step.step_index)
                else:
                    # Check if step has no actions but description suggests it should
                    for a in actions.steps:
                        if a.step_index == step.step_index:
                            if not a.actions and self._description_has_actions(step.full_description):
                                missing_steps.append(step.step_index)
                            break
        
        if missing_steps:
            self.log(f"Detected {len(missing_steps)} steps with potentially missing actions: {missing_steps}")
        
        prompt = self.build_prompt(
            guide,
            actions=actions,
            tools=tools,
            hands=hands,
            parts_text=parts_text,
        )
        
        system_prompt = self.get_system_prompt()
        
        self.log(f"Reviewing {guide.num_steps} steps")
        response = self.client.complete(prompt, system_prompt=system_prompt)
        
        result = self.parse_response(response, guide)
        
        # Add pre-detected missing steps if not already in result
        for idx in missing_steps:
            if idx not in result.missing_steps:
                result.missing_steps.append(idx)
        
        if result.missing_steps:
            result.overall_valid = False
        
        # Log feedback to the pipeline logger
        if logger:
            for step_result in result.steps:
                if not step_result.is_valid:
                    if step_result.feedback_for_action_agent:
                        logger.reviewer_feedback(
                            step_index=step_result.step_index,
                            target_agent="ActionAgent",
                            issue="; ".join(step_result.issues),
                            suggestion=step_result.feedback_for_action_agent,
                            severity="warning" if step_result.confidence > 0.5 else "error",
                        )
                    if step_result.feedback_for_tool_agent:
                        logger.reviewer_feedback(
                            step_index=step_result.step_index,
                            target_agent="ToolAgent",
                            issue="; ".join(step_result.issues),
                            suggestion=step_result.feedback_for_tool_agent,
                            severity="warning",
                        )
                    if step_result.feedback_for_hands_agent:
                        logger.reviewer_feedback(
                            step_index=step_result.step_index,
                            target_agent="HandsAgent",
                            issue="; ".join(step_result.issues),
                            suggestion=step_result.feedback_for_hands_agent,
                            severity="warning",
                        )
        
        return result

    def get_refinement_requests(self, result: ReviewResult) -> list[RefinementRequest]:
        """
        Get list of refinement requests from review result.
        
        Returns list of requests that can be passed to agents for refinement.
        """
        requests = []
        
        for step_result in result.steps:
            if not step_result.is_valid:
                if step_result.feedback_for_action_agent:
                    requests.append(RefinementRequest(
                        step_index=step_result.step_index,
                        agent_name="ActionAgent",
                        feedback=step_result.feedback_for_action_agent,
                        original_issues=step_result.issues,
                    ))
                if step_result.feedback_for_tool_agent:
                    requests.append(RefinementRequest(
                        step_index=step_result.step_index,
                        agent_name="ToolAgent",
                        feedback=step_result.feedback_for_tool_agent,
                        original_issues=step_result.issues,
                    ))
                if step_result.feedback_for_hands_agent:
                    requests.append(RefinementRequest(
                        step_index=step_result.step_index,
                        agent_name="HandsAgent",
                        feedback=step_result.feedback_for_hands_agent,
                        original_issues=step_result.issues,
                    ))
        
        return requests

    def _description_has_actions(self, description: str) -> bool:
        """Check if a description likely contains actionable instructions."""
        action_indicators = [
            "remove", "pull", "push", "lift", "insert", "disconnect",
            "squeeze", "press", "turn", "rotate", "unplug", "plug",
            "use", "grab", "grasp", "hold", "release", "cut", "pry",
        ]
        desc_lower = description.lower()
        return any(indicator in desc_lower for indicator in action_indicators)
