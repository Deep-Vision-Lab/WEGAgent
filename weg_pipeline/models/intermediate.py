"""
Intermediate data models used between agents during pipeline execution.
"""
from typing import Any, Optional

from pydantic import BaseModel, Field


# === Action Quadruple (new format) ===

class ActionQuadruple(BaseModel):
    """
    Represents a single atomic action as a quadruple:
    <Action, Tool, Component, Hands>
    """
    action: str = Field(..., description="One-word action verb (e.g., 'remove', 'pull', 'insert')")
    precise_action: Optional[str] = Field(None, description="More precise/technical action verb (e.g., 'unscrew' instead of 'remove')")
    tool: Optional[str] = Field(None, description="Tool used for this action (null if bare hands)")
    component: str = Field(..., description="Component being manipulated (e.g., 'connector', 'screw')")
    hands: int = Field(..., ge=0, le=2, description="Number of hands needed (0, 1, or 2)")
    full_action: str = Field(..., description="Original full action sentence")


# === Action Agent Output ===

class ActionExtractionResult(BaseModel):
    """Output from Action Agent for a single step."""
    step_index: int
    task_name: Optional[str] = Field(None, description="Title/name of this step's task")
    actions: list[str] = Field(default_factory=list, description="List of atomic actions (full sentences)")
    action_quadruples: list[ActionQuadruple] = Field(
        default_factory=list, 
        description="Structured quadruple for each action"
    )
    hints: list[str] = Field(default_factory=list, description="Hints, tips, warnings (non-action info)")
    reasoning: Optional[str] = Field(None, description="Agent's reasoning (for debugging)")


class ActionsPerGuide(BaseModel):
    """Complete action extraction output for all steps."""
    guide_id: int
    steps: list[ActionExtractionResult] = Field(default_factory=list)


# === Tool Agent Output ===

class ToolExtractionResult(BaseModel):
    """Output from Tool Agent for a single step."""
    step_index: int
    tools: list[str] = Field(default_factory=list, description="Tools used in this step")
    primary_tool: Optional[str] = Field(None, description="Main tool for this step")
    reasoning: Optional[str] = None


class ToolsPerGuide(BaseModel):
    """Complete tool extraction output for all steps."""
    guide_id: int
    global_toolbox: list[str] = Field(default_factory=list, description="Tools from guide header")
    steps: list[ToolExtractionResult] = Field(default_factory=list)


# === Hands Agent Output ===

class HandsEstimationResult(BaseModel):
    """Output from Hands Agent for a single step."""
    step_index: int
    hands: int = Field(..., ge=0, le=2, description="Number of hands: 0, 1, or 2")
    reasoning: Optional[str] = None


class HandsPerGuide(BaseModel):
    """Complete hands estimation output for all steps."""
    guide_id: int
    steps: list[HandsEstimationResult] = Field(default_factory=list)


# === Part Text Agent Output ===

class PartTextResult(BaseModel):
    """Output from Part Text Agent for a single step."""
    step_index: int
    parts: list[str] = Field(default_factory=list, description="Part names extracted from text")
    primary_part: Optional[str] = Field(None, description="Main part being manipulated")
    reasoning: Optional[str] = None


class PartsTextPerGuide(BaseModel):
    """Complete part text extraction output for all steps."""
    guide_id: int
    steps: list[PartTextResult] = Field(default_factory=list)


# === Part Vision Agent Output ===

class PartBBoxResult(BaseModel):
    """Bounding box detection result for a component (and optionally its containing part)."""
    name: str = Field(..., description="Component name")
    bbox: list[int] = Field(..., min_length=4, max_length=4, description="[x1, y1, x2, y2] for the component")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    marked_by_annotation: bool = Field(
        default=False, 
        description="Whether this part was identified via a red circle or visual annotation"
    )
    # V2 fields for containing part
    part_name: Optional[str] = Field(None, description="Name of the containing part (e.g., 'panel')")
    part_bbox: Optional[list[int]] = Field(None, min_length=4, max_length=4, description="[x1, y1, x2, y2] for the containing part")
    part_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class PartVisionResult(BaseModel):
    """Output from Part Vision Agent for a single step image."""
    step_index: int
    image_index: int = Field(default=1, description="Which image in the step (1-based)")
    image_path: str
    image_size: tuple[int, int] = Field(..., description="(width, height)")
    parts: list[PartBBoxResult] = Field(default_factory=list)
    raw_model_output: Optional[str] = None


class PartsVisionPerGuide(BaseModel):
    """Complete part vision output for all steps."""
    guide_id: int
    steps: list[PartVisionResult] = Field(default_factory=list)


# === V2 Pipeline: Per-Action Models ===

class ComponentPerAction(BaseModel):
    """Component and part extraction for a single action (V2 pipeline)."""
    action_id: int = Field(..., description="Index into action_quadruples (0-based)")
    component: str = Field(..., description="Physical component being touched/manipulated (what hand/tool touches)")
    part: Optional[str] = Field(None, description="The larger part containing the component (context)")


class ToolPerAction(BaseModel):
    """Tool identification for a single action (V2 pipeline)."""
    action_id: int = Field(..., description="Index into action_quadruples (0-based)")
    tool: Optional[str] = Field(None, description="Tool name or None if bare hands")


class HandsPerAction(BaseModel):
    """Hands estimation for a single action (V2 pipeline)."""
    action_id: int = Field(..., description="Index into action_quadruples (0-based)")
    hands: int = Field(..., ge=0, le=2, description="Number of hands: 0, 1, or 2")


# === V2 Part Text Agent Output (Enhanced) ===

class PartTextResultV2(BaseModel):
    """Output from Part Text Agent for a single step (V2 pipeline)."""
    step_index: int
    device_type: str = Field(default="appliance", description="Type of device being repaired")
    parts: list[str] = Field(default_factory=list, description="Part names extracted from text")
    primary_part: Optional[str] = Field(None, description="Main part being manipulated")
    components_per_action: list[ComponentPerAction] = Field(
        default_factory=list,
        description="Physical component touched for each action"
    )


class PartsTextPerGuideV2(BaseModel):
    """Complete part text extraction output for all steps (V2 pipeline)."""
    guide_id: int
    device_type: str = Field(default="appliance", description="Device type for the guide")
    steps: list[PartTextResultV2] = Field(default_factory=list)


# === V2 Tool Agent Output (Enhanced) ===

class ToolExtractionResultV2(BaseModel):
    """Output from Tool Agent for a single step (V2 pipeline)."""
    step_index: int
    tools: list[str] = Field(default_factory=list, description="Tools used in this step")
    primary_tool: Optional[str] = Field(None, description="Main tool for this step")
    tools_per_action: list[ToolPerAction] = Field(
        default_factory=list,
        description="Tool for each action"
    )


class ToolsPerGuideV2(BaseModel):
    """Complete tool extraction output for all steps (V2 pipeline)."""
    guide_id: int
    global_toolbox: list[str] = Field(default_factory=list, description="Tools from guide header")
    steps: list[ToolExtractionResultV2] = Field(default_factory=list)


# === V2 Hands Agent Output (Enhanced) ===

class HandsEstimationResultV2(BaseModel):
    """Output from Hands Agent for a single step (V2 pipeline)."""
    step_index: int
    hands: int = Field(default=1, ge=0, le=2, description="Step-level hands (backward compat)")
    hands_per_action: list[HandsPerAction] = Field(
        default_factory=list,
        description="Hands for each action"
    )


class HandsPerGuideV2(BaseModel):
    """Complete hands estimation output for all steps (V2 pipeline)."""
    guide_id: int
    steps: list[HandsEstimationResultV2] = Field(default_factory=list)


# === Pipeline State (for LangGraph) ===

class PipelineState(BaseModel):
    """Complete state passed through the LangGraph pipeline."""
    # Input
    guide_path: str
    guide: Optional[dict[str, Any]] = None  # Raw pre-WEG JSON

    # Intermediate outputs
    actions: Optional[ActionsPerGuide] = None
    tools: Optional[ToolsPerGuide] = None
    hands: Optional[HandsPerGuide] = None
    parts_text: Optional[PartsTextPerGuide] = None
    parts_vision: Optional[PartsVisionPerGuide] = None

    # Final output
    weg: Optional[dict[str, Any]] = None

    # Metadata
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    completed_agents: list[str] = Field(default_factory=list)
