"""
Pydantic models for WEG output (final structured format)
"""
from typing import Any, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """2D bounding box in pixel coordinates [x1, y1, x2, y2]."""
    x1: int = Field(..., description="Left x coordinate")
    y1: int = Field(..., description="Top y coordinate")
    x2: int = Field(..., description="Right x coordinate")
    y2: int = Field(..., description="Bottom y coordinate")

    @property
    def as_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class PartInfo(BaseModel):
    """Information about a part being manipulated in a step."""
    part_id: int = Field(..., description="Unique identifier for this part within the step (1-based)")
    name: str = Field(..., description="Name of the part")
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box in image coordinates")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Detection confidence")
    image_path: Optional[str] = Field(None, description="Path to image where part was detected")


class ActionQuadrupleOutput(BaseModel):
    """
    Structured representation of an atomic action.
    Format: <Action, Tool, Component, Hands>
    
    Actions are stored in execution order within the step.
    The part_id links to the corresponding entry in parts_all for visual localization.
    """
    action: str = Field(..., description="One-word action verb (e.g., 'remove', 'pull', 'insert')")
    precise_action: Optional[str] = Field(None, description="More precise/technical action verb (e.g., 'unscrew' instead of 'remove')")
    tool: Optional[str] = Field(None, description="Tool used for this action (null if bare hands)")
    component: str = Field(..., description="Component being manipulated")
    part_id: Optional[int] = Field(None, description="Reference to parts_all entry (matches PartInfo.part_id). Null if component not visually detected.")
    hands: int = Field(..., ge=0, le=2, description="Number of hands needed (0, 1, or 2)")
    full_action: str = Field(..., description="Original full action sentence")


class WEGStep(BaseModel):
    """A single step in the WEG format."""
    step_id: int = Field(..., description="1-based step identifier")
    task_name: Optional[str] = Field(None, description="Title/name of this step's task (e.g., 'Reassembly notes')")
    description: str = Field(..., description="Full step description")
    actions: list[str] = Field(default_factory=list, description="Atomic actions in this step (full sentences)")
    action_quadruples: list[ActionQuadrupleOutput] = Field(
        default_factory=list, 
        description="Structured quadruple for each action: <Action, Tool, Component, Hands>. Actions are in execution order."
    )
    hints: list[str] = Field(default_factory=list, description="Tips, warnings, notes (non-action info)")
    # Note: tools are now only in action_quadruples (tool field per action)
    primary_part: Optional[PartInfo] = Field(None, description="Primary part being manipulated in this step")
    parts_all: list[PartInfo] = Field(
        default_factory=list, 
        description="All components from action_quadruples with their bounding boxes detected by vision"
    )
    # Note: hands is per-action in action_quadruples, NOT per-step
    geometric_location: Optional[str] = Field(None, description="Spatial location hint")


class WEGHeader(BaseModel):
    """Header information for the WEG file."""
    title: str = Field(..., description="Guide title")
    description: str = Field("", description="Guide description/summary")
    toolbox: list[str] = Field(default_factory=list, description="Required tools")
    parts_list: list[str] = Field(default_factory=list, description="Required parts")
    source_url: Optional[str] = Field(None, description="Original guide URL")
    guide_id: Optional[int] = Field(None, description="Original guide ID")


class WEGDocument(BaseModel):
    """Complete WEG document structure."""
    header: WEGHeader = Field(..., description="Document header")
    steps: list[WEGStep] = Field(default_factory=list, description="Ordered list of steps")

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump(exclude_none=True)
