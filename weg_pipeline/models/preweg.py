"""
Pydantic models for Pre-WEG data (input from crawler)
"""
from typing import Any, Optional

from pydantic import BaseModel, Field


class StepImage(BaseModel):
    """Image associated with a step."""
    src_url: str = Field(..., description="Original image URL")
    saved_path: str = Field(..., description="Local path to downloaded image")


class PreWEGStep(BaseModel):
    """A single step in the pre-WEG guide."""
    step_index: int = Field(..., description="1-based step index")
    step_id_ifixit: Optional[int] = Field(None, description="iFixit's internal step ID")
    full_description: str = Field("", description="Full text description of the step")
    images: list[StepImage] = Field(default_factory=list, description="Images for this step")


class PreWEGGuide(BaseModel):
    """Complete pre-WEG guide structure (input to pipeline)."""
    guide_id: int = Field(..., description="Unique guide identifier")
    source_url: Optional[str] = Field(None, description="Original URL of the guide")
    title: str = Field(..., description="Guide title")
    device: Optional[str] = Field(None, description="Device being repaired")
    summary: Optional[str] = Field(None, description="Guide summary/introduction")
    toolbox: list[str] = Field(default_factory=list, description="List of tools needed")
    parts: list[str] = Field(default_factory=list, description="List of parts needed")
    relevant_urls: list[str] = Field(default_factory=list, description="Related URLs")
    steps: list[PreWEGStep] = Field(default_factory=list, description="Guide steps")
    images: list[dict[str, Any]] = Field(default_factory=list, description="All guide images")

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def get_step(self, index: int) -> Optional[PreWEGStep]:
        """Get step by 1-based index."""
        for step in self.steps:
            if step.step_index == index:
                return step
        return None
