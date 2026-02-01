"""
Pydantic models for Pre-WEG data (input from crawler)
"""
import re
from typing import Any, Optional

from pydantic import BaseModel, Field, computed_field


# Common appliance/device keywords for extraction
DEVICE_KEYWORDS = [
    "refrigerator", "fridge", "freezer",
    "dishwasher", 
    "washing machine", "washer", "dryer",
    "microwave", "oven", "stove", "range",
    "air conditioner", "ac unit", "hvac",
    "garbage disposal", "disposal",
    "ice maker", "icemaker",
    "water heater",
    "furnace", "boiler",
    "vacuum", "vacuum cleaner",
    "blender", "mixer", "food processor",
    "coffee maker", "coffeemaker", "espresso machine",
    "toaster", "toaster oven",
]


def extract_device_type(title: str, device: Optional[str] = None) -> str:
    """
    Extract device type from guide title or device field.
    
    Examples:
        "Samsung Refrigerator Ice Maker Replacement" -> "refrigerator"
        "GE Dishwasher Spray Arm Replacement" -> "dishwasher"
    """
    # Try device field first
    if device:
        device_lower = device.lower()
        for keyword in DEVICE_KEYWORDS:
            if keyword in device_lower:
                # Return the singular main word
                return keyword.split()[0] if " " in keyword else keyword
    
    # Try title
    title_lower = title.lower()
    for keyword in DEVICE_KEYWORDS:
        if keyword in title_lower:
            return keyword.split()[0] if " " in keyword else keyword
    
    # Fallback
    return "appliance"


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
    
    @property
    def device_type(self) -> str:
        """Extract device type from title/device field."""
        return extract_device_type(self.title, self.device)

    def get_step(self, index: int) -> Optional[PreWEGStep]:
        """Get step by 1-based index."""
        for step in self.steps:
            if step.step_index == index:
                return step
        return None
