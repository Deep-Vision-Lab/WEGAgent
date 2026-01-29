"""
Part Vision Agent - Localizes parts in images with bounding boxes.

Uses VLM (Gemini/GPT-4o) to detect and localize parts mentioned in step text.
"""
from pathlib import Path
from typing import Optional

from ..models.preweg import PreWEGGuide, PreWEGStep
from ..models.intermediate import PartBBoxResult, PartVisionResult, PartsVisionPerGuide
from ..utils.llm_utils import extract_json_from_response
from ..utils.image_utils import get_image_size, normalize_bbox, is_normalized_bbox
from .base_agent import BaseAgent


class PartVisionAgent(BaseAgent):
    """
    Localizes parts in repair step images using bounding boxes.
    
    Given a step's text and image, identifies and locates the parts
    that are being manipulated.
    """

    AGENT_NAME = "PartVisionAgent"
    REQUIRES_VISION = True
    DEFAULT_PROVIDER = "google"
    DEFAULT_MODEL = "gemini-2.5-flash"

    def get_system_prompt(self) -> str:
        return """You are an expert at visual component detection in appliance repair images.

Your task is to LOCATE SPECIFIC COMPONENTS in the image. You will be given:
1. A list of exact component names to find (extracted from action descriptions)
2. Step description providing context about what's happening
3. Hints that may indicate component locations

=== YOUR MISSION ===
For EACH component in the provided list, find and bound it in the image with a precise bounding box.

=== VISUAL ANNOTATION DETECTION (CRITICAL!) ===
Repair images often contain RED CIRCLES, ARROWS, or HIGHLIGHTS marking important components:

1. **Red Circles typically mark:**
   - Screws/bolts to remove
   - Connectors to disconnect  
   - Tabs to press/release
   - Key action locations

2. **Use annotations to locate components:**
   - If looking for "mounting screw" and you see red circles → those likely mark the screws
   - Match visual annotations to the component names you're searching for
   - Red circles = high confidence locations for the described components

=== BOUNDING BOX FORMAT ===
Coordinates as [x1, y1, x2, y2]:
- x1, y1 = top-left corner
- x2, y2 = bottom-right corner  
- NORMALIZED values (0.0 to 1.0) relative to image dimensions

=== DETECTION GUIDELINES ===
- Find EACH component from the provided list
- Be precise - tightly bound the actual component, not surrounding area
- If a component appears multiple times (e.g., 2 screws), return each separately
- Confidence: 0.9+ if clearly visible/marked, 0.7-0.9 if reasonably certain, <0.7 if uncertain
- If a component is NOT visible in the image, omit it (don't guess)

Return ONLY valid JSON. No explanations outside JSON."""

    def build_prompt(
        self,
        guide: PreWEGGuide,
        step_index: int = 1,
        components_to_find: Optional[list[str]] = None,
        description_context: Optional[str] = None,
        hints_context: Optional[list[str]] = None,
        **kwargs,
    ) -> str:
        """
        Build prompt for component localization.
        
        Args:
            guide: Pre-WEG guide
            step_index: Which step (1-based)
            components_to_find: List of component names from action_quadruples
            description_context: Step description for context
            hints_context: Hints that may indicate component locations
        """
        step = guide.get_step(step_index)
        if not step:
            raise ValueError(f"Step {step_index} not found")

        # Build components list section
        components_section = ""
        if components_to_find and len(components_to_find) > 0:
            unique_components = list(dict.fromkeys(components_to_find))  # Preserve order, remove duplicates
            components_list = "\n".join(f"  - {comp}" for comp in unique_components)
            components_section = f"""
=== COMPONENTS TO LOCATE ===
Find and bound EACH of these components in the image:
{components_list}

For each component, create a precise bounding box around it.
If a component appears multiple times (e.g., "screw" → 2 screws visible), return each instance."""
        else:
            components_section = """
=== COMPONENTS TO LOCATE ===
No specific components provided. Identify the PRIMARY component(s) being manipulated based on the description."""

        # Build description context section
        description_section = ""
        if description_context:
            description_section = f"""
=== STEP DESCRIPTION (Context) ===
\"\"\"{description_context}\"\"\"

This tells you WHAT is happening in this step. Use it to understand which components are important."""

        # Build hints section
        hints_section = ""
        if hints_context and len(hints_context) > 0:
            hints_list = "\n".join(f"  • {hint}" for hint in hints_context)
            hints_section = f"""
=== HINTS (Location Clues) ===
These hints may help locate components:
{hints_list}"""

        return f"""Locate the specified components in this repair image.

=== CONTEXT ===
Guide: {guide.title}
Step {step_index} of {guide.num_steps}
{components_section}
{description_section}
{hints_section}

=== DETECTION PROCESS ===

1. **Scan for Visual Markers**
   Look for RED CIRCLES, ARROWS, or HIGHLIGHTS - these mark key components.

2. **Match Components to Visual Cues**
   For each component in the list:
   - Is there a red circle/marker near where this component should be?
   - Does the description mention where it's located?
   - Do the hints provide location info?

3. **Create Precise Bounding Boxes**
   For each found component, draw a tight box around the ACTUAL component.

=== RETURN FORMAT ===
{{
  "components_found": [
    {{
      "name": "exact component name from the list",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95,
      "marked_by_annotation": true
    }},
    {{
      "name": "another component",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.85,
      "marked_by_annotation": false
    }}
  ],
  "reasoning": "Brief explanation of how you found each component"
}}

=== RULES ===
- Coordinates MUST be normalized (0.0 to 1.0)
- Use the EXACT component name from the provided list
- Return multiple entries if the same component appears multiple times
- Only return components you can actually see in the image
- High confidence (0.9+) for clearly marked components"""

    def parse_response(
        self,
        response: str,
        guide: PreWEGGuide,
        step_index: int = 1,
        image_path: Optional[Path | str] = None,
        **kwargs,
    ) -> PartVisionResult:
        """Parse VLM response into structured part detections."""
        from ..utils.logger import get_logger
        
        # Get image size for bbox normalization
        image_size = (0, 0)
        if image_path:
            try:
                image_size = get_image_size(image_path)
            except Exception:
                pass

        try:
            data = extract_json_from_response(response)
        except ValueError as e:
            self.log_error(f"Failed to parse JSON: {e}")
            return PartVisionResult(
                step_index=step_index,
                image_index=1,
                image_path=str(image_path) if image_path else "",
                image_size=image_size,
                parts=[],
                raw_model_output=response,
            )

        # Handle different response formats
        if isinstance(data, dict):
            # New format: components_found
            parts_data = data.get("components_found", data.get("parts", []))
            
            # Log reasoning if available
            logger = get_logger()
            if logger:
                reasoning = data.get("reasoning", "")
                if reasoning:
                    logger.agent_reasoning(
                        self.AGENT_NAME,
                        step_index,
                        f"Component detection for step {step_index}",
                        reasoning,
                        f"Found {len(parts_data)} components",
                    )
        elif isinstance(data, list):
            parts_data = data
        else:
            parts_data = []

        parts = []
        for item in parts_data:
            if not isinstance(item, dict):
                continue

            name = item.get("name", "").strip()
            bbox = item.get("bbox") or item.get("bbox_xyxy") or item.get("bounding_box")
            confidence = item.get("confidence")

            if not name or not bbox or not isinstance(bbox, list) or len(bbox) != 4:
                continue

            # Normalize bbox to pixel coordinates
            try:
                if is_normalized_bbox(bbox) and image_size[0] > 0:
                    bbox = normalize_bbox(bbox, image_size[0], image_size[1], input_normalized=True)
                else:
                    bbox = [int(x) for x in bbox]
            except Exception:
                continue

            # Validate confidence
            if confidence is not None:
                try:
                    confidence = float(confidence)
                    confidence = max(0.0, min(1.0, confidence))
                except (ValueError, TypeError):
                    confidence = None

            # Check if part was marked by visual annotation
            marked_by_annotation = item.get("marked_by_annotation", False)
            if not isinstance(marked_by_annotation, bool):
                marked_by_annotation = str(marked_by_annotation).lower() in ("true", "1", "yes")

            parts.append(PartBBoxResult(
                name=name,
                bbox=bbox,
                confidence=confidence,
                marked_by_annotation=marked_by_annotation,
            ))

        return PartVisionResult(
            step_index=step_index,
            image_index=kwargs.get("image_index", 1),
            image_path=str(image_path) if image_path else "",
            image_size=image_size,
            parts=parts,
            raw_model_output=response,
        )

    def run_on_step(
        self,
        guide: PreWEGGuide,
        step_index: int,
        image_path: Path | str,
        components_to_find: Optional[list[str]] = None,
        description_context: Optional[str] = None,
        hints_context: Optional[list[str]] = None,
        image_index: int = 1,
    ) -> PartVisionResult:
        """
        Run component localization on a single step image.
        
        Args:
            guide: Pre-WEG guide
            step_index: Which step (1-based)
            image_path: Path to the step's image
            components_to_find: List of component names from action_quadruples
            description_context: Step description for context
            hints_context: Hints that may indicate component locations
            image_index: Which image in the step (1-based)
        
        Returns:
            PartVisionResult with detected components and bboxes
        """
        prompt = self.build_prompt(
            guide,
            step_index=step_index,
            components_to_find=components_to_find,
            description_context=description_context,
            hints_context=hints_context,
        )

        system_prompt = self.get_system_prompt()

        self.log(f"Processing step {step_index}, image {image_index}")
        response = self.client.complete_with_image(
            prompt,
            image_path,
            system_prompt=system_prompt,
        )

        return self.parse_response(
            response,
            guide,
            step_index=step_index,
            image_path=image_path,
            image_index=image_index,
        )

    def run_all_steps(
        self,
        guide: PreWEGGuide,
        components_per_step: Optional[dict[int, list[str]]] = None,
        descriptions_per_step: Optional[dict[int, str]] = None,
        hints_per_step: Optional[dict[int, list[str]]] = None,
    ) -> PartsVisionPerGuide:
        """
        Run component localization on all steps that have images.
        
        Args:
            guide: Pre-WEG guide
            components_per_step: Dict mapping step_index → list of component names from action_quadruples
            descriptions_per_step: Dict mapping step_index → step description
            hints_per_step: Dict mapping step_index → list of hints
        
        Returns:
            PartsVisionPerGuide with all detections
        """
        results = []

        for step in sorted(guide.steps, key=lambda s: s.step_index):
            if not step.images:
                continue

            components_to_find = None
            if components_per_step:
                components_to_find = components_per_step.get(step.step_index)

            description_context = None
            if descriptions_per_step:
                description_context = descriptions_per_step.get(step.step_index)
            elif step.full_description:
                description_context = step.full_description

            hints_context = None
            if hints_per_step:
                hints_context = hints_per_step.get(step.step_index)

            for img_idx, img in enumerate(step.images, start=1):
                image_path = Path(img.saved_path)
                if not image_path.exists():
                    self.log(f"Image not found: {image_path}", style="yellow")
                    continue

                result = self.run_on_step(
                    guide,
                    step.step_index,
                    image_path,
                    components_to_find=components_to_find,
                    description_context=description_context,
                    hints_context=hints_context,
                    image_index=img_idx,
                )
                results.append(result)

        return PartsVisionPerGuide(guide_id=guide.guide_id, steps=results)
