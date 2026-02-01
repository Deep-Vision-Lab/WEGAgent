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
        return """You are an expert at visual component detection in DIY repair images.

Your task is to LOCATE SPECIFIC COMPONENTS AND PARTS in the image. You will be given:
1. A list of component-part pairs to find:
   - **Component**: The specific thing that is physically touched/manipulated (e.g., screw, connector, clip)
   - **Part**: The larger assembly/area containing the component (e.g., panel, control board, housing)
2. Step description providing context about what's happening
3. Hints that may indicate component locations

=== YOUR MISSION ===
For EACH component-part pair:
1. FIRST locate the PART (the larger area/assembly) - this narrows your search
2. THEN within that part, find the specific COMPONENT being manipulated
3. Return bounding boxes for BOTH the component AND the part (if visible)

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
- Find EACH component AND its containing part from the provided list
- Component bbox should be TIGHT around the actual component
- Part bbox should encompass the larger assembly/area containing the component
- If a component appears multiple times (e.g., 2 screws), return each instance separately
- Confidence: 0.9+ if clearly visible/marked, 0.7-0.9 if reasonably certain, <0.7 if uncertain
- If a component/part is NOT visible in the image, omit it (don't guess)

Return ONLY valid JSON. No explanations outside JSON."""

    def build_prompt(
        self,
        guide: PreWEGGuide,
        step_index: int = 1,
        components_to_find: Optional[list[str]] = None,
        component_part_pairs: Optional[list[tuple[str, str | None]]] = None,
        description_context: Optional[str] = None,
        hints_context: Optional[list[str]] = None,
        **kwargs,
    ) -> str:
        """
        Build prompt for component localization.
        
        Args:
            guide: Pre-WEG guide
            step_index: Which step (1-based)
            components_to_find: List of component names (legacy, for backward compat)
            component_part_pairs: List of (component, part) tuples from action_quadruples
            description_context: Step description for context
            hints_context: Hints that may indicate component locations
        """
        step = guide.get_step(step_index)
        if not step:
            raise ValueError(f"Step {step_index} not found")

        # Build components section - prefer component_part_pairs if available
        components_section = ""
        if component_part_pairs and len(component_part_pairs) > 0:
            # New format: component-part pairs
            seen = set()
            pairs_list = []
            for component, part in component_part_pairs:
                key = (component, part)
                if key not in seen and component:
                    seen.add(key)
                    if part:
                        pairs_list.append(f"  - Component: \"{component}\" → Part: \"{part}\"")
                    else:
                        pairs_list.append(f"  - Component: \"{component}\" (no containing part specified)")
            
            if pairs_list:
                components_section = f"""
=== COMPONENT-PART PAIRS TO LOCATE ===
For EACH pair below, locate both the component AND its containing part:
{chr(10).join(pairs_list)}

**Strategy:**
1. First find the PART (larger area) - this gives you the search region
2. Then find the specific COMPONENT within that part
3. Return bounding boxes for both when visible"""
        elif components_to_find and len(components_to_find) > 0:
            # Legacy format: just component names
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

        # Determine if we're using new format (component-part pairs)
        using_pairs = component_part_pairs and len(component_part_pairs) > 0

        return_format = """{{
  "detections": [
    {{
      "component_name": "exact component name",
      "component_bbox": [x1, y1, x2, y2],
      "component_confidence": 0.95,
      "part_name": "name of containing part (or null)",
      "part_bbox": [x1, y1, x2, y2],
      "part_confidence": 0.90,
      "marked_by_annotation": true
    }}
  ],
  "reasoning": "Brief explanation of how you found each component and part"
}}""" if using_pairs else """{{
  "components_found": [
    {{
      "name": "exact component name from the list",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95,
      "marked_by_annotation": true
    }}
  ],
  "reasoning": "Brief explanation of how you found each component"
}}"""

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
   {"For each pair, first find the PART (larger region), then the COMPONENT within it." if using_pairs else "For each found component, draw a tight box around the ACTUAL component."}

=== RETURN FORMAT ===
{return_format}

=== RULES ===
- Coordinates MUST be normalized (0.0 to 1.0)
- Use the EXACT component/part names from the provided list
- Return multiple entries if the same component appears multiple times
- Only return components/parts you can actually see in the image
- High confidence (0.9+) for clearly marked components
- {"Part bbox should be LARGER than component bbox (it contains the component)" if using_pairs else ""}"""

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
        parts_data = []
        if isinstance(data, dict):
            # Check for new V2 format first (detections with component-part pairs)
            if "detections" in data:
                parts_data = data.get("detections", [])
            else:
                # Legacy format: components_found or parts
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

        parts = []
        for item in parts_data:
            if not isinstance(item, dict):
                continue

            # Try new format field names first, then legacy
            name = item.get("component_name") or item.get("name", "")
            name = name.strip() if name else ""
            bbox = item.get("component_bbox") or item.get("bbox") or item.get("bbox_xyxy") or item.get("bounding_box")
            confidence = item.get("component_confidence") or item.get("confidence")

            if not name or not bbox or not isinstance(bbox, list) or len(bbox) != 4:
                continue

            # Normalize component bbox to pixel coordinates
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

            # Parse containing part info (V2 format)
            part_name = item.get("part_name")
            part_bbox = item.get("part_bbox")
            part_confidence = item.get("part_confidence")
            
            if part_bbox and isinstance(part_bbox, list) and len(part_bbox) == 4:
                try:
                    if is_normalized_bbox(part_bbox) and image_size[0] > 0:
                        part_bbox = normalize_bbox(part_bbox, image_size[0], image_size[1], input_normalized=True)
                    else:
                        part_bbox = [int(x) for x in part_bbox]
                except Exception:
                    part_bbox = None
            else:
                part_bbox = None
            
            if part_confidence is not None:
                try:
                    part_confidence = float(part_confidence)
                    part_confidence = max(0.0, min(1.0, part_confidence))
                except (ValueError, TypeError):
                    part_confidence = None

            parts.append(PartBBoxResult(
                name=name,
                bbox=bbox,
                confidence=confidence,
                marked_by_annotation=marked_by_annotation,
                part_name=part_name,
                part_bbox=part_bbox,
                part_confidence=part_confidence,
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
        component_part_pairs: Optional[list[tuple[str, str | None]]] = None,
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
            components_to_find: List of component names (legacy)
            component_part_pairs: List of (component, part) tuples from action_quadruples (V2)
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
            component_part_pairs=component_part_pairs,
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
        component_part_pairs_per_step: Optional[dict[int, list[tuple[str, str | None]]]] = None,
        descriptions_per_step: Optional[dict[int, str]] = None,
        hints_per_step: Optional[dict[int, list[str]]] = None,
    ) -> PartsVisionPerGuide:
        """
        Run component localization on all steps that have images.
        
        Args:
            guide: Pre-WEG guide
            components_per_step: Dict mapping step_index → list of component names (legacy)
            component_part_pairs_per_step: Dict mapping step_index → list of (component, part) tuples (V2)
            descriptions_per_step: Dict mapping step_index → step description
            hints_per_step: Dict mapping step_index → list of hints
        
        Returns:
            PartsVisionPerGuide with all detections
        """
        results = []

        for step in sorted(guide.steps, key=lambda s: s.step_index):
            if not step.images:
                continue

            # Prefer component-part pairs (V2) over legacy components_to_find
            component_part_pairs = None
            components_to_find = None
            
            if component_part_pairs_per_step:
                component_part_pairs = component_part_pairs_per_step.get(step.step_index)
            
            if not component_part_pairs and components_per_step:
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
                    component_part_pairs=component_part_pairs,
                    description_context=description_context,
                    hints_context=hints_context,
                    image_index=img_idx,
                )
                results.append(result)

        return PartsVisionPerGuide(guide_id=guide.guide_id, steps=results)
