"""
WEG Combiner - Merges all agent outputs into the final WEG format.
"""
from typing import Any, Optional

from ..models.weg import WEGDocument, WEGHeader, WEGStep, PartInfo, BoundingBox, ActionQuadrupleOutput
from ..models.intermediate import (
    ActionsPerGuide,
    ToolsPerGuide,
    HandsPerGuide,
    PartsTextPerGuide,
    PartsVisionPerGuide,
    ActionExtractionResult,
)


def combine_to_weg(
    guide: dict[str, Any],
    actions: Optional[ActionsPerGuide] = None,
    tools: Optional[ToolsPerGuide] = None,
    hands: Optional[HandsPerGuide] = None,
    parts_text: Optional[PartsTextPerGuide] = None,
    parts_vision: Optional[PartsVisionPerGuide] = None,
) -> dict[str, Any]:
    """
    Combine all extraction results into the final WEG document.
    
    Args:
        guide: Original pre-WEG guide data
        actions: Action extraction results (with quadruples, hints, task_name)
        tools: Tool extraction results (used only for global_toolbox in header)
        hands: Hands estimation results (deprecated - hands is per-action in quadruples)
        parts_text: Part text extraction results (for primary_part fallback)
        parts_vision: Part vision detection results (for parts_all with bboxes)
    
    Returns:
        WEG document as dictionary
    """
    # Build lookup maps for quick access
    actions_result_map: dict[int, ActionExtractionResult] = {}
    if actions:
        for step in actions.steps:
            actions_result_map[step.step_index] = step

    # Note: tools are now in action_quadruples, tools agent only provides global_toolbox
    # Note: hands is now per-action in action_quadruples, not per-step

    parts_text_map = {}
    primary_part_text_map = {}
    if parts_text:
        for step in parts_text.steps:
            parts_text_map[step.step_index] = step.parts
            primary_part_text_map[step.step_index] = step.primary_part

    # Part vision is indexed by (step_index, image_index)
    parts_vision_map = {}
    if parts_vision:
        for step in parts_vision.steps:
            key = (step.step_index, step.image_index)
            parts_vision_map[key] = step

    # Build header (toolbox from tools agent's global_toolbox or guide)
    header = WEGHeader(
        title=guide.get("title", "Unknown Guide"),
        description=guide.get("summary", ""),
        toolbox=tools.global_toolbox if tools else guide.get("toolbox", []),
        parts_list=guide.get("parts", []),
        source_url=guide.get("source_url"),
        guide_id=guide.get("guide_id"),
    )

    # Build steps
    steps = []
    guide_steps = guide.get("steps", [])
    guide_steps_sorted = sorted(
        [s for s in guide_steps if isinstance(s, dict)],
        key=lambda s: s.get("step_index", 0),
    )

    for step_data in guide_steps_sorted:
        step_index = step_data.get("step_index", 0)

        # Get action extraction result (includes actions, quadruples, hints, task_name)
        action_result = actions_result_map.get(step_index)
        
        step_actions = action_result.actions if action_result else []
        task_name = action_result.task_name if action_result else None
        hints = action_result.hints if action_result else []
        
        # Get primary part name from text extraction (used for primary_part fallback)
        primary_part_name = primary_part_text_map.get(step_index)

        # Build parts_all from vision detections with part_ids
        parts_all = []
        part_id_counter = 1
        component_to_part_id = {}  # Map component name (lowercase) -> part_id
        
        vision_result = parts_vision_map.get((step_index, 1))
        if vision_result and vision_result.parts:
            for detection in vision_result.parts:
                part_info = PartInfo(
                    part_id=part_id_counter,
                    name=detection.name,
                    bbox=BoundingBox(
                        x1=detection.bbox[0],
                        y1=detection.bbox[1],
                        x2=detection.bbox[2],
                        y2=detection.bbox[3],
                    ),
                    confidence=detection.confidence,
                    image_path=vision_result.image_path,
                )
                parts_all.append(part_info)
                # Map the component name to this part_id for linking
                component_to_part_id[detection.name.lower()] = part_id_counter
                part_id_counter += 1

        # Convert quadruples to output format with part_id linking
        action_quadruples = []
        if action_result and action_result.action_quadruples:
            for q in action_result.action_quadruples:
                # Try to find matching part_id for this component
                part_id = None
                component_lower = q.component.lower()
                
                # Exact match first
                if component_lower in component_to_part_id:
                    part_id = component_to_part_id[component_lower]
                else:
                    # Fuzzy match: check if component name is contained in any part name or vice versa
                    for part_name, pid in component_to_part_id.items():
                        if component_lower in part_name or part_name in component_lower:
                            part_id = pid
                            break
                
                action_quadruples.append(ActionQuadrupleOutput(
                    action=q.action,
                    precise_action=q.precise_action,
                    tool=q.tool,
                    component=q.component,
                    part_id=part_id,
                    hands=q.hands,
                    full_action=q.full_action,
                ))

        # Determine primary_part
        primary_part = None
        
        if parts_all:
            # Try to match with text-extracted primary part name
            if primary_part_name:
                for part in parts_all:
                    if primary_part_name.lower() in part.name.lower():
                        primary_part = part
                        break
            
            # If no match, use highest confidence detection
            if primary_part is None:
                primary_part = max(parts_all, key=lambda p: p.confidence or 0.0)
        
        # If no vision detection but we have text-extracted primary part, create partial info
        if primary_part is None and primary_part_name:
            primary_part = PartInfo(part_id=1, name=primary_part_name)

        # Build step
        weg_step = WEGStep(
            step_id=step_index,
            task_name=task_name,
            description=step_data.get("full_description", ""),
            actions=step_actions,
            action_quadruples=action_quadruples,
            hints=hints,
            primary_part=primary_part,
            parts_all=parts_all,
        )
        steps.append(weg_step)

    # Build final document
    weg_doc = WEGDocument(header=header, steps=steps)

    return weg_doc.to_dict()
