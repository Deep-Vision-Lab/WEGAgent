"""
WEG Combiner - Merges all agent outputs into the final WEG format.

V1 Combiner: combine_to_weg() - Uses action_quadruples with pre-filled fields
V2 Combiner: combine_to_weg_v2() - Builds quadruples from separate agent outputs
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
    # V2 imports
    PartsTextPerGuideV2,
    ToolsPerGuideV2,
    HandsPerGuideV2,
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


def combine_to_weg_v2(
    guide: dict[str, Any],
    actions: Optional[ActionsPerGuide] = None,
    parts_text: Optional[PartsTextPerGuideV2] = None,
    tools: Optional[ToolsPerGuideV2] = None,
    hands: Optional[HandsPerGuideV2] = None,
    parts_vision: Optional[PartsVisionPerGuide] = None,
) -> dict[str, Any]:
    """
    V2 Combiner: Build WEG by merging outputs from specialized agents.
    
    Key Difference from V1:
    - In V1, action_quadruples already have tool/component/hands filled
    - In V2, we BUILD the quadruples by merging:
      * action + action_verb from ActionAgent
      * component from PartTextAgent (per action_id)
      * tool from ToolAgent (per action_id)
      * hands from HandsAgent (per action_id)
    
    Args:
        guide: Original pre-WEG guide data
        actions: Action extraction results (actions only, no tool/component/hands)
        parts_text: Part Text V2 results (parts list + components per action)
        tools: Tool V2 results (tools per action)
        hands: Hands V2 results (hands per action)
        parts_vision: Part vision detection results (for parts_all with bboxes)
    
    Returns:
        WEG document as dictionary
    """
    # Build lookup maps for quick access
    actions_result_map: dict[int, ActionExtractionResult] = {}
    if actions:
        for step in actions.steps:
            actions_result_map[step.step_index] = step
    
    # V2: Build lookup maps for per-action data
    # components_map[step_index][action_id] -> (component, part)
    components_map: dict[int, dict[int, tuple[str, Optional[str]]]] = {}
    primary_part_text_map: dict[int, Optional[str]] = {}
    if parts_text:
        for step in parts_text.steps:
            components_map[step.step_index] = {}
            for comp in step.components_per_action:
                components_map[step.step_index][comp.action_id] = (comp.component, comp.part)
            primary_part_text_map[step.step_index] = step.primary_part
    
    # tools_map[step_index][action_id] -> tool
    tools_map: dict[int, dict[int, Optional[str]]] = {}
    if tools:
        for step in tools.steps:
            tools_map[step.step_index] = {}
            for tool_info in step.tools_per_action:
                tools_map[step.step_index][tool_info.action_id] = tool_info.tool
    
    # hands_map[step_index][action_id] -> hands
    hands_map: dict[int, dict[int, int]] = {}
    if hands:
        for step in hands.steps:
            hands_map[step.step_index] = {}
            for hands_info in step.hands_per_action:
                hands_map[step.step_index][hands_info.action_id] = hands_info.hands

    # Part vision lookup
    parts_vision_map = {}
    if parts_vision:
        for step in parts_vision.steps:
            key = (step.step_index, step.image_index)
            parts_vision_map[key] = step

    # Build header
    # Get global toolbox from tools result or guide
    global_toolbox = []
    if tools and hasattr(tools, 'global_toolbox') and tools.global_toolbox:
        global_toolbox = tools.global_toolbox
    else:
        global_toolbox = guide.get("toolbox", [])
    
    header = WEGHeader(
        title=guide.get("title", "Unknown Guide"),
        description=guide.get("summary", ""),
        toolbox=global_toolbox,
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

        # Get action extraction result
        action_result = actions_result_map.get(step_index)
        
        step_actions = action_result.actions if action_result else []
        task_name = action_result.task_name if action_result else None
        hints = action_result.hints if action_result else []
        
        # Get primary part name from V2 parts text
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
                component_to_part_id[detection.name.lower()] = part_id_counter
                part_id_counter += 1

        # === V2: Build action_quadruples by merging all agent outputs ===
        action_quadruples = []
        if action_result and action_result.action_quadruples:
            step_components = components_map.get(step_index, {})
            step_tools = tools_map.get(step_index, {})
            step_hands = hands_map.get(step_index, {})
            
            for action_id, q in enumerate(action_result.action_quadruples):
                # Get component AND part from PartTextAgent (V2)
                component_data = step_components.get(action_id, ("", None))
                component = component_data[0] if isinstance(component_data, tuple) else component_data
                part = component_data[1] if isinstance(component_data, tuple) else None
                
                # Get tool from ToolAgent (V2)
                tool = step_tools.get(action_id)
                
                # Get hands from HandsAgent (V2)
                hands_count = step_hands.get(action_id, 1)
                
                # Try to find matching part_id for this component
                part_id = None
                if component:
                    component_lower = component.lower()
                    
                    # Exact match first
                    if component_lower in component_to_part_id:
                        part_id = component_to_part_id[component_lower]
                    else:
                        # Fuzzy match
                        for part_name, pid in component_to_part_id.items():
                            if component_lower in part_name or part_name in component_lower:
                                part_id = pid
                                break
                
                action_quadruples.append(ActionQuadrupleOutput(
                    action=q.action,  # action verb from ActionAgent
                    precise_action=q.precise_action if hasattr(q, 'precise_action') else None,
                    tool=tool,  # from ToolAgent
                    component=component,  # from PartTextAgent (what you touch)
                    part=part,  # from PartTextAgent (what contains the component)
                    part_id=part_id,
                    hands=hands_count,  # from HandsAgent
                    full_action=q.full_action,  # from ActionAgent
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
