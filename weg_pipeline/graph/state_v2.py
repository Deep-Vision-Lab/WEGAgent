"""
Pipeline State Definition for LangGraph V2.

V2 Pipeline: Each agent focuses on ONE task.
- Action Agent: Extract actions only
- Part Text Agent: Extract parts AND components per action
- Tool Agent: Extract tools per action
- Hands Agent: Extract hands per action
- Reviewer: Validates each agent output and triggers refinement
- Combiner: Builds WEG from all agent outputs
"""
import operator
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from ..models.intermediate import (
    ActionsPerGuide,
    PartsTextPerGuideV2,
    ToolsPerGuideV2,
    HandsPerGuideV2,
    PartsVisionPerGuide,
)
from ..agents.reviewer_agent import ReviewResult


class PipelineStateV2(TypedDict, total=False):
    """
    V2 Pipeline State.
    
    Key difference from V1:
    - actions only contains actions (no tool/component/hands in quadruples)
    - parts_text_v2, tools_v2, hands_v2 contain per-action data
    - Combiner builds the quadruples by merging all outputs
    """
    # === Input ===
    guide_path: str  # Path to pre-WEG JSON
    guide: dict[str, Any]  # Loaded pre-WEG data

    # === V2 Agent Outputs ===
    # Action Agent output (actions only, no quadruple fields)
    actions: Optional[ActionsPerGuide]
    
    # Part Text Agent V2 output (parts + components per action)
    parts_text_v2: Optional[PartsTextPerGuideV2]
    
    # Tool Agent V2 output (tools per action)
    tools_v2: Optional[ToolsPerGuideV2]
    
    # Hands Agent V2 output (hands per action)
    hands_v2: Optional[HandsPerGuideV2]
    
    # Part Vision Agent (remains same as V1)
    parts_vision: Optional[PartsVisionPerGuide]

    # === Review Results ===
    # Per-agent review results
    review_actions: Optional[ReviewResult]
    review_parts_text: Optional[ReviewResult]
    review_tools: Optional[ReviewResult]
    review_hands: Optional[ReviewResult]
    
    # Overall review status
    all_reviews_passed: bool

    # === Final Output ===
    weg: Optional[dict[str, Any]]  # Final WEG document
    weg_path: Optional[str]  # Path where WEG was saved

    # === Metadata ===
    errors: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]
    completed_agents: Annotated[list[str], operator.add]

    # === Configuration ===
    config: dict[str, Any]

    # === Refinement Tracking ===
    # Note: Each node should only update its own agent's count
    action_agent_refinements: int
    part_text_agent_refinements: int
    tool_agent_refinements: int
    hands_agent_refinements: int

    suffix: str  # Suffix used for output WEG file 
