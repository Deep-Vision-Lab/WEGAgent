"""
Pipeline State Definition for LangGraph.

This module defines the state that flows through the pipeline graph.
"""
import operator
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from ..models.intermediate import (
    ActionsPerGuide,
    ToolsPerGuide,
    HandsPerGuide,
    PartsTextPerGuide,
    PartsVisionPerGuide,
)
from ..agents.reviewer_agent import ReviewResult


class PipelineState(TypedDict, total=False):
    """
    State dictionary passed through the LangGraph pipeline.
    
    This state accumulates results from each agent and carries
    all necessary data for the combiner to produce the final WEG.
    """
    # === Input ===
    guide_path: str  # Path to pre-WEG JSON
    guide: dict[str, Any]  # Loaded pre-WEG data

    # === Agent Outputs ===
    actions: Optional[ActionsPerGuide]
    tools: Optional[ToolsPerGuide]
    hands: Optional[HandsPerGuide]
    parts_text: Optional[PartsTextPerGuide]
    parts_vision: Optional[PartsVisionPerGuide]

    # === Review Results ===
    review: Optional[ReviewResult]  # Reviewer agent output
    review_passed: bool  # Whether review passed validation

    # === Final Output ===
    weg: Optional[dict[str, Any]]  # Final WEG document
    weg_path: Optional[str]  # Path where WEG was saved

    # === Metadata ===
    # Using Annotated with operator.add to allow concurrent updates from parallel agents
    errors: Annotated[list[str], operator.add]  # Errors encountered during processing
    warnings: Annotated[list[str], operator.add]  # Non-fatal warnings
    completed_agents: Annotated[list[str], operator.add]  # List of agents that completed successfully

    # === Configuration ===
    config: dict[str, Any]  # Pipeline configuration
