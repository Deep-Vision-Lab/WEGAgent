"""
LangGraph Pipeline V2 Construction.

V2 Pipeline Architecture:
========================
Each agent focuses on ONE task, with reviewer validation after each.

Flow:
1. load_guide
2. Action Agent → Review Actions
3. (Parallel after actions reviewed):
   - Part Text Agent → Review Parts Text → Part Vision Agent
   - Tool Agent → Review Tools
   - Hands Agent → Review Hands
4. Sync
5. V2 Combiner (builds quadruples from all outputs)

The key difference from V1:
- Action Agent extracts ONLY actions (no tool/component/hands)
- Tool/Component/Hands come from separate specialized agents
- Combiner assembles the full quadruples
"""
from pathlib import Path
from typing import Any, Optional

from langgraph.graph import StateGraph, START, END
from rich.console import Console

from ..utils.io_utils import save_json, get_output_path
from ..utils.logger import create_logger, get_logger, LogLevel
from .state_v2 import PipelineStateV2
from .nodes_v2 import (
    load_guide_node_v2,
    action_agent_node_v2,
    review_actions_node_v2,
    part_text_agent_node_v2,
    review_parts_text_node_v2,
    tool_agent_node_v2,
    review_tools_node_v2,
    hands_agent_node_v2,
    review_hands_node_v2,
    part_vision_agent_node_v2,
    sync_node_v2,
    combiner_node_v2,
)

console = Console()


def create_pipeline_v2(parallel_agents: bool = True) -> StateGraph:
    """
    Create the V2 LangGraph pipeline.
    
    Args:
        parallel_agents: If True, run extraction agents in parallel after actions.
                        If False, run sequentially (useful for debugging).
    
    Returns:
        Compiled StateGraph
    
    Pipeline Structure:
    -------------------
    V2 enforces a specific flow:
    1. Actions must be extracted and reviewed FIRST
    2. Other agents (PartText, Tool, Hands) depend on actions
    3. PartVision depends on PartText
    4. All converge at combiner
    """
    graph = StateGraph(PipelineStateV2)

    # Add all nodes
    graph.add_node("load_guide", load_guide_node_v2)
    
    # Action extraction and review (must complete before others)
    graph.add_node("action_agent", action_agent_node_v2)
    graph.add_node("review_actions", review_actions_node_v2)
    
    # Parallel agents (all depend on actions)
    graph.add_node("part_text_agent", part_text_agent_node_v2)
    graph.add_node("review_parts_text", review_parts_text_node_v2)
    graph.add_node("tool_agent", tool_agent_node_v2)
    graph.add_node("review_tools", review_tools_node_v2)
    graph.add_node("hands_agent", hands_agent_node_v2)
    graph.add_node("review_hands", review_hands_node_v2)
    
    # Part vision depends on part text
    graph.add_node("part_vision_agent", part_vision_agent_node_v2)
    
    # Sync and combine
    graph.add_node("sync", sync_node_v2)
    graph.add_node("combiner", combiner_node_v2)

    # === Define Edges ===
    
    # Start with loading guide, then extract actions
    graph.add_edge(START, "load_guide")
    graph.add_edge("load_guide", "action_agent")
    graph.add_edge("action_agent", "review_actions")
    
    if parallel_agents:
        # After actions are reviewed, fan out to parallel agents
        # All three agents can run in parallel since they only need actions
        graph.add_edge("review_actions", "part_text_agent")
        graph.add_edge("review_actions", "tool_agent")
        graph.add_edge("review_actions", "hands_agent")
        
        # Part text -> review -> vision (sequential dependency)
        graph.add_edge("part_text_agent", "review_parts_text")
        graph.add_edge("review_parts_text", "part_vision_agent")
        
        # Tool and hands agents have their own review
        graph.add_edge("tool_agent", "review_tools")
        graph.add_edge("hands_agent", "review_hands")
        
        # All branches converge at sync
        graph.add_edge("part_vision_agent", "sync")
        graph.add_edge("review_tools", "sync")
        graph.add_edge("review_hands", "sync")
    else:
        # Sequential execution (for debugging)
        graph.add_edge("review_actions", "part_text_agent")
        graph.add_edge("part_text_agent", "review_parts_text")
        graph.add_edge("review_parts_text", "tool_agent")
        graph.add_edge("tool_agent", "review_tools")
        graph.add_edge("review_tools", "hands_agent")
        graph.add_edge("hands_agent", "review_hands")
        graph.add_edge("review_hands", "part_vision_agent")
        graph.add_edge("part_vision_agent", "sync")
    
    # Sync to combiner to end
    graph.add_edge("sync", "combiner")
    graph.add_edge("combiner", END)

    return graph.compile()


def run_pipeline_v2(
    guide_path: str | Path,
    config: Optional[dict[str, Any]] = None,
    parallel: bool = True,
    save_logs: bool = True,
    suffix: str = "New_WEG_v2.json",
) -> PipelineStateV2:
    """
    Run the V2 WEG extraction pipeline on a guide.
    
    V2 Pipeline: Each agent focuses on ONE task.
    - Action Agent: Extract actions only
    - Part Text Agent: Extract parts + components per action
    - Tool Agent: Extract tools per action
    - Hands Agent: Estimate hands per action
    - Reviewer: Validates each agent's output
    - Combiner: Builds full quadruples
    
    Args:
        guide_path: Path to pre-WEG JSON file
        config: Optional configuration dict with:
            - llm_model: Model for text agents (default: gpt-4o-mini)
            - llm_provider: Provider for LLM (default: openai)
            - vlm_model: Model for vision agents (default: gemini-1.5-flash)
            - vlm_provider: Provider for VLM (default: google)
            - verbose: Print progress (default: True)
        parallel: Run agents in parallel where possible (default: True)
        save_logs: Save detailed logs to JSON file (default: True)
    
    Returns:
        Final pipeline state with WEG and all intermediate results
    """
    config = config or {}

    # Extract guide_id from path
    guide_path = Path(guide_path)
    guide_id = guide_path.stem.replace("_preWEG", "")
    
    # Initialize logger
    logger = create_logger(guide_id)
    logger.log(LogLevel.INFO, "Pipeline", "Starting V2 WEG extraction pipeline", details={
        "guide_path": str(guide_path),
        "parallel": parallel,
        "version": "v2",
    })

    # Default configuration
    default_config = {
        "llm_model": "gpt-4o-mini",
        "llm_provider": "openai",
        "vlm_model": "gemini-1.5-flash",
        "vlm_provider": "google",
        "verbose": True,
        "max_refinement_iterations": 2,
    }
    default_config.update(config)

    # Create initial state
    initial_state: PipelineStateV2 = {
        "guide_path": str(guide_path),
        "guide": None,
        "actions": None,
        "parts_text_v2": None,
        "tools_v2": None,
        "hands_v2": None,
        "parts_vision": None,
        "review_actions": None,
        "review_parts_text": None,
        "review_tools": None,
        "review_hands": None,
        "all_reviews_passed": False,
        "weg": None,
        "weg_path": None,
        "errors": [],
        "warnings": [],
        "completed_agents": [],
        "config": default_config,
        "action_agent_refinements": 0,
        "part_text_agent_refinements": 0,
        "tool_agent_refinements": 0,
        "hands_agent_refinements": 0,
        "suffix": suffix,
    }

    # Create and run pipeline
    console.print("[bold cyan]Creating V2 WEG Pipeline...[/bold cyan]")
    console.print("[dim]V2: Specialized agents, each focusing on one task[/dim]")
    
    pipeline = create_pipeline_v2(parallel_agents=parallel)

    console.print("[bold cyan]Running V2 Pipeline...[/bold cyan]")
    console.print("=" * 50)

    final_state = pipeline.invoke(initial_state)

    console.print("=" * 50)

    # Report results
    if final_state.get("weg"):
        console.print("[bold green]✓ V2 Pipeline completed successfully![/bold green]")
        if final_state.get("weg_path"):
            console.print(f"  WEG saved to: {final_state['weg_path']}")
    else:
        console.print("[bold red]✗ V2 Pipeline failed to produce WEG[/bold red]")

    # Report review status
    if final_state.get("all_reviews_passed"):
        console.print("[green]✓ All agent reviews passed[/green]")
    else:
        console.print("[yellow]⚠ Some agent reviews flagged issues[/yellow]")
        for review_key in ["review_actions", "review_parts_text", "review_tools", "review_hands"]:
            review = final_state.get(review_key)
            if review and not review.overall_valid:
                agent_name = review_key.replace("review_", "").replace("_", " ").title()
                console.print(f"  - {agent_name}: {review.summary if hasattr(review, 'summary') else 'Issues found'}")

    # Report refinement counts
    refinement_counts = {
        "action_agent": final_state.get("action_agent_refinements", 0),
        "part_text_agent": final_state.get("part_text_agent_refinements", 0),
        "tool_agent": final_state.get("tool_agent_refinements", 0),
        "hands_agent": final_state.get("hands_agent_refinements", 0),
    }
    if any(v > 0 for v in refinement_counts.values()):
        console.print(f"[dim]Refinement iterations: {refinement_counts}[/dim]")

    if final_state.get("errors"):
        console.print("[bold red]Errors:[/bold red]")
        for err in final_state["errors"]:
            console.print(f"  - {err}")

    if final_state.get("warnings"):
        warnings = final_state["warnings"]
        console.print(f"[bold yellow]Warnings ({len(warnings)} total):[/bold yellow]")
        for warn in warnings[:10]:
            console.print(f"  - {warn}")
        if len(warnings) > 10:
            console.print(f"  ... and {len(warnings) - 10} more")

    console.print(f"\nCompleted agents: {final_state.get('completed_agents', [])}")

    # Save logs
    if save_logs:
        output_dir = guide_path.parent / "logs"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / f"{guide_id}_pipeline_v2_log.json"
        logger.save_json_log(str(log_path))
        console.print(f"[dim]Pipeline logs saved to: {log_path}[/dim]")
    
    logger.log(LogLevel.INFO, "Pipeline", "V2 Pipeline completed", details={
        "success": bool(final_state.get("weg")),
        "errors_count": len(final_state.get("errors", [])),
        "warnings_count": len(final_state.get("warnings", [])),
        "refinements": refinement_counts,
    })

    return final_state
