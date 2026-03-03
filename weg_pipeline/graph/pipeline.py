"""
LangGraph Pipeline Construction.

This module builds and runs the WEG extraction pipeline using LangGraph.
"""
from pathlib import Path
from typing import Any, Optional

from langgraph.graph import StateGraph, START, END
from rich.console import Console

from ..combiner.weg_combiner import combine_to_weg
from ..utils.io_utils import save_json, get_output_path
from ..utils.logger import create_logger, get_logger, LogLevel
from .state import PipelineState
from .nodes import (
    load_guide_node,
    action_agent_node,
    tool_agent_node,
    hands_agent_node,
    part_text_agent_node,
    part_vision_agent_node,
    reviewer_agent_node,
)

console = Console()


def synchronization_node(state: PipelineState) -> dict[str, Any]:
    """
    Synchronization barrier node.
    
    This node ensures all parallel agents have completed before 
    proceeding to the reviewer and combiner.
    """
    # Check if already synced
    completed = state.get("completed_agents", [])
    if "sync" in completed:
        console.print("[dim]Sync already completed, skipping...[/dim]")
        return {}
    
    console.print(f"[dim]Sync point reached. Completed agents: {completed}[/dim]")
    
    # Check which agents completed
    expected_agents = {"action_agent", "tool_agent", "hands_agent", "part_vision_agent"}
    completed_set = set(completed)
    
    missing = expected_agents - completed_set
    if missing:
        console.print(f"[yellow]Warning: Some agents may not have completed: {missing}[/yellow]")
    
    return {
        "completed_agents": ["sync"],
    }


def combiner_node(state: PipelineState) -> dict[str, Any]:
    """Combine all agent outputs into final WEG."""
    # Check if combiner has already run to prevent duplicate execution
    completed = state.get("completed_agents", [])
    if "combiner" in completed:
        console.print("[yellow]Combiner already ran, skipping...[/yellow]")
        return {}
    
    console.print("[bold green]Running Combiner...[/bold green]")

    try:
        weg = combine_to_weg(
            guide=state.get("guide", {}),
            actions=state.get("actions"),
            tools=state.get("tools"),
            hands=state.get("hands"),
            parts_text=state.get("parts_text"),
            parts_vision=state.get("parts_vision"),
        )

        # Save WEG
        guide_path = state.get("guide_path", "")
        if guide_path:
            guide_id = weg.get("header", {}).get("guide_id", "")
            weg_path = get_output_path(guide_path, f"{guide_id}_WEG.json")
            save_json(weg_path, weg)
            console.print(f"[bold green]WEG saved to: {weg_path}[/bold green]")
            return {
                "weg": weg,
                "weg_path": str(weg_path),
                "completed_agents": ["combiner"],
            }

        return {
            "weg": weg,
            "completed_agents": ["combiner"],
        }
    except Exception as e:
        console.print(f"[red]Combiner error: {e}[/red]")
        return {
            "errors": [f"Combiner failed: {e}"],
            "completed_agents": ["combiner"],
        }


def create_pipeline(parallel_agents: bool = True, enable_reviewer: bool = True) -> StateGraph:
    """
    Create the LangGraph pipeline for WEG extraction.
    
    Args:
        parallel_agents: If True, run extraction agents in parallel.
                        If False, run sequentially (useful for debugging).
        enable_reviewer: If True, run reviewer agent before combiner.
    
    Returns:
        Compiled StateGraph
    """
    # Create graph
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("load_guide", load_guide_node)
    graph.add_node("action_agent", action_agent_node)
    graph.add_node("tool_agent", tool_agent_node)
    graph.add_node("hands_agent", hands_agent_node)
    graph.add_node("part_text_agent", part_text_agent_node)
    graph.add_node("part_vision_agent", part_vision_agent_node)
    graph.add_node("sync", synchronization_node)
    
    if enable_reviewer:
        graph.add_node("reviewer", reviewer_agent_node)
    
    graph.add_node("combiner", combiner_node)

    # Define edges
    # Always start with loading the guide
    graph.add_edge(START, "load_guide")

    if parallel_agents:
        # Parallel execution strategy:
        # 1. load_guide fans out to multiple agents
        # 2. part_text_agent -> part_vision_agent (sequential dependency)
        # 3. All branches converge at sync node
        # 4. sync -> reviewer -> combiner
        
        # Fan out from load_guide to parallel agents
        graph.add_edge("load_guide", "action_agent")
        graph.add_edge("load_guide", "tool_agent")
        graph.add_edge("load_guide", "hands_agent")
        graph.add_edge("load_guide", "part_text_agent")

        # Part vision depends on part text (to get part names)
        graph.add_edge("part_text_agent", "part_vision_agent")

        # Converge all branches at sync node
        graph.add_edge("action_agent", "sync")
        graph.add_edge("tool_agent", "sync")
        graph.add_edge("hands_agent", "sync")
        graph.add_edge("part_vision_agent", "sync")
        
        # After sync, run reviewer then combiner
        if enable_reviewer:
            graph.add_edge("sync", "reviewer")
            graph.add_edge("reviewer", "combiner")
        else:
            graph.add_edge("sync", "combiner")
    else:
        # Sequential execution
        graph.add_edge("load_guide", "action_agent")
        graph.add_edge("action_agent", "tool_agent")
        graph.add_edge("tool_agent", "hands_agent")
        graph.add_edge("hands_agent", "part_text_agent")
        graph.add_edge("part_text_agent", "part_vision_agent")
        graph.add_edge("part_vision_agent", "sync")
        
        if enable_reviewer:
            graph.add_edge("sync", "reviewer")
            graph.add_edge("reviewer", "combiner")
        else:
            graph.add_edge("sync", "combiner")

    # End after combiner
    graph.add_edge("combiner", END)

    return graph.compile()


def run_pipeline(
    guide_path: str | Path,
    config: Optional[dict[str, Any]] = None,
    parallel: bool = True,
    enable_reviewer: bool = True,
    save_logs: bool = True,
) -> PipelineState:
    """
    Run the complete WEG extraction pipeline on a guide.
    
    Args:
        guide_path: Path to pre-WEG JSON file
        config: Optional configuration dict with:
            - llm_model: Model for text agents (default: gpt-4o-mini)
            - llm_provider: Provider for LLM (default: openai)
            - vlm_model: Model for vision agents (default: gemini-1.5-flash)
            - vlm_provider: Provider for VLM (default: google)
            - verbose: Print progress (default: True)
        parallel: Run agents in parallel (default: True)
        enable_reviewer: Run reviewer agent to validate outputs (default: True)
        save_logs: Save detailed logs to JSON file (default: True)
    
    Returns:
        Final pipeline state with WEG and all intermediate results
    """
    config = config or {}

    # Extract guide_id from path for logging
    guide_path = Path(guide_path)
    guide_id = guide_path.stem.replace("_preWEG", "")
    
    # Initialize logger
    logger = create_logger(guide_id)
    logger.log(LogLevel.INFO, "Pipeline", "Starting WEG extraction pipeline", details={
        "guide_path": str(guide_path),
        "parallel": parallel,
        "enable_reviewer": enable_reviewer,
    })

    # Default configuration
    default_config = {
        "llm_model": "gpt-4o-mini",
        "llm_provider": "openai",
        "vlm_model": "gemini-1.5-flash",
        "vlm_provider": "google",
        "verbose": True,
    }
    default_config.update(config)

    # Create initial state
    initial_state: PipelineState = {
        "guide_path": str(guide_path),
        "guide": None,
        "actions": None,
        "tools": None,
        "hands": None,
        "parts_text": None,
        "parts_vision": None,
        "review": None,
        "review_passed": False,
        "weg": None,
        "weg_path": None,
        "errors": [],
        "warnings": [],
        "completed_agents": [],
        "config": default_config,
    }

    # Create and run pipeline
    console.print("[bold cyan]Creating WEG Pipeline...[/bold cyan]")
    if enable_reviewer:
        console.print("[dim]Reviewer agent: enabled[/dim]")
    pipeline = create_pipeline(parallel_agents=parallel, enable_reviewer=enable_reviewer)

    console.print("[bold cyan]Running Pipeline...[/bold cyan]")
    console.print("=" * 50)

    final_state = pipeline.invoke(initial_state)

    console.print("=" * 50)

    # Report results
    if final_state.get("weg"):
        console.print("[bold green]✓ Pipeline completed successfully![/bold green]")
        if final_state.get("weg_path"):
            console.print(f"  WEG saved to: {final_state['weg_path']}")
    else:
        console.print("[bold red]✗ Pipeline failed to produce WEG[/bold red]")

    # Report review status
    if final_state.get("review"):
        review = final_state["review"]
        if final_state.get("review_passed"):
            console.print("[green]✓ Review validation passed[/green]")
        else:
            console.print(f"[yellow]⚠ Review flagged issues[/yellow]")
            if hasattr(review, 'missing_steps') and review.missing_steps:
                console.print(f"  Missing/incomplete steps: {review.missing_steps}")

    if final_state.get("errors"):
        console.print("[bold red]Errors:[/bold red]")
        for err in final_state["errors"]:
            console.print(f"  - {err}")

    if final_state.get("warnings"):
        # Limit warnings to first 10
        warnings = final_state["warnings"]
        console.print(f"[bold yellow]Warnings ({len(warnings)} total):[/bold yellow]")
        for warn in warnings[:10]:
            console.print(f"  - {warn}")
        if len(warnings) > 10:
            console.print(f"  ... and {len(warnings) - 10} more")

    console.print(f"\nCompleted agents: {final_state.get('completed_agents', [])}")

    # Save logs
    if save_logs:
        # Get output directory from guide path
        output_dir = guide_path.parent / "logs"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / f"{guide_id}_pipeline_log.json"
        logger.save_json_log(str(log_path))
        console.print(f"[dim]Pipeline logs saved to: {log_path}[/dim]")
    
    logger.log(LogLevel.INFO, "Pipeline", "Pipeline completed", details={
        "success": bool(final_state.get("weg")),
        "errors_count": len(final_state.get("errors", [])),
        "warnings_count": len(final_state.get("warnings", [])),
    })

    return final_state
