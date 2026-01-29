"""
Graph Node Functions for the WEG Pipeline.

Each node function processes state and returns updates to be merged.
"""
from pathlib import Path
from typing import Any

from rich.console import Console

from ..agents import ActionAgent, ToolAgent, HandsAgent, PartTextAgent, PartVisionAgent, ReviewerAgent
from ..models.preweg import PreWEGGuide
from ..utils.io_utils import load_json
from ..utils.logger import get_logger, LogLevel
from .state import PipelineState

console = Console()


def load_guide_node(state: PipelineState) -> dict[str, Any]:
    """Load the pre-WEG guide from disk."""
    guide_path = state["guide_path"]
    console.print(f"[bold blue]Loading guide: {guide_path}[/bold blue]")
    
    logger = get_logger()

    try:
        guide_data = load_json(guide_path)
        if logger:
            logger.log(LogLevel.INFO, "Pipeline", f"Loaded guide with {len(guide_data.get('steps', []))} steps")
        return {
            "guide": guide_data,
            "completed_agents": ["load_guide"],
        }
    except Exception as e:
        if logger:
            logger.log(LogLevel.ERROR, "Pipeline", f"Failed to load guide: {e}")
        return {
            "errors": [f"Failed to load guide: {e}"],
        }


def action_agent_node(state: PipelineState) -> dict[str, Any]:
    """Run the Action Agent."""
    console.print("[bold magenta]Running Action Agent...[/bold magenta]")
    
    logger = get_logger()

    guide_data = state.get("guide")
    if not guide_data:
        return {
            "errors": ["No guide loaded for action agent"],
            "completed_agents": ["action_agent"],
        }

    try:
        guide = PreWEGGuide(**guide_data)
        config = state.get("config", {})

        agent = ActionAgent(
            model=config.get("vlm_model", "gemini-1.5-flash"),
            provider=config.get("vlm_provider", "google"),
            verbose=config.get("verbose", True),
        )
        
        if logger:
            logger.agent_start("ActionAgent")

        # Run without images for now (faster), can be enhanced later
        actions = agent.run(guide)
        
        if logger:
            total_actions = sum(len(s.actions) for s in actions.steps)
            logger.agent_complete("ActionAgent", result_summary=f"Extracted {total_actions} actions from {len(actions.steps)} steps")

        return {
            "actions": actions,
            "completed_agents": ["action_agent"],
        }
    except Exception as e:
        console.print(f"[red]Action Agent error: {e}[/red]")
        if logger:
            logger.agent_error("ActionAgent", str(e))
        return {
            "errors": [f"Action agent failed: {e}"],
            "completed_agents": ["action_agent"],
        }


def tool_agent_node(state: PipelineState) -> dict[str, Any]:
    """Run the Tool Agent."""
    console.print("[bold magenta]Running Tool Agent...[/bold magenta]")
    
    logger = get_logger()

    guide_data = state.get("guide")
    if not guide_data:
        return {
            "errors": ["No guide loaded for tool agent"],
            "completed_agents": ["tool_agent"],
        }

    try:
        guide = PreWEGGuide(**guide_data)
        config = state.get("config", {})

        agent = ToolAgent(
            model=config.get("llm_model", "gpt-4o-mini"),
            provider=config.get("llm_provider", "openai"),
            verbose=config.get("verbose", True),
        )

        tools = agent.run(guide)

        return {
            "tools": tools,
            "completed_agents": ["tool_agent"],
        }
    except Exception as e:
        console.print(f"[red]Tool Agent error: {e}[/red]")
        return {
            "errors": [f"Tool agent failed: {e}"],
            "completed_agents": ["tool_agent"],
        }


def hands_agent_node(state: PipelineState) -> dict[str, Any]:
    """Run the Hands Agent."""
    console.print("[bold magenta]Running Hands Agent...[/bold magenta]")

    guide_data = state.get("guide")
    if not guide_data:
        return {
            "errors": ["No guide loaded for hands agent"],
            "completed_agents": ["hands_agent"],
        }

    try:
        guide = PreWEGGuide(**guide_data)
        config = state.get("config", {})

        agent = HandsAgent(
            model=config.get("llm_model", "gpt-4o-mini"),
            provider=config.get("llm_provider", "openai"),
            verbose=config.get("verbose", True),
        )

        hands = agent.run(guide)

        return {
            "hands": hands,
            "completed_agents": ["hands_agent"],
        }
    except Exception as e:
        console.print(f"[red]Hands Agent error: {e}[/red]")
        return {
            "errors": [f"Hands agent failed: {e}"],
            "completed_agents": ["hands_agent"],
        }


def part_text_agent_node(state: PipelineState) -> dict[str, Any]:
    """Run the Part Text Agent."""
    console.print("[bold magenta]Running Part Text Agent...[/bold magenta]")

    guide_data = state.get("guide")
    if not guide_data:
        return {
            "errors": ["No guide loaded for part text agent"],
            "completed_agents": ["part_text_agent"],
        }

    try:
        guide = PreWEGGuide(**guide_data)
        config = state.get("config", {})

        agent = PartTextAgent(
            model=config.get("llm_model", "gpt-4o-mini"),
            provider=config.get("llm_provider", "openai"),
            verbose=config.get("verbose", True),
        )

        parts_text = agent.run(guide)

        return {
            "parts_text": parts_text,
            "completed_agents": ["part_text_agent"],
        }
    except Exception as e:
        console.print(f"[red]Part Text Agent error: {e}[/red]")
        return {
            "errors": [f"Part text agent failed: {e}"],
            "completed_agents": ["part_text_agent"],
        }


def part_vision_agent_node(state: PipelineState) -> dict[str, Any]:
    """Run the Part Vision Agent."""
    console.print("[bold magenta]Running Part Vision Agent...[/bold magenta]")

    guide_data = state.get("guide")
    if not guide_data:
        return {
            "errors": ["No guide loaded for part vision agent"],
            "completed_agents": ["part_vision_agent"],
        }

    try:
        guide = PreWEGGuide(**guide_data)
        config = state.get("config", {})

        agent = PartVisionAgent(
            model=config.get("vlm_model", "gemini-1.5-flash"),
            provider=config.get("vlm_provider", "google"),
            verbose=config.get("verbose", True),
        )

        # Extract components from action_quadruples for each step
        components_per_step = {}
        descriptions_per_step = {}
        hints_per_step = {}
        
        actions_result = state.get("actions")
        if actions_result and hasattr(actions_result, "steps"):
            for step_result in actions_result.steps:
                # Get all components from action_quadruples
                components = []
                if step_result.action_quadruples:
                    for quad in step_result.action_quadruples:
                        if quad.component:
                            components.append(quad.component)
                if components:
                    components_per_step[step_result.step_index] = components
                
                # Get hints for this step
                if step_result.hints:
                    hints_per_step[step_result.step_index] = step_result.hints
        
        # Get descriptions from guide steps
        for step in guide.steps:
            if step.full_description:
                descriptions_per_step[step.step_index] = step.full_description

        parts_vision = agent.run_all_steps(
            guide, 
            components_per_step=components_per_step,
            descriptions_per_step=descriptions_per_step,
            hints_per_step=hints_per_step,
        )

        return {
            "parts_vision": parts_vision,
            "completed_agents": ["part_vision_agent"],
        }
    except Exception as e:
        console.print(f"[red]Part Vision Agent error: {e}[/red]")
        return {
            "errors": [f"Part vision agent failed: {e}"],
            "completed_agents": ["part_vision_agent"],
        }


def reviewer_agent_node(state: PipelineState) -> dict[str, Any]:
    """
    Run the Reviewer Agent to validate all extraction results.
    
    This node runs AFTER all extraction agents complete and BEFORE the combiner.
    It validates the quality and completeness of extracted data.
    When issues are found, it triggers refinement of specific steps.
    """
    # Check if reviewer already ran
    completed = state.get("completed_agents", [])
    if "reviewer_agent" in completed:
        console.print("[dim]Reviewer already ran, skipping...[/dim]")
        return {}
    
    console.print("[bold cyan]Running Reviewer Agent...[/bold cyan]")
    
    logger = get_logger()

    guide_data = state.get("guide")
    if not guide_data:
        return {
            "errors": ["No guide loaded for reviewer agent"],
            "completed_agents": ["reviewer_agent"],
            "review_passed": False,
        }

    try:
        guide = PreWEGGuide(**guide_data)
        config = state.get("config", {})
        max_refinement_iterations = config.get("max_refinement_iterations", 2)

        reviewer = ReviewerAgent(
            model=config.get("llm_model", "claude-3-5-haiku-20241022"),
            provider=config.get("llm_provider", "anthropic"),
            verbose=config.get("verbose", True),
        )
        
        if logger:
            logger.agent_start("ReviewerAgent")

        # Get current actions for potential refinement
        current_actions = state.get("actions")

        # Review all agent outputs
        review_result = reviewer.review(
            guide=guide,
            actions=current_actions,
            tools=state.get("tools"),
            hands=state.get("hands"),
            parts_text=state.get("parts_text"),
        )

        # Log review summary
        console.print(f"[bold]Review Summary:[/bold] {review_result.summary}")
        
        if review_result.missing_steps:
            console.print(f"[yellow]⚠ Missing/incomplete steps: {review_result.missing_steps}[/yellow]")
        
        # Count issues
        issue_count = sum(len(step.issues) for step in review_result.steps)
        if issue_count > 0:
            console.print(f"[yellow]⚠ Total issues found: {issue_count}[/yellow]")
            # Print first few issues
            for step in review_result.steps[:5]:
                if step.issues:
                    console.print(f"  Step {step.step_index}: {step.issues}")
        
        # === REFINEMENT LOOP ===
        # If there are issues and refinement is enabled, ask agents to fix
        refinement_requests = reviewer.get_refinement_requests(review_result)
        
        if refinement_requests and max_refinement_iterations > 0 and current_actions:
            console.print(f"\n[bold cyan]Starting Refinement Loop ({len(refinement_requests)} requests)...[/bold cyan]")
            
            # Create ActionAgent for refinement
            action_agent = ActionAgent(
                model=config.get("vlm_model", "gemini-1.5-flash"),
                provider=config.get("vlm_provider", "google"),
                verbose=config.get("verbose", True),
            )
            
            # Process ActionAgent refinements
            refined_steps = {}
            for request in refinement_requests:
                if request.agent_name == "ActionAgent":
                    console.print(f"  [cyan]Refining step {request.step_index}...[/cyan]")
                    
                    # Find original result for this step
                    original_result = None
                    for step in current_actions.steps:
                        if step.step_index == request.step_index:
                            original_result = step
                            break
                    
                    if logger:
                        logger.log(
                            LogLevel.AGENT_ACTION,
                            "ActionAgent",
                            f"Received refinement request",
                            step_index=request.step_index,
                            details={"feedback": request.feedback[:200]},
                        )
                    
                    # Refine this step
                    refined_result = action_agent.refine_step(
                        guide=guide,
                        step_index=request.step_index,
                        feedback=request.feedback,
                        original_result=original_result,
                    )
                    
                    refined_steps[request.step_index] = refined_result
                    
                    if logger:
                        logger.agent_response_to_feedback(
                            "ActionAgent",
                            request.step_index,
                            original_result.action_quadruples if original_result else [],
                            refined_result.action_quadruples,
                            f"Refined based on: {request.feedback[:100]}",
                        )
            
            # Merge refined steps back into actions
            if refined_steps:
                # Get existing step indices
                existing_indices = {step.step_index for step in current_actions.steps}
                
                new_steps = []
                for step in current_actions.steps:
                    if step.step_index in refined_steps:
                        new_steps.append(refined_steps[step.step_index])
                    else:
                        new_steps.append(step)
                
                # Add NEW steps that weren't in original actions (e.g., steps 22-27)
                for step_idx, refined_step in refined_steps.items():
                    if step_idx not in existing_indices:
                        new_steps.append(refined_step)
                
                # Sort by step index to maintain order
                new_steps.sort(key=lambda s: s.step_index)
                
                # Update actions with refined versions
                from ..models.intermediate import ActionsPerGuide
                current_actions = ActionsPerGuide(
                    guide_id=current_actions.guide_id,
                    steps=new_steps,
                )
                
                console.print(f"  [green]✓ Refined {len(refined_steps)} steps[/green]")
        
        if review_result.overall_valid:
            console.print("[green]✓ Review passed[/green]")
        else:
            console.print("[yellow]⚠ Review flagged issues (proceeding with warnings)[/yellow]")
        
        if logger:
            logger.agent_complete(
                "ReviewerAgent", 
                result_summary=f"{'Passed' if review_result.overall_valid else 'Flagged issues'}, "
                              f"{issue_count} issues, {len(refinement_requests)} refinements"
            )

        # Convert review result to dict for state
        result = {
            "review": review_result,
            "review_passed": review_result.overall_valid,
            "warnings": [f"Step {step.step_index}: {issue}" 
                        for step in review_result.steps 
                        for issue in step.issues],
            "completed_agents": ["reviewer_agent"],
        }
        
        # If we refined actions, include the updated version
        if refinement_requests and current_actions:
            result["actions"] = current_actions
        
        return result
        
    except Exception as e:
        console.print(f"[red]Reviewer Agent error: {e}[/red]")
        import traceback
        traceback.print_exc()
        if logger:
            logger.agent_error("ReviewerAgent", str(e))
        return {
            "errors": [f"Reviewer agent failed: {e}"],
            "completed_agents": ["reviewer_agent"],
            "review_passed": False,
        }
