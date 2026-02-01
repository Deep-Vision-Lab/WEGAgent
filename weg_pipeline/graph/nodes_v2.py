"""
Graph Node Functions for the WEG Pipeline V2.

V2 Pipeline: Each agent focuses on ONE task.
- Action Agent: Extract actions only (no tool/component/hands in quadruples)
- Part Text Agent: Extract parts AND components per action  
- Tool Agent: Extract tools per action
- Hands Agent: Extract hands per action
- Reviewer: Validates each agent's output
- Combiner: Builds WEG quadruples from all agent outputs
"""
from pathlib import Path
from typing import Any

from rich.console import Console

from ..agents import ActionAgent, ToolAgent, HandsAgent, PartTextAgent, PartVisionAgent, ReviewerAgent
from ..models.preweg import PreWEGGuide
from ..utils.io_utils import load_json
from ..utils.logger import get_logger, LogLevel
from .state_v2 import PipelineStateV2

console = Console()

MAX_REFINEMENTS = 2  # Maximum refinement attempts per agent


def load_guide_node_v2(state: PipelineStateV2) -> dict[str, Any]:
    """Load the pre-WEG guide from disk."""
    guide_path = state["guide_path"]
    console.print(f"[bold blue]Loading guide: {guide_path}[/bold blue]")
    
    logger = get_logger()

    try:
        guide_data = load_json(guide_path)
        
        # Extract device type for context
        guide = PreWEGGuide(**guide_data)
        console.print(f"[dim]Device type detected: {guide.device_type}[/dim]")
        
        if logger:
            logger.log(LogLevel.INFO, "Pipeline", f"Loaded guide with {len(guide_data.get('steps', []))} steps")
        
        return {
            "guide": guide_data,
            "completed_agents": ["load_guide"],
            "refinement_counts": {},  # Initialize refinement tracking
        }
    except Exception as e:
        if logger:
            logger.log(LogLevel.ERROR, "Pipeline", f"Failed to load guide: {e}")
        return {
            "errors": [f"Failed to load guide: {e}"],
        }


# =============================================================================
# ACTION AGENT NODE (V2: Actions only)
# =============================================================================

def action_agent_node_v2(state: PipelineStateV2) -> dict[str, Any]:
    """
    V2 Action Agent: Extract ONLY actions.
    
    Does NOT fill in tool/component/hands - those come from other agents.
    """
    console.print("[bold magenta]Running Action Agent (V2: Actions Only)...[/bold magenta]")
    
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

        # V2: Extract actions only
        actions = agent.run_actions_only(guide)
        
        if logger:
            total_actions = sum(len(s.action_quadruples) for s in actions.steps)
            logger.agent_complete(
                "ActionAgent", 
                result_summary=f"Extracted {total_actions} actions from {len(actions.steps)} steps"
            )

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


def review_actions_node_v2(state: PipelineStateV2) -> dict[str, Any]:
    """
    Review Action Agent output and trigger refinement if needed.
    """
    console.print("[bold cyan]Reviewing Action Agent output...[/bold cyan]")
    
    logger = get_logger()
    
    guide_data = state.get("guide")
    actions = state.get("actions")
    
    if not guide_data or not actions:
        return {
            "review_actions": None,
            "completed_agents": ["review_actions"],
        }

    try:
        guide = PreWEGGuide(**guide_data)
        config = state.get("config", {})
        
        reviewer = ReviewerAgent(
            model=config.get("llm_model", "claude-3-5-haiku-20241022"),
            provider=config.get("llm_provider", "anthropic"),
            verbose=config.get("verbose", True),
        )
        
        # Review actions
        review_result = reviewer.review_actions_v2(guide, actions)
        
        console.print(f"[bold]Review: {review_result.summary}[/bold]")
        
        # Check if refinement needed and allowed
        action_refinements = state.get("action_agent_refinements", 0)
        
        if not review_result.overall_valid and action_refinements < MAX_REFINEMENTS:
            console.print(f"[yellow]⚠ Issues found, triggering refinement (attempt {action_refinements + 1}/{MAX_REFINEMENTS})...[/yellow]")
            
            # Create ActionAgent for refinement
            action_agent = ActionAgent(
                model=config.get("vlm_model", "gemini-1.5-flash"),
                provider=config.get("vlm_provider", "google"),
                verbose=config.get("verbose", True),
            )
            
            # Get refinement requests
            refinement_requests = reviewer.get_refinement_requests(review_result)
            
            refined_steps = {}
            for request in refinement_requests:
                if request.agent_name == "ActionAgent":
                    console.print(f"  [cyan]Refining step {request.step_index}...[/cyan]")
                    
                    # V2: Refine actions only
                    refined_result = action_agent.refine_actions_only(
                        guide=guide,
                        step_index=request.step_index,
                        feedback=request.feedback,
                    )
                    
                    refined_steps[request.step_index] = refined_result
            
            # Merge refined steps
            if refined_steps:
                from ..models.intermediate import ActionsPerGuide
                
                new_steps = []
                existing_indices = {step.step_index for step in actions.steps}
                
                for step in actions.steps:
                    if step.step_index in refined_steps:
                        new_steps.append(refined_steps[step.step_index])
                    else:
                        new_steps.append(step)
                
                # Add new steps not in original
                for step_idx, refined_step in refined_steps.items():
                    if step_idx not in existing_indices:
                        new_steps.append(refined_step)
                
                new_steps.sort(key=lambda s: s.step_index)
                
                actions = ActionsPerGuide(
                    guide_id=actions.guide_id,
                    steps=new_steps,
                )
                
                console.print(f"  [green]✓ Refined {len(refined_steps)} steps[/green]")
            
            return {
                "actions": actions,
                "review_actions": review_result,
                "action_agent_refinements": action_refinements + 1,
                "completed_agents": ["review_actions"],
            }
        
        if review_result.overall_valid:
            console.print("[green]✓ Actions review passed[/green]")
        else:
            console.print(f"[yellow]⚠ Actions review: max refinements reached[/yellow]")
        
        return {
            "review_actions": review_result,
            "completed_agents": ["review_actions"],
        }
        
    except Exception as e:
        console.print(f"[red]Review Actions error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return {
            "errors": [f"Review actions failed: {e}"],
            "completed_agents": ["review_actions"],
        }


# =============================================================================
# PART TEXT AGENT NODE (V2: Parts + Components per action)
# =============================================================================

def part_text_agent_node_v2(state: PipelineStateV2) -> dict[str, Any]:
    """
    V2 Part Text Agent: Extract parts AND components per action.
    
    Requires actions to be extracted first.
    """
    console.print("[bold magenta]Running Part Text Agent (V2: Parts + Components)...[/bold magenta]")

    guide_data = state.get("guide")
    actions = state.get("actions")
    
    if not guide_data:
        return {
            "errors": ["No guide loaded for part text agent"],
            "completed_agents": ["part_text_agent"],
        }
    
    if not actions:
        return {
            "errors": ["No actions available for part text agent (run action agent first)"],
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

        # V2: Run with actions context
        parts_text = agent.run_with_actions(guide, actions)

        return {
            "parts_text_v2": parts_text,
            "completed_agents": ["part_text_agent"],
        }
    except Exception as e:
        console.print(f"[red]Part Text Agent error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return {
            "errors": [f"Part text agent failed: {e}"],
            "completed_agents": ["part_text_agent"],
        }


def review_parts_text_node_v2(state: PipelineStateV2) -> dict[str, Any]:
    """
    Review Part Text Agent output.
    
    Critical: Validates that components are PHYSICAL PARTS, not device names.
    """
    console.print("[bold cyan]Reviewing Part Text Agent output...[/bold cyan]")
    
    guide_data = state.get("guide")
    actions = state.get("actions")
    parts_text = state.get("parts_text_v2")
    
    if not guide_data or not actions or not parts_text:
        return {
            "review_parts_text": None,
            "completed_agents": ["review_parts_text"],
        }

    try:
        guide = PreWEGGuide(**guide_data)
        config = state.get("config", {})
        
        reviewer = ReviewerAgent(
            model=config.get("llm_model", "claude-3-5-haiku-20241022"),
            provider=config.get("llm_provider", "anthropic"),
            verbose=config.get("verbose", True),
        )
        
        review_result = reviewer.review_parts_text_v2(guide, actions, parts_text)
        
        console.print(f"[bold]Review: {review_result.summary}[/bold]")
        
        # Check for refinement
        part_refinements = state.get("part_text_agent_refinements", 0)
        
        if not review_result.overall_valid and part_refinements < MAX_REFINEMENTS:
            console.print(f"[yellow]⚠ Issues found, triggering refinement (attempt {part_refinements + 1}/{MAX_REFINEMENTS})...[/yellow]")
            
            part_agent = PartTextAgent(
                model=config.get("llm_model", "gpt-4o-mini"),
                provider=config.get("llm_provider", "openai"),
                verbose=config.get("verbose", True),
            )
            
            # Extract feedback for each step and refine individually
            refined_steps = {}
            for step_result in review_result.steps:
                if not step_result.is_valid and step_result.feedback_for_action_agent:
                    feedback = step_result.feedback_for_action_agent
                    if feedback.startswith("[PartTextAgent]"):
                        feedback = feedback[len("[PartTextAgent]"):].strip()
                    
                    console.print(f"  [cyan]Refining step {step_result.step_index}...[/cyan]")
                    
                    # Refine individual step
                    refined_step = part_agent.refine_with_actions(
                        guide=guide,
                        actions=actions,
                        step_index=step_result.step_index,
                        feedback=feedback,
                    )
                    refined_steps[step_result.step_index] = refined_step
            
            # Merge refined steps back into parts_text
            if refined_steps:
                from ..models.intermediate import PartsTextPerGuideV2
                
                new_steps = []
                for step in parts_text.steps:
                    if step.step_index in refined_steps:
                        new_steps.append(refined_steps[step.step_index])
                    else:
                        new_steps.append(step)
                
                parts_text = PartsTextPerGuideV2(
                    guide_id=parts_text.guide_id,
                    device_type=parts_text.device_type,
                    steps=new_steps,
                )
                
                console.print(f"  [green]✓ Refined {len(refined_steps)} steps[/green]")
            
            return {
                "parts_text_v2": parts_text,
                "review_parts_text": review_result,
                "part_text_agent_refinements": part_refinements + 1,
                "completed_agents": ["review_parts_text"],
            }
        
        if review_result.overall_valid:
            console.print("[green]✓ Parts/Components review passed[/green]")
        else:
            console.print(f"[yellow]⚠ Parts review: max refinements reached[/yellow]")
        
        return {
            "review_parts_text": review_result,
            "completed_agents": ["review_parts_text"],
        }
        
    except Exception as e:
        console.print(f"[red]Review Parts Text error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return {
            "errors": [f"Review parts text failed: {e}"],
            "completed_agents": ["review_parts_text"],
        }


# =============================================================================
# TOOL AGENT NODE (V2: Tools per action)
# =============================================================================

def tool_agent_node_v2(state: PipelineStateV2) -> dict[str, Any]:
    """
    V2 Tool Agent: Extract tools per action.
    
    Requires actions to be extracted first.
    """
    console.print("[bold magenta]Running Tool Agent (V2: Per-Action)...[/bold magenta]")

    guide_data = state.get("guide")
    actions = state.get("actions")
    
    if not guide_data:
        return {
            "errors": ["No guide loaded for tool agent"],
            "completed_agents": ["tool_agent"],
        }
    
    if not actions:
        return {
            "errors": ["No actions available for tool agent (run action agent first)"],
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

        # V2: Run for actions
        tools = agent.run_for_actions(guide, actions)

        return {
            "tools_v2": tools,
            "completed_agents": ["tool_agent"],
        }
    except Exception as e:
        console.print(f"[red]Tool Agent error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return {
            "errors": [f"Tool agent failed: {e}"],
            "completed_agents": ["tool_agent"],
        }


def review_tools_node_v2(state: PipelineStateV2) -> dict[str, Any]:
    """Review Tool Agent output."""
    console.print("[bold cyan]Reviewing Tool Agent output...[/bold cyan]")
    
    guide_data = state.get("guide")
    actions = state.get("actions")
    tools = state.get("tools_v2")
    
    if not guide_data or not actions or not tools:
        return {
            "review_tools": None,
            "completed_agents": ["review_tools"],
        }

    try:
        guide = PreWEGGuide(**guide_data)
        config = state.get("config", {})
        
        reviewer = ReviewerAgent(
            model=config.get("llm_model", "claude-3-5-haiku-20241022"),
            provider=config.get("llm_provider", "anthropic"),
            verbose=config.get("verbose", True),
        )
        
        review_result = reviewer.review_tools_v2(guide, actions, tools)
        
        console.print(f"[bold]Review: {review_result.summary}[/bold]")
        
        # Refinement logic
        tool_refinements = state.get("tool_agent_refinements", 0)
        
        if not review_result.overall_valid and tool_refinements < MAX_REFINEMENTS:
            console.print(f"[yellow]⚠ Issues found, triggering refinement...[/yellow]")
            
            tool_agent = ToolAgent(
                model=config.get("llm_model", "gpt-4o-mini"),
                provider=config.get("llm_provider", "openai"),
                verbose=config.get("verbose", True),
            )
            
            # Refine each step individually
            refined_steps = {}
            for step_result in review_result.steps:
                if not step_result.is_valid and step_result.feedback_for_action_agent:
                    console.print(f"  [cyan]Refining step {step_result.step_index}...[/cyan]")
                    
                    refined_step = tool_agent.refine_for_actions(
                        guide=guide,
                        actions=actions,
                        step_index=step_result.step_index,
                        feedback=step_result.feedback_for_action_agent,
                    )
                    refined_steps[step_result.step_index] = refined_step
            
            # Merge refined steps
            if refined_steps:
                from ..models.intermediate import ToolsPerGuideV2
                
                new_steps = []
                for step in tools.steps:
                    if step.step_index in refined_steps:
                        new_steps.append(refined_steps[step.step_index])
                    else:
                        new_steps.append(step)
                
                tools = ToolsPerGuideV2(
                    guide_id=tools.guide_id,
                    global_toolbox=tools.global_toolbox,
                    steps=new_steps,
                )
                
                console.print(f"  [green]✓ Refined {len(refined_steps)} steps[/green]")
            
            return {
                "tools_v2": tools,
                "review_tools": review_result,
                "tool_agent_refinements": tool_refinements + 1,
                "completed_agents": ["review_tools"],
            }
        
        if review_result.overall_valid:
            console.print("[green]✓ Tools review passed[/green]")
        
        return {
            "review_tools": review_result,
            "completed_agents": ["review_tools"],
        }
        
    except Exception as e:
        console.print(f"[red]Review Tools error: {e}[/red]")
        return {
            "errors": [f"Review tools failed: {e}"],
            "completed_agents": ["review_tools"],
        }


# =============================================================================
# HANDS AGENT NODE (V2: Hands per action)
# =============================================================================

def hands_agent_node_v2(state: PipelineStateV2) -> dict[str, Any]:
    """
    V2 Hands Agent: Estimate hands per action.
    
    Requires actions to be extracted first.
    """
    console.print("[bold magenta]Running Hands Agent (V2: Per-Action)...[/bold magenta]")

    guide_data = state.get("guide")
    actions = state.get("actions")
    
    if not guide_data:
        return {
            "errors": ["No guide loaded for hands agent"],
            "completed_agents": ["hands_agent"],
        }
    
    if not actions:
        return {
            "errors": ["No actions available for hands agent (run action agent first)"],
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

        # V2: Run for actions
        hands = agent.run_for_actions(guide, actions)

        return {
            "hands_v2": hands,
            "completed_agents": ["hands_agent"],
        }
    except Exception as e:
        console.print(f"[red]Hands Agent error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return {
            "errors": [f"Hands agent failed: {e}"],
            "completed_agents": ["hands_agent"],
        }


def review_hands_node_v2(state: PipelineStateV2) -> dict[str, Any]:
    """Review Hands Agent output."""
    console.print("[bold cyan]Reviewing Hands Agent output...[/bold cyan]")
    
    guide_data = state.get("guide")
    actions = state.get("actions")
    hands = state.get("hands_v2")
    
    if not guide_data or not actions or not hands:
        return {
            "review_hands": None,
            "completed_agents": ["review_hands"],
        }

    try:
        guide = PreWEGGuide(**guide_data)
        config = state.get("config", {})
        
        reviewer = ReviewerAgent(
            model=config.get("llm_model", "claude-3-5-haiku-20241022"),
            provider=config.get("llm_provider", "anthropic"),
            verbose=config.get("verbose", True),
        )
        
        review_result = reviewer.review_hands_v2(guide, actions, hands)
        
        console.print(f"[bold]Review: {review_result.summary}[/bold]")
        
        # Refinement logic
        hands_refinements = state.get("hands_agent_refinements", 0)
        
        if not review_result.overall_valid and hands_refinements < MAX_REFINEMENTS:
            console.print(f"[yellow]⚠ Issues found, triggering refinement...[/yellow]")
            
            hands_agent = HandsAgent(
                model=config.get("llm_model", "gpt-4o-mini"),
                provider=config.get("llm_provider", "openai"),
                verbose=config.get("verbose", True),
            )
            
            # Refine each step individually
            refined_steps = {}
            for step_result in review_result.steps:
                if not step_result.is_valid and step_result.feedback_for_action_agent:
                    console.print(f"  [cyan]Refining step {step_result.step_index}...[/cyan]")
                    
                    refined_step = hands_agent.refine_for_actions(
                        guide=guide,
                        actions=actions,
                        step_index=step_result.step_index,
                        feedback=step_result.feedback_for_action_agent,
                    )
                    refined_steps[step_result.step_index] = refined_step
            
            # Merge refined steps
            if refined_steps:
                from ..models.intermediate import HandsPerGuideV2
                
                new_steps = []
                for step in hands.steps:
                    if step.step_index in refined_steps:
                        new_steps.append(refined_steps[step.step_index])
                    else:
                        new_steps.append(step)
                
                hands = HandsPerGuideV2(
                    guide_id=hands.guide_id,
                    steps=new_steps,
                )
                
                console.print(f"  [green]✓ Refined {len(refined_steps)} steps[/green]")
            
            return {
                "hands_v2": hands,
                "review_hands": review_result,
                "hands_agent_refinements": hands_refinements + 1,
                "completed_agents": ["review_hands"],
            }
        
        if review_result.overall_valid:
            console.print("[green]✓ Hands review passed[/green]")
        
        return {
            "review_hands": review_result,
            "completed_agents": ["review_hands"],
        }
        
    except Exception as e:
        console.print(f"[red]Review Hands error: {e}[/red]")
        return {
            "errors": [f"Review hands failed: {e}"],
            "completed_agents": ["review_hands"],
        }


# =============================================================================
# PART VISION AGENT NODE (Same as V1)
# =============================================================================

def part_vision_agent_node_v2(state: PipelineStateV2) -> dict[str, Any]:
    """
    V2 Part Vision Agent: Extract visual bounding boxes.
    
    Uses components AND parts from parts_text_v2 for context.
    """
    console.print("[bold magenta]Running Part Vision Agent (V2)...[/bold magenta]")

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

        # Extract component-part pairs from V2 parts_text
        component_part_pairs_per_step = {}
        components_per_step = {}  # Legacy fallback
        descriptions_per_step = {}
        hints_per_step = {}
        
        parts_text_v2 = state.get("parts_text_v2")
        actions = state.get("actions")
        
        if parts_text_v2 and hasattr(parts_text_v2, "steps"):
            for step_result in parts_text_v2.steps:
                # Get component-part pairs from V2 format
                pairs = []
                components = []
                if step_result.components_per_action:
                    for comp in step_result.components_per_action:
                        if comp.component:
                            # Build (component, part) tuple
                            pairs.append((comp.component, comp.part))
                            components.append(comp.component)
                
                if pairs:
                    # Remove duplicates while preserving order
                    seen = set()
                    unique_pairs = []
                    for p in pairs:
                        if p not in seen:
                            seen.add(p)
                            unique_pairs.append(p)
                    component_part_pairs_per_step[step_result.step_index] = unique_pairs
                
                if components:
                    # Legacy fallback - just components
                    seen = set()
                    unique_components = []
                    for c in components:
                        if c not in seen:
                            seen.add(c)
                            unique_components.append(c)
                    components_per_step[step_result.step_index] = unique_components
        
        # Get hints from actions
        if actions and hasattr(actions, "steps"):
            for step_result in actions.steps:
                if step_result.hints:
                    hints_per_step[step_result.step_index] = step_result.hints
        
        # Get descriptions from guide steps
        for step in guide.steps:
            if step.full_description:
                descriptions_per_step[step.step_index] = step.full_description

        parts_vision = agent.run_all_steps(
            guide, 
            components_per_step=components_per_step,
            component_part_pairs_per_step=component_part_pairs_per_step,
            descriptions_per_step=descriptions_per_step,
            hints_per_step=hints_per_step,
        )

        return {
            "parts_vision": parts_vision,
            "completed_agents": ["part_vision_agent"],
        }
    except Exception as e:
        console.print(f"[red]Part Vision Agent error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return {
            "errors": [f"Part vision agent failed: {e}"],
            "completed_agents": ["part_vision_agent"],
        }


# =============================================================================
# SYNC AND COMBINER NODES
# =============================================================================

def sync_node_v2(state: PipelineStateV2) -> dict[str, Any]:
    """
    Synchronization barrier for V2 pipeline.
    
    Ensures all parallel agents (Tool, Hands after their reviews) complete.
    """
    completed = state.get("completed_agents", [])
    if "sync_v2" in completed:
        console.print("[dim]V2 Sync already completed, skipping...[/dim]")
        return {}
    
    console.print(f"[dim]V2 Sync point reached. Completed agents: {completed}[/dim]")
    
    # Check reviews passed
    all_passed = all([
        state.get("review_actions") is None or state.get("review_actions", {}).overall_valid if state.get("review_actions") else True,
        state.get("review_parts_text") is None or state.get("review_parts_text", {}).overall_valid if state.get("review_parts_text") else True,
        state.get("review_tools") is None or state.get("review_tools", {}).overall_valid if state.get("review_tools") else True,
        state.get("review_hands") is None or state.get("review_hands", {}).overall_valid if state.get("review_hands") else True,
    ])
    
    return {
        "all_reviews_passed": all_passed,
        "completed_agents": ["sync_v2"],
    }


def combiner_node_v2(state: PipelineStateV2) -> dict[str, Any]:
    """
    V2 Combiner: Build WEG quadruples from all agent outputs.
    
    This is where the action quadruples are fully assembled:
    - action + verb from Action Agent
    - component from Part Text Agent
    - tool from Tool Agent
    - hands from Hands Agent
    """
    completed = state.get("completed_agents", [])
    if "combiner_v2" in completed:
        console.print("[yellow]V2 Combiner already ran, skipping...[/yellow]")
        return {}
    
    console.print("[bold green]Running V2 Combiner...[/bold green]")
    
    from ..combiner.weg_combiner import combine_to_weg_v2
    from ..utils.io_utils import save_json, get_output_path

    try:
        weg = combine_to_weg_v2(
            guide=state.get("guide", {}),
            actions=state.get("actions"),
            parts_text=state.get("parts_text_v2"),
            tools=state.get("tools_v2"),
            hands=state.get("hands_v2"),
            parts_vision=state.get("parts_vision"),
        )

        # Save WEG
        guide_path = state.get("guide_path", "")
        if guide_path:
            weg_name = state["suffix"] + state["config"]["llm_model"] + "_" + state["config"]["vlm_model"]
            weg_path = get_output_path(guide_path, f"_{weg_name}_WEG_v2.json")
            save_json(weg_path, weg)
            console.print(f"[bold green]V2 WEG saved to: {weg_path}[/bold green]")
            return {
                "weg": weg,
                "weg_path": str(weg_path),
                "completed_agents": ["combiner_v2"],
            }

        return {
            "weg": weg,
            "completed_agents": ["combiner_v2"],
        }
    except Exception as e:
        console.print(f"[red]V2 Combiner error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return {
            "errors": [f"V2 Combiner failed: {e}"],
            "completed_agents": ["combiner_v2"],
        }
