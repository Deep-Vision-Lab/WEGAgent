#!/usr/bin/env python3
"""
Graph Similarity Comparator

Compares PreWEG (source guide) and WEG (extracted) graphs to measure
extraction quality and content alignment.

The comparison validates that WEG correctly extracts information FROM the
original description text, checking:
- Actions: Is the extracted action verb present in the description?
- Components: Is the extracted component mentioned in the description?
- Tools: Is the extracted tool mentioned in the description or toolbox?

Usage:
    python scripts/compare_graphs.py --guide <guide.json> --weg <weg.json> [-o results]
    python scripts/compare_graphs.py  # Uses default test files
"""

import json
import argparse
import re
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any
from collections import Counter

# Import the graph builder from visualize_guide_graph
from visualize_guide_graph import GuideGraph, StepNode


# =============================================================================
# Text Matching Utilities
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Remove markdown artifacts
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}([^_]+)_{1,3}', r'\1', text)
    # Remove bullet points
    text = re.sub(r'^-\s*', '', text, flags=re.MULTILINE)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def text_contains_phrase(text: str, phrase: str) -> bool:
    """
    Check if text contains the phrase (case-insensitive).
    Handles multi-word phrases and partial matches.
    """
    text_lower = normalize_text(text)
    phrase_lower = phrase.lower().strip()
    
    # Direct containment
    if phrase_lower in text_lower:
        return True
    
    # Check word-by-word for multi-word phrases
    phrase_words = phrase_lower.split()
    if len(phrase_words) > 1:
        # Check if all words appear in text
        return all(word in text_lower for word in phrase_words)
    
    # For single words, check with word boundaries (but allow verb forms)
    word = phrase_words[0]
    # Match word or common verb forms (e.g., "run" matches "running", "runs")
    pattern = rf'\b{re.escape(word)}(?:s|ed|ing|e)?\b'
    return bool(re.search(pattern, text_lower))


def fuzzy_component_match(component: str, text: str) -> bool:
    """
    Check if component is mentioned in text with fuzzy matching.
    Handles variations like "lower rack" vs "dishwasher lower rack".
    """
    text_lower = normalize_text(text)
    component_lower = component.lower().strip()
    
    # Direct match
    if component_lower in text_lower:
        return True
    
    # Check if main noun is present (last word is usually the main noun)
    words = component_lower.split()
    if len(words) > 1:
        main_noun = words[-1]  # e.g., "rack" from "lower rack"
        # Check if main noun with any preceding word exists
        if main_noun in text_lower:
            return True
    
    return False


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ActionValidation:
    """Validation result for a single extracted action."""
    action: str
    component: str
    tool: Optional[str]
    full_action: str
    
    # Validation results
    action_in_description: bool = False
    component_in_description: bool = False
    tool_valid: bool = True  # True if no tool or tool is valid
    
    def is_valid(self) -> bool:
        """Overall validation: action and component must be in description."""
        return self.action_in_description and self.component_in_description


@dataclass
class StepExtractionQuality:
    """Extraction quality metrics for a single step."""
    step_id: int
    description: str  # Original PreWEG description (source of truth)
    weg_task_name: str  # Generated (not compared)
    
    # Extraction counts
    num_actions: int = 0
    num_valid_actions: int = 0
    num_components: int = 0
    num_valid_components: int = 0
    num_tools: int = 0
    num_valid_tools: int = 0
    num_parts: int = 0
    num_hints: int = 0
    
    # Detailed validation
    action_validations: list = field(default_factory=list)
    extracted_components: list = field(default_factory=list)
    valid_components: list = field(default_factory=list)
    invalid_components: list = field(default_factory=list)
    extracted_tools: list = field(default_factory=list)
    valid_tools: list = field(default_factory=list)
    invalid_tools: list = field(default_factory=list)
    
    # Quality scores (0-1)
    action_precision: float = 1.0  # % of extracted actions found in description
    component_precision: float = 1.0  # % of extracted components found in description
    tool_precision: float = 1.0  # % of extracted tools that are valid
    
    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "description": self.description[:150] + "..." if len(self.description) > 150 else self.description,
            "weg_task_name": self.weg_task_name,
            "quality_scores": {
                "action_precision": round(self.action_precision, 3),
                "component_precision": round(self.component_precision, 3),
                "tool_precision": round(self.tool_precision, 3),
            },
            "extraction_counts": {
                "actions": f"{self.num_valid_actions}/{self.num_actions}",
                "components": f"{self.num_valid_components}/{self.num_components}",
                "tools": f"{self.num_valid_tools}/{self.num_tools}",
                "parts_detected": self.num_parts,
                "hints_generated": self.num_hints,
            },
            "action_details": [
                {
                    "action": av.action,
                    "component": av.component,
                    "tool": av.tool,
                    "action_valid": av.action_in_description,
                    "component_valid": av.component_in_description,
                    "overall_valid": av.is_valid(),
                }
                for av in self.action_validations
            ],
            "component_analysis": {
                "valid": self.valid_components,
                "invalid": self.invalid_components,
            },
            "tool_analysis": {
                "valid": self.valid_tools,
                "invalid": self.invalid_tools,
            },
        }


@dataclass
class ExtractionQualityReport:
    """Complete extraction quality report."""
    preweg_path: str
    weg_path: str
    guide_title: str
    guide_id: str
    global_toolbox: list = field(default_factory=list)
    
    # Structural
    preweg_step_count: int = 0
    weg_step_count: int = 0
    step_count_match: bool = True
    
    # Per-step quality
    step_qualities: list[StepExtractionQuality] = field(default_factory=list)
    
    # Aggregated scores
    mean_action_precision: float = 0.0
    mean_component_precision: float = 0.0
    mean_tool_precision: float = 0.0
    
    # Overall extraction statistics
    total_actions: int = 0
    valid_actions: int = 0
    total_components: int = 0
    valid_components: int = 0
    total_tools_used: int = 0
    valid_tools: int = 0
    total_parts_detected: int = 0
    total_hints_generated: int = 0
    
    # Completeness metrics
    steps_with_actions: int = 0
    steps_with_parts: int = 0
    steps_with_hints: int = 0
    
    # Overall score
    overall_quality_score: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "metadata": {
                "preweg_path": self.preweg_path,
                "weg_path": self.weg_path,
                "guide_title": self.guide_title,
                "guide_id": self.guide_id,
                "global_toolbox": self.global_toolbox,
            },
            "structural": {
                "preweg_step_count": self.preweg_step_count,
                "weg_step_count": self.weg_step_count,
                "step_count_match": self.step_count_match,
            },
            "aggregated_precision": {
                "action_precision": round(self.mean_action_precision, 3),
                "component_precision": round(self.mean_component_precision, 3),
                "tool_precision": round(self.mean_tool_precision, 3),
            },
            "extraction_totals": {
                "actions": f"{self.valid_actions}/{self.total_actions} valid",
                "components": f"{self.valid_components}/{self.total_components} valid",
                "tools": f"{self.valid_tools}/{self.total_tools_used} valid",
                "parts_detected": self.total_parts_detected,
                "hints_generated": self.total_hints_generated,
            },
            "completeness": {
                "steps_with_actions": f"{self.steps_with_actions}/{self.weg_step_count}",
                "steps_with_parts": f"{self.steps_with_parts}/{self.weg_step_count}",
                "steps_with_hints": f"{self.steps_with_hints}/{self.weg_step_count}",
            },
            "overall_quality_score": round(self.overall_quality_score, 3),
            "per_step_analysis": [sq.to_dict() for sq in self.step_qualities],
        }


class GraphComparator:
    """
    Compare a PreWEG graph with its corresponding WEG graph.
    
    Validates that WEG correctly extracts information FROM the original text:
    - Actions should be verbs present in the description
    - Components should be nouns/phrases mentioned in the description
    - Tools should be mentioned in description or global toolbox
    """
    
    # Weights for overall score
    WEIGHTS = {
        "structural": 0.10,
        "action_precision": 0.35,
        "component_precision": 0.30,
        "tool_precision": 0.10,
        "completeness": 0.15,
    }
    
    def __init__(self, preweg_graph: GuideGraph, weg_graph: GuideGraph):
        self.preweg = preweg_graph
        self.weg = weg_graph
        
        # Get global toolbox for tool validation
        self.global_toolbox = set(
            t.lower() for t in self.preweg.metadata.get("toolbox", [])
        )
        
        # Validate graphs
        if self.preweg.is_weg:
            raise ValueError("First graph should be PreWEG (source guide)")
        if not self.weg.is_weg:
            raise ValueError("Second graph should be WEG (extracted)")
    
    def compare(self) -> ExtractionQualityReport:
        """
        Perform extraction quality analysis.
        """
        report = ExtractionQualityReport(
            preweg_path=str(self.preweg.json_path),
            weg_path=str(self.weg.json_path),
            guide_title=self.preweg.metadata.get("title", "Unknown"),
            guide_id=str(self.preweg.metadata.get("guide_id", "")),
            global_toolbox=list(self.global_toolbox),
        )
        
        # Structural comparison
        report.preweg_step_count = self.preweg.graph.number_of_nodes()
        report.weg_step_count = self.weg.graph.number_of_nodes()
        report.step_count_match = report.preweg_step_count == report.weg_step_count
        
        # Per-step quality analysis
        step_qualities = self._analyze_steps()
        report.step_qualities = step_qualities
        
        # Aggregate metrics
        if step_qualities:
            # Precision scores (only count steps with extractions)
            action_precisions = [sq.action_precision for sq in step_qualities if sq.num_actions > 0]
            component_precisions = [sq.component_precision for sq in step_qualities if sq.num_components > 0]
            tool_precisions = [sq.tool_precision for sq in step_qualities if sq.num_tools > 0]
            
            report.mean_action_precision = sum(action_precisions) / len(action_precisions) if action_precisions else 1.0
            report.mean_component_precision = sum(component_precisions) / len(component_precisions) if component_precisions else 1.0
            report.mean_tool_precision = sum(tool_precisions) / len(tool_precisions) if tool_precisions else 1.0
            
            # Totals
            report.total_actions = sum(sq.num_actions for sq in step_qualities)
            report.valid_actions = sum(sq.num_valid_actions for sq in step_qualities)
            report.total_components = sum(sq.num_components for sq in step_qualities)
            report.valid_components = sum(sq.num_valid_components for sq in step_qualities)
            report.total_tools_used = sum(sq.num_tools for sq in step_qualities)
            report.valid_tools = sum(sq.num_valid_tools for sq in step_qualities)
            report.total_parts_detected = sum(sq.num_parts for sq in step_qualities)
            report.total_hints_generated = sum(sq.num_hints for sq in step_qualities)
            
            # Completeness
            report.steps_with_actions = sum(1 for sq in step_qualities if sq.num_actions > 0)
            report.steps_with_parts = sum(1 for sq in step_qualities if sq.num_parts > 0)
            report.steps_with_hints = sum(1 for sq in step_qualities if sq.num_hints > 0)
        
        # Overall score
        report.overall_quality_score = self._calculate_overall_score(report)
        
        return report
    
    def _analyze_steps(self) -> list[StepExtractionQuality]:
        """Analyze extraction quality for each step."""
        qualities = []
        
        # Get step data from both graphs
        preweg_steps = {
            node_id: self.preweg.graph.nodes[node_id]["data"]
            for node_id in sorted(self.preweg.graph.nodes())
        }
        weg_steps = {
            node_id: self.weg.graph.nodes[node_id]["data"]
            for node_id in sorted(self.weg.graph.nodes())
        }
        
        # Analyze aligned steps
        all_step_ids = sorted(set(preweg_steps.keys()) | set(weg_steps.keys()))
        
        for step_id in all_step_ids:
            preweg_node = preweg_steps.get(step_id)
            weg_node = weg_steps.get(step_id)
            
            if preweg_node and weg_node:
                quality = self._analyze_single_step(step_id, preweg_node, weg_node)
                qualities.append(quality)
            elif weg_node:
                # Step only in WEG (shouldn't happen normally)
                quality = StepExtractionQuality(
                    step_id=step_id,
                    description="[NO SOURCE - step not in PreWEG]",
                    weg_task_name=weg_node.task_name,
                )
                qualities.append(quality)
        
        return qualities
    
    def _analyze_single_step(
        self, step_id: int, preweg_node: StepNode, weg_node: StepNode
    ) -> StepExtractionQuality:
        """Analyze extraction quality for a single step."""
        
        # Source description (ground truth)
        description = preweg_node.description
        
        quality = StepExtractionQuality(
            step_id=step_id,
            description=description,
            weg_task_name=weg_node.task_name,
        )
        
        # Analyze action quadruples
        action_validations = []
        for action_obj in weg_node.actions:
            av = ActionValidation(
                action=action_obj.action,
                component=action_obj.component,
                tool=action_obj.tool,
                full_action=action_obj.full_action,
            )
            
            # Validate action is in description
            av.action_in_description = text_contains_phrase(description, action_obj.action)
            
            # Validate component is in description
            if action_obj.component:
                av.component_in_description = fuzzy_component_match(action_obj.component, description)
            else:
                av.component_in_description = True  # No component to validate
            
            # Validate tool (should be in toolbox or description)
            if action_obj.tool:
                tool_lower = action_obj.tool.lower()
                av.tool_valid = (
                    tool_lower in self.global_toolbox or
                    any(tool_lower in t for t in self.global_toolbox) or
                    text_contains_phrase(description, action_obj.tool)
                )
            
            action_validations.append(av)
        
        quality.action_validations = action_validations
        quality.num_actions = len(action_validations)
        quality.num_valid_actions = sum(1 for av in action_validations if av.action_in_description)
        
        # Extract and validate components from action quadruples
        components = [av.component for av in action_validations if av.component]
        valid_comps = [c for c in components if fuzzy_component_match(c, description)]
        invalid_comps = [c for c in components if not fuzzy_component_match(c, description)]
        
        quality.extracted_components = components
        quality.valid_components = valid_comps
        quality.invalid_components = invalid_comps
        quality.num_components = len(components)
        quality.num_valid_components = len(valid_comps)
        
        # Validate tools
        tools = weg_node.tools
        valid_tools = []
        invalid_tools = []
        for tool in tools:
            tool_lower = tool.lower()
            if (tool_lower in self.global_toolbox or
                any(tool_lower in t or t in tool_lower for t in self.global_toolbox) or
                text_contains_phrase(description, tool)):
                valid_tools.append(tool)
            else:
                invalid_tools.append(tool)
        
        quality.extracted_tools = tools
        quality.valid_tools = valid_tools
        quality.invalid_tools = invalid_tools
        quality.num_tools = len(tools)
        quality.num_valid_tools = len(valid_tools)
        
        # Count parts and hints
        quality.num_parts = len(weg_node.parts)
        quality.num_hints = len(weg_node.hints)
        
        # Calculate precision scores
        quality.action_precision = quality.num_valid_actions / quality.num_actions if quality.num_actions > 0 else 1.0
        quality.component_precision = quality.num_valid_components / quality.num_components if quality.num_components > 0 else 1.0
        quality.tool_precision = quality.num_valid_tools / quality.num_tools if quality.num_tools > 0 else 1.0
        
        return quality
    
    def _calculate_overall_score(self, report: ExtractionQualityReport) -> float:
        """Calculate weighted overall quality score."""
        
        # Structural score
        structural_score = 1.0 if report.step_count_match else 0.5
        
        # Precision scores
        action_score = report.mean_action_precision
        component_score = report.mean_component_precision
        tool_score = report.mean_tool_precision
        
        # Completeness score
        completeness_indicators = []
        if report.weg_step_count > 0:
            completeness_indicators.append(report.steps_with_actions / report.weg_step_count)
            completeness_indicators.append(report.steps_with_parts / report.weg_step_count)
            completeness_indicators.append(min(report.steps_with_hints / report.weg_step_count, 1.0))
        completeness_score = sum(completeness_indicators) / len(completeness_indicators) if completeness_indicators else 0.0
        
        # Weighted combination
        overall = (
            self.WEIGHTS["structural"] * structural_score +
            self.WEIGHTS["action_precision"] * action_score +
            self.WEIGHTS["component_precision"] * component_score +
            self.WEIGHTS["tool_precision"] * tool_score +
            self.WEIGHTS["completeness"] * completeness_score
        )
        
        return overall


# =============================================================================
# Report Generation
# =============================================================================

def print_report(report: ExtractionQualityReport):
    """Print a human-readable report to console."""
    print("\n" + "=" * 70)
    print("WEG EXTRACTION QUALITY REPORT")
    print("=" * 70)
    
    print(f"\nGuide: {report.guide_title}")
    print(f"ID: {report.guide_id}")
    if report.global_toolbox:
        print(f"Toolbox: {', '.join(report.global_toolbox)}")
    
    print("\n" + "-" * 40)
    print("STRUCTURAL ALIGNMENT")
    print("-" * 40)
    print(f"PreWEG Steps: {report.preweg_step_count}")
    print(f"WEG Steps:    {report.weg_step_count}")
    print(f"Match:        {'✅ Yes' if report.step_count_match else '❌ No'}")
    
    print("\n" + "-" * 40)
    print("EXTRACTION PRECISION (Higher = Better)")
    print("-" * 40)
    print(f"Action Precision:    {report.mean_action_precision:.1%}  ({report.valid_actions}/{report.total_actions} valid)")
    print(f"Component Precision: {report.mean_component_precision:.1%}  ({report.valid_components}/{report.total_components} valid)")
    print(f"Tool Precision:      {report.mean_tool_precision:.1%}  ({report.valid_tools}/{report.total_tools_used} valid)")
    
    print("\n" + "-" * 40)
    print("EXTRACTION COMPLETENESS")
    print("-" * 40)
    print(f"Steps with Actions: {report.steps_with_actions}/{report.weg_step_count}")
    print(f"Steps with Parts:   {report.steps_with_parts}/{report.weg_step_count}")
    print(f"Steps with Hints:   {report.steps_with_hints}/{report.weg_step_count}")
    print(f"Total Parts Detected: {report.total_parts_detected}")
    print(f"Total Hints Generated: {report.total_hints_generated}")
    
    print("\n" + "-" * 40)
    print("PER-STEP ANALYSIS")
    print("-" * 40)
    
    for sq in report.step_qualities:
        # Overall status
        avg_precision = (sq.action_precision + sq.component_precision) / 2
        status = "✅" if avg_precision >= 0.8 else "⚠️" if avg_precision >= 0.5 else "❌"
        
        print(f"\nStep {sq.step_id}: {status} (Action: {sq.action_precision:.0%}, Component: {sq.component_precision:.0%})")
        print(f"  📝 Description: {sq.description[:80]}...")
        print(f"  🏷️  Task Name: {sq.weg_task_name}")
        
        # Action details
        for av in sq.action_validations:
            action_status = "✓" if av.action_in_description else "✗"
            comp_status = "✓" if av.component_in_description else "✗"
            print(f"  ⚡ Action: [{action_status}] \"{av.action}\" → Component: [{comp_status}] \"{av.component}\"")
        
        # Invalid extractions
        if sq.invalid_components:
            print(f"  ⚠️  Invalid components (not in description): {sq.invalid_components}")
        if sq.invalid_tools:
            print(f"  ⚠️  Invalid tools (not in toolbox/description): {sq.invalid_tools}")
        
        # Parts and hints
        print(f"  📦 Parts: {sq.num_parts} | 💡 Hints: {sq.num_hints}")
    
    print("\n" + "=" * 70)
    print(f"OVERALL QUALITY SCORE: {report.overall_quality_score:.1%}")
    print("=" * 70)
    
    # Interpretation
    if report.overall_quality_score >= 0.9:
        interpretation = "🌟 Excellent - WEG extraction is highly accurate"
    elif report.overall_quality_score >= 0.75:
        interpretation = "✅ Good - WEG extraction is reliable"
    elif report.overall_quality_score >= 0.6:
        interpretation = "⚠️ Moderate - Some extraction issues need attention"
    else:
        interpretation = "❌ Poor - Significant extraction errors detected"
    
    print(f"\n{interpretation}\n")


def save_report(report: ExtractionQualityReport, output_path: str) -> str:
    """Save report to JSON file."""
    os.makedirs(output_path, exist_ok=True)
    
    filename = f"extraction_quality_{report.guide_id}.json"
    full_path = os.path.join(output_path, filename)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    
    return full_path


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare PreWEG and WEG graphs for similarity analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare using default test files
  python scripts/compare_graphs.py
  
  # Compare specific files
  python scripts/compare_graphs.py --guide path/to/guide.json --weg path/to/guide_WEG.json
  
  # Save report to JSON
  python scripts/compare_graphs.py -o results --json
        """
    )
    
    parser.add_argument(
        "--guide",
        type=str,
        default='data/preweg/appliances/dishwashers/whirlpool/whirlpool_dishwasher/guides/157559_remove_and_clean_the_whirlpool_dishwasher_spray_arm.json',
        help="Path to PreWEG guide JSON file",
    )
    
    parser.add_argument(
        "--weg",
        type=str,
        default='data/preweg/appliances/dishwashers/whirlpool/whirlpool_dishwasher/guides/157559_remove_and_clean_the_whirlpool_dishwasher_spray_arm_WEG.json',
        help="Path to WEG JSON file",
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default='results',
        help="Output directory for reports",
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Save detailed report to JSON file",
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print overall score (no detailed output)",
    )
    
    args = parser.parse_args()

    
    # Validate paths
    guide_path = Path(args.guide)
    weg_path = Path(args.weg)
    
    if not guide_path.exists():
        print(f"Error: Guide file not found: {guide_path}")
        exit(1)
    if not weg_path.exists():
        print(f"Error: WEG file not found: {weg_path}")
        exit(1)
    
    # Load graphs
    print(f"Loading PreWEG: {guide_path}")
    preweg_graph = GuideGraph(str(guide_path))
    
    print(f"Loading WEG: {weg_path}")
    weg_graph = GuideGraph(str(weg_path))
    
    # Compare
    print("\nComparing graphs...")
    comparator = GraphComparator(preweg_graph, weg_graph)
    report = comparator.compare()
    
    # Output
    if args.quiet:
        print(f"\nOverall Score: {report.overall_score:.1%}")
    else:
        print_report(report)
    
    # Save JSON if requested
    if args.json:
        json_path = save_report(report, args.output)
        print(f"\nReport saved to: {json_path}")
    
    print("Done!")


if __name__ == "__main__":
    main()
