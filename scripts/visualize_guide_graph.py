#!/usr/bin/env python3
"""
Guide Graph Builder and Visualizer

This script builds and visualizes a state graph from either:
- PreWEG guide JSON (raw crawled data)
- WEG JSON (processed guide with extracted actions, tools, parts)

Usage:
    python scripts/visualize_guide_graph.py <path_to_json> [--output <output_path>] [--format png|svg|pdf]
"""

import json
import argparse
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any
import textwrap
import os

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
except ImportError:
    print("Please install required packages: pip install networkx matplotlib")
    exit(1)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ActionQuadruple:
    """Represents an action with its components."""
    action: str
    tool: Optional[str] = None
    component: str = ""
    hands: int = 1
    precise_action: Optional[str] = None
    full_action: str = ""


@dataclass
class PartEntity:
    """Represents a part/component in the guide."""
    name: str
    bbox: Optional[dict] = None
    confidence: float = 0.0
    image_path: str = ""


@dataclass
class StepNode:
    """Node representing a single step in the guide graph."""
    step_id: int
    task_name: str
    description: str
    
    # Extracted content (WEG)
    actions: list[ActionQuadruple] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    parts: list[PartEntity] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    
    # Raw content (PreWEG)
    images: list[dict] = field(default_factory=list)
    
    # Derived features
    action_verbs: set[str] = field(default_factory=set)
    component_names: set[str] = field(default_factory=set)
    
    # Source type
    is_weg: bool = False


# =============================================================================
# Graph Builder
# =============================================================================

class GuideGraph:
    """
    Graph representation of a repair guide.
    
    Builds a directed graph where:
    - Nodes = Steps
    - Edges = Sequential transitions between steps
    - Node attributes = Actions, tools, parts, etc.
    """
    
    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        self.graph = nx.DiGraph()
        self.metadata: dict[str, Any] = {}
        self.is_weg = False
        
        # Load and build
        self.data = self._load_json()
        self._detect_format()
        self._extract_metadata()
        self._build_graph()
    
    def _load_json(self) -> dict:
        """Load JSON file."""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _detect_format(self):
        """Detect if this is a WEG or PreWEG file."""
        # WEG files have "header" and steps with "action_quadruples"
        if "header" in self.data:
            self.is_weg = True
        elif self.data.get("steps") and "action_quadruples" in self.data["steps"][0]:
            self.is_weg = True
        else:
            self.is_weg = False
    
    def _extract_metadata(self):
        """Extract guide metadata."""
        if self.is_weg:
            header = self.data.get("header", {})
            self.metadata = {
                "title": header.get("title", "Unknown"),
                "guide_id": header.get("guide_id", ""),
                "toolbox": header.get("toolbox", []),
                "source_url": header.get("source_url", ""),
                "format": "WEG",
            }
        else:
            self.metadata = {
                "title": self.data.get("title", "Unknown"),
                "guide_id": self.data.get("guide_id", ""),
                "toolbox": self.data.get("toolbox", []),
                "source_url": self.data.get("source_url", ""),
                "format": "PreWEG",
            }
    
    def _build_graph(self):
        """Build the NetworkX graph from steps."""
        steps = self.data.get("steps", [])
        
        for i, step in enumerate(steps):
            node = self._create_node(step)
            self.graph.add_node(
                node.step_id,
                data=node,
                label=self._create_node_label(node),
            )
            
            # Add sequential edge to next step
            if i > 0:
                prev_step = steps[i - 1]
                prev_id = prev_step.get("step_id") if self.is_weg else prev_step.get("step_index")
                self.graph.add_edge(prev_id, node.step_id, relation="next")
    
    def _create_node(self, step: dict) -> StepNode:
        """Create a StepNode from step data."""
        if self.is_weg:
            return self._create_weg_node(step)
        else:
            return self._create_preweg_node(step)
    
    def _create_weg_node(self, step: dict) -> StepNode:
        """Create node from WEG step data."""
        # Parse action quadruples
        actions = []
        action_verbs = set()
        component_names = set()
        
        for aq in step.get("action_quadruples", []):
            action = ActionQuadruple(
                action=aq.get("action", ""),
                tool=aq.get("tool"),
                component=aq.get("component", ""),
                hands=aq.get("hands", 1),
                precise_action=aq.get("precise_action"),
                full_action=aq.get("full_action", ""),
            )
            actions.append(action)
            action_verbs.add(action.action.lower())
            if action.component:
                component_names.add(action.component.lower())
        
        # Parse parts
        parts = []
        for p in step.get("parts_all", []):
            part = PartEntity(
                name=p.get("name", ""),
                bbox=p.get("bbox"),
                confidence=p.get("confidence", 0.0),
                image_path=p.get("image_path", ""),
            )
            parts.append(part)
            component_names.add(part.name.lower())
        
        # Extract tools from action_quadruples
        tools = []
        for quadruple in step.get("action_quadruples", []):
            if quadruple.get("tool"):
                tools.append(quadruple.get("tool"))
        tools = list(set(tools))  # Remove duplicates
        
        return StepNode(
            step_id=step.get("step_id", 0),
            task_name=step.get("task_name", ""),
            description=step.get("description", ""),
            actions=actions,
            tools=tools,
            parts=parts,
            hints=step.get("hints", []),
            action_verbs=action_verbs,
            component_names=component_names,
            is_weg=True,
        )
    
    def _create_preweg_node(self, step: dict) -> StepNode:
        """Create node from PreWEG step data."""
        description = step.get("full_description", "")
        
        # Clean markdown artifacts (***word***, **word**, *word*)
        description = self._clean_markdown(description)
        
        # Extract basic action verbs from description
        action_verbs = self._extract_verbs_from_text(description)
        
        # Parse images
        images = step.get("images", [])
        
        return StepNode(
            step_id=step.get("step_index", 0),
            task_name="",  # PreWEG doesn't have task names
            description=description,
            images=images,
            action_verbs=action_verbs,
            is_weg=False,
        )
    
    def _clean_markdown(self, text: str) -> str:
        """Remove markdown formatting artifacts from text."""
        # Remove bold/italic markers: ***word***, **word**, *word*
        text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
        # Remove underscores: ___word___, __word__, _word_
        text = re.sub(r'_{1,3}([^_]+)_{1,3}', r'\1', text)
        # Clean up multiple spaces
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    def _extract_verbs_from_text(self, text: str) -> set[str]:
        """Extract common action verbs from text."""
        common_verbs = {
            "remove", "install", "lift", "pull", "push", "insert", "press",
            "unplug", "disconnect", "connect", "unscrew", "screw", "loosen",
            "tighten", "open", "close", "slide", "rotate", "turn", "flip",
            "squeeze", "pinch", "grasp", "hold", "release", "detach", "attach",
            "pry", "cut", "position", "align", "lower", "raise", "swing",
        }
        
        text_lower = text.lower()
        found_verbs = set()
        for verb in common_verbs:
            if verb in text_lower:
                found_verbs.add(verb)
        
        return found_verbs
    
    def _create_node_label(self, node: StepNode) -> str:
        """Create a label for visualization."""
        if node.is_weg and node.task_name:
            return f"Step {node.step_id}\n{node.task_name}"
        else:
            # Truncate description for label
            desc = node.description[:50] + "..." if len(node.description) > 50 else node.description
            desc = desc.replace("\n", " ").replace("- ", "")
            return f"Step {node.step_id}\n{desc}"
    
    def get_summary(self) -> dict:
        """Get a summary of the graph."""
        nodes = list(self.graph.nodes(data=True))
        
        all_actions = set()
        all_tools = set()
        all_components = set()
        
        for _, data in nodes:
            node: StepNode = data["data"]
            all_actions.update(node.action_verbs)
            all_tools.update(node.tools)
            all_components.update(node.component_names)
        
        return {
            "format": self.metadata["format"],
            "title": self.metadata["title"],
            "guide_id": self.metadata["guide_id"],
            "num_steps": len(nodes),
            "num_edges": self.graph.number_of_edges(),
            "unique_actions": sorted(all_actions),
            "unique_tools": sorted(all_tools),
            "unique_components": sorted(list(all_components)[:20]),  # Limit for display
        }
    
    def to_dict(self) -> dict:
        """
        Export the graph to a dictionary format suitable for JSON serialization.
        
        Returns:
            Dictionary with metadata, nodes, and edges.
        """
        # Build nodes list
        nodes_data = []
        for node_id in sorted(self.graph.nodes()):
            node: StepNode = self.graph.nodes[node_id]["data"]
            
            node_dict = {
                "step_id": node.step_id,
                "task_name": node.task_name,
                "description": node.description,
                "is_weg": node.is_weg,
                "actions": [
                    {
                        "action": a.action,
                        "tool": a.tool,
                        "component": a.component,
                        "hands": a.hands,
                        "precise_action": a.precise_action,
                        "full_action": a.full_action,
                    }
                    for a in node.actions
                ],
                "tools": node.tools,
                "parts": [
                    {
                        "name": p.name,
                        "bbox": p.bbox,
                        "confidence": p.confidence,
                        "image_path": p.image_path,
                    }
                    for p in node.parts
                ],
                "hints": node.hints,
                "action_verbs": sorted(node.action_verbs),
                "component_names": sorted(node.component_names),
                "images": node.images,
            }
            nodes_data.append(node_dict)
        
        # Build edges list
        edges_data = []
        for src, dst, edge_data in self.graph.edges(data=True):
            edges_data.append({
                "source": src,
                "target": dst,
                "relation": edge_data.get("relation", "next"),
            })
        
        return {
            "metadata": self.metadata,
            "summary": self.get_summary(),
            "nodes": nodes_data,
            "edges": edges_data,
        }
    
    def export_json(self, output_path: str) -> str:
        """
        Export the graph to a JSON file.
        
        Args:
            output_path: Directory to save the JSON file.
            
        Returns:
            Path to the saved JSON file.
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Generate filename based on format
        filename = f"{self.metadata['format'].lower()}_graph.json"
        full_path = os.path.join(output_path, filename)
        
        # Export to JSON
        graph_dict = self.to_dict()
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(graph_dict, f, indent=2, ensure_ascii=False)
        
        return full_path


# =============================================================================
# Visualization
# =============================================================================

class GraphVisualizer:
    """Visualize a GuideGraph with clean, human-readable layout."""
    
    # Color schemes
    COLORS = {
        "weg_node": "#4CAF50",      # Green for WEG
        "weg_node_light": "#C8E6C9",
        "preweg_node": "#2196F3",   # Blue for PreWEG
        "preweg_node_light": "#BBDEFB",
        "edge": "#616161",          # Gray edges
        "text": "#212121",          # Dark text
        "bg": "#FAFAFA",            # Light background
    }
    
    def __init__(self, guide_graph: GuideGraph, format: str):
        self.guide_graph = guide_graph
        self.graph = guide_graph.graph
        self.format = format
    
    def visualize(
        self,
        output_path: Optional[str] = None,
        figsize: tuple[int, int] = None,
        show_details: bool = True,
    ):
        """
        Create a clean, human-readable visualization of the guide graph.
        """
        num_nodes = len(self.graph.nodes())
        
        # Dynamic figure size based on number of steps
        if figsize is None:
            width = max(14, num_nodes * 2.5)
            height = 8
            figsize = (width, height)
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_facecolor("white")
        
        # Create clean horizontal layout
        pos = self._create_clean_layout(num_nodes)
        
        # Draw connecting lines (edges)
        self._draw_edges(ax, pos)
        
        # Draw step boxes with content
        self._draw_step_boxes(ax, pos)
        
        # Add title and legend
        title = f"{self.guide_graph.metadata['format']} Graph: {self.guide_graph.metadata['title']}"
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        self._add_legend(ax)
        
        # Clean up axes
        ax.set_xlim(-0.5, num_nodes * 2.5 + 0.5)
        ax.set_ylim(-1.5, 2)
        ax.axis("off")
        
        plt.tight_layout()
        
        # Save or show
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            filename = 'weg_graph.png' if self.format == 'weg' else 'preweg_graph.png'
            full_path = os.path.join(output_path, filename)
            plt.savefig(full_path, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"Graph saved to: {full_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _create_clean_layout(self, num_nodes: int) -> dict:
        """Create a clean horizontal layout with good spacing."""
        pos = {}
        spacing = 2.5  # Space between nodes
        
        for i, node_id in enumerate(sorted(self.graph.nodes())):
            x = i * spacing
            y = 0.5  # Center line
            pos[node_id] = (x, y)
        
        return pos
    
    def _draw_edges(self, ax, pos):
        """Draw curved arrows between steps."""
        edges = list(self.graph.edges())
        
        for src, dst in edges:
            x1, y1 = pos[src]
            x2, y2 = pos[dst]
            
            # Draw arrow
            ax.annotate(
                "",
                xy=(x2 - 0.4, y2),
                xytext=(x1 + 0.4, y1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=self.COLORS["edge"],
                    lw=2,
                    connectionstyle="arc3,rad=0",
                ),
            )
    
    def _draw_step_boxes(self, ax, pos):
        """Draw step boxes with all content in a single readable box."""
        is_weg = self.guide_graph.is_weg
        node_color = self.COLORS["weg_node"] if is_weg else self.COLORS["preweg_node"]
        light_color = self.COLORS["weg_node_light"] if is_weg else self.COLORS["preweg_node_light"]
        
        for node_id in sorted(self.graph.nodes()):
            x, y = pos[node_id]
            node_data: StepNode = self.graph.nodes[node_id]["data"]
            
            # === Draw main circle node ===
            circle = plt.Circle(
                (x, y), 0.35,
                facecolor=node_color,
                edgecolor="white",
                linewidth=2,
                zorder=10,
            )
            ax.add_patch(circle)
            
            # Step number in circle
            ax.text(
                x, y, str(node_id),
                ha="center", va="center",
                fontsize=12, fontweight="bold",
                color="white",
                zorder=11,
            )
            
            # === Build content box ===
            content_lines = []
            
            # Task name or description (first line, bold)
            if node_data.task_name:
                title = node_data.task_name
            else:
                # Clean and truncate description
                desc = node_data.description.replace("\n", " ").replace("- ", "")
                title = desc[:45] + "..." if len(desc) > 45 else desc
            content_lines.append(title)
            
            # Actions
            if node_data.action_verbs:
                verbs = ", ".join(sorted(node_data.action_verbs)[:4])
                content_lines.append(f"⚡ {verbs}")
            
            # Tools (WEG only)
            if node_data.tools:
                tools = ", ".join(node_data.tools[:2])
                if len(tools) > 30:
                    tools = tools[:30] + "..."
                content_lines.append(f"🔧 {tools}")
            
            # Parts/Components (WEG only)
            if node_data.parts:
                parts = ", ".join([p.name for p in node_data.parts[:2]])
                if len(parts) > 30:
                    parts = parts[:30] + "..."
                content_lines.append(f"📦 {parts}")
            
            # === Draw content box below node ===
            box_text = "\n".join(content_lines)
            box_y = y - 0.7  # Position below the circle
            
            # Calculate box height based on content
            num_lines = len(content_lines)
            box_height = 0.15 + num_lines * 0.12
            
            # Draw rounded rectangle background
            box_width = 1.8
            rect = FancyBboxPatch(
                (x - box_width/2, box_y - box_height),
                box_width, box_height,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor=light_color,
                edgecolor=node_color,
                linewidth=1.5,
                zorder=5,
            )
            ax.add_patch(rect)
            
            # Draw connecting line from circle to box
            ax.plot(
                [x, x], [y - 0.35, box_y],
                color=node_color, linewidth=1.5, linestyle="-",
                zorder=4,
            )
            
            # Add text content
            self._draw_box_text(ax, x, box_y, content_lines, node_color)
    
    def _draw_box_text(self, ax, x, y, lines, accent_color):
        """Draw formatted text in the content box."""
        line_height = 0.12
        current_y = y - 0.06
        
        for i, line in enumerate(lines):
            # First line is title (bold)
            if i == 0:
                ax.text(
                    x, current_y, line,
                    ha="center", va="top",
                    fontsize=8, fontweight="bold",
                    color=self.COLORS["text"],
                    zorder=15,
                )
            else:
                ax.text(
                    x, current_y, line,
                    ha="center", va="top",
                    fontsize=7,
                    color=self.COLORS["text"],
                    zorder=15,
                )
            current_y -= line_height
    
    def _add_legend(self, ax):
        """Add legend explaining the visualization."""
        legend_elements = [
            mpatches.Patch(
                facecolor=self.COLORS["weg_node"] if self.guide_graph.is_weg else self.COLORS["preweg_node"],
                edgecolor="white",
                label=f"{self.guide_graph.metadata['format']} Step"
            ),
        ]
        
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            framealpha=0.9,
            fontsize=9,
        )
    
    def visualize_detailed(self, output_path: Optional[str] = None):
        """
        Create a detailed multi-panel visualization.
        
        Shows:
        - Main graph
        - Step details table
        - Action/Tool/Component summary
        """
        fig = plt.figure(figsize=(18, 12))
        
        # Create grid
        gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.25, wspace=0.15)
        
        # Main graph (top, spanning both columns)
        ax_graph = fig.add_subplot(gs[0, :])
        self._draw_main_graph_detailed(ax_graph)
        
        # Step details (bottom left)
        ax_steps = fig.add_subplot(gs[1, 0])
        self._draw_step_table(ax_steps)
        
        # Summary stats (bottom right)
        ax_summary = fig.add_subplot(gs[1, 1])
        self._draw_summary(ax_summary)
        
        # Title
        fig.suptitle(
            f"{self.guide_graph.metadata['format']} Analysis: {self.guide_graph.metadata['title']}",
            fontsize=14,
            fontweight="bold",
        )
        
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            filename = 'detailed_weg_graph.png' if self.format == 'weg' else 'detailed_preweg_graph.png'
            full_path = os.path.join(output_path, filename)
            plt.savefig(full_path, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"Detailed graph saved to: {full_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _draw_main_graph_detailed(self, ax):
        """Draw the main graph for detailed view."""
        num_nodes = len(self.graph.nodes())
        pos = {}
        spacing = 1.8
        
        for i, node_id in enumerate(sorted(self.graph.nodes())):
            pos[node_id] = (i * spacing, 0)
        
        node_color = self.COLORS["weg_node"] if self.guide_graph.is_weg else self.COLORS["preweg_node"]
        
        # Draw edges
        for src, dst in self.graph.edges():
            x1, y1 = pos[src]
            x2, y2 = pos[dst]
            ax.annotate(
                "", xy=(x2 - 0.3, y2), xytext=(x1 + 0.3, y1),
                arrowprops=dict(arrowstyle="-|>", color=self.COLORS["edge"], lw=1.5),
            )
        
        # Draw nodes
        for node_id in sorted(self.graph.nodes()):
            x, y = pos[node_id]
            node_data: StepNode = self.graph.nodes[node_id]["data"]
            
            circle = plt.Circle((x, y), 0.25, facecolor=node_color, edgecolor="white", lw=2, zorder=10)
            ax.add_patch(circle)
            
            ax.text(x, y, str(node_id), ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white", zorder=11)
            
            # Task name below
            if node_data.task_name:
                label = node_data.task_name
            else:
                label = node_data.description[:25] + "..."
            label = "\n".join(textwrap.wrap(label, 15))
            ax.text(x, y - 0.45, label, ha="center", va="top", fontsize=7, zorder=11)
        
        ax.set_xlim(-0.5, num_nodes * spacing)
        ax.set_ylim(-1, 0.8)
        ax.set_title("Step Flow", fontsize=11, fontweight="bold")
        ax.axis("off")
    def _draw_step_table(self, ax):
        """Draw a table of step details."""
        ax.axis("off")
        
        # Prepare table data
        headers = ["Step", "Task/Description", "Actions", "Tools"]
        rows = []
        
        for node_id in sorted(self.graph.nodes()):
            node: StepNode = self.graph.nodes[node_id]["data"]
            
            # Task or truncated description
            if node.task_name:
                task = node.task_name
            else:
                task = node.description[:40] + "..." if len(node.description) > 40 else node.description
            task = task.replace("\n", " ")
            
            # Actions
            actions = ", ".join(sorted(node.action_verbs)[:3]) if node.action_verbs else "-"
            
            # Tools
            tools = ", ".join(node.tools[:2]) if node.tools else "-"
            
            rows.append([str(node_id), task[:35], actions[:30], tools[:25]])
        
        if rows:
            table = ax.table(
                cellText=rows,
                colLabels=headers,
                cellLoc="left",
                loc="center",
                colWidths=[0.08, 0.42, 0.25, 0.25],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            
            # Style header
            for i in range(len(headers)):
                table[(0, i)].set_facecolor("#E0E0E0")
                table[(0, i)].set_text_props(weight="bold")
        
        ax.set_title("Step Details", fontsize=11, fontweight="bold")
    
    def _draw_summary(self, ax):
        """Draw summary statistics."""
        ax.axis("off")
        
        summary = self.guide_graph.get_summary()
        
        text_lines = [
            f"Format: {summary['format']}",
            f"Guide ID: {summary['guide_id']}",
            f"Total Steps: {summary['num_steps']}",
            f"",
            f"Unique Actions ({len(summary['unique_actions'])}):",
            f"  {', '.join(summary['unique_actions'][:8])}",
            f"",
            f"Tools ({len(summary['unique_tools'])}):",
            f"  {', '.join(summary['unique_tools'][:5]) or 'None detected'}",
            f"",
            f"Components ({len(summary['unique_components'])}):",
            f"  {', '.join(summary['unique_components'][:6]) or 'None detected'}",
        ]
        
        ax.text(
            0.05, 0.95,
            "\n".join(text_lines),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="#BDBDBD"),
        )
        
        ax.set_title("Summary Statistics", fontsize=12, fontweight="bold")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build and visualize a state graph from guide JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize a PreWEG guide
  python scripts/visualize_guide_graph.py guide
  
  # Visualize a WEG file with custom output
  python scripts/visualize_guide_graph.py weg -o results
  
  # Create detailed visualization
  python scripts/visualize_guide_graph.py guide --detailed -o results
  
  # Export graph as JSON
  python scripts/visualize_guide_graph.py weg --json -o results
  
  # Export JSON and visualize
  python scripts/visualize_guide_graph.py guide --json --detailed -o results
        """
    )
    
    parser.add_argument(
        "type",
        type=str,
        choices=["guide", "weg"],
        help="Type of file to process (guide=PreWEG, weg=WEG)",
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default='results',
        help="Output directory for the visualization and JSON files",
    )
    
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Create a detailed multi-panel visualization",
    )
    
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print graph summary to console",
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Export the graph structure to a JSON file",
    )
    
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization (useful when only exporting JSON)",
    )
    
    args = parser.parse_args()
    
    # Validate input
    guide = 'data/preweg/appliances/dishwashers/whirlpool/whirlpool_dishwasher/guides/157559_remove_and_clean_the_whirlpool_dishwasher_spray_arm.json'
    weg = 'data/preweg/appliances/dishwashers/whirlpool/whirlpool_dishwasher/guides/157559_remove_and_clean_the_whirlpool_dishwasher_spray_arm_WEG.json'
    json_path = Path(guide if args.type == 'guide' else weg)
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        exit(1)
    
    # Build graph
    print(f"Loading: {json_path}")
    guide_graph = GuideGraph(str(json_path))
    
    print(f"Detected format: {guide_graph.metadata['format']}")
    print(f"Title: {guide_graph.metadata['title']}")
    print(f"Steps: {guide_graph.graph.number_of_nodes()}")
    
    # Print summary if requested
    if args.summary:
        print("\n" + "=" * 50)
        print("GRAPH SUMMARY")
        print("=" * 50)
        summary = guide_graph.get_summary()
        for key, value in summary.items():
            if isinstance(value, list):
                print(f"{key}: {', '.join(map(str, value[:10]))}")
            else:
                print(f"{key}: {value}")
    
    # Export JSON if requested
    if args.json:
        json_output = guide_graph.export_json(args.output)
        print(f"\nGraph JSON exported to: {json_output}")
    
    # Visualize (unless --no-viz is set)
    if not args.no_viz:
        visualizer = GraphVisualizer(guide_graph, format=args.type)
        
        if args.detailed:
            visualizer.visualize_detailed(output_path=args.output)
        else:
            visualizer.visualize(output_path=args.output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
