# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================
from typing import List, Dict, Any
from langgraph.graph import StateGraph, END
import re
from log import logger
import traceback
from models import SectionData
from typing import Dict, List, Optional, Any, Union, TypeVar, Callable, Tuple, cast
import time
import os
from datetime import datetime


# Type variables for better type hinting
T = TypeVar("T")
StateT = TypeVar("StateT", bound="State")


def safe_invoke(
    chain_func: Callable,
    inputs: Dict[str, Any],
    default_value: T,
    error_msg: str,
    max_retries: int=3,
) -> T:
    """
    Safely invoke a chain with retry logic and error handling

    Args:
        chain_func: The chain function to invoke
        inputs: The inputs to the chain
        default_value: The default value to return if the chain fails
        error_msg: The error message to log if the chain fails
        max_retries: Maximum number of retry attempts

    Returns:
        The result of the chain or the default value
    """
    for attempt in range(max_retries):
        try:
            return chain_func.invoke(inputs)
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt+1} failed: {str(e)}. Retrying...")
                time.sleep(1)  # Add a small delay between retries
            else:
                logger.error(f"{error_msg}: {str(e)}")
                logger.debug(f"Exception details: {traceback.format_exc()}")
                return default_value


def visualize_graph(graph: StateGraph):
    dot = Digraph(format="png")
    dot.node("START", shape="ellipse", color="green")
    dot.node("END", shape="ellipse", color="red")

    for node_name in graph.nodes:
        dot.node(node_name, shape="box", style="rounded")

    for edge in graph.edges:
        dot.edge(edge.source, edge.target)

    return dot


def prepare_sections_data(
    outline: Dict[str, Any], sections_content: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Format section data from outline and sections_content for evaluation."""
    sections_data = []
    conclusion_sections = []

    # Sort the sections by their index if available
    logger.info(f"prepare_sections_data outline: {outline}")
    # logger.info(f"prepare_sections_data sections_content: {sections_content}")
    if "sections" in outline:
        outline = outline["sections"]

    sorted_sections = sorted(
        outline.items(),
        key=lambda x: x[1].get("section_index", float("inf")),
    )
    for section_id, section_data in sorted_sections:
        is_conclusion = section_data["is_conclusion"]
        if is_conclusion and "conclustion" not in section_data["section_title"].lower():
            logger.info(f"Skip conclustion section: {section_data}")
            conclusion_sections.append(section_data)
            continue
        section_name = section_id
        key_points = section_data.get("key_points", [])

        # Extract summaries from sections_content
        content_items = sections_content.get(section_name, [])
        section_summaries_text = "\n\n".join(
            f"[{item['section_point']}]: {item['section_summary']}"
            for item in content_items
            if "section_point" in item and "section_summary" in item
        )

        sections_data.append(
            {
                "name": section_name,
                "index": section_data.get("index", 0),
                "key_points": key_points,
                "summary": section_summaries_text or "No summary available",
            }
        )

    return sections_data,conclusion_sections


def aggregate_references(
    text: str, snippet_references: List[Dict[str, Any]], local_ref_map: Dict[Any, int]
) -> str:
    """
    Replaces citation markers in the text based on a mapping to final indices.
    Also handles potential aggregation of consecutive identical citations (optional).

    Args:
        text: The original text containing citation markers (e.g., "[1]", "[URL]").
              The exact format of markers depends on how they were initially generated.
              Assuming markers correspond to the order in snippet_references or use URLs.
        snippet_references: The list of reference dictionaries as they appeared in the original snippet.
        local_ref_map: A dictionary mapping original reference identifiers (e.g., URL or temp ID)
                       to their final index number (1-based).

    Returns:
        The text with citation markers replaced by their final index numbers (e.g., "[5]").
    """
    logger.debug(
        f"aggregate_references called. Text length: {len(text)}, Snippet refs: {len(snippet_references)}, Map size: {len(local_ref_map)}"
    )
    # logger.debug(f"Local ref map: {local_ref_map}") # Can be verbose

    # This implementation assumes citation markers in the original text are simple placeholders
    # that correspond *positionally* to the snippet_references list, OR that they contain URLs.
    # A more robust implementation might need a specific marker format like "[REF:URL]" or "[REF:INDEX]".

    # Let's assume a simple positional replacement for now, using the index 'i'
    # and the original identifier derived from snippet_references[i].

    # We need a way to reliably find the *original* markers in the text.
    # If the original markers were just "[1]", "[2]" corresponding to snippet_references order:
    current_text = text
    try:
        # Iterate through the original references and their intended final indices
        for i, ref in enumerate(snippet_references):
            ref_url = ref.get("url")
            original_identifier = (
                ref_url if ref_url else f"__temp_ref_{i}__"
            )  # Reconstruct potential temp ID used in map key
            original_marker_pattern = (
                rf"\[{i+1}\]"  # Assuming original markers were [1], [2], ...
            )

            if original_identifier in local_ref_map:
                final_index = local_ref_map[original_identifier]
                replacement_marker = f"[{final_index}]"
                # Replace the *original* marker pattern with the *final* marker
                # Use re.sub for safety, replacing only the specific original index
                current_text = re.sub(
                    original_marker_pattern, replacement_marker, current_text
                )
            else:
                # This case should ideally not happen if the map was built correctly
                logger.warning(
                    f"Original identifier '{original_identifier}' not found in local_ref_map for text snippet."
                )
                # Keep the original marker? Or replace with placeholder? Keeping original for now.
                pass  # current_text remains unchanged for this marker

        # --- Optional: Aggregate consecutive identical *final* markers ---
        # Example: "Text [5][5][5] more text [6]" -> "Text [5] more text [6]"
        # This regex finds consecutive identical bracketed numbers
        aggregation_pattern = r"(\[\d+\])(?:\s*\1)+"

        def replace_consecutive(match):
            return match.group(1)  # Return only the first occurrence

        final_text = re.sub(aggregation_pattern, replace_consecutive, current_text)

        return final_text

    except Exception as e:
        logger.error(f"Error during reference aggregation/replacement: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return text  # Return original text on error


def format_sections_for_global_reflection(
    sections: Dict[str, SectionData],
) -> Dict[str, List[Dict[str, Any]]]:
    """Format sections data for global reflection input"""
    formatted_sections = {}

    for section_name, section_data in sections.items():
        section_list = []

        reflection_data = section_data.reflection_results
        if not reflection_data and section_data.content:
            reflection_data = section_data.content

        for key_point, content in reflection_data.items():
            # Skip empty content
            if not content:
                continue

            section_list.append(
                {
                    "section_index": content.get("section_index", 0),
                    "parent_section": content.get("parent_section", ""),
                    "search_query": content.get("search_query", ""),
                    "section_point": content.get("section_key_point", ""),
                    "section_text": content.get("section_text", ""),
                    "section_summary": content.get("section_summary", ""),
                    "main_figure_data": content.get("main_figure_data", ""),
                    "main_figure_caption": content.get("main_figure_caption", ""),
                    "reportIndexList": content.get("reportIndexList", []),
                }
            )
        if section_list:
            formatted_sections[section_name] = section_list

    return formatted_sections


def generate_mind_map_from_outline(processed_data, abstract_conclusion, save_to_file=True, output_dir="./output"):
    from graphviz import Digraph
    """
    Generate mind map representations (Mermaid and Graphviz) from the processed outline structure.
    Optionally save Graphviz visualization as PNG file.

    Args:
        processed_data: The processed data containing outline and sections
        abstract_conclusion: Abstract and conclusion data
        save_to_file: Whether to save the mind map as PNG file
        output_dir: Directory to save the output files

    Returns:
        dict: Contains both 'mermaid' and 'graphviz' representations of the mind map,
              and optionally 'png_path' if saved to file
    """
    try:
        paper_title = processed_data.get("paper_title", "Research Paper")
        sections_content = processed_data.get("sections_content", [])

        # Generate Mermaid mind map
        mermaid_content = generate_mermaid_mind_map(paper_title, sections_content, abstract_conclusion)
        # Generate Graphviz mind map
        graphviz_content = generate_graphviz_mind_map(paper_title, sections_content, abstract_conclusion)

        result = {
            "mermaid": mermaid_content,
            "graphviz": graphviz_content
        }

        # Save PNG file if requested
        if save_to_file:
            try:
                png_path = save_mind_map_as_png(paper_title, sections_content, abstract_conclusion, output_dir)
                result["png_path"] = png_path
                logger.info(f"Mind map saved as PNG: {png_path}")
            except Exception as e:
                logger.error(f"Failed to save mind map as PNG: {e}")
                result["png_path"] = None

        return result

    except Exception as e:
        logger.error(f"Error in generate_mind_map_from_outline: {e}")
        return {
            "mermaid": f"# Error\n\nCould not generate mind map: {str(e)}",
            "graphviz": f"digraph G {{ error [label=\"Error: {str(e)}\"] }}",
            "png_path": None
        }

def save_mind_map_as_png(paper_title, sections_content, abstract_conclusion, output_dir="./output"):
    """
    Generate and save mind map as PNG file using Graphviz.

    Args:
        paper_title: Title of the paper
        sections_content: Processed sections content
        abstract_conclusion: Abstract and conclusion data
        output_dir: Directory to save the output file

    Returns:
        str: Path to the saved PNG file
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Create Graphviz Digraph object with better layout settings
        dot = Digraph(comment='Paper Mind Map')

        # Use LR (Left to Right) layout for better width/height balance
        dot.attr(rankdir='LR')

        # Set graph attributes for better layout
        dot.attr(size='12,8!')  # Constrain size to 12x8 inches
        dot.attr(ratio='auto')  # Auto ratio for better proportions
        dot.attr(nodesep='0.5')  # Space between nodes at same rank
        dot.attr(ranksep='1.0')  # Space between ranks
        dot.attr(splines='ortho')  # Use orthogonal edges for cleaner look

        # Node attributes
        dot.attr('node', shape='box', style='rounded,filled', fontname='Arial',
                fontsize='10', width='2', height='0.5')

        # Edge attributes
        dot.attr('edge', color='gray', arrowsize='0.8', penwidth='1.5')

        # Root node
        root_id = "root"
        dot.node(root_id, sanitize_graphviz_text(paper_title),
                shape='ellipse', fillcolor='lightblue',
                fontsize='12', fontweight='bold', width='3', height='1')

        node_counter = 1

        # Add abstract (title only)
        # abstract_text = abstract_conclusion.get("Abstract", "")
        # if abstract_text:
        #     abstract_id = f"node{node_counter}"
        #     dot.node(abstract_id, "Abstract", fillcolor='lightyellow',
        #             fontweight='bold')
        #     dot.edge(root_id, abstract_id)
        #     node_counter += 1

        # Sort sections by index
        sorted_sections = sorted(sections_content, key=lambda x: x.get("section_index", 0))

        # Group sections into clusters for better layout
        section_clusters = []
        sections_per_cluster = max(1, len(sorted_sections) // 3)  # Divide into roughly 3 columns

        for i in range(0, len(sorted_sections), sections_per_cluster):
            cluster_sections = sorted_sections[i:i + sections_per_cluster]
            section_clusters.append(cluster_sections)

        # Add sections with clustering
        cluster_counter = 0
        for cluster in section_clusters:
            if len(cluster) > 1:
                # Create a subgraph for this cluster
                with dot.subgraph(name=f'cluster_{cluster_counter}') as cluster_graph:
                    cluster_graph.attr(style='invis')  # Invisible cluster border

                    for section in cluster:
                        section_id = f"section{section.get('section_index', node_counter)}"
                        section_title = section.get("section_title_refine", section.get("section_title", "Untitled"))

                        cluster_graph.node(section_id, sanitize_graphviz_text(section_title),
                                         fillcolor='lightgreen', fontweight='bold')
                        dot.edge(root_id, section_id)

                        # Add subsections more compactly
                        structure_parts = section.get("structure_parts", [])
                        subsections = [part for part in structure_parts
                                     if part.get("type") == "subsection" and part.get("level", 0) <= 3]

                        # Limit subsections to avoid overcrowding
                        subsections = subsections[:3]  # Show only first 3 subsections

                        for i, subsection in enumerate(subsections):
                            subsection_title = subsection.get("title_refine", subsection.get("title", ""))
                            if subsection_title:
                                subsection_id = f"{section_id}_sub{i}"
                                cluster_graph.node(subsection_id, sanitize_graphviz_text(subsection_title),
                                                 fillcolor='lightcoral', fontsize='9',
                                                 width='1.5', height='0.4')
                                dot.edge(section_id, subsection_id)

                        node_counter += 1

                cluster_counter += 1
            else:
                # Single section, add normally
                for section in cluster:
                    section_id = f"section{section.get('section_index', node_counter)}"
                    section_title = section.get("section_title_refine", section.get("section_title", "Untitled"))

                    dot.node(section_id, sanitize_graphviz_text(section_title),
                           fillcolor='lightgreen', fontweight='bold')
                    dot.edge(root_id, section_id)

                    # Add subsections
                    structure_parts = section.get("structure_parts", [])
                    subsections = [part for part in structure_parts
                                 if part.get("type") == "subsection" and part.get("level", 0) <= 3]

                    # Limit subsections
                    subsections = subsections[:3]

                    for i, subsection in enumerate(subsections):
                        subsection_title = subsection.get("title_refine", subsection.get("title", ""))
                        if subsection_title:
                            subsection_id = f"{section_id}_sub{i}"
                            dot.node(subsection_id, sanitize_graphviz_text(subsection_title),
                                   fillcolor='lightcoral', fontsize='9',
                                   width='1.5', height='0.4')
                            dot.edge(section_id, subsection_id)

                    node_counter += 1

        # Add conclusion (title only)
        # conclusion_text = abstract_conclusion.get("Conclusion", "")
        # if conclusion_text:
        #     conclusion_id = f"node{node_counter}"
        #     dot.node(conclusion_id, "Conclusion", fillcolor='lightpink',
        #             fontweight='bold')
        #     dot.edge(root_id, conclusion_id)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c for c in paper_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')[:50]  # Limit filename length
        filename = f"mind_map_{safe_title}_{timestamp}"

        # Save as PNG with higher DPI for better quality
        output_path = os.path.join(output_dir, filename)
        dot.attr(dpi='400')  # Higher DPI for better quality
        dot.render(output_path, format='png', cleanup=True)

        png_path = f"{output_path}.png"
        logger.info(f"Mind map successfully saved to: {png_path}")

        return png_path

    except Exception as e:
        logger.error(f"Error saving mind map as PNG: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def generate_graphviz_mind_map(paper_title, sections_content, abstract_conclusion):
    """Generate a Graphviz DOT representation with improved layout."""
    dot_lines = [
        "digraph PaperMindMap {",
        "  rankdir=LR;",  # Left to Right layout
        "  size=\"12,8!\";",  # Constrain size
        "  ratio=auto;",
        "  nodesep=0.5;",
        "  ranksep=1.0;",
        "  splines=ortho;",
        "  node [shape=box, style=\"rounded,filled\", fontname=\"Arial\", fontsize=10, width=2, height=0.5];",
        "  edge [color=gray, arrowsize=0.8, penwidth=1.5];",
        ""
    ]

    # Root node
    root_id = "root"
    dot_lines.append(f'  {root_id} [label="{sanitize_graphviz_text(paper_title)}", shape=ellipse, style=filled, fillcolor=lightblue, fontsize=12, fontweight=bold, width=3, height=1];')

    node_counter = 1

    # Add abstract (title only)
    abstract_text = abstract_conclusion.get("Abstract", "")
    if abstract_text:
        abstract_id = f"node{node_counter}"
        dot_lines.append(f'  {abstract_id} [label="Abstract", fillcolor=lightyellow, style=filled, fontweight=bold];')
        dot_lines.append(f"  {root_id} -> {abstract_id};")
        node_counter += 1

    # Sort sections by index
    sorted_sections = sorted(sections_content, key=lambda x: x.get("section_index", 0))

    # Add sections with better organization
    for section in sorted_sections:
        section_id = f"section{section.get('section_index', node_counter)}"
        section_title = section.get("section_title_refine", section.get("section_title", "Untitled"))

        dot_lines.append(f'  {section_id} [label="{sanitize_graphviz_text(section_title)}", fillcolor=lightgreen, style=filled, fontweight=bold];')
        dot_lines.append(f"  {root_id} -> {section_id};")

        # Add subsections (limit to 3 for better layout)
        structure_parts = section.get("structure_parts", [])
        subsections = [part for part in structure_parts if part.get("type") == "subsection" and part.get("level", 0) <= 3]
        subsections = subsections[:3]  # Limit to first 3

        for i, subsection in enumerate(subsections):
            subsection_title = subsection.get("title_refine", subsection.get("title", ""))
            if subsection_title:
                subsection_id = f"{section_id}_sub{i}"
                dot_lines.append(f'  {subsection_id} [label="{sanitize_graphviz_text(subsection_title)}", fillcolor=lightcoral, style=filled, fontsize=9, width=1.5, height=0.4];')
                dot_lines.append(f"  {section_id} -> {subsection_id};")

        node_counter += 1

    # Add conclusion (title only)
    # conclusion_text = abstract_conclusion.get("Conclusion", "")
    # if conclusion_text:
    #     conclusion_id = f"node{node_counter}"
    #     dot_lines.append(f'  {conclusion_id} [label="Conclusion", fillcolor=lightpink, style=filled, fontweight=bold];')
    #     dot_lines.append(f"  {root_id} -> {conclusion_id};")

    dot_lines.append("}")
    return "\n".join(dot_lines)


def generate_mind_map_png_only(processed_data, abstract_conclusion, output_dir="./output"):
    """
    Convenience function to generate and save only the PNG mind map.

    Args:
        processed_data: The processed data containing outline and sections
        abstract_conclusion: Abstract and conclusion data
        output_dir: Directory to save the output file

    Returns:
        str: Path to the saved PNG file or None if failed
    """
    try:
        paper_title = processed_data.get("paper_title", "Research Paper")
        sections_content = processed_data.get("sections_content", [])

        return save_mind_map_as_png(paper_title, sections_content, abstract_conclusion, output_dir)

    except Exception as e:
        logger.error(f"Error generating PNG mind map: {e}")
        return None


def generate_mermaid_mind_map(paper_title, sections_content, abstract_conclusion):
    """Generate a Mermaid mind map representation."""
    mermaid_lines = ["mindmap"]
    mermaid_lines.append("  root)(" + sanitize_mermaid_text(paper_title) + ")")

    # Add abstract if available (title only)
    abstract_text = abstract_conclusion.get("Abstract", "")
    if abstract_text:
        mermaid_lines.append("    Abstract")

    # Sort sections by index
    sorted_sections = sorted(sections_content, key=lambda x: x.get("section_index", 0))

    # Add sections and subsections
    for section in sorted_sections:
        section_title = section.get("section_title_refine", section.get("section_title", "Untitled"))
        mermaid_lines.append(f"    {sanitize_mermaid_text(section_title)}")

        # Add subsections from structure_parts
        structure_parts = section.get("structure_parts", [])
        subsections = [part for part in structure_parts if part.get("type") == "subsection" and part.get("level", 0) <= 3]

        for subsection in subsections:
            subsection_title = subsection.get("title_refine", subsection.get("title", ""))
            if subsection_title and subsection.get("level", 0) == 3:  # Only level 3 subsections
                mermaid_lines.append(f"      {sanitize_mermaid_text(subsection_title)}")

    # Add conclusion if available (title only)
    # conclusion_text = abstract_conclusion.get("Conclusion", "")
    # if conclusion_text:
    #     mermaid_lines.append("    Conclusion")

    return "\n".join(mermaid_lines)



def sanitize_mermaid_text(text):
    """Sanitize text for Mermaid mind map format."""
    if not text:
        return ""
    # Remove or replace characters that might break Mermaid syntax
    text = text.replace("(", "").replace(")", "")
    text = text.replace("[", "").replace("]", "")
    text = text.replace("{", "").replace("}", "")
    text = text.replace('"', "'")
    text = text.replace('\n', ' ').replace('\r', ' ')
    # Limit length for readability
    if len(text) > 40:
        text = text[:37] + "..."
    return text.strip()


def sanitize_graphviz_text(text):
    """Sanitize text for Graphviz DOT format with line wrapping at 30 characters."""
    if not text:
        return ""

    # Escape quotes and backslashes
    text = text.replace('\\', '\\\\').replace('"', '\\"')
    text = text.replace('\n', ' ').replace('\r', '')

    # Split into words and wrap at 30 characters per line
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        # If adding this word would exceed 30 characters, start a new line
        if current_line and len(current_line + " " + word) > 30:
            lines.append(current_line)
            current_line = word
        elif current_line:
            current_line += " " + word
        else:
            current_line = word

    # Add the last line if it exists
    if current_line:
        lines.append(current_line)

    # Join lines with Graphviz line break syntax
    return "\\n".join(lines)

def extract_content_preview(content):
    """Extract a meaningful preview from content text."""
    if not content:
        return ""

    # Remove citations and clean up
    content = re.sub(r'\[\d+\]', '', content)
    content = content.strip()

    # Try to get first sentence
    sentences = content.split('.')
    if sentences and len(sentences[0]) > 10:
        preview = sentences[0].strip()
        if len(preview) > 60:
            preview = preview[:57] + "..."
        return preview

    # Fallback to first 60 characters
    if len(content) > 60:
        return content[:57] + "..."

    return content
