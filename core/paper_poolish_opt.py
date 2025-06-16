# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================
import copy
from log import logger
from utils import aggregate_references,generate_mind_map_from_outline
import asyncio
import traceback
import re
from prompt_manager import get_section_name_refinement_prompt
from model_factory import llm_map
from langchain_core.output_parsers import StrOutputParser
import time
from configuration import DEFAULT_MODEL_FOR_SECTION_NAME_REFLECTION, global_semaphores
from prompt_manager import get_subsection_intro_content_prompt


async def optimize_section_titles_async(section_data: dict, paper_title: str):
    """
    Evaluates and refines the title of a given section and its subsection titles
    for clarity and accuracy using an LLM.
    Modifies the section_data dictionary in place by adding 'section_title_refine'
    and 'title_refine' (for subsections) keys.

    Args:
        section_data: The dictionary for a single processed section.
        paper_title: The title of the paper for context.
    """
    section_log_title = section_data.get("section_title", "Untitled Section")
    logger.info(f"Starting title optimization for section: '{section_log_title}'...")

    if not section_data:
        logger.warning("Received empty section data for title optimization.")
        return

    try:
        # Select a model for title refinement
        refinement_llm = llm_map[DEFAULT_MODEL_FOR_SECTION_NAME_REFLECTION].with_config(
            {"temperature": 0.2}
        )

        items_to_refine = []
        # --- Collect titles to refine from this section ---
        # 1. Top-level section title
        original_title = section_data.get("section_title", "Untitled Section")
        # Skip optimization for standard sections
        if original_title.lower() not in [
            "introduction",
            "conclusion",
            "abstract",
            "references",
        ]:
            content_preview = section_data.get(
                "section_text", ""
            )  # Use full text for context
            items_to_refine.append(
                {
                    "type": "section",
                    "data": section_data,  # Reference to the section dict
                    "original_title": original_title,
                    "content_preview": content_preview,
                    "log_prefix": f"Section '{original_title}'",
                }
            )
        else:
            # Ensure the key exists even if not refined
            section_data["section_title_refine"] = original_title

        # 2. Subsection titles within structure_parts
        for part_index, part in enumerate(section_data.get("structure_parts", [])):
            if part.get("type") == "subsection":
                original_subsection_title = part.get("title", "Untitled Subsection")
                # Avoid refining error messages or empty titles
                if (
                    original_subsection_title
                    and not original_subsection_title.startswith("[Error")
                ):
                    # Use subsection content for preview if available
                    subsection_content_preview = part.get("content", "")
                    items_to_refine.append(
                        {
                            "type": "subsection",
                            "data": part,  # Reference to the part dict
                            "original_title": original_subsection_title,
                            "content_preview": subsection_content_preview,
                            "log_prefix": f"Subsection '{original_subsection_title}' (in Section '{original_title}')",
                        }
                    )
                else:
                    # Ensure the key exists even if not refined
                    part["title_refine"] = original_subsection_title

        # --- Prepare and execute refinement tasks ---
        tasks = []
        title_changes = []

        async def refine_single_title_with_semaphore(item_info):
            async with global_semaphores.section_name_refine_semaphore:
                original_title = item_info["original_title"]
                log_prefix = item_info["log_prefix"]
                logger.debug(f"Acquired semaphore for refining title: {log_prefix}")
                try:
                    prompt = get_section_name_refinement_prompt(
                        paper_title=paper_title,
                        section_name=original_title,
                        content_preview=item_info["content_preview"],
                    )
                    chain = prompt | refinement_llm | StrOutputParser()
                    refined_title = await chain.ainvoke({})
                    refined_title = refined_title.strip().strip('"').strip("'")

                    # Determine the key to update based on type
                    target_key = (
                        "section_title_refine"
                        if item_info["type"] == "section"
                        else "title_refine"
                    )

                    if refined_title and refined_title != original_title:
                        return (
                            item_info["data"],
                            target_key,
                            original_title,
                            refined_title,
                            item_info["type"],  # Pass type for logging
                        )
                    return (
                        item_info["data"],
                        target_key,
                        original_title,
                        original_title,
                        item_info["type"],  # Pass type for logging
                    )

                except Exception as title_err:
                    logger.warning(f"Error refining title {log_prefix}: {title_err}")
                    target_key = (
                        "section_title_refine"
                        if item_info["type"] == "section"
                        else "title_refine"
                    )
                    return (
                        item_info["data"],
                        target_key,
                        original_title,
                        original_title,
                        item_info["type"],  # Pass type for logging
                    )

        for item in items_to_refine:
            tasks.append(refine_single_title_with_semaphore(item))

        # Execute refinement tasks concurrently
        if tasks:
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            logger.info(
                f"Title refinement LLM calls for {len(tasks)} items in section '{section_log_title}' took {end_time - start_time:.2f} seconds."
            )

            # Apply the changes back to the referenced dictionaries
            for (
                item_data,
                target_key,
                original_title,
                refined_title,
                item_type,
            ) in results:
                item_data[target_key] = refined_title  # Add/update the refinement key
                if refined_title != original_title:
                    logger.info(
                        f"Refined {item_type.capitalize()} title in section '{section_log_title}' from '{original_title}' to '{refined_title}'"
                    )
                    title_changes.append((original_title, refined_title))

        # Log summary of changes for this section
        if title_changes:
            logger.info(
                f"Optimized {len(title_changes)} titles within section '{section_log_title}'."
            )
        else:
            # No need to log if no changes were made for this specific section
            logger.info(
                f"No titles required optimization in section '{section_log_title}'."
            )

    except Exception as e:
        logger.error(
            f"Error during title optimization for section '{section_log_title}': {e}"
        )
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Ensure 'title_refine' keys exist as fallback if error occurs mid-process
        if "section_title_refine" not in section_data:
            section_data["section_title_refine"] = section_data.get(
                "section_title", "Untitled Section"
            )
        for part in section_data.get("structure_parts", []):
            if part.get("type") == "subsection" and "title_refine" not in part:
                part["title_refine"] = part.get("title", "Untitled Subsection")


async def process_poolish_data(
    paper_title, outline, sections_content, abstract_conclusion
):
    """
    Process poolish data to organize content based on the outline structure,
    deduplicate references globally, and generate final markdown with globally numbered citations.

    Args:
        outline (dict): The outline structure of the document.
        sections_content (dict): The content of each section (keyed by section ID or name).
        abstract_conclusion (dict): Contains abstract and conclusion information.

    Returns:
        dict: The processed poolish data including final markdown.
    """
    try:
        logger.info("Processing poolish data...")
        # Avoid logging potentially large inputs in production
        logger.info(f"Outline: {outline}")
        logger.info(f"Sections content: {sections_content}")
        logger.info(f"Abstract and conclusion: {abstract_conclusion}")

        processed_data = {
            "paper_title": paper_title,
            "outline": copy.deepcopy(outline),
            "sections_content": [],  # Will store processed section data including text with local citation numbers
            "reportIndexList": [],  # Will store the final global, deduplicated reference list
            "Abstract": abstract_conclusion.get("Abstract", ""),
            "Conclusion": abstract_conclusion.get("Conclusion", ""),
            "markdown_content": "",
        }
        conclustion_section_content = abstract_conclusion.get(
            "conclusion_sections_processed", {}
        )

        section_tasks = []
        section_order = []  # 记录section的顺序

        for section_name, section_structure in outline.items():
            section_tasks.append(
                process_section_with_error_handling(
                    section_name,
                    section_structure,
                    sections_content,
                    processed_data,
                    paper_title,
                    conclustion_section_content,
                )
            )
            section_order.append(section_structure.get("section_index", 0))

        await asyncio.gather(*section_tasks)
        # 按照section_index重新排序sections_content
        processed_data["sections_content"].sort(key=lambda x: x.get("section_index", 0))

        if not processed_data["sections_content"]:
            logger.warning("No sections processed successfully. Using fallback.")
            # Fallback might need adjustment based on expected output format
            fallback_data = create_fallback_data(
                outline, sections_content, abstract_conclusion
            )
            # Try to generate basic markdown for fallback
            fallback_data["markdown_content"] = generate_markdown_from_processed(
                fallback_data, abstract_conclusion
            )
            return fallback_data

        # --- Global Reference Deduplication and Renumbering ---
        try:
            # 1. Create Global Reference List and URL-to-Global-Index Map
            global_report_index_list = []
            global_url_to_final_index_map = {}
            temp_global_refs = {}  # Use dict for deduplication based on URL
            logger.info(f"processed_data: {processed_data['sections_content']}")

            for section in processed_data["sections_content"]:
                for ref in section.get("reportIndexList", []):
                    ref_url = ref.get("url")
                    if ref_url and ref_url not in temp_global_refs:
                        temp_global_refs[ref_url] = ref
                    elif (
                        not ref_url
                    ):  # Handle refs without URL - add based on title/other? For now, add directly.
                        # This might lead to duplicates if title isn't unique.
                        # Consider a more robust non-URL ref handling if needed.
                        temp_key = ref.get(
                            "title", f"__no_url_ref_{len(temp_global_refs)}__"
                        )
                        if temp_key not in temp_global_refs:
                            temp_global_refs[temp_key] = ref

            # Assign final global indices
            global_report_index_list = list(temp_global_refs.values())
            for i, ref in enumerate(global_report_index_list, 1):
                ref_url = ref.get("url")
                if ref_url:
                    global_url_to_final_index_map[ref_url] = i
                else:
                    # Map non-URL refs based on the temporary key used for deduplication
                    temp_key = ref.get(
                        "title", f"__no_url_ref_{i-1}__"
                    )  # Reconstruct potential temp key
                    # This mapping might be less reliable if titles changed or weren't unique
                    # Only map if we are reasonably sure it's the same ref added earlier
                    if (
                        temp_key in temp_global_refs
                        and temp_global_refs[temp_key] == ref
                    ):
                        global_url_to_final_index_map[temp_key] = i

            logger.info(f"global_url_to_final_index_map: {global_url_to_final_index_map}")
            processed_data["reportIndexList"] = (
                global_report_index_list  # Store the final global list
            )

            # 2. Renumber Citations in Section Text
            for section in processed_data["sections_content"]:
                section_local_refs = section.get("reportIndexList", [])
                if not section_local_refs:
                    continue  # Skip if section had no references

                # Create a map from local index (1-based) within this section to global index
                section_local_to_global_map = {}
                for local_idx, ref in enumerate(section_local_refs, 0):
                    ref_url = ref.get("url")
                    global_idx = None
                    if ref_url and ref_url in global_url_to_final_index_map:
                        global_idx = global_url_to_final_index_map[ref_url]
                    elif not ref_url:
                        # Try to find global index for non-URL ref using title/temp key
                        temp_key = ref.get(
                            "title", f"__no_url_ref_{local_idx-1}__"
                        )  # Approximate temp key
                        if temp_key in global_url_to_final_index_map:
                            global_idx = global_url_to_final_index_map[temp_key]

                    if global_idx is not None:
                        section_local_to_global_map[local_idx] = global_idx
                    else:
                        logger.warning(
                            f"Could not map local ref {local_idx} (URL: {ref_url}, Title: {ref.get('title')}) in section '{section.get('section_title')}' to global index."
                        )
                        # Keep original number or use placeholder? Using original for now.
                        section_local_to_global_map[local_idx] = local_idx

                # Define replacement function for re.sub
                def replace_citation(match):
                    local_index = int(match.group(1))
                    global_index = section_local_to_global_map.get(
                        local_index, local_index
                    )  # Default to local if mapping failed
                    return f"[{global_index}]"

                # Apply replacement to section_text
                # Use regex to find citations like [1], [12], etc.
                section["section_text"] = re.sub(
                    r"\[(\d+)\]", replace_citation, section["section_text"]
                )

        except Exception as e:
            logger.error(f"Error during global reference renumbering: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Proceed with potentially incorrect citation numbers in markdown

        # --- Mind Map Generation ---
        try:
            processed_data["mind_map"] = generate_mind_map_from_outline(
                processed_data, abstract_conclusion
            )
            logger.info("Successfully generated mind map from outline structure")
        except Exception as e:
            logger.error(f"Error generating mind map: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            processed_data["mind_map"] = {
                "mermaid": "# Error Generating Mind Map\n\nCould not generate visualization.",
                "graphviz": "digraph G { error [label=\"Error generating mind map\"] }"
            }

        # --- Markdown Generation ---
        try:
            processed_data["markdown_content"] = generate_markdown_from_processed(
                processed_data, abstract_conclusion, conclustion_section_content
            )
        except Exception as e:
            logger.error(f"Error generating final markdown: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Attempt fallback markdown generation if possible
            processed_data["markdown_content"] = (
                "# Error Generating Markdown\n\nCould not assemble final document."
            )

        # logger.info(f"process_poolish_data final processed_data: {processed_data}") # Avoid logging large output
        return processed_data

    except Exception as e:
        logger.error(f"Critical error in process_poolish_data: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Fallback might need adjustment based on expected output format
        fallback_data = create_fallback_data(
            outline, sections_content, abstract_conclusion
        )
        fallback_data["markdown_content"] = (
            "# Error Processing Data\n\nCould not generate document."
        )
        return fallback_data


def generate_markdown_from_processed(
    processed_data, abstract_conclusion, conclustion_section_content
):
    """Generates the final markdown string from processed data."""
    markdown_content = []
    conclusion_added_in_sections = (
        False  # Flag to track if conclusion was part of sections
    )

    # Add title if available (assuming it might be in processed_data top level)
    if processed_data.get("paper_title"):  # Add a check for title key if applicable
        markdown_content.append(f"# {processed_data['paper_title']}\n")

    # Add abstract
    # Use the key consistent with initialization ('Abstract')
    abstract_text = processed_data.get("Abstract", "")
    if abstract_text:
        markdown_content.append(f"## Abstract\n\n{abstract_text}\n")

    # Sort sections by index
    sorted_sections = sorted(
        processed_data.get("sections_content", []),
        key=lambda x: x.get("section_index", 0),
    )

    # Add section content (now with globally numbered citations)
    for section in sorted_sections:
        section_title = section.get(
            "section_title_refine", section.get("section_title", "Untitled Section")
        )
        section_text = section.get("section_text", "")
        if not section_text:
            section_text = conclustion_section_content.get(
                section["section_title"], {}
            ).get("section_text", "")
            logger.info(f"find conclusion text: {section_text}")

        if not section_text:
            logger.info("section_text is empty")
            continue

        if section_title.lower() == "conclusion" and processed_data.get("Conclusion"):
            # If the conclusion content was placed here during processing
            markdown_content.append(f"## {section_title}\n\n{section_text}")
            conclusion_added_in_sections = True
        else:
            markdown_content.append(f"## {section_title}\n\n{section_text}")

    # Add Conclusion if it wasn't part of the main sections
    # Use the key consistent with initialization ('Conclusion')
    conclusion_text = processed_data.get("Conclusion", "")
    if conclusion_text and not conclusion_added_in_sections:
        markdown_content.append(f"## Conclusion\n\n{conclusion_text}\n")

    # Add References using the globally deduplicated and ordered list
    global_references = processed_data.get("reportIndexList", [])
    if global_references:
        markdown_content.append("## References\n")
        for i, ref in enumerate(global_references, 1):
        # for i, ref in enumerate(global_references, 0):
            # Format reference entry (adjust formatting as needed)
            ref_text = f"{i}. {ref.get('title', 'Untitled')}"
            authors = ref.get("authors")
            if authors:
                # Handle potential list or string format for authors
                if isinstance(authors, list):
                    ref_text += f". {', '.join(authors)}"
                else:
                    ref_text += f". {authors}"
            venue = ref.get("venue")
            if venue:
                ref_text += f". *{venue}*"
            year = ref.get("year")
            if year:
                ref_text += f", {year}"
            url = ref.get("url")
            # if url: # Optionally add URL
            #     ref_text += f". Available: {url}"
            markdown_content.append(ref_text)

    # Join all parts with double newlines
    return "\n\n".join(markdown_content)


def create_fallback_data(outline, sections_content, abstract_conclusion):
    """Create fallback data using original content when processing fails"""
    logger.info("Creating fallback data from original content")

    fallback_data = {
        "outline": copy.deepcopy(outline),
        "sections_content": [],
        "reportIndexList": [],
        "Abstract": abstract_conclusion.get("Abstract", ""),  # Consistent key
        "Conclusion": abstract_conclusion.get("Conclusion", ""),  # Consistent key
        "markdown_content": "",  # Add markdown field
    }

    all_refs_temp = {}

    # Create a simplified version of the sections content
    for section_name, section_info in outline.items():
        section_data = {
            "section_index": section_info.get("section_index", 0),
            "section_title": section_info.get("section_title", section_name),
            "section_text": "",
            "section_summary": "",
            # Fallback doesn't include figures/structure parts for simplicity
            "reportIndexList": [],
        }

        combined_text = []
        section_local_refs = []

        try:
            # Try to find content matching section_name or key_point
            content_found = False
            if section_name in sections_content:
                contents = sections_content[section_name]
                content_found = True
            # If not found by name, maybe check key_points? (More complex)
            # else: ...

            if content_found:
                for content in contents:
                    text = content.get("section_text", "")
                    refs = content.get("reportIndexList", [])
                    if text:
                        combined_text.append(text)
                    if refs:
                        section_local_refs.extend(refs)
                        for ref in refs:  # Collect for global fallback list
                            ref_url = ref.get("url")
                            if ref_url and ref_url not in all_refs_temp:
                                all_refs_temp[ref_url] = ref
                            elif not ref_url:
                                temp_key = ref.get(
                                    "title", f"__fallback_no_url_{len(all_refs_temp)}__"
                                )
                                if temp_key not in all_refs_temp:
                                    all_refs_temp[temp_key] = ref

                section_data["section_text"] = "\n\n".join(combined_text)
                section_data["reportIndexList"] = (
                    section_local_refs  # Keep local refs for potential fallback renumbering
                )

            else:
                logger.warning(
                    f"No direct content found for fallback section {section_name}"
                )
                section_data["section_text"] = (
                    f"[Content for section '{section_name}' not found]"
                )

        except Exception as inner_e:
            logger.error(
                f"Error creating fallback text for section {section_name}: {inner_e}"
            )
            section_data["section_text"] = (
                f"[Error processing fallback content for section '{section_name}']"
            )

        fallback_data["sections_content"].append(section_data)

    # Create global fallback reference list
    fallback_data["reportIndexList"] = list(all_refs_temp.values())

    # Attempt basic markdown generation for fallback
    fallback_data["markdown_content"] = generate_markdown_from_processed(
        fallback_data, abstract_conclusion
    )

    return fallback_data


async def process_subsection_structure(
    subsection_name,
    subsection_info,
    sections_content,
    section_data,  # Contains the cumulative 'reportIndexList' and the 'url_to_final_index_map'
    header_level=3,
    subsection_index=0,
):
    # ... (rest of the function remains the same)
    try:
        logger.info(
            f"Processing subsection structure: {subsection_name} at level {header_level}"
        )
        result_parts = []

        if "url_to_final_index_map" not in section_data:
            section_data["url_to_final_index_map"] = {
                ref.get("url"): idx + 1
                for idx, ref in enumerate(section_data.get("reportIndexList", []))
                if ref.get("url")
            }

        key_points = subsection_info.get("key_points", [])

        if not key_points:
            # Case 1: Leaf node
            logger.info(
                f"Subsection '{subsection_name}' has no key_points. Treating as content node."
            )
            subsection_part = {
                "type": "subsection",
                "title": subsection_name,
                "content": "",
                "level": header_level,
                "index": subsection_index,
                "hierarchical_index": (subsection_index,),
                "main_figure_data": None,
                "main_figure_caption": None,
            }
            point_content = await find_content_for_key_point(
                subsection_name,
                sections_content,
                section_data["reportIndexList"],
                section_data["url_to_final_index_map"],
            )
            if point_content["text"]:
                subsection_part["content"] = point_content["text"]
                if point_content.get("main_figure_data"):
                    subsection_part["main_figure_data"] = point_content[
                        "main_figure_data"
                    ]
                    subsection_part["main_figure_caption"] = point_content.get(
                        "main_figure_caption", ""
                    )
            else:
                logger.warning(
                    f"No content found for leaf subsection '{subsection_name}' using its name as key."
                )
            result_parts.append(subsection_part)

        else:
            # Case 2: Directory node
            logger.info(
                f"Subsection '{subsection_name}' has key_points. Treating as directory node."
            )
            current_header_part = {
                "type": "subsection",
                "title": subsection_name,
                "content": "",
                "level": header_level,
                "index": subsection_index,
                "hierarchical_index": (subsection_index,),
                "main_figure_data": None,  # Initialize figure data for the header part
                "main_figure_caption": None,
            }
            # result_parts.append(current_header_part)

            nested_index = 0
            key_point_parts = []
            key_point_texts = []  # Store text content for each key point

            # Iterate through the key_points listed under this subsection
            for key_point_name in key_points:
                # Directly find content for this key_point using find_content_for_key_point
                point_content = await find_content_for_key_point(
                    key_point_name,
                    sections_content,
                    section_data["reportIndexList"],
                    section_data["url_to_final_index_map"],
                )
                key_point_texts.append(point_content["text"])

                # Create a part for this key_point's content if text or figure exists
                if point_content["text"] or point_content.get("main_figure_data"):
                    key_point_part = {
                        "type": "subsection",  # Treat key_point content as a sub-part for structure
                        "title": key_point_name,  # Use key_point name as title for this content block
                        "content": point_content["text"],
                        "level": header_level + 1,  # Content is one level deeper
                        "index": nested_index,
                        "hierarchical_index": (subsection_index, nested_index),
                        "main_figure_data": point_content.get("main_figure_data", ""),
                        "main_figure_caption": point_content.get(
                            "main_figure_caption", ""
                        ),
                    }
                    key_point_parts.append(key_point_part)

                    # result_parts.append(key_point_part)

                else:
                    logger.warning(
                        f"No content or figure found for key_point '{key_point_name}' listed under '{subsection_name}'. Skipping part creation."
                    )

                nested_index += 1

            # Now generate the intro content for the header part
            # Only generate intro if we have key points with content
            if key_point_parts:
                try:
                    # Get paper title from section_data or use a placeholder
                    paper_title = section_data.get("paper_title", "Research Paper")
                    # Extract user query if available, or use placeholder
                    user_query = section_data.get("user_query", "Research query")
                    # Create list of key point names for those that have content
                    key_point_names = [part["title"] for part in key_point_parts]
                    # Create prompt for generating intro content
                    intro_prompt = get_subsection_intro_content_prompt(
                        paper_title=paper_title,
                        user_query=user_query,
                        subsection_name=subsection_name,
                        key_points=key_point_names,
                        key_points_content=key_point_texts,
                    )

                    # Use the same model as for title refinement, with low temperature
                    intro_llm = llm_map[
                        DEFAULT_MODEL_FOR_SECTION_NAME_REFLECTION
                    ].with_config({"temperature": 0.2})

                    # Generate intro content
                    intro_chain = intro_prompt | intro_llm | StrOutputParser()
                    intro_content = await intro_chain.ainvoke({})

                    # Set intro content to the header part
                    current_header_part["content"] = intro_content.strip()
                    logger.info(
                        f"Generated introduction content for subsection '{subsection_name}'"
                    )

                except Exception as e:
                    logger.warning(
                        f"Error generating intro for subsection '{subsection_name}': {e}"
                    )
                    # Continue without an intro if there's an error

            # Add the header part first, then all key point parts
            result_parts.append(current_header_part)
            result_parts.extend(key_point_parts)

        return result_parts

    except Exception as e:
        logger.error(f"Error processing subsection structure {subsection_name}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return [
            {
                "type": "subsection",
                "title": subsection_name,
                "content": f"[Error processing subsection: {str(e)}]",
                "level": header_level,
                "index": subsection_index,
            }
        ]


async def process_section(
    section_name,
    section_structure,
    sections_content,
    processed_data_list_container,  # Changed to pass the container list
    paper_title: str,  # Add paper_title argument
    conclustion_section_content,
):
    """
    处理单个部分的内容，按照目录结构组织内容，处理引用去重和编号更新 (local scope),
    并优化该部分的标题。
    Appends the processed section_data dictionary to the processed_data_list_container.
    """
    # ... (existing section processing logic remains largely the same)
    logger.info(f"process_section section_name: {section_name}")

    section_index = section_structure["section_index"]
    section_title = section_structure["section_title"]
    subsections = section_structure.get("subsection_info", {})

    section_text = ""
    if section_title in conclustion_section_content:
        section_text = conclustion_section_content[section_title]["section_text"]
        if section_text:
            section_text = section_text + "\n\n"

    section_data = {
        "section_index": section_index,
        "section_title": section_title,
        "section_text": section_text,  # Will contain text with LOCAL citation numbers
        "section_summary": "",
        "reportIndexList": [],  # Will contain unique references LOCAL to this section
        "structure_parts": [],
        "url_to_final_index_map": {},  # Local map used during processing
        # Initialize refinement key - will be overwritten if optimized
        "section_title_refine": section_title,
    }

    # --- Start Copied Logic ---
    logger.info(
        f"Processing section: {section_title} with section index {section_index}"
    )

    section_structure_parts = []

    if subsections:
        subsection_parts = []
        subsection_index = 0
        for subsection_name, subsection_info in subsections.items():
            # Ensure subsection parts also get an initial 'title_refine'
            subsection_result = await process_subsection_structure(
                subsection_name,
                subsection_info,
                sections_content,
                section_data,
                header_level=3,
                subsection_index=subsection_index,
            )
            if subsection_result:
                for part in subsection_result:
                    if part.get("type") == "subsection" and "title_refine" not in part:
                        part["title_refine"] = part.get("title", "Untitled Subsection")
                subsection_parts.extend(subsection_result)
            subsection_index += 1
        section_structure_parts.extend(subsection_parts)
    else:
        if "key_points" in section_structure and section_structure["key_points"]:
            section_key_points = section_structure["key_points"]
            logger.info(
                f"Section {section_title} has key points but no subsections. Processing key points directly."
            )
            key_point_index = 0
            for key_point in section_key_points:
                point_content = await find_content_for_key_point(
                    key_point,
                    sections_content,
                    section_data["reportIndexList"],
                    section_data["url_to_final_index_map"],
                )
                if point_content["text"] or point_content.get(
                    "main_figure_data"
                ):  # Check figure too
                    section_structure_parts.append(
                        {
                            "type": "main_section_content",  # Or adjust type if needed
                            "title": None,  # Key points often don't have explicit titles here
                            "content": point_content["text"],
                            "level": 2,  # Or adjust level
                            "index": key_point_index,
                            "hierarchical_index": (key_point_index,),
                            "key_point": key_point,
                            "main_figure_data": point_content.get("main_figure_data"),
                            "main_figure_caption": point_content.get(
                                "main_figure_caption"
                            ),
                            # No title_refine needed for this type usually
                        }
                    )
                key_point_index += 1

    # --- Call Title Optimization ---
    section_data["structure_parts"] = sorted(
        section_structure_parts, key=lambda x: x.get("hierarchical_index", ())
    )
    await optimize_section_titles_async(section_data, paper_title)

    # --- Text Assembly Logic (Consider if this needs adjustment based on refined titles) ---
    # Current logic assembles text BEFORE title refinement. If refined subsection titles
    # should appear in the final markdown's section_text, this assembly might need
    # to happen AFTER title refinement, or generate_markdown_from_processed needs
    # to be smarter about using structure_parts. Assuming current assembly is okay for now.
    section_text_parts = []
    for part in section_data["structure_parts"]:
        part_text = ""
        # Use original title for assembly, refinement happens later
        part_title = part.get("title_refine", part.get("title", ""))

        if part["type"] == "main_section_content":
            if part["content"]:
                part_text = part["content"]
        elif part["type"] == "subsection":
            # Use original title for structure generation
            header = "#" * part["level"] + " " + part_title
            if part["content"]:
                # Check if content is not an error message before adding header+content
                if not str(part.get("content", "")).startswith(
                    "[Error processing subsection"
                ):
                    part_text = f"{header}\n\n{part['content']}"
                # else: only add header if content is empty but not an error (handled below)
            # Add header even if content is empty, unless it's an error placeholder
            if not part_text and not str(part.get("content", "")).startswith(
                "[Error processing subsection"
            ):
                part_text = header

        if part_text:
            section_text_parts.append(part_text)

        # Figure handling (remains the same)
        add_figure = False
        if add_figure and part.get("main_figure_data"):
            figure_caption = part.get(
                "main_figure_caption", f"Figure for {part_title or 'section'}"
            )
            figure_markdown = f"\n\n![{figure_caption}]({part['main_figure_data']})\n"
            if figure_caption:
                figure_markdown += f"*{figure_caption}*\n"
            # Append figure markdown after the corresponding text part
            if section_text_parts and section_text_parts[-1] == part_text:
                section_text_parts[-1] += figure_markdown
            elif part_text:  # Append if part_text was just added
                section_text_parts.append(figure_markdown.strip())
            else:  # Append if no text but figure exists (e.g., subsection header only + figure)
                section_text_parts.append(figure_markdown.strip())

    section_data["section_text"] += "\n\n".join(filter(None, section_text_parts))
    # --- End Text Assembly Logic ---

    # Now section_data contains 'section_title_refine' and 'title_refine' in structure_parts

    logger.info(
        f"Finished processing section {section_title}. Local unique ref count: {len(section_data['reportIndexList'])}"
    )

    if "url_to_final_index_map" in section_data:
        del section_data["url_to_final_index_map"]  # Clean up temp map

    # Append the result (now including refined titles) to the list provided by the caller
    processed_data_list_container.append(section_data)


async def find_content_for_key_point(
    key_point, sections_content, cumulative_references_list, url_to_final_index_map
):
    # ... (find_content_for_key_point function remains the same)
    """
    Find content associated with a specific key point, deduplicate references against
    a cumulative list (local to the section being processed), update the list and map,
    and return text with LOCAL citation numbers.
    """
    result = {
        "text": "",
        "references": [],
        "main_figure_data": "",
        "main_figure_caption": "",
    }

    for content_list in sections_content.values():
        for content in content_list:
            content_key_point = content.get("section_point") or content.get(
                "section_key_point"
            )
            if content_key_point == key_point:
                try:
                    original_text = content.get("section_text", "")
                    snippet_references = content.get("reportIndexList", [])
                    local_ref_map = {}

                    if not snippet_references:
                        modified_text = original_text
                    else:
                        for i, ref in enumerate(snippet_references):
                            ref_url = ref.get("url")
                            original_identifier = (
                                ref_url if ref_url else f"__temp_ref_{key_point}_{i}__"
                            )

                            if not ref_url:
                                logger.warning(
                                    f"Reference missing URL for key_point '{key_point}': {ref.get('title')}"
                                )
                                # Add non-URL ref, assign next local index
                                if not any(
                                    existing_ref == ref
                                    for existing_ref in cumulative_references_list
                                ):
                                    cumulative_references_list.append(ref)
                                final_index = (
                                    cumulative_references_list.index(ref) + 1
                                )  # Find its index in the list
                                local_ref_map[original_identifier] = final_index
                                continue

                            if ref_url in url_to_final_index_map:
                                final_index = url_to_final_index_map[ref_url]
                                local_ref_map[original_identifier] = final_index
                            else:
                                # New unique reference (local scope)
                                cumulative_references_list.append(ref)
                                final_index = len(cumulative_references_list)
                                url_to_final_index_map[ref_url] = final_index
                                local_ref_map[original_identifier] = final_index

                        # Assume aggregate_references replaces based on local_ref_map
                        modified_text = aggregate_references(
                            original_text, snippet_references, local_ref_map
                        )

                    result["text"] = modified_text
                    if "main_figure_data" in content:
                        result["main_figure_data"] = content["main_figure_data"]
                    if "main_figure_caption" in content:
                        result["main_figure_caption"] = content["main_figure_caption"]
                    return result

                except Exception as e:
                    logger.error(
                        f"Error processing content for key point '{key_point}': {e}"
                    )
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    result["text"] = content.get(
                        "section_text", f"[Error processing content for {key_point}]"
                    )
                    if "main_figure_data" in content:
                        result["main_figure_data"] = content["main_figure_data"]
                    if "main_figure_caption" in content:
                        result["main_figure_caption"] = content["main_figure_caption"]
                    return result

    logger.warning(f"No content found for key point: {key_point}")
    return result



async def process_section_with_error_handling(
    section_name,
    section_info,
    sections_content,
    processed_data,
    paper_title: str,
    conclustion_section_content,
):
    """Process a section with error handling, appending result to processed_data['sections_content']"""
    try:
        # Pass the list where results should be appended and the paper_title
        await process_section(
            section_name,
            section_info,
            sections_content,
            processed_data["sections_content"],
            paper_title,  # Pass paper_title here
            conclustion_section_content,
        )
    except Exception as e:
        logger.error(f"Error processing section {section_name}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Create fallback section data
        fallback_title = section_info.get("section_title", section_name)
        section_data = {
            "section_index": section_info.get("section_index", 0),
            "section_title": fallback_title,
            "section_title_refine": fallback_title,  # Add fallback refine key
            "section_text": f"[Error processing section: {str(e)}]",
            "section_summary": "",
            "reportIndexList": [],  # Fallback has no refs
            "structure_parts": [],  # Add empty structure parts for consistency
        }
        # Add fallback section data directly to the list in the main dictionary
        processed_data["sections_content"].append(section_data)


