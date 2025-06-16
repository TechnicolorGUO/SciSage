# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================


from dataclasses import dataclass, field, asdict
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import os
from enum import Enum


class QueryIntent(BaseModel):
    research_field: str = Field(
        description="The academic or research field the query belongs to"
    )
    paper_type: str = Field(description="The type of query")
    topic: str = Field(description="Specific topic of the paper")
    explanation: Optional[str] = Field(description="Brief explanation for the classification")


class Reference(BaseModel):
    title: str = Field(description="Title of the reference")
    authors: Optional[str] = Field(description="Authors of the reference")
    conference: Optional[str] = Field(description="Conference of the reference")
    source: Optional[str] = Field(description="Source (e.g., journal, arXiv)")
    url: Optional[str] = Field(description="URL if available")
    abstract: Optional[str] = Field(
        default="", description="Abstract of the reference")


class Figure(BaseModel):
    caption: str = Field(description="Caption of the figure")
    description: str = Field(description="Description of the figure content")
    placeholder_path: Optional[str] = Field(
        default=None, description="Placeholder for figure file path"
    )


class SectionContent(BaseModel):
    section_name: str = Field(description="Name of the section")
    section_index: Optional[Any] = Field(description="Index of the section")
    parent_section: Optional[str] = Field(
        default=None, description="Parent section name"
    )
    section_key_points: str = Field(description="The key point of this section")
    section_search_querys: str = Field(description="Search query ")

    section_text: str = Field(description="The generated content from the RAG service")
    main_figure_data: str = Field("The base64 encoded image data for the main figure")
    main_figure_caption: str = Field("The caption for the main figure")
    reportIndexList: List[Reference] = Field(description="List of reference entities")

    task_id: str = Field(
        description="Unique identifier for the task, used for tracking"
    )


class SectionSummary(BaseModel):
    section_name: str = Field(description="Name of the section")
    summary: List[str] = Field(
        description="Up to 3 sentences summarizing the section content"
    )
    parent_section: Optional[str] = Field(
        default=None, description="Parent section name, if any"
    )


class Feedback(BaseModel):
    meets_requirements: bool = Field(
        description="Whether the content meets requirements"
    )
    reasons: Optional[List[str]] = Field(
        default=None, description="Reasons for not meeting requirements"
    )
    suggested_improvements: Optional[List[str]] = Field(
        default=None, description="Suggestions for improvement"
    )


class GlobalReflection(BaseModel):
    meets_requirements: bool = Field(
        description="Whether the full paper meets requirements"
    )
    reasons: Optional[List[str]] = Field(
        default=None, description="Reasons for not meeting requirements"
    )
    section_feedback: Optional[Dict[str, List[str]]] = Field(
        default=None, description="Per-section improvement suggestions"
    )


class FinalPaper(BaseModel):
    user_query: str = Field(description="Original user query")
    meta_info: Dict[str, Any] = Field(description="Meta info of the paper")
    title: str = Field(description="Paper title")
    abstract: str = Field(description="Paper abstract")
    sections: Dict[str, SectionContent] = Field(
        description="Section contents with references and figures"
    )
    conclusion: Optional[str] = Field(default=None, description="Conclusion content")
    global_references: List[Reference] = Field(
        description="Globally numbered references"
    )


class IntermediateState(BaseModel):
    user_name: str = Field(default="anonymous", description="User identifier")
    user_query: str = Field(description="Original user query")
    timestamp: str = Field(description="Timestamp of process start")
    data: Dict[str, Any] = Field(description="Intermediate results with descriptions")


def generate_filename_prefix(user_name: str, user_query: str, timestamp: str) -> str:
    """Generate a unique filename prefix for intermediate storage."""
    query_snippet = user_query[:50].replace(" ", "_").replace("/", "_")
    return f"{user_name}_{query_snippet}_{timestamp}"


def initialize_intermediate_state(user_name: str, user_query: str) -> IntermediateState:
    """Initialize the intermediate state dictionary."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return IntermediateState(
        user_name=user_name, user_query=user_query, timestamp=timestamp, data={}
    )


@dataclass
class SectionData:
    """Data structure for section information"""

    section_name: str
    parent_section: str
    section_index: int
    key_points: List[str] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    content: Dict[str, Any] = field(default_factory=dict)
    reflection_results: Dict[str, Any] = field(default_factory=dict)
    section_summary: str = ""
