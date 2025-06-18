# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================


import json
import requests
from typing import Dict, Any, List, Optional, Tuple, Callable, Union


def create_reflection_fallback(
    params: Dict[str, Any], error_message: str
) -> Dict[str, Any]:
    """Create fallback response for section reflection failures"""
    return {
        "section_content": {
            "parent_section": params.get("parent_section", ""),
            "section_text": params.get("section_text", ""),
            "section_key_point": params.get("section_key_point", ""),
            "section_summary": f"[Summary could not be generated due to reflection error: {error_message}]",
            "section_index": params.get("section_index", 0),
            "search_query": params.get("search_query", ""),
            "reportIndexList": params.get("reportIndexList", []),
            "main_figure_data": params.get("main_figure_data", ""),
            "main_figure_caption": params.get("main_figure_caption", ""),
        },
        "search_queries": [],
        "reflection_result": {
            "meets_requirements": False,
            "feedback": f"Reflection failed with error: {error_message}",
            "improvement_queries": [],
        },
    }

