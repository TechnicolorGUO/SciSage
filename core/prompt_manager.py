# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] : prompt manager
# ==================================================================
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
import json
from typing import Dict, Any, List, Optional, Tuple, Union, TypeVar, cast

# -----------------------------------------------------------------------------
# Template generator
# -----------------------------------------------------------------------------


def get_seed_outline_template(paper_type: str) -> List[Dict]:
    """Return a list of dictionaries describing the outline template for a given
    paper type.  Each dictionary must include:  title, key_points, and an optional
    subsections list that follows the same schema recursively.
    """
    # ------------------------------- SURVEY ---------------------------------- #
    paper_type = paper_type.lower()
    if paper_type == "survey":
        return [
            {
                "title": "Introduction",
                "key_points": [
                    f"Background and motivation for <TOPIC>",
                    f"Definition and significance of <TOPIC>",
                    "Scope and organization of this survey",
                ],
            },
            {
                "title": "Task Definition and Background",
                "key_points": [
                    f"Formal definition of the <TOPIC> task",
                    "Comparison with cascaded paradigms",
                    "Key terminology and prerequisites",
                ],
            },
            {
                "title": "Key Methods and Models",
                "key_points": [
                    f"Overview of mainstream approaches in <TOPIC>",
                    "Taxonomy by model architecture",
                    "Multilingual and zero‑shot techniques",
                ],
                "subsections": [
                    {
                        "title": "Encoder‑Decoder Models",
                        "key_points": [
                            "Attention‑based seq2seq frameworks",
                            "Strengths and limitations",
                        ],
                    },
                    {
                        "title": "Transducer & RNN‑T Models",
                        "key_points": [
                            "Streaming capability",
                            "Alignment handling",
                        ],
                    },
                    {
                        "title": "Massively Multilingual Models",
                        "key_points": [
                            "Parameter sharing strategies",
                            "Cross‑lingual transfer effectiveness",
                        ],
                    },
                ],
            },
            {
                "title": "System Architectures and Benchmarks",
                "key_points": [
                    "Pipeline vs. end‑to‑end deployment trade‑offs",
                    "Latency and resource considerations",
                ],
                "subsections": [
                    {
                        "title": "Datasets",
                        "key_points": [
                            "MuST‑C, CoVoST, Europarl‑ST, etc.",
                            "Domain coverage and size analysis",
                        ],
                    },
                    {
                        "title": "Evaluation Metrics",
                        "key_points": [
                            "BLEU, WER, METEOR, COMET",
                            "Metric suitability for speech translation",
                        ],
                    },
                ],
            },
            {
                "title": "Challenges and Future Directions",
                "key_points": [
                    "Robustness to noisy input and domain shift",
                    "Low‑resource language support",
                    "Ethical, privacy, and bias considerations",
                ],
                "subsections": [
                    {
                        "title": "Technical Challenges",
                        "key_points": [
                            "Data scarcity",
                            "Latency/quality trade‑off",
                        ],
                    },
                    {
                        "title": "Ethical Challenges",
                        "key_points": [
                            "Bias amplification",
                            "Responsible deployment",
                        ],
                    },
                    {
                        "title": "Future Research Directions",
                        "key_points": [
                            "Context‑aware translation",
                            "Universal speech translation",
                        ],
                    },
                ],
            },
            {
                "title": "Conclusion",
                "key_points": [
                    "Summary of survey findings",
                    "Outlook on field trajectory",
                ],
            },
        ]
    # ------------------------------- METHOD ---------------------------------- #
    elif paper_type == "method":
        return [
            {
                "title": "Introduction",
                "key_points": [
                    f"Motivation for a new method in <TOPIC>",
                    "Limitations of existing approaches",
                    "Key contributions",
                ],
            },
            {
                "title": "Related Work",
                "key_points": [
                    f"Overview of prior work in <TOPIC>",
                    "Comparison with baselines",
                ],
            },
            {
                "title": "Problem Definition",
                "key_points": [
                    "Formal problem statement",
                    "Input/Output specification",
                    "Evaluation criteria",
                ],
            },
            {
                "title": "Proposed Method",
                "key_points": ["High‑level overview", "Design rationale"],
                "subsections": [
                    {
                        "title": "Model Architecture",
                        "key_points": [
                            "Component description",
                            "Complexity analysis",
                        ],
                    },
                    {
                        "title": "Training Procedure",
                        "key_points": [
                            "Objective functions",
                            "Optimization tricks",
                        ],
                    },
                    {
                        "title": "Algorithm Details",
                        "key_points": [
                            "Pseudocode",
                            "Theoretical justification",
                        ],
                    },
                ],
            },
            {
                "title": "Experiments",
                "key_points": ["Setup and baselines", "Result highlights"],
                "subsections": [
                    {
                        "title": "Datasets",
                        "key_points": ["Dataset description", "Preprocessing"],
                    },
                    {
                        "title": "Results",
                        "key_points": ["Quantitative metrics", "Qualitative examples"],
                    },
                    {
                        "title": "Ablation Studies",
                        "key_points": [
                            "Component contribution",
                            "Sensitivity analysis",
                        ],
                    },
                ],
            },
            {
                "title": "Discussion",
                "key_points": [
                    "Strengths and limitations",
                    "Scalability considerations",
                    "Potential improvements",
                ],
            },
            {
                "title": "Conclusion",
                "key_points": ["Summary of method", "Future work"],
            },
        ]

    # ----------------------------- APPLICATION -------------------------------- #
    elif paper_type == "application":
        return [
            {
                "title": "Introduction",
                "key_points": [
                    f"Practical motivation for applying <TOPIC>",
                    "Problem context and requirements",
                    "Paper structure",
                ],
            },
            {
                "title": "System Design and Architecture",
                "key_points": ["Overall system overview", "Design principles"],
                "subsections": [
                    {
                        "title": "Component Breakdown",
                        "key_points": [
                            "Data ingestion",
                            "Model serving",
                            "User interface",
                        ],
                    },
                    {
                        "title": "Deployment Considerations",
                        "key_points": ["Scalability", "Latency", "Monitoring"],
                    },
                ],
            },
            {
                "title": "Implementation Details",
                "key_points": ["Data preprocessing", "Training setup"],
                "subsections": [
                    {
                        "title": "Data Pipeline",
                        "key_points": ["Collection", "Cleaning", "Augmentation"],
                    },
                    {
                        "title": "Model Training",
                        "key_points": ["Hyper‑parameters", "Compute resources"],
                    },
                ],
            },
            {
                "title": "Case Study or Evaluation",
                "key_points": ["Evaluation metrics", "Result analysis"],
                "subsections": [
                    {
                        "title": "Use Case Demonstration",
                        "key_points": ["Scenario description", "Qualitative results"],
                    },
                    {
                        "title": "Quantitative Evaluation",
                        "key_points": ["Benchmark comparison", "User study"],
                    },
                ],
            },
            {
                "title": "Limitations and Lessons Learned",
                "key_points": [
                    "System bottlenecks",
                    "Deployment challenges",
                    "Unexpected outcomes",
                ],
            },
            {
                "title": "Conclusion",
                "key_points": ["Summary of impact", "Future improvements"],
            },
        ]

    # ------------------------------ ANALYSIS ---------------------------------- #
    elif paper_type == "analysis":
        return [
            {
                "title": "Introduction",
                "key_points": [
                    f"Rationale for an analysis of <TOPIC>",
                    "Scope and contributions",
                ],
            },
            {
                "title": "Theoretical Framework",
                "key_points": ["Analytical tools", "Assumptions"],
            },
            {
                "title": "Error and Behavior Analysis",
                "key_points": ["Performance bottlenecks", "Statistical behavior"],
                "subsections": [
                    {
                        "title": "Quantitative Analysis",
                        "key_points": ["Metric breakdown", "Statistical tests"],
                    },
                    {
                        "title": "Qualitative Analysis",
                        "key_points": ["Failure case taxonomy", "Visualization"],
                    },
                ],
            },
            {
                "title": "Case Studies",
                "key_points": ["Representative examples", "Insight extraction"],
            },
            {
                "title": "Implications and Recommendations",
                "key_points": ["Research implications", "Design guidance"],
            },
            {
                "title": "Conclusion",
                "key_points": ["Summary of findings", "Future analytical work"],
            },
        ]

    # ------------------------------ POSITION ---------------------------------- #
    elif paper_type == "position":
        return [
            {
                "title": "Introduction",
                "key_points": [
                    f"Context and controversy around <TOPIC>",
                    "Author’s stance",
                    "Paper structure",
                ],
            },
            {
                "title": "Background and Related Arguments",
                "key_points": ["Existing viewpoints", "Supporting literature"],
            },
            {
                "title": "Argument for the Position",
                "key_points": ["Core arguments", "Evidence"],
            },
            {
                "title": "Counterarguments and Rebuttals",
                "key_points": ["Key criticisms", "Rebuttal"],
            },
            {
                "title": "Conclusion",
                "key_points": ["Restate position", "Implications", "Call to action"],
            },
        ]

    # ------------------------------ THEORY ------------------------------------ #
    elif paper_type == "theory":
        return [
            {
                "title": "Introduction",
                "key_points": [
                    f"Theoretical motivation for studying <TOPIC>",
                    "Key questions",
                ],
            },
            {
                "title": "Preliminaries and Definitions",
                "key_points": ["Notation", "Formal definitions"],
            },
            {
                "title": "Main Theorems",
                "key_points": ["Theorem statements", "Proof sketches"],
            },
            {
                "title": "Proofs",
                "key_points": ["Detailed proofs", "Lemma usage"],
            },
            {
                "title": "Applications and Implications",
                "key_points": ["Practical impact", "Broader insights"],
            },
            {
                "title": "Conclusion",
                "key_points": ["Summary", "Open questions"],
            },
        ]

    # ----------------------------- BENCHMARK ---------------------------------- #
    elif paper_type == "benchmark":
        return [
            {
                "title": "Introduction",
                "key_points": [
                    f"Need for benchmarking in <TOPIC>",
                    "Paper contributions",
                ],
            },
            {
                "title": "Benchmark Design",
                "key_points": ["Dataset description", "Task definition"],
            },
            {
                "title": "Evaluation Protocol",
                "key_points": ["Metrics", "Baselines"],
            },
            {
                "title": "Experimental Results",
                "key_points": ["Model performance", "Leaderboard"],
            },
            {
                "title": "Analysis and Insights",
                "key_points": ["Performance patterns", "Limitations"],
            },
            {
                "title": "Conclusion",
                "key_points": ["Summary of benchmark", "Future work"],
            },
        ]

    # ------------------------------ DATASET ----------------------------------- #
    elif paper_type == "dataset":
        return [
            {
                "title": "Introduction",
                "key_points": [
                    f"Motivation for creating a <TOPIC> dataset",
                    "Target use cases",
                ],
            },
            {
                "title": "Data Collection",
                "key_points": ["Source selection", "Collection methodology"],
            },
            {
                "title": "Annotation Process",
                "key_points": ["Guidelines", "Annotator details", "IAA"],
            },
            {
                "title": "Dataset Statistics and Properties",
                "key_points": ["Size and distribution", "Characteristics"],
            },
            {
                "title": "Baselines and Benchmarks",
                "key_points": ["Baseline models", "Evaluation results"],
            },
            {
                "title": "Conclusion",
                "key_points": ["Dataset summary", "Future extensions"],
            },
        ]

    # ------------------------------------------------------------------------- #
    else:
        return {}


def get_intent_classification_prompt(
    query, format_instructions: str
) -> ChatPromptTemplate:
    """
    Generates a prompt template for classifying the intent of a user query.

    Args:
        format_instructions: Instructions for the expected output format (e.g., Pydantic schema).

    Returns:
        A ChatPromptTemplate object configured for intent classification.
    """
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=f"""You are an expert in classifying user queries for academic research purposes.
Your task is to analyze the given user query and extract the following information:

1.  **Research Domain**: Identify the broad academic field the query falls into.
    Examples: Computer Science, Medicine, Physics, Sociology, History, Linguistics.
    Be as specific as reasonably possible (e.g., "Machine Learning" if clearly indicated within Computer Science, otherwise "Computer Science").

2.  **Query Type**: Determine the type of information or paper the user is likely seeking.
    You MUST choose one of the following predefined types:
    `survey, method, application, analysis, position, theory, benchmark, dataset, OTHER`.
    If none of the specific types fit well, use `OTHER`.

3.  **Research Topic**: Pinpoint the specific subject, concept, or entities at the core of the query.
    This should be a concise phrase representing the main focus. For example, if the query is "latest advancements in using LLMs for code generation", the topic could be "LLMs for code generation".

Strictly adhere to the output schema provided below. Return *only* the JSON object and nothing else.

Output Schema Instructions:
{format_instructions}"""
            ),
            HumanMessage(content=f"User Query: {query}"),
        ]
    )


def get_language_detection_prompt(
    query, format_instructions: str
) -> ChatPromptTemplate:
    """
    Generates a prompt template for detecting language and translating if necessary.

    Args:
        format_instructions: Instructions for the expected output format (e.g., Pydantic schema).

    Returns:
        A ChatPromptTemplate object configured for language detection and translation.
    """
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=f"""You are a language detection and translation expert.
Analyze the provided query to:
1. Determine its language.
2. If not English, translate it to English preserving all meaning.
3. If already in English, return it unchanged in the 'translated_query' field.

Output Schema Instructions:
{format_instructions}

Provide the analysis strictly following the schema. Return *only* the JSON object."""
            ),
            HumanMessage(
                content=f"User Query: {query}"
            ),  # Placeholder for the user query
        ]
    )



def get_query_rewrite_prompt(
    query, research_domain, query_type, format_instructions: str
) -> ChatPromptTemplate:
    """
    Generates a prompt template for evaluating and potentially rewriting a user query.

    Args:
        format_instructions: Instructions for the expected output format (e.g., Pydantic schema).

    Returns:
        A ChatPromptTemplate object configured for query rewriting.
    """
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=f"""You are a query rewriting expert.
Evaluate the query and determine if it needs rewriting by checking for:
1. Semantic clarity issues
2. Ambiguity
3. Contextual fit for search/research scenarios
4. Overly complex or verbose phrasing

If rewriting is needed, create a version that:
- Maintains the same semantic meaning
- Is more precise and concise
- Is better suited for search/research purposes

Output Schema Instructions:
{format_instructions}

Provide the analysis strictly following the schema. Return *only* the JSON object."""
            ),
            HumanMessage(
                content=f"""
Query: {query}
Research domain: {research_domain}
Query type: {query_type}

Evaluate and rewrite the query if necessary based on the context provided.
"""
            ),
        ]
    )

def get_general_query_rewrite_prompt(
    query: str, format_instructions: str
) -> ChatPromptTemplate:
    """
    Generates a prompt template for evaluating and potentially rewriting a general (non-academic) user query.
    This function handles general queries that don't require academic research focus.

    Args:
        query: The user query to evaluate and potentially rewrite.
        format_instructions: Instructions for the expected output format (e.g., Pydantic schema).

    Returns:
        A ChatPromptTemplate object configured for general query rewriting.
    """
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=f"""You are a query optimization expert specializing in improving general information queries.
Evaluate the provided query and determine if it needs rewriting by checking for:
1. Clarity and specificity - Is the query clear about what information is being sought?
2. Language precision - Are the terms used appropriate and specific enough?
3. Completeness - Does the query contain enough context for effective information retrieval?
4. Redundancy or verbosity - Can the query be made more concise without losing meaning?

If rewriting is needed, create a version that:
- Maintains the original intent and language (preserve Chinese if the query is in Chinese)
- Improves clarity and specificity for better search results
- Uses appropriate terminology for the topic domain
- Is concise but comprehensive

Important:
- Preserve the original language of the query (do NOT translate)
- Focus on improving searchability and clarity
- Maintain the general/informational nature of the query

Output Schema Instructions:
{format_instructions}

Provide the analysis strictly following the schema. Return *only* the JSON object."""
            ),
            HumanMessage(
                content=f"""
Query to evaluate: {query}

Evaluate and rewrite this general query if necessary. Remember to:
1. Keep the same language as the original query
2. Improve clarity and searchability
3. Maintain the general information focus
4. Make it more specific and actionable if needed
"""
            ),
        ]
    )


def get_subsection_intro_content_prompt(
    paper_title: str,
    user_query: str,
    subsection_name: str,
    key_points: list,
    key_points_content: list,
) -> ChatPromptTemplate:
    """
    Generates a prompt template for creating introductory content for a subsection header
    based on its key points and their content.

    Args:
        paper_title: The title of the paper.
        user_query: The original user query.
        subsection_name: The name of the subsection.
        key_points: List of key point titles under this subsection.
        key_points_content: List of content texts for each key point.

    Returns:
        A ChatPromptTemplate object configured for generating subsection introductions.
    """
    # Format key points and their content for better prompt readability
    key_points_info = "\n\n".join([f"Key Point: {kp}" for kp in key_points])

    # Include brief previews of the content (first 200 chars) to inform the introduction
    content_previews = "\n\n".join(
        [
            f"Content Preview for '{kp}': {content[:4000]}..."
            for kp, content in zip(key_points, key_points_content)
            if content
        ]
    )

    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are an expert academic writer. Your task is to write a concise introductory paragraph for a subsection of a research paper. This paragraph should:
1. Appear at the beginning of the subsection, before individual key points are discussed
2. Provide a smooth introduction to the topics covered in the key points
3. Briefly outline what readers should expect in this subsection
4. Create logical flow and connections between the subsection title and its key points
5. Maintain scholarly tone appropriate for academic writing

Write 2-4 sentences that effectively introduce this subsection, creating coherence across the topics that follow. Do NOT simply list the key points. Instead, synthesize their themes into a cohesive introduction."""
            ),
            HumanMessage(
                content=f"""Paper Title: {paper_title}
User Query: {user_query}
Subsection Name: {subsection_name}

The subsection contains these key points:
{key_points_info}

Content information for these key points:
{content_previews}

Based on this information, write a concise introductory paragraph for the subsection '{subsection_name}'
that will appear before the individual key points are presented.
"""
            ),
        ]
    )


def get_section_summary_intro_prompt(
    paper_title: str, user_query: str, section_name: str, key_points_info: str
) -> ChatPromptTemplate:
    """
    Generates a prompt template for creating introductory content for a section header
    based on its key points.

    Args:
        paper_title: The title of the paper.
        user_query: The original user query.
        section_name: The name of the section.
        key_points_info: A string containing summarized information from the section's key points.

    Returns:
        A ChatPromptTemplate object configured for generating section introductions.
    """
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are an expert academic writer. Your task is to write a concise introductory paragraph for a section of a research paper. This paragraph should:
1. Appear at the beginning of the section, before individual key points are discussed.
2. Provide a smooth introduction to the topics covered in the key points.
3. Briefly outline what readers should expect in this section.
4. Create logical flow and connections between the section title and its key points.
5. Maintain a scholarly tone appropriate for academic writing.

Write 2-4 sentences that effectively introduce this section, creating coherence across the topics that follow. Do NOT simply list the key points. Instead, synthesize their themes into a cohesive introduction."""
            ),
            HumanMessage(
                content=f"""
Paper Title: {paper_title}
User Query: {user_query}
Section Name: {section_name}

Key points to be covered in this section:
{key_points_info}

Based on this information, write a concise introductory paragraph for the section '{section_name}'
that will appear before the individual key points are presented.
"""
            ),
        ]
    )


def get_conclusion_section_content_prompt(
    paper_title: str, user_query: str, section_title: str, section_summaries: list
) -> ChatPromptTemplate:
    """
    Generates a prompt template for creating content for a conclusion-related section.

    Args:
        paper_title: The title of the paper.
        user_query: The original user query.
        section_title: The title of the conclusion section to generate.
        section_summaries: Summaries of all non-conclusion sections of the paper.

    Returns:
        A ChatPromptTemplate object configured for generating conclusion section content.
    """
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are an academic writing expert. Generate content for a specific conclusion-related
                section of a research paper. The content should:
                1. Be directly relevant to the section title
                2. Reference key findings and themes from the paper's sections
                3. Maintain scholarly tone appropriate for academic publication
                4. Be detailed and comprehensive (300-500 words)
                5. Follow logical organization and flow
                6. Connect well with the paper's overall narrative"""
            ),
            HumanMessage(
                content=f"""
Paper Title: {paper_title}
User Query: {user_query}
Section Title: {section_title}

Paper Structure and Content:
{json.dumps(section_summaries, indent=2, ensure_ascii=False)}

Based on this information, generate comprehensive content for the "{section_title}" section that:
- Synthesizes key findings from the paper's main sections
- Aligns with the specific focus suggested by the section title
- Provides appropriate closure or discussion as expected in this type of section
- Maintains academic rigor and depth
"""
            ),
        ]
    )


def get_abstract_prompt(
    paper_title: str, user_query: str, outline: dict, summaries: list
) -> ChatPromptTemplate:
    """
    Generates a prompt template for creating the paper's abstract.

    Args:
        paper_title: The title of the paper.
        user_query: The original user query.
        outline: The structured outline of the paper.
        summaries: A list of summaries for the paper's sections.

    Returns:
        A ChatPromptTemplate object configured for abstract generation.
    """
    # Using json.dumps for better formatting of complex data structures in the prompt
    outline_str = json.dumps(outline, indent=2, ensure_ascii=False)
    summaries_str = json.dumps(summaries, indent=2, ensure_ascii=False)

    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are an academic writing specialist responsible for creating abstracts for research papers.
Generate a comprehensive academic abstract that concisely represents the paper's content.
The abstract should:
1. Be approximately 200-250 words in one paragraph
2. Clearly state the paper's purpose, scope, and research question/objective
3. Mention key methodologies, approaches, or theoretical frameworks used
4. Highlight the most significant findings, contributions, or results
5. Indicate the implications, significance, or practical applications of the work
6. Maintain academic tone and language appropriate for scholarly publication
7. Use present tense for established facts and general statements, past tense for specific study findings
8. Follow the standard abstract structure: Background/Problem → Methods/Approach → Results/Findings → Conclusions/Implications
9. Avoid citations, abbreviations without definitions, and overly technical jargon
10. Ensure the abstract can stand alone as a complete summary

Generate ONLY the abstract content without any title or heading. Do not include "Abstract:" or any other label."""
            ),
            HumanMessage(
                content=f"""
Paper Title: {paper_title}
User Query: {user_query}

Paper Outline:
{outline_str}

Section Summaries:
{summaries_str}

Based on this information, generate the abstract content for this research paper without any title or heading.
"""
            ),
        ]
    )


def get_conclusion_prompt(
    paper_title: str, user_query: str, abstract: str, outline: dict, summaries: list
) -> ChatPromptTemplate:
    """
    Generates a prompt template for creating the paper's conclusion.

    Args:
        paper_title: The title of the paper.
        user_query: The original user query.
        abstract: The generated abstract of the paper.
        outline: The structured outline of the paper.
        summaries: A list of summaries for the paper's sections.

    Returns:
        A ChatPromptTemplate object configured for conclusion generation.
    """
    # Using json.dumps for better formatting
    outline_str = json.dumps(outline, indent=2, ensure_ascii=False)
    summaries_str = json.dumps(summaries, indent=2, ensure_ascii=False)
    abstract_str = abstract if abstract else "No abstract was generated or provided."

    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are an academic writing specialist responsible for crafting compelling conclusions for research papers.
Generate a comprehensive and insightful academic conclusion that effectively synthesizes the paper's main points and offers a forward-looking perspective.
The conclusion should:
1.  Be appropriately sized for an academic paper, typically 300-500 words, presented in one or at most two well-structured paragraphs.
2.  Begin by briefly restating the paper's main objectives or research questions.
3.  Succinctly summarize the most significant findings and contributions, drawing directly from the provided section summaries and abstract.
4.  Clearly articulate how these findings address the initial objectives or research questions.
5.  Discuss the broader implications and significance of the work. What is the impact of these findings on the field?
6.  Acknowledge any important limitations of the study or unresolved questions.
7.  Suggest specific and relevant directions for future research that build upon the current work.
8.  Conclude with a strong, memorable closing statement that reinforces the overall importance and contribution of the research.
9.  Maintain a formal academic tone and precise language suitable for scholarly publication.
10. Ensure the conclusion flows logically and provides a sense of closure to the paper.

Generate ONLY the conclusion content. Do not include any title, heading, or labels such as "Conclusion:"."""
            ),
            HumanMessage(
                content=f"""
Paper Title: {paper_title}
User Query (for overall context): {user_query}
Paper Abstract:
{abstract_str}

Paper Outline:
{outline_str}

Section Summaries:
{summaries_str}

Based on all the provided information (title, user query, abstract, outline, and section summaries), please generate the conclusion content for this research paper.
Ensure it synthesizes the key aspects, discusses implications, limitations, and future work, and ends with a strong closing statement.
Adhere strictly to the requirements outlined in the system message, especially regarding content, length, and formatting (no title/heading).
"""
            ),
        ]
    )


def get_abstract_conclusion_evaluation_prompt(
    paper_title: str,
    user_query: str,
    abstract: str,
    conclusion: str,
    summaries: dict,  # Assuming summaries might be a dict {section_name: summary} here
    format_instructions: str,
) -> ChatPromptTemplate:
    """
    Generates a prompt template for evaluating the generated abstract and conclusion.

    Args:
        paper_title: The title of the paper.
        user_query: The original user query.
        abstract: The generated abstract.
        conclusion: The generated conclusion.
        summaries: A dictionary containing section summaries.
        format_instructions: Instructions for the expected output format (e.g., Pydantic schema).

    Returns:
        A ChatPromptTemplate object configured for evaluating the abstract and conclusion.
    """
    summaries_str = json.dumps(summaries, indent=2, ensure_ascii=False)
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are an academic editor evaluating the abstract and conclusion of a research paper.
Assess whether they meet academic standards, considering:
- Comprehensiveness: Do they cover the key aspects of the paper?
- Coherence: Do they present a cohesive narrative?
- Balance: Do they give appropriate attention to all major paper components?
- Academic tone: Do they maintain scholarly language?
- Alignment: Are they consistent with the paper's structure, title, and research question?

Provide detailed feedback if improvements are needed, following the specified output schema."""
            ),
            HumanMessage(
                content=f"""
Paper Title: {paper_title}
User Query: {user_query}

Paper Outline With Summaries:
{summaries_str}

ABSTRACT:
{abstract}

CONCLUSION:
{conclusion}

Evaluate if the abstract and conclusion meet academic requirements for this paper.
Output Schema Instructions:
{format_instructions}

Please provide your evaluation strictly following the schema.
"""
            ),
        ]
    )


def get_research_field_prompt(
    user_query: str, format_instructions: str
) -> ChatPromptTemplate:
    """
    Generates a prompt template to determine the research field, paper type, and topic from a user query.

    Args:
        user_query: The user's initial query.
        format_instructions: Instructions for the expected output format (e.g., Pydantic schema).

    Returns:
        A ChatPromptTemplate object configured for research field analysis.
    """
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are an academic research expert. Analyze the user query to determine its research field, the likely type of paper requested (e.g., survey, research article, review), and the specific research topic."
            ),
            HumanMessage(
                content=f"""
User query: {user_query}

Please analyze this query and determine:
1. The primary research field it belongs to.
2. The most likely type of paper being requested.
3. The specific research topic derived from the query.

Output Schema Instructions:
{format_instructions}

Provide the analysis strictly following the schema.
"""
            ),
        ]
    )


def get_outline_generation_prompt(
    field: str,
    paper_type: str,
    topic: str,
    user_query: str,
    format_instructions: str,
    max_sections: int = 4,
    min_depth: int = 2,
    seed_outline: Optional[List[Dict]] = None,  # 新添加的参数
) -> ChatPromptTemplate:
    """
    Generates a prompt template for creating a detailed paper outline with structural constraints.

    Args:
        field: The determined research field.
        paper_type: The determined paper type.
        topic: The specific research topic.
        user_query: The original user query.
        format_instructions: Instructions for the expected output format (e.g., Pydantic schema).
        max_sections: The maximum number of top-level sections allowed.
        min_depth: The minimum required depth of the outline hierarchy (e.g., 2 means sections must have subsections).
        seed_outline: Optional reference outline template to guide the generation.

    Returns:
        A ChatPromptTemplate object configured for outline generation.
    """
    # 处理种子大纲
    seed_outline_text = ""
    if seed_outline:
        seed_outline_text = f"""

REFERENCE OUTLINE TEMPLATE:
The following is a reference outline template that you should use as guidance for structure and organization:
{json.dumps(seed_outline, indent=2, ensure_ascii=False)}

Please use this template as inspiration for:
- Section organization and logical flow
- Types of key points to include
- Appropriate depth and structure for this paper type
- Standard conventions for {paper_type} papers in {field}

Adapt the template to specifically address the topic "{topic}" and user query, ensuring all content is relevant and specific to the research focus."""

    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=f"""You are an academic paper writing expert. Please generate a detailed paper outline for the specified research topic, including ONLY title and section structure (do NOT include an abstract).
Ensure the outline follows academic writing standards with a clear structure and rigorous logic.

IMPORTANT CONSIDERATIONS:
- Tailor the outline specifically to the research field: {field}
- Structure appropriate for paper type: {paper_type}
- Include terminology and structural elements specific to <{topic}>
- Section names should reflect standard conventions in <{field}> for <{paper_type}> papers <{seed_outline_text}>

For each section's content:
- Make it specific enough to avoid generic results.
- Include technical details to capture implementation information where relevant.
- Ensure diversity to cover all aspects of the section plan (e.g., theory, methods, applications, challenges).
- Focus on content that can be supported by authoritative sources like documentation, technical blogs, and academic papers.

STRUCTURE CONSTRAINTS:
- The outline should have AT MOST <{max_sections}> top-level sections.
- The outline should have AT LEAST <{min_depth}> levels of hierarchy (i.e., sections must have subsections if min_depth >= 1).
- Each section should contain 3-5 key points that summarize the main content.

STRUCTURE COHERENCE & FLOW:
- Ensure each section contributes to a coherent narrative that progresses from background and motivation, through methodology and results, to discussion and implications.
- Design transitions between sections to be natural and context-aware: the end of one section should logically set up the beginning of the next.
- Avoid content overlap or repetition across sections and subsections; each part should serve a unique purpose in the overall paper structure.
- Each section should answer a specific research subquestion or tackle a distinct component of the user query.
- Emphasize clarity in section purpose: for example, make it explicit how the methodology follows from the problem statement, and how the evaluation supports the proposed solution.
"""),
            HumanMessage(
                content=f"""
Please generate a paper outline for the following research topic:
- Research field: {field}
- Paper type: {paper_type}
- Specific topic: {topic}
- User query: {user_query}

Output Schema Instructions:
{format_instructions}

Important: Return *only* a valid JSON object that strictly matches the schema. Do not include any text before or after the JSON.
Adhere to the structure constraints: maximum {max_sections} top-level sections and minimum hierarchy depth of {min_depth}.
Remember to arrange sections and subsections in a logical order that follows academic writing conventions for {field}.
DO NOT include an abstract in the outline - only generate the title and section structure.
"""
            ),
        ]
    )





def get_outline_generation_strick_prompt(
    field: str, paper_type: str, topic: str, user_query: str
) -> ChatPromptTemplate:
    """
    Generates a prompt template for creating a paper outline with a predefined JSON structure example.
    (Note: Less flexible than using format_instructions with Pydantic, potentially more brittle).

    Args:
        field: The determined research field.
        paper_type: The determined paper type.
        topic: The specific research topic.
        user_query: The original user query.

    Returns:
        A ChatPromptTemplate object configured for strict JSON outline generation.
    """
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are an academic paper writing expert. Please generate a detailed paper outline for the specified research topic, including title, abstract, and section structure.
Ensure the outline follows academic writing standards with a clear structure and rigorous logic. For each section's content:
- Make it specific enough to avoid generic results.
- Include technical details to capture implementation information where relevant.
- Ensure diversity to cover all aspects of the section plan (e.g., theory, methods, applications, challenges).
- Focus on content that can be supported by authoritative sources like documentation, technical blogs, and academic papers."""
            ),
            HumanMessage(
                content=f"""
Please generate a paper outline for the following research topic:
- Research field: {field}
- Paper type: {paper_type}
- Specific topic: {topic}
- User query: {user_query}

The outline should have the following structure:

```json
{{
    "title": "Your Paper Title Here",
    "abstract": "Brief abstract describing the paper...",
    "sections": [
    {{
        "title": "Introduction",
        "key_point": [
        "Point 1 about the introduction",
        "Point 2 about the introduction"
        ]
    }},
    // More sections...
    {{
        "title": "Method",
        "key_point": [
        "Point 1 about the method",
        "Point 2 about the method"
        // ... 3-5 points total
        ]
    }}
    ]
}}
```

Return only the JSON object conforming to this structure, without any explanations or additional text.
"""
            ),
        ]
    )


def get_outline_synthesis_prompt(
    outlines_json: str,
    field: str,
    paper_type: str,
    topic: str,
    format_instructions: str,
) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=f"""You are an academic writing expert. Your task is to synthesize several draft outlines into a single optimal paper outline.

Your goal is to:
- Combine the strengths of each draft: retain well-structured sections, useful terminology, and technically detailed key points.
- Ensure logical coherence and progression between sections: e.g., background → problem → methods → evaluation → discussion.
- Remove redundant or overlapping sections/subsections.
- Unify structure depth and formatting (e.g., consistent section/subsection hierarchy across all sections).
- Prioritize clarity and structure: make sure transitions between sections are natural and progressive.
- Ensure that the final outline fully covers the topic scope and is aligned with the conventions of {paper_type} papers in {field}.
- The synthesized outline should be feasible to write as a real academic paper and should avoid overly abstract or generic section titles.

You MUST return a well-structured, technically specific outline that reflects best academic practices."""
            ),
            HumanMessage(
                content=f"""
Please synthesize the following multiple paper outlines into one optimal outline:

Draft Outlines (JSON):
{outlines_json}

Context:

- Research field: {field}
- Paper type: {paper_type}
- Specific topic: {topic}

Output Schema Instructions:
{format_instructions}

IMPORTANT: Return only the final JSON object, strictly following the schema. Do NOT include any extra commentary or text.
"""
            ),
        ]
    )


def get_outline_reflection_prompt(
    user_query: str,
    field: str,
    paper_type: str,
    topic: str,
    outline_json: str,
    format_instructions: str,
) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are an academic paper writing reviewer. Your role is to critically assess the provided paper outline with a focus on its suitability for high-quality academic writing.

Your evaluation should consider the following criteria:

STRUCTURE AND LOGIC:
- Does the outline exhibit a clear and coherent structure from introduction to conclusion?
- Are the sections arranged in a logical progression (e.g., background → methodology → results → discussion)?
- Do transitions between adjacent sections make sense and show contextual continuity?
- Are there redundant or overly similar subsections that indicate structural inefficiency?
- Does each section serve a unique and non-overlapping role in advancing the paper’s argument or contribution?

CONTENT ALIGNMENT:
- Does the outline fully address the user query and match the scope of the topic?
- Are all major aspects of the paper type (e.g., problem definition, solution design, evaluation) present?
- Does each section contain specific, relevant, and academically appropriate key points (i.e., not vague or generic)?

FEASIBILITY & ACADEMIC STANDARDS:
- Is the proposed structure feasible for a standard academic paper in the specified field and format?
- Is the depth appropriate—i.e., not too shallow or overly detailed for an outline?

Return your judgment in JSON format, indicating whether the outline meets academic standards, and if not, explain why in a clear, actionable list of reasons.

Use the following output format strictly:
"""
            ),
            HumanMessage(
                content=f"""Please evaluate whether the following paper outline meets the requirements for academic paper writing:
Context:

- User query: {user_query}
- Research field: {field}
- Paper type: {paper_type}
- Specific topic: {topic}

Paper Outline (JSON):
{outline_json}

Output Schema Instructions:
{format_instructions}

IMPORTANT: Respond with only a valid JSON object that strictly follows the schema. Do NOT include any explanatory text before or after the JSON. The JSON should contain "meets_requirements" (boolean) and "reasons" (list of strings detailing issues if requirements are not met, otherwise null or empty list).
"""
            ),
        ]
    )


# def get_outline_generation_prompt_v2(
#     field: str,
#     paper_type: str,
#     topic: str,
#     user_query: str,
#     format_instructions: str,
#     max_sections: int = 4,
#     min_depth: int = 2,
#     seed_outline: Optional[List[Dict]] = None,  # 新添加的参数
# ) -> ChatPromptTemplate:
#     """
#     Generates a prompt template for creating a detailed paper outline with structural constraints.

#     Args:
#         field: The determined research field.
#         paper_type: The determined paper type.
#         topic: The specific research topic.
#         user_query: The original user query.
#         format_instructions: Instructions for the expected output format (e.g., Pydantic schema).
#         max_sections: The maximum number of top-level sections allowed.
#         min_depth: The minimum required depth of the outline hierarchy (e.g., 2 means sections must have subsections).
#         seed_outline: Optional reference outline template to guide the generation.

#     Returns:
#         A ChatPromptTemplate object configured for outline generation.
#     """
#     # 处理种子大纲
#     seed_outline_text = ""
#     if seed_outline:
#         seed_outline_text = f"""

# REFERENCE OUTLINE TEMPLATE:
# The following is a reference outline template that you should use as guidance for structure and organization:
# {json.dumps(seed_outline, indent=2, ensure_ascii=False)}

# Please use this template as inspiration for:
# - Section organization and logical flow
# - Types of key points to include
# - Appropriate depth and structure for this paper type
# - Standard conventions for {paper_type} papers in {field}

# Adapt the template to specifically address the topic "{topic}" and user query, ensuring all content is relevant and specific to the research focus."""

#     return ChatPromptTemplate.from_messages(
#         [
#             SystemMessage(content=f"""You are an expert in academic research paper structuring. Your task is to generate a high-quality, logically structured, and critically informed **survey outline** for the research topic below.

# Your output will be rigorously evaluated based on:
# - Structural Coherence & Narrative Logic
# - Conceptual Depth & Thematic Coverage
# - Critical Thinking & Scholarly Synthesis

# Please adhere to the following principles when constructing the outline:

# [STRUCTURE & FLOW]
# - Begin with **background and motivation**, and progress through **key themes**, **methodological discussions**, **challenges or debates**, and finally **future directions or open questions**.
# - Ensure smooth and logical transitions between sections: each section must clearly build upon the previous one.
# - Avoid redundancy. Each section and subsection must serve a unique role in the overall narrative arc of the paper.
# - Explicitly define the conceptual function of each section (e.g., literature synthesis, methods comparison, emerging trends).

# [THEMATIC & CONCEPTUAL COVERAGE]
# - The outline must **comprehensively address all major concepts, techniques, or debates** relevant to the topic.
# - Avoid overly generic titles or key points. Focus on **domain-specific**, technically precise formulations.
# - Ensure inclusion of both **foundational theories** and **recent advancements**, where appropriate.

# [CRITICAL DEPTH & SYNTHESIS]
# - At least one subsection per section should:
#   - Identify **research gaps**, **methodological limitations**, or **contradictions** in existing work.
#   - Propose **unifying perspectives**, **emerging frameworks**, or **open research questions**.

# [STRUCTURE CONSTRAINTS]
# - At most {max_sections} top-level sections.
# - At least {min_depth} levels of hierarchy.
# - Each section must contain 2–4 key points, each representing a concise yet substantial contribution to the section's purpose.

# {seed_outline_text}
# """),
#             HumanMessage(
#                 content=f"""
# Please generate a paper outline for the following research topic:
# - Research field: {field}
# - Paper type: {paper_type}
# - Specific topic: {topic}
# - User query: {user_query}

# Output Schema Instructions:
# {format_instructions}

# Important: Return *only* a valid JSON object that strictly matches the schema. Do not include any text before or after the JSON.
# Remember to arrange sections and subsections in a logical order that follows academic writing conventions for {field}.
# DO NOT include an abstract in the outline - only generate the title and section structure.
# """
#             ),
#         ]
#     )

# def get_outline_reflection_prompt_v2(
#     user_query: str,
#     field: str,
#     paper_type: str,
#     topic: str,
#     outline_json: str,
#     format_instructions: str,
# ) -> ChatPromptTemplate:
#     return ChatPromptTemplate.from_messages(
#         [
#             SystemMessage(
#     content="""You are a critical academic reviewer specializing in scholarly paper structure and quality. Your task is to evaluate a generated paper outline for its academic rigor, conceptual completeness, and logical coherence.

# Please assess the outline according to the following key criteria:

# ---

# [STRUCTURAL COHERENCE & NARRATIVE LOGIC]
# - Does the outline follow a logical academic structure from introduction to conclusion?
# - Is there a clear progression between sections (e.g., motivation → problem → methodology → evaluation → discussion)?
# - Do transitions between adjacent sections show narrative continuity, or do they appear abrupt or disconnected?
# - Are there any redundant, overlapping, or overly similar sections or subsections?
# - Does each section serve a **distinct logical purpose** within the overall flow of the paper?

# ---

# [CONCEPTUAL DEPTH & THEMATIC COVERAGE]
# - Does the outline comprehensively address all **major themes, methods, or perspectives** relevant to the topic?
# - Are any key subtopics or canonical approaches **missing or underdeveloped**?
# - Do the key points demonstrate **field-specific knowledge**, or are they overly vague and general?
# - Does the outline reflect **awareness of current developments, debates, or challenges** in the research area?

# ---

# [CRITICAL THINKING & SCHOLARLY SYNTHESIS]
# - Does the outline incorporate **critical reflection**, such as highlighting research gaps, tensions, or unresolved questions?
# - Does it attempt to synthesize or contrast competing approaches or perspectives?
# - Are at least some sections structured to facilitate **argumentation**, **evaluation**, or **integration of evidence** rather than pure description?

# ---

# Return your judgment as a JSON object that clearly states:
# - Whether the outline meets academic expectations ("meets_requirements": true/false)
# - If not, a list of clear, **actionable**, and **constructive** reasons explaining the deficiencies

# Output must strictly follow the provided JSON schema:
# """
# ),
#             HumanMessage(
#                 content=f"""Please evaluate whether the following paper outline meets the requirements for academic paper writing:
# Context:

# - User query: {user_query}
# - Research field: {field}
# - Paper type: {paper_type}
# - Specific topic: {topic}

# Paper Outline (JSON):
# {outline_json}

# Output Schema Instructions:
# {format_instructions}

# IMPORTANT: Respond with only a valid JSON object that strictly follows the schema. Do NOT include any explanatory text before or after the JSON. The JSON should contain "meets_requirements" (boolean) and "reasons" (list of strings detailing issues if requirements are not met, otherwise null or empty list).
# """
#             ),
#         ]
#     )


# def get_outline_improve_prompt_v2(
#     user_query: str,
#     field: str,
#     paper_type: str,
#     topic: str,
#     outline: str,  # JSON string
#     improvement_feedback: str,
#     schema: str,
# ) -> ChatPromptTemplate:
#     return ChatPromptTemplate.from_messages(
#         [
#             SystemMessage(
#                 content=f"""You are an expert academic editor. Your task is to revise a survey paper outline based on detailed evaluation feedback.

# Your revision **must address weaknesses** identified across the following core dimensions:

# 1. **Structural Coherence & Narrative Logic**
#    - Ensure clear logical progression across sections and subsections.
#    - Fix any imbalances in depth or breadth between parts.
#    - Add smooth and purposeful transitions (e.g., from background → challenges → approaches → synthesis → future work).
#    - Remove structural redundancy or disconnected parts.

# 2. **Conceptual Depth & Thematic Coverage**
#    - Expand or adjust the outline to ensure comprehensive and balanced coverage of key concepts, theories, and subfields relevant to the topic.
#    - Avoid overfitting to niche topics at the expense of foundational ideas.
#    - Integrate historical evolution and current state-of-the-art where appropriate.

# 3. **Critical Thinking & Scholarly Synthesis**
#    - Make space for analysis of methodological debates, knowledge gaps, and unresolved questions.
#    - Ensure synthesis of perspectives across schools of thought.
#    - Highlight how sections interact to form a coherent scholarly argument, not a list of items.

# Additional Requirements:
# - Follow academic norms for a {paper_type} in the field of {field}.
# - Ensure consistency in terminology and depth across sections.
# - Clarify any vague section titles or poorly scoped parts.
# - All content must stay tightly aligned with the topic: "{topic}" and the original user query.

# Your final output must strictly follow the provided JSON schema and **must not include any explanations or extra text**."""
#             ),
#             HumanMessage(
#                 content=f"""
# Please revise and improve the following paper outline based on critical feedback:

# Context:
# - User query: {user_query}
# - Research field: {field}
# - Paper type: {paper_type}
# - Specific topic: {topic}

# Current Outline (JSON):
# {outline}

# Feedback / Areas for Improvement:
# {improvement_feedback}

# Output Schema Instructions:
# {schema}

# IMPORTANT:
# - Fully implement all improvement points from the feedback.
# - Do not remove essential ideas unless redundancy or irrelevance is clearly indicated.
# - Return a valid JSON object only, strictly matching the schema.
# """
#             ),
#         ]
#     )

def get_outline_generation_prompt_v2(
    field: str,
    paper_type: str,
    topic: str,
    user_query: str,
    format_instructions: str,
    max_sections: int = 4,
    min_depth: int = 2,
    seed_outline: Optional[List[Dict]] = None,
) -> ChatPromptTemplate:
    """
    Generates a prompt template for creating a detailed paper outline with structural constraints.

    Args:
        field: The determined research field.
        paper_type: The determined paper type.
        topic: The specific research topic.
        user_query: The original user query.
        format_instructions: Instructions for the expected output format (e.g., Pydantic schema).
        max_sections: The maximum number of top-level sections allowed.
        min_depth: The minimum required depth of the outline hierarchy (e.g., 2 means sections must have subsections).
        seed_outline: Optional reference outline template to guide the generation.

    Returns:
        A ChatPromptTemplate object configured for outline generation.
    """
    # 处理种子大纲
    seed_outline_text = ""
    if seed_outline:
        seed_outline_text = f"""

REFERENCE OUTLINE TEMPLATE:
The following is a reference outline template that you should use as guidance for structure and organization:
{json.dumps(seed_outline, indent=2, ensure_ascii=False)}

Please use this template as inspiration for:
- Section organization and logical flow
- Types of key points to include
- Appropriate depth and structure for this paper type
- Standard conventions for {paper_type} papers in {field}

Adapt the template to specifically address the topic "{topic}" and user query, ensuring all content is relevant and specific to the research focus."""

    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=f"""You are an expert in academic research paper structuring. Your task is to generate a high-quality, logically structured, and critically informed **survey outline** for the research topic below.

Your output will be rigorously evaluated based on:
- Structural Coherence & Narrative Logic
- Conceptual Depth & Thematic Coverage
- Critical Thinking & Scholarly Synthesis

Please adhere to the following principles when constructing the outline:

[STRUCTURE & FLOW]
- Begin with **background and motivation**, and progress through **key themes**, **methodological discussions**, **challenges or debates**, and finally **future directions or open questions**.
- Ensure smooth and logical transitions between sections: each section must clearly build upon the previous one.
- Avoid redundancy. Each section and subsection must serve a unique role in the overall narrative arc of the paper.
- Explicitly define the conceptual function of each section (e.g., literature synthesis, methods comparison, emerging trends).

[THEMATIC & CONCEPTUAL COVERAGE]
- The outline must **comprehensively address all major concepts, techniques, or debates** relevant to the topic.
- Avoid overly generic titles or key points. Focus on **domain-specific**, technically precise formulations.
- Ensure inclusion of both **foundational theories** and **recent advancements**, where appropriate.

[CRITICAL DEPTH & SYNTHESIS]
- At least one subsection per section should:
  - Identify **research gaps**, **methodological limitations**, or **contradictions** in existing work.
  - Propose **unifying perspectives**, **emerging frameworks**, or **open research questions**.

[STRUCTURE CONSTRAINTS - STRICTLY ENFORCED]
**CRITICAL: These constraints are MANDATORY and CANNOT be violated:**

1. **Maximum Sections Constraint**:
   - You MUST generate EXACTLY {max_sections} top-level sections or fewer
   - Count carefully: Introduction, Method, Results, Discussion, etc. each count as ONE section
   - If you exceed {max_sections} sections, the outline will be REJECTED

2. **Minimum Depth Constraint**:
   - You MUST ensure the outline has AT LEAST {min_depth} levels of hierarchy
   - Level 1 = Sections (e.g., "Introduction")
   - Level 2 = Subsections (e.g., "Background and Motivation")
   - Level 3 = Sub-subsections (if min_depth >= 3)
   - If min_depth = 2: ALL sections must have subsections
   - If min_depth = 3: At least some subsections must have sub-subsections

3. **Key Points Constraint**:
   - Each section/subsection must contain EXACTLY 2-4 key points
   - Each key point should represent a concise yet substantial contribution
   - Key points must be specific and actionable, not generic placeholders

**VALIDATION CHECKLIST BEFORE SUBMISSION:**
□ Total top-level sections ≤ {max_sections}
□ Hierarchy depth ≥ {min_depth} levels
□ Each section has 2-4 meaningful key points
□ No section lacks the required subsection structure (if min_depth ≥ 2)

{seed_outline_text}
"""),
            HumanMessage(
                content=f"""
Please generate a paper outline for the following research topic:
- Research field: {field}
- Paper type: {paper_type}
- Specific topic: {topic}
- User query: {user_query}

**CRITICAL CONSTRAINTS (MUST BE FOLLOWED):**
- Maximum {max_sections} top-level sections (count them!)
- Minimum {min_depth} levels of hierarchy (verify depth!)
- Each section/subsection must have 2-4 key points

Output Schema Instructions:
{format_instructions}

**IMPORTANT:**
1. Before generating, plan your section count to ensure ≤ {max_sections}
2. Verify hierarchy depth meets ≥ {min_depth} requirement
3. Return ONLY a valid JSON object that strictly matches the schema
4. DO NOT include an abstract in the outline - only generate the title and section structure
5. Each key point must be substantial and specific to the topic

**Final Check:** Count your sections and verify hierarchy depth before submitting!
"""
            ),
        ]
    )


def get_outline_reflection_prompt_v2(
    user_query: str,
    field: str,
    paper_type: str,
    topic: str,
    outline_json: str,
    format_instructions: str,
    max_sections: int = 4,  # 新增参数
    min_depth: int = 2,     # 新增参数
) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=f"""You are a critical academic reviewer specializing in scholarly paper structure and quality. Your task is to evaluate a generated paper outline for its academic rigor, conceptual completeness, and logical coherence.

Please assess the outline according to the following key criteria:

---

[STRUCTURAL CONSTRAINTS VALIDATION - CRITICAL]
**These constraints are MANDATORY and must be strictly enforced:**

1. **Section Count Validation**:
   - The outline MUST have no more than {max_sections} top-level sections
   - Count each main section (e.g., Introduction, Methods, Results, etc.)
   - If > {max_sections} sections detected → AUTOMATIC FAILURE

2. **Hierarchy Depth Validation**:
   - The outline MUST have at least {min_depth} levels of hierarchy
   - Level 1 = Main sections, Level 2 = Subsections, Level 3+ = Sub-subsections
   - If depth < {min_depth} → AUTOMATIC FAILURE
   - Verify that required subsection structure exists

3. **Key Points Validation**:
   - Each section/subsection must contain 2-4 key points
   - Key points must be specific and substantial, not generic
   - Empty or poorly defined key points → FAILURE

---

[STRUCTURAL COHERENCE & NARRATIVE LOGIC]
- Does the outline follow a logical academic structure from introduction to conclusion?
- Is there a clear progression between sections (e.g., motivation → problem → methodology → evaluation → discussion)?
- Do transitions between adjacent sections show narrative continuity, or do they appear abrupt or disconnected?
- Are there any redundant, overlapping, or overly similar sections or subsections?
- Does each section serve a **distinct logical purpose** within the overall flow of the paper?

---

[CONCEPTUAL DEPTH & THEMATIC COVERAGE]
- Does the outline comprehensively address all **major themes, methods, or perspectives** relevant to the topic?
- Are any key subtopics or canonical approaches **missing or underdeveloped**?
- Do the key points demonstrate **field-specific knowledge**, or are they overly vague and general?
- Does the outline reflect **awareness of current developments, debates, or challenges** in the research area?

---

[CRITICAL THINKING & SCHOLARLY SYNTHESIS]
- Does the outline incorporate **critical reflection**, such as highlighting research gaps, tensions, or unresolved questions?
- Does it attempt to synthesize or contrast competing approaches or perspectives?
- Are at least some sections structured to facilitate **argumentation**, **evaluation**, or **integration of evidence** rather than pure description?

---

**EVALUATION PRIORITY ORDER:**
1. FIRST: Check structural constraints ({max_sections} sections max, {min_depth} depth min)
2. SECOND: Assess academic quality and coherence
3. THIRD: Evaluate conceptual depth and synthesis

Return your judgment as a JSON object that clearly states:
- Whether the outline meets academic expectations ("meets_requirements": true/false)
- If not, a list of clear, **actionable**, and **constructive** reasons explaining the deficiencies
- **Structural constraint violations must be listed first and marked as critical**

Output must strictly follow the provided JSON schema:
"""
            ),
            HumanMessage(
                content=f"""Please evaluate whether the following paper outline meets the requirements for academic paper writing:

**STRUCTURAL CONSTRAINTS TO VALIDATE:**
- Maximum {max_sections} top-level sections allowed
- Minimum {min_depth} levels of hierarchy required
- Each section must have 2-4 key points

Context:
- User query: {user_query}
- Research field: {field}
- Paper type: {paper_type}
- Specific topic: {topic}

Paper Outline (JSON):
{outline_json}

**CRITICAL:** First verify structural constraints, then assess academic quality.

Output Schema Instructions:
{format_instructions}

IMPORTANT: Respond with only a valid JSON object that strictly follows the schema. Do NOT include any explanatory text before or after the JSON. The JSON should contain "meets_requirements" (boolean) and "reasons" (list of strings detailing issues if requirements are not met, with structural violations listed first).
"""
            ),
        ]
    )


def get_outline_improve_prompt_v2(
    user_query: str,
    field: str,
    paper_type: str,
    topic: str,
    outline: str,  # JSON string
    improvement_feedback: str,
    schema: str,
    max_sections: int = 4,  # 新增参数
    min_depth: int = 2,     # 新增参数
) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=f"""You are an expert academic editor. Your task is to revise a survey paper outline based on detailed evaluation feedback.

**CRITICAL STRUCTURAL CONSTRAINTS - MUST BE RESPECTED:**

1. **Maximum Sections**: You MUST NOT exceed {max_sections} top-level sections
2. **Minimum Depth**: You MUST ensure at least {min_depth} levels of hierarchy
3. **Key Points**: Each section/subsection must have exactly 2-4 substantial key points

**These constraints are NON-NEGOTIABLE and violations will result in rejection.**

Your revision **must address weaknesses** identified across the following core dimensions:

1. **Structural Coherence & Narrative Logic**
   - Ensure clear logical progression across sections and subsections.
   - Fix any imbalances in depth or breadth between parts.
   - Add smooth and purposeful transitions (e.g., from background → challenges → approaches → synthesis → future work).
   - Remove structural redundancy or disconnected parts.
   - **MAINTAIN section count ≤ {max_sections}**

2. **Conceptual Depth & Thematic Coverage**
   - Expand or adjust the outline to ensure comprehensive and balanced coverage of key concepts, theories, and subfields relevant to the topic.
   - Avoid overfitting to niche topics at the expense of foundational ideas.
   - Integrate historical evolution and current state-of-the-art where appropriate.

3. **Critical Thinking & Scholarly Synthesis**
   - Make space for analysis of methodological debates, knowledge gaps, and unresolved questions.
   - Ensure synthesis of perspectives across schools of thought.
   - Highlight how sections interact to form a coherent scholarly argument, not a list of items.

**REVISION STRATEGY:**
- If feedback indicates too many sections: **MERGE** related sections rather than just removing content
- If feedback indicates insufficient depth: **ADD** subsections within existing sections, not new top-level sections
- If hierarchy is too shallow: **RESTRUCTURE** existing content into deeper hierarchies

Additional Requirements:
- Follow academic norms for a {paper_type} in the field of {field}.
- Ensure consistency in terminology and depth across sections.
- Clarify any vague section titles or poorly scoped parts.
- All content must stay tightly aligned with the topic: "{topic}" and the original user query.

**VALIDATION BEFORE SUBMISSION:**
□ Section count ≤ {max_sections}
□ Hierarchy depth ≥ {min_depth}
□ All sections have 2-4 key points
□ Addresses all feedback points

Your final output must strictly follow the provided JSON schema and **must not include any explanations or extra text**."""
            ),
            HumanMessage(
                content=f"""
Please revise and improve the following paper outline based on critical feedback:

**MANDATORY CONSTRAINTS:**
- Maximum {max_sections} top-level sections (currently enforced)
- Minimum {min_depth} levels of hierarchy (currently enforced)
- Each section must have 2-4 key points

Context:
- User query: {user_query}
- Research field: {field}
- Paper type: {paper_type}
- Specific topic: {topic}

Current Outline (JSON):
{outline}

Feedback / Areas for Improvement:
{improvement_feedback}

**REVISION INSTRUCTIONS:**
1. Address ALL feedback points while respecting structural constraints
2. If reducing sections: merge content intelligently, don't just delete
3. If increasing depth: restructure within existing sections
4. Maintain academic quality and logical flow

Output Schema Instructions:
{schema}

IMPORTANT:
- Fully implement all improvement points from the feedback.
- Respect the {max_sections} section limit and {min_depth} depth requirement.
- Return a valid JSON object only, strictly matching the schema.
- Count your sections before submitting!
"""
            ),
        ]
    )


def get_query_generation_prompt(
    paper_title: str, section_title: str, content_point: str
) -> ChatPromptTemplate:
    """
    Generates a prompt template for creating a targeted web search query for a specific content point.
    Args:
        paper_title: The title of the paper.
        section_title: The title of the section containing the content point.
        content_point: The specific key point or detail requiring a search query.

    Returns:
        A ChatPromptTemplate object configured for search query generation.
    """
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are an expert at crafting effective search queries for academic and technical research.
Generate ONE concise and specific web search query based on the provided paper title, section title, and content point. The query should:
1. Be highly specific to the content point.
2. FUse keywords likely to yield authoritative sources (academic papers, technical docs, reputable blogs).
3. Avoid overly broad terms.
4. Be formatted as a natural language query suitable for search engines like Google Scholar, Semantic Scholar, arXiv, etc.
5. Exclude quotation marks or special search operators unless essential. Output only the generated search query string."""
            ),
            HumanMessage(
                content=f"""
Paper Title: {paper_title}
Section: {section_title}
Content Point: {content_point}

Generate a single, optimized search query for this content point. Return only the query string.
"""
            ),
        ]
    )

def get_query_generation_prompt_v2(
    paper_title: str, section_title: str, content_point: str
) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""
You are a research assistant skilled in generating precise and context-aware search queries for academic surveys and literature reviews.

Your task is to generate **one single, well-targeted** search query that supports deeper investigation of a specific content point from a scholarly paper. The goal is to help researchers find **high-quality, thematically relevant, and critically useful** sources from platforms such as Google Scholar, Semantic Scholar, or arXiv.

[Instructions for Query Generation]
- The query must directly address the **content point**, not just general topics.
- Use terminology that reflects a **graduate-level understanding** of the field (technical terms preferred over general phrases).
- **Avoid redundancy** and do not include terms that are implied or overly broad.
- If the content point includes a methodological conflict, theoretical gap, or a proposal for future work, reflect that nuance in the query.
- The query should read as a **natural, fluent sentence fragment**, as if typed into an academic search engine.
- **Do not** include quotation marks, Boolean operators, or special characters unless essential for disambiguation.
- Output **only** the query string. No explanation or additional text.
"""
            ),
            HumanMessage(
                content=f"""
Paper Title: {paper_title}
Section: {section_title}
Content Point: {content_point}

Generate a single, optimized search query for this content point. Return only the query string.
"""
            ),
        ]
    )


def get_outline_conclusion_judge_prompt(section_title: str) -> ChatPromptTemplate:
    """
    Generates a prompt template to quickly classify if a section title likely refers
    to a Conclusion, Summary, Discussion, or References section.

    This is used to potentially skip detailed processing for these standard closing sections.

    Args:
        section_title: The section title to classify.

    Returns:
        A ChatPromptTemplate object configured for simple title classification,
        expecting a "true" or "false" string response.
    """
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are an academic structure classifier. Determine if the given section title typically represents a 'Conclusion', 'Summary', 'Discussion', or 'References' section in a standard academic paper."
            ),
            HumanMessage(
                content=f"""
Does the following section title likely belong to a CONCLUSION, SUMMARY, DISCUSSION, or REFERENCE section in an academic paper?

Section Title: "{section_title}"

Respond with the single word "true" if it likely does, otherwise respond with the single word "false".
"""
            ),
        ]
    )


def get_outline_improve_prompt(
    user_query: str,
    field: str,
    paper_type: str,
    topic: str,
    outline: str,  # Assuming outline is passed as a JSON string
    improvement_feedback: str,  # Feedback on what needs improvement
    schema: str,  # Schema instructions for the improved outline
) -> ChatPromptTemplate:
    """
    Generates a prompt template to revise and improve a paper outline based on specific feedback.

    This is typically used after an initial outline generation and reflection step identifies weaknesses.

    Args:
        user_query: The original user query.
        field: The research field.
        paper_type: The paper type.
        topic: The specific research topic.
        outline: The JSON string of the current outline needing improvement.
        improvement_feedback: Text describing the issues or areas needing revision.
        schema: Instructions for the expected output format (e.g., Pydantic schema for the revised outline).

    Returns:
        A ChatPromptTemplate object configured for outline revision, expecting a JSON output.
    """
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=f"""You are an expert academic editor. Your task is to revise a paper outline based on specific critical feedback.

Your revision should:
- Fully address each point raised in the feedback.
- Improve the logical structure and flow between sections and subsections.
- Remove any structural redundancy or overlapping content.
- Ensure that section titles and content follow {paper_type} conventions in {field}.
- Unify terminology and hierarchy depth across sections (e.g., if one section has subsections, others of similar scope should too).
- Improve clarity and completeness, ensuring that the outline can serve as a solid foundation for a full academic paper.
- Ensure smooth logical transitions between sections (e.g., from background → problem → method → evaluation).
- Align closely with the topic: "{topic}" and the user's query.

Your output must strictly follow the provided JSON schema."""
            ),
            HumanMessage(
                content=f"""
Please revise and improve the following paper outline based on critical feedback:

Context:
- User query: {user_query}
- Research field: {field}
- Paper type: {paper_type}
- Specific topic: {topic}

Current Outline (JSON):
{outline}

Feedback / Areas for Improvement:
{improvement_feedback}

Output Schema Instructions:
{schema}

IMPORTANT:
- Implement all suggested changes from the feedback.
- Return only a revised JSON object conforming to the schema.
- Do NOT include explanations, justifications, or any additional text outside the JSON.
"""
            ),
        ]
    )



def get_section_reflection_evaluation_system_prompts() -> List[str]:
    """
    Provides a list of system prompts (personas) for evaluating a draft subsection in the context of its key point.
    """

    return [

        # Persona 1: Standard Academic Reviewer
        """You are a professional academic reviewer evaluating a short section of a paper that corresponds to a specific key point in the outline.
Your task is to assess the academic quality of this section based on the following dimensions:
1. Academic Formality and Tone – Is the style formal, terminology precise, and tone appropriate for a peer-reviewed journal?
2. Clarity and Logical Flow – Are ideas expressed with clarity and logically connected?
3. Redundancy – Does each sentence add new value, or is there repetition?
4. Coverage of Key Point – Does the content fully address the intended key point?
5. Alignment – Is the content consistent with the section’s objective and overall paper structure?

Please provide a concise evaluation for each dimension (1–2 sentences), and conclude with specific improvement suggestions if needed.
Assume that references will be added later; focus on **content quality** and **academic adequacy**.
""",

        # Persona 2: Senior Professor / Mentor
        """You are a senior professor mentoring a junior researcher. You are reviewing a short subsection draft meant to elaborate a specific key point in the paper.
Assess the draft based on:
1. Depth and Analytical Rigor – Does the writing show thoughtful engagement with the topic and meaningful synthesis of ideas?
2. Conceptual Clarity – Are key terms and ideas well defined and well explained?
3. Technical Richness – Are models, data, or results integrated where appropriate?
4. Original Thought – Does the content offer a new angle, synthesis, or perspective?
If the writing is insufficient in any dimension, explain what is missing and suggest precise edits or external search queries that would help strengthen this subsection.""",

        # Persona 3: Critical Peer Reviewer (Journal Style)
        """You are a critical peer reviewer for a top-tier academic journal. You are asked to rigorously evaluate one short paragraph tied to a single key point in a section draft.
Your assessment should focus on:
1. Accuracy and Validity – Are the claims plausible, logically sound, and in line with the literature?
2. Argumentation Quality – Is there a clear, persuasive, and precise argument related to this key point?
3. Contribution to Section Argument – Does this content meaningfully advance the argument of the section?
4. Weaknesses and Gaps – Are there missing assumptions, unsupported assertions, or unexplained terms?

End your review with constructive, targeted feedback: what should be clarified, cut, or expanded, and why. Suggest revisions in terms of sentence restructuring, topic refocus, or improved specificity.
""",
    ]


def get_section_reflection_evaluation_system_prompts_v2() -> List[str]:
    """
    Provides a list of system prompts (personas) for evaluating a draft subsection in the context of its key point.
    """

    return [

        # Persona 1: Standard Academic Reviewer
        """You are a professional academic reviewer evaluating a short section of a paper corresponding to a specific key point in the outline.
Your task is to assess the academic quality of this section based on the following five dimensions:

1. **Academic Formality and Tone** – The writing must adhere strictly to academic standards: precise terminology, no colloquial expressions, and consistently formal tone. Even minor informality warrants deduction.
2. **Clarity and Logical Flow** – Ideas must be expressed with exceptional clarity and logical progression. Sentences should build upon each other and transitions must be seamless.
3. **Redundancy** – Each sentence must add distinct informational value. Repetition, unless structurally necessary, should be flagged.
4. **Coverage of Key Point** – The section should fully and directly address the assigned key point, neither omitting necessary content nor digressing.
5. **Alignment with Section and Paper Structure** – Content should fit naturally into the broader section and contribute meaningfully to the overall structure and goals of the paper.

Provide 1–2 concise sentences of evaluation per dimension. Conclude with specific suggestions for improvement if applicable.
Assume citations will be added later; focus solely on content quality and academic adequacy.
""",

        # Persona 2: Senior Professor / Mentor
        """You are a senior professor mentoring a junior researcher. You are reviewing a draft subsection intended to elaborate a specific key point in an academic survey.

Assess the draft rigorously on the following dimensions:

1. **Depth and Analytical Rigor** – Does the section show deep engagement, critical interpretation of literature, and an ability to synthesize perspectives beyond summary?
2. **Conceptual Clarity** – Are all key terms and ideas clearly defined and precisely explained? Are conceptual relationships made explicit?
3. **Technical Richness** – Are relevant models, data, methodologies, or empirical findings introduced appropriately and meaningfully?
4. **Original Thought** – Does the writing offer fresh perspectives, new frameworks, or insightful synthesis that adds value beyond summarization?

For any shortcomings, clearly identify what is lacking and suggest targeted edits or further research queries (e.g., “search for recent meta-analyses on X”) to guide improvement.
""",

        # Persona 3: Critical Peer Reviewer (Journal Style)
        """You are a critical peer reviewer for a top-tier academic journal. You are evaluating a paragraph-length subsection that aims to develop a specific key point.

Your review should assess the paragraph according to:

1. **Accuracy and Validity** – Are the claims factually accurate, methodologically sound, and supported by existing scholarly consensus?
2. **Argumentation Quality** – Is the argument clearly articulated, internally consistent, and appropriately scoped for the subsection's focus?
3. **Contribution to Section Argument** – Does this paragraph advance the broader argument of the section and connect to its thematic purpose?
4. **Critical Gaps and Weaknesses** – Are there missing assumptions, oversights in logic, ambiguous terms, or unsupported generalizations?

End your review with direct, constructive feedback. Recommend what to clarify, cut, or expand—with reasoning. Offer sentence-level or conceptual restructuring suggestions where relevant.
""",
    ]




def get_section_reflection_eval_prompt(
    prompt_text: str,  # The specific system prompt/persona for this evaluation
    paper_title: str,
    user_query: str,
    section_name: str,
    parent_section: Optional[str],  # Made optional to handle top-level sections
    section_key_point: str,
    section_content: dict,  # Content related to the specific key point
    format_instructions: str,  # Schema for the evaluation output
) -> ChatPromptTemplate:
    """
    Generates a prompt template for evaluating a specific key point's content within a section during the reflection phase.

    Uses a specific evaluator persona (defined by prompt_text) to assess the quality and alignment of the generated content.

    Args:
        prompt_text: The system message defining the reviewer's persona and focus.
        paper_title: The title of the paper.
        user_query: The original user query.
        section_name: The name of the section being evaluated.
        parent_section: The name of the parent section, if applicable.
        section_key_point: The specific key point within the section being evaluated.
        section_content: The generated content corresponding to this key point.
        format_instructions: Instructions for the expected output format (e.g., Pydantic schema).

    Returns:
        A ChatPromptTemplate object configured for detailed section content evaluation, expecting JSON output.
    """
    # Ensure parent_section is handled gracefully if None or empty
    parent_section_str = parent_section if parent_section else "N/A (Top-level section)"
    # Format section content for readability in the prompt
    content_str = json.dumps(section_content, indent=2, ensure_ascii=False)

    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=prompt_text),
            HumanMessage(
                content=f"""
Paper Title: {paper_title}
User Query: {user_query}
Section Name: {section_name}
Parent Section: {parent_section_str}
Specific Key Point Being Evaluated: {section_key_point}

Generated Content for this Key Point:
{content_str}

Evaluate if this specific content adequately addresses the key point "{section_key_point}" within the context of section "{section_name}" and the overall paper "{paper_title}". Consider:
1. Relevance & Alignment: Does the content directly and accurately address the key point? Does it fit logically within the section and paper?
2. Depth & Comprehensiveness: Is the content sufficiently detailed and thorough for an academic paper? Does it cover the necessary aspects of the key point?
3. Academic Quality: Is the writing style scholarly? Is evidence presented appropriately (even if citations are placeholders)? Is the technical information accurate?
4. User Query Connection: Does this content contribute to answering the original user query?

If the content does NOT meet expectations:
- Provide specific and actionable feedback, pinpointing the exact weaknesses (e.g., "lacks technical detail on X", "explanation of Y is unclear", "doesn't connect back to the main argument").
- Then, suggest 1–2 focused, fully-formed search queries that address each specific weakness for this particular key point.
    Each query must:
    - Be standalone and precise, suitable for academic search engines.
    - Include a clear subject, relevant concepts, and context (e.g., methodology, framework, limitation).
    - Avoid vague terms like "this" or incomplete references. (e.g., "limitations of transformer-based models in long document summarization" or "empirical evaluation of attention mechanisms in multi-modal learning")

---

Output Schema Instructions:
{format_instructions}

Provide your evaluation strictly following the schema. Return *only* the JSON object.
"""
            ),
        ]
    )


def get_section_summary_intro_prompt(
    paper_title: str, user_query: str, section_name: str, key_points_info: str
) -> ChatPromptTemplate:
    """
    Generates a prompt template for creating an introductory summary paragraph for a section.

    Args:
        paper_title: The title of the paper.
        user_query: The original user query that initiated the paper generation.
        section_name: The name of the section for which the summary is being generated.
        key_points_info: A string containing summarized information from the section's key points.

    Returns:
        A ChatPromptTemplate object configured for generating section introductions.
    """
    # Note: Corrected typos in the HumanMessage content (extra quotes)
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are an expert academic writer. Your task is to write a concise introductory paragraph for a specific section of a research paper. This paragraph should be placed at the *beginning* of the section to give the reader an overview of what will be discussed. Synthesize the key information provided below from the section's key points. Focus on creating a smooth, engaging, and informative introduction. Do NOT simply list the key points. Aim for 4-8 sentences."""
            ),
            HumanMessage(
                content=f"""Paper Title: {paper_title}
User Query: {user_query}
Section Name: {section_name}

Key Points Information:
{key_points_info}

Based on the information above, please generate the introductory paragraph for the section '{section_name}'.
"""
            ),
        ]
    )


def get_section_name_refinement_prompt(
    paper_title: str, section_name: str, content_preview: str
) -> ChatPromptTemplate:
    """
    Generates a prompt template for refining a section's title based on its content.

    Args:
        paper_title: The title of the paper.
        section_name: The current title of the section.
        content_preview: A short preview or summary of the section's content.

    Returns:
        A ChatPromptTemplate object configured for section title refinement.
    """
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are an expert academic editor. Your task is to evaluate a section title for a research paper based on its content preview and the paper's overall title. The section title should be concise, clear, and accurately reflect the content. If the current title is good, return it exactly as is. If it can be improved (e.g., too vague, too long, inaccurate), generate a better title. Output *only* the final chosen or generated title, with no explanations or quotation marks."""
            ),
            HumanMessage(
                content=f"""Paper Title: {paper_title}
Current Section Title: {section_name}

Section Content Preview:
{content_preview}

Evaluate the 'Current Section Title'. If it is suitable, return it. If not, provide a concise, clear, and accurate alternative title based on the preview. Output only the final title:"""
            ),
        ]
    )





def get_section_summary_prompt_template(
    paper_title: str, section_name: str, section_text: str
) -> ChatPromptTemplate:
    """
    Generates a prompt template for creating a concise summary of a given section's text.

    Args:
        paper_title: The title of the paper.
        section_name: The name of the section to summarize.
        section_text: The full text content of the section.

    Returns:
        A ChatPromptTemplate object configured for section summarization, expecting 1-3 sentences.
    """
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
        content="""You are an academic summarizer. Your task is to create a very concise summary of the provided research paper section text.
Extract only the most crucial information or findings presented in the text.
Aim for 1-3 key sentences that capture the essence of the content."""),
            HumanMessage(
                content=f"""
Paper Title: {paper_title}
Section Name: {section_name}

Section Content to Summarize:
{section_text}

Create a concise summary capturing the absolute key points or findings of this section's content in 1-3 complete sentences.
Return exactly 1-3 complete sentences.
"""
            ),
        ]
    )


def get_section_summary_prompt_template_v2(
    paper_title: str, section_name: str, section_text: str
) -> ChatPromptTemplate:
    """
    Generates a prompt template for creating a concise summary of a given section's text.

    Args:
        paper_title: The title of the paper.
        section_name: The name of the section to summarize.
        section_text: The full text content of the section.

    Returns:
        A ChatPromptTemplate object configured for section summarization, expecting 1-3 sentences.
    """
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are an academic summarizer. Your task is to produce a highly concise and scholarly summary of the provided research paper section.
Your summary must:
- Extract only the most essential concepts, findings, or arguments.
- Be written in formal academic style with precise terminology.
- Avoid any redundancy, superficial phrasing, or generic commentary.
- Emphasize any critical insights, methodological tensions, or future-oriented claims if present.

The summary must be structured clearly, logically, and informatively, in exactly 1 to 3 full sentences.
"""
            ),
            HumanMessage(
                content=f"""
Paper Title: {paper_title}
Section Name: {section_name}

Section Content:
{section_text}

Write a formal, information-dense summary of this section in exactly 1 to 3 complete sentences. Focus on the most crucial information, arguments, or findings. Highlight any critical insights or gaps if they are discussed.
"""
            ),
        ]
    )


def get_global_reflection_eval_system() -> List[str]:
    """
    Provides a list of different system prompts (personas) for evaluating the entire paper
    structure and content during the global reflection phase.

    These prompts focus on a holistic assessment of the paper based on its outline and
    section summaries/content, checking for overall coherence, completeness, and quality.

    Returns:
        A list of strings, each a system prompt defining a global evaluator persona.
    """
    return [
        # Persona 1: Journal Editor (Holistic View)
        """You are an academic journal editor evaluating a complete research paper based on its outline and section summaries/content.
Assess the paper's overall quality and readiness for publication. Consider:
- Coherence and logical flow: Does the paper progress logically from introduction to conclusion? Are transitions between sections smooth?
- Comprehensive coverage: Does the paper adequately cover the topic defined by the title and outline? Are there obvious gaps?
- Argument strength: Is the central argument clear and well-supported throughout the sections (based on summaries/content)?
- Structural integrity: Does the paper adhere to standard academic structure? Is the balance between sections appropriate?
- Alignment: Does the paper consistently address the user query and fulfill the promise of the title and outline?
Provide detailed feedback, identifying specific sections needing improvement if the paper doesn't meet high academic standards.""",
        # Persona 2: Senior Professor (Mentorship View)
        """You are a senior professor reviewing a complete paper manuscript based on its outline and section summaries/content.
Evaluate the paper's overall quality, structure, and potential impact. Focus on:
- Organization: Is the content structured logically and effectively?
- Consistency: Are the arguments and findings presented consistently across sections?
- Completeness: Does the paper include all necessary components (intro, methods, results, discussion, conclusion)? Are sections sufficiently developed (based on summaries/content)?
- Clarity and Tone: Is the overall presentation clear and maintain a scholarly tone?
If improvements are needed, suggest specific sections requiring revision and explain why.""",
        # Persona 3: Critical Peer Reviewer (Rigour View)
        """As a peer reviewer for a prestigious academic journal, critically assess this complete paper based on its outline and section summaries/content.
Focus on:
- Overall contribution: Does the paper (as represented by summaries/content) seem to offer a significant contribution to the field?
- Methodological soundness (inferred): Does the structure suggest a sound methodology was likely followed?
- Argumentative rigor: Is the overall argument coherent and logically sound across sections?
- Literature connection (inferred): Does the structure suggest appropriate engagement with existing literature?
Provide constructive criticism, highlighting major strengths and weaknesses, and suggest specific section-level improvements required for potential publication.""",
    ]


def get_global_reflection_eval_system_v2() -> List[str]:
    """
    Provides a list of different system prompts (personas) for evaluating the entire paper
    structure and content during the global reflection phase.

    These prompts focus on a holistic assessment of the paper based on its outline and
    section summaries/content, checking for overall coherence, completeness, and quality.

    Returns:
        A list of strings, each a system prompt defining a global evaluator persona.
    """
    return [
        # Persona 1: Journal Editor (Holistic View)
        """You are an experienced academic journal editor. Based on the outline and section summaries/content of the paper, evaluate its overall quality and readiness for publication.

Strictly assess the following five aspects:
1. **Logical Coherence**: Does the paper progress clearly and logically from introduction to conclusion? Are transitions between sections smooth?
2. **Thematic Completeness**: Does the content fully cover the topic defined by the title and outline? Are all essential themes and debates included?
3. **Argument Strength**: Is there a clear central argument? Is it supported consistently across sections?
4. **Structural Integrity**: Does the structure conform to standard academic expectations (e.g., intro, methods, results, discussion)? Are section lengths balanced?
5. **Alignment with Intent**: Does the paper consistently address the research purpose stated in its title and introduction?

Provide detailed feedback across these aspects. Identify specific section-level issues and indicate if the manuscript would likely require minor revision, major revision, or rejection for publication in a top-tier journal.
""",

        # Persona 2: Senior Professor (Mentorship View)
        """You are a senior academic mentoring a junior scholar. You are reviewing a full academic paper draft via its outline and section summaries/content.

Provide constructive, in-depth evaluation on:
1. **Conceptual Organization**: Is the paper logically structured, with sections building upon one another meaningfully?
2. **Depth and Coverage**: Does the paper offer both sufficient depth in analysis and broad coverage of essential subtopics?
3. **Consistency of Thought**: Are ideas and arguments presented consistently across sections? Are there contradictions or logical leaps?
4. **Academic Tone and Readability**: Is the writing likely to be clear, scholarly, and free of redundancy?
5. **Potential Impact**: Does the paper demonstrate scholarly potential (e.g., by identifying gaps, offering insight, or proposing future research directions)?

Give mentoring-style advice, identifying strengths to preserve and areas to revise. Suggest concrete changes that would elevate the paper to a publishable standard.
""",

        # Persona 3: Critical Peer Reviewer (Rigour View)
        """You are a critical peer reviewer for a top academic journal. Evaluate this paper holistically based on its outline and section summaries/content.

Your evaluation should focus on:
1. **Scholarly Contribution**: Does the paper appear to contribute something original or important to its field?
2. **Methodological Soundness (Inferred)**: Does the structure imply that appropriate methods and analytical strategies were applied?
3. **Argumentative Rigor**: Are the arguments logically developed and connected across sections?
4. **Engagement with Literature (Inferred)**: Does the outline reflect awareness of the field’s foundational works and current debates?
5. **Readiness for Publication**: Based on its structure and content summaries, would this paper likely be publishable with minor revision, major revision, or is it fundamentally flawed?

Provide a clear rationale for your recommendation, citing strengths and weaknesses by section. Maintain a professional, objective tone.
"""
    ]


def get_global_reflection_eval_paper_prompt(
    prompt_text: str,  # The specific system prompt/persona for this evaluation
    paper_title: str,
    user_query: str,
    outline: Dict[str, Any],  # The full paper outline
    sections_data: List[
        Dict[str, Any]
    ],  # List of section data (e.g., summaries or key content)
    # format_instructions: str # Removed as JSON format is now embedded in prompt
) -> ChatPromptTemplate:
    """
    Generates a prompt template for evaluating an entire paper holistically based on its
    outline and section data (summaries or key content) during global reflection.

    Uses a specific evaluator persona and expects a structured JSON output detailing
    the assessment and required improvement actions.

    Args:
        prompt_text: The system message defining the reviewer's persona and focus.
        paper_title: The title of the paper.
        user_query: The original user query.
        outline: The complete paper outline structure.
        sections_data: A list of dictionaries, each representing a section's key information
                       (e.g., title, summary, key points).

    Returns:
        A ChatPromptTemplate object configured for holistic paper evaluation, expecting JSON output.
    """
    # Format complex data structures for better readability in the prompt
    outline_str = json.dumps(outline, indent=2, ensure_ascii=False)
    sections_data_str = json.dumps(sections_data, indent=2, ensure_ascii=False)

    # Define expected JSON format directly in the prompt for clarity and robustness
    json_format_instructions = """
Format your response as a JSON object with these exact keys:
{
    "meets_requirements": boolean, // Does the paper, as represented, meet high academic standards overall?
    "feedback": "string", // Detailed overall feedback. Explain why it meets/doesn't meet requirements. Mention strengths and weaknesses.
    "improvement_actions": [ // List of specific actions needed. Empty list if meets_requirements is true.
        {
            "section": "string", // Name of the key point needing improvement (e.g., "Algorithm Details")
            "issues": ["string", ...], // List of specific issues identified in this section (e.g., "Lacks sufficient detail", "Argument is unclear", "Doesn't connect to previous section")
            "rewrite": boolean // Should this entire section/point be rewritten (true) or just revised (false)?
        },
        ... // More actions if needed
    ]
}"""

    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=prompt_text),
            HumanMessage(
                content=f"""
Paper Title: {paper_title}
User Query: {user_query}

Paper Full Outline:
{outline_str}

Section Data (Summaries/Key Content):
{sections_data_str}

Evaluate if this paper, as represented by its outline and section data, meets high academic requirements holistically. Consider:
1. Overall Structure & Flow: Is the paper organized logically? Do sections connect well?
2. Completeness & Depth: Does the paper seem to cover the topic comprehensively based on the outline and summaries/content? Is sufficient depth indicated?
3. Coherence & Consistency: Is there a clear, consistent argument or narrative throughout?
4. Alignment: Does the paper strongly align with the title, user query, and outline?
5. Academic Standards: Does the overall structure and summarized content suggest adherence to scholarly standards?

If it doesn't meet requirements, provide specific, actionable feedback identifying the weakest sections or aspects.

{json_format_instructions}

Make sure each action has the exact fields: "section" , "issues" (as a list), and "rewrite" (boolean).
Ensure your output is *only* the JSON object conforming to this structure. Do not include any text before or after the JSON.
"""
            ),
        ]
    )


def get_issue_analysis_prompt(
    section_name: str,
    key_points: List[str],
    issues: List[str],
    format_instructions: str,  # Pydantic schema instructions
) -> ChatPromptTemplate:
    """
    Generates a prompt to analyze improvement issues against existing key points.

    Args:
        section_name: The name of the section.
        key_points: The list of existing key points in the section.
        issues: The list of improvement issues identified for the section.
        format_instructions: Instructions for the expected JSON output format.

    Returns:
        A ChatPromptTemplate object for issue analysis.
    """
    key_points_str = (
        "\n".join([f"- {kp}" for kp in key_points]) if key_points else "None"
    )
    issues_str = "\n".join([f"- {issue}" for issue in issues])

    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are an academic content analyst. Your task is to map improvement issues to the most relevant existing key points within a section. If an issue doesn't clearly relate to any existing key point, mark it for needing a new key point.
Analyze each issue provided and determine its relationship to the existing key points.
Output your analysis strictly following the provided JSON schema."""
            ),
            HumanMessage(
                content=f"""
Section Name: {section_name}

Existing Key Points:
{key_points_str}

Improvement Issues to Analyze:
{issues_str}

For each issue, determine:
1. If it strongly relates to one of the 'Existing Key Points'. Map the issue to that key point.
2. If it does NOT strongly relate to any existing key point, mark it as needing a new key point.

Output Schema Instructions:
{format_instructions}

Provide the analysis strictly following the schema. Return *only* the JSON object.
"""
            ),
        ]
    )


def get_new_key_point_generation_prompt(
    section_name: str,
    issue: str,  # The specific issue requiring a new key point
    paper_title: str,
    user_query: str,
) -> ChatPromptTemplate:
    """
    Generates a prompt to create a new key point based on an unaddressed issue.

    Args:
        section_name: The name of the section where the new key point will belong.
        issue: The specific improvement issue that needs a new key point.
        paper_title: The title of the paper.
        user_query: The original user query.

    Returns:
        A ChatPromptTemplate object for generating a new key point string.
    """
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are an academic outline developer. Based on an identified improvement issue that isn't covered by existing key points, generate a concise and actionable new key point for the paper outline.
The new key point should:
- Directly address the core of the provided issue.
- Be suitable for inclusion in the specified section.
- Be phrased clearly and concisely, similar to other key points in an outline.
Output *only* the generated key point string."""
            ),
            HumanMessage(
                content=f"""
Context:
- Paper Title: {paper_title}
- User Query: {user_query}
- Section Name: {section_name}

Issue Requiring New Key Point:
{issue}

Generate a single, concise new key point for the section '{section_name}' that specifically addresses this issue.
Return ONLY the key point string.
"""
            ),
        ]
    )



def get_enhanced_search_query_prompt(
    paper_title: str,
    user_query: str,
    section_name: str,
    key_point: str,
    original_query: Optional[str],  # Made optional as it might not always exist
    improvement_issues: List[str],
) -> ChatPromptTemplate:
    """
    Generates a prompt template for creating a new, improved search query specifically
    designed to address identified weaknesses in the content for a particular key point.

    This is used during reflection phases when content is deemed insufficient and needs
    better source material.

    Args:
        paper_title: The title of the paper.
        user_query: The original user query.
        section_name: The name of the section containing the key point.
        key_point: The specific key point needing better content.
        original_query: The previous search query used for this point (if any).
        improvement_issues: A list of strings describing the problems with the current content.

    Returns:
        A ChatPromptTemplate object configured for generating targeted search queries,
        expecting a single query string as output.
    """
    # Format the list of issues for clear presentation in the prompt
    issues_str = "\n- ".join(improvement_issues) if improvement_issues else "N/A"
    # Handle optional original query
    original_query_str = original_query if original_query else "None"

    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are a research assistant specializing in refining search strategies for academic writing.
Generate ONE highly effective and specific search query designed to find information that directly addresses the identified weaknesses in the content for a given key point.
The query should be:
1. Laser-focused on the key point AND the specific issues raised.
2. Use precise academic or technical terms relevant to the issues.
3. Formatted for optimal results in academic search engines (like Google Scholar, PubMed, arXiv).
4. Concise and clear.
Output *only* the single generated search query string."""
            ),
            HumanMessage(
                content=f"""
Context:
- Paper Title: {paper_title}
- User Query: {user_query}
- Section Name: {section_name}
- Key Point to Improve: {key_point}
- Original Search Query Used: {original_query_str}

Identified Issues with Current Content for this Key Point:
- {issues_str}

Generate ONE specific search query that will help find information to directly address these issues for this key point.
Return ONLY the search query string.
"""
            ),
        ]
    )


def get_enhanced_search_query_prompt_v2(
    paper_title: str,
    user_query: str,
    section_name: str,
    key_point: str,
    original_query: Optional[str],  # Made optional as it might not always exist
    improvement_issues: List[str],
) -> ChatPromptTemplate:
    """
    Generates a prompt template for creating a new, improved search query specifically
    designed to address identified weaknesses in the content for a particular key point.

    This is used during reflection phases when content is deemed insufficient and needs
    better source material.

    Args:
        paper_title: The title of the paper.
        user_query: The original user query.
        section_name: The name of the section containing the key point.
        key_point: The specific key point needing better content.
        original_query: The previous search query used for this point (if any).
        improvement_issues: A list of strings describing the problems with the current content.

    Returns:
        A ChatPromptTemplate object configured for generating targeted search queries,
        expecting a single query string as output.
    """
    # Format the list of issues for clear presentation in the prompt
    issues_str = "\n- ".join(improvement_issues) if improvement_issues else "N/A"
    # Handle optional original query
    original_query_str = original_query if original_query else "None"

    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are an expert research assistant tasked with refining search strategies for academic writing.
Generate ONE precise and effective search query to identify high-quality sources that directly address the identified weaknesses in the content for a specific key point. The query must:

[Instructions for Query Generation]
- Target sources that resolve the specific issues listed, focusing on the key point within the context of the paper’s topic and section.
- Use terminology that reflects a **graduate-level understanding** of the field (technical terms preferred over general phrases).
- **Avoid redundancy** and do not include terms that are implied or overly broad.
- If the content point includes a methodological conflict, theoretical gap, or a proposal for future work, reflect that nuance in the query.
- The query should read as a **natural, fluent sentence fragment**, as if typed into an academic search engine.
- **Do not** include quotation marks, Boolean operators, or special characters unless essential for disambiguation.
- Output **only** the query string. No explanation or additional text.
"""
            ),
            HumanMessage(
                content=f"""
Context:
- Paper Title: {paper_title}
- Section Name: {section_name}
- Key Point to Address: {key_point}
- Original User Query: {user_query}
- Original Search Query (if any): {original_query_str}
- Identified Weaknesses in Current Content:
  - {issues_str}

Generate ONE search query to find sources that:
1. Directly address the listed weaknesses for the key point.
2. Cover foundational theories, recent debates, or interdisciplinary perspectives relevant to the topic.
3. Include critical analyses of methodologies, scholarly disagreements, or unresolved questions.
4. Suggest novel insights or future research directions.
Output ONLY the search query string.
"""
            ),
        ]
    )

def get_query_type_classification_prompt(
    query: str, format_instructions: str
) -> ChatPromptTemplate:
    """
    Generates a prompt template for classifying the type of user query: academic or general.

    Args:
        query: The user query to classify.
        format_instructions: Instructions for the expected output format (e.g., Pydantic schema).

    Returns:
        A ChatPromptTemplate object configured for query type classification.
    """
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=f"""You are an expert query classifier specializing in distinguishing between academic research queries and general information queries.

Your task is to analyze the given user query and classify it into one of two categories:

**ACADEMIC RESEARCH QUERIES** are characterized by:
- Focus on scientific research, technical methods, theoretical frameworks
- Contains professional terminology, algorithm names, technical concepts
- Seeks research papers, academic materials, technical documentation
- Examples: machine learning algorithms, deep learning models, natural language processing techniques, theoretical analysis, empirical studies

**GENERAL INFORMATION QUERIES** are characterized by:
- Focus on news, policies, regulations, business information
- Contains current events, market dynamics, industry reports
- Seeks latest news, real-time information, official policies
- Examples: government policies, market trends, news events, company updates, product information

Analyze the query content, terminology used, and likely information-seeking intent to make your classification.

Provide your analysis with:
1. Classification (academic or general)
2. Confidence level (0.0 to 1.0)
3. Clear reasoning for your decision

Output Schema Instructions:
{format_instructions}

Provide the classification strictly following the schema. Return *only* the JSON object."""
            ),
            HumanMessage(
                content=f"""Please classify the following user query:

Query: "{query}"

Analyze this query and determine whether it is seeking academic research information or general information. Consider the terminology used, the type of information likely being sought, and the overall context.

Return your classification following the specified schema format."""
            ),
        ]
    )