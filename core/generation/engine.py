# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================

import os
import re
import spacy

import asyncio

import numpy as np
from log import logger
import traceback
import requests
import tqdm
from generation.global_config import rerank_model_url



def remove_citations(sent):
    return (
        re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent))
        .replace(" |", "")
        .replace("]", "")
    )


def process_paragraph(text):
    text = text.replace("<cit.>", "")
    text = remove_citations(text)
    return text


def process_ctx(ctxs):
    logger.debug("process ctx")
    for one in ctxs:
        text = one["text"]
        cleaned_text = re.sub(r"\[\d+\]", "", text)
        cleaned_text = process_paragraph(cleaned_text)
        one["text"] = cleaned_text
    return ctxs


def rerank_paragraphs_bge(
    query, paragraphs, reranker, norm_cite=False, start_index=0, use_abstract=False
):
    logger.info("run rerank_paragraphs_bge ...")
    paragraphs = [p for p in paragraphs if p["text"] is not None]
    logger.debug(f"use_abstract?: {use_abstract}")
    logger.debug(f"norm_cite?: {norm_cite}")
    if use_abstract is True and p["text"] != p["abstract"]:
        paragraph_texts = [
            (
                p["title"] + "\n" + p["abstract"] + "\n" + p["text"]
                if "title" in p and "abstract" in p
                else p["text"]
            )
            for p in paragraphs
        ]
    else:
        paragraph_texts = [
            (
                p["title"] + " " + p["text"]
                if "title" in p and p["title"] is not None
                else p["text"]
            )
            for p in paragraphs
        ]

    logger.info(
        f"rerank_paragraphs_bge paragraph_texts: {len(paragraph_texts)}, {paragraph_texts}"
    )

    data = {
        "messages": [[query, p] for p in paragraph_texts],
        "model": "bge_rerank",
        "problem_type": "rerank",
    }
    response = requests.post(rerank_model_url, json=data).json()
    try:

        scores = response["scores"]
    except:
        scores = [0] * len(paragraph_texts)

    if type(scores) is float:
        result_dic = {0: scores}
    else:
        result_dic = {p_id: score for p_id, score in enumerate(scores)}
    if (
        norm_cite is True
        and len(
            [
                item["citation_counts"]
                for item in paragraphs
                if "citation_counts" in item and item["citation_counts"] is not None
            ]
        )
        > 0
    ):
        # add normalized scores
        max_citations = max(
            [
                item["citation_counts"]
                for item in paragraphs
                if "citation_counts" in item and item["citation_counts"] is not None
            ]
        )
        for p_id in result_dic:
            if (
                "citation_counts" in paragraphs[p_id]
                and paragraphs[p_id]["citation_counts"] is not None
            ):
                result_dic[p_id] = result_dic[p_id] + (
                    paragraphs[p_id]["citation_counts"] / max_citations
                )
    p_ids = sorted(result_dic.items(), key=lambda x: x[1], reverse=True)
    new_orders = []
    id_mapping = {}
    for i, p_id in enumerate(p_ids):
        new_orders.append(paragraphs[p_id[0]])
        id_mapping[i] = int(p_id[0])
    return new_orders, result_dic, id_mapping


def create_prompt_with_llama3_format(
    prompt,
    system_message="You are a helpful AI assistant for scientific literature review. Please carefully follow user's instruction and help them to understand the most recent papers.",
):
    if system_message is not None:
        formatted_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{0}<|eot_id|>".format(
            system_message
        )
    else:
        formatted_text = "<|begin_of_text|>"
    formatted_text += (
        "<|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|>"
    )
    formatted_text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return formatted_text


def load_hf_tokenizer(
    model_name_or_path,
    tokenizer_name_or_path=None,
    use_fast_tokenizer=True,
    padding_side="left",
    token=os.getenv("HF_TOKEN", None),
):
    from transformers import AutoTokenizer

    # Need to explicitly import the olmo tokenizer.
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, use_fast=use_fast_tokenizer, token=token
        )
    except:
        # some tokenizers (e.g., GPTNeoXTokenizer) don't have the slow or fast version, so we just roll back to the default one
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, token=token)
    # set padding side to left for batch generation
    tokenizer.padding_side = padding_side
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
