import json
from difflib import SequenceMatcher
"""
Evaluate the overlap, overlap count, and F1 score between the references of the generated papers and the human-written references in the test set.

jsonl_path: Evaluation set SurveyEval_test

jsonl_path_ourBenchMark: Our benchmark, path to surveyscope

our_data_papers_file_path: Path to the folder containing the generated papers to be evaluated
"""

def remove_version_tags(text):
    """
    Remove 'v1', 'v2', 'v3' from a string.

    Args:
        text (str): input string

    Returns:
        str: string after removing version tags
    """
    return re.sub(r'\bv[1-3]\b', '', text)


def string_similarity(s1, s2):
    s1 = s1.lower()
    s2 = s2.lower()
    return SequenceMatcher(None, s1, s2).ratio()


def is_similar(sentence, paragraph, threshold=0.8):
    sentence = sentence.strip().lower()
    sub_sentences = re.split(r'[.!?。\n]', paragraph.lower())

    for sub in sub_sentences:
        if string_similarity(sentence, sub.strip()) >= threshold:
            print("sentence:", sentence)
            print("sub:", sub.strip())
            return 1
    return 0


def read_jsonl_file(filepath):
    """
    Read a .jsonl file and return a list of parsed JSON objects, one per line.

    Args:
        filepath (str): path to the .jsonl file

    Returns:
        List[dict]: list of JSON objects from each line
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}, content: {line}")
    return data


def write_jsonl(data_list, save_path):
    """
    Write a list of dictionaries to a JSONL file.

    Args:
    - data_list: List[dict], data to write, each dict is one JSON line.
    - save_path: str, path to save the file, e.g., "output.jsonl"
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')


import re

def extract_all_second_sentences(text):
    """
    Extract from the text:
    1. The second sentence following any [xxx] or 【xxx】 brackets;
    2. The second sentence of every paragraph after line breaks.

    Returns a list of all extracted sentences.
    """
    def get_second_sentence(fragment):
        # Simple sentence splitting: split by . ! ? followed by space or end of string
        sentences = re.split(r'(?<=[.?!。？！])\s+', fragment.strip())
        return sentences[1] if len(sentences) > 1 else None

    results = set()  # use set to deduplicate

    # Process all fragments after [xxx] or 【xxx】
    for match in re.finditer(r'[\[【][^\]】]+[\]】](.*?)(?=[\[【]|\n|$)', text, re.DOTALL):
        fragment = match.group(1).strip()
        second = get_second_sentence(fragment)
        if second:
            results.add(second)
    # Process all paragraphs after line breaks
    for paragraph in text.split('\n'):
        second = get_second_sentence(paragraph)
        if second:
            results.add(second)

    return list(results)
import os
def load_all_jsonl_from_folder(folder_path):
    """
    Traverse all .jsonl files in a folder and merge their contents into a single list.

    Args:
        folder_path: path to the folder containing multiple .jsonl files

    Returns:
        list: combined list of all entries from all .jsonl files
    """
    all_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # skip empty lines
                        try:
                            data = json.loads(line)
                            all_data.append(data)
                        except json.JSONDecodeError as e:
                            print(f"Could not parse a line in file {filename} as JSON: {e}")

    return all_data
def clean_arxiv_url(url):
    """
    Remove version numbers (like v1, v2, v3) from arXiv URLs.

    Args:
        url (str): string containing an arXiv URL

    Returns:
        str: URL with version numbers removed
    """
    return re.sub(r'(http://arxiv\.org/abs/\d+\.\d+)(v\d+)', r'\1', url)

def optimized_similarity_match(datas_SurveyScope, generate_papers_file_path):
    reference_n = 0
    SurveyScope_references_n = 0
    paper_references_n = 0

    # Step 1: 缓存所有 .jsonl 文件路径
    jsonl_files = []
    for root, _, files in os.walk(generate_papers_file_path):
        for filename in files:
            if filename.endswith('.jsonl'):
                jsonl_files.append((filename, os.path.join(root, filename)))

    # Step 2: 遍历 SurveyScope 数据
    for item in datas_SurveyScope:
        item_title = item["title"]
        item_references = item["references"]

        matched = False
        matched_papers = []

        for filename, full_path in jsonl_files:
            if is_similar(item_title, filename) > 0:
                matched = True
                datas = read_jsonl_file(full_path)
                for data in datas:
                    papers = data["papers"]
                    paper_references_n += len(papers)
                    matched_papers.extend(papers)


        if matched:
            SurveyScope_references_n += len(item_references)

            for reference in item_references:
                reference_title = reference["title"]
                for paper in matched_papers:
                    if is_similar(paper["title"], reference_title) > 0:
                        reference_n += 1

    return reference_n, SurveyScope_references_n, paper_references_n

from tqdm import tqdm
import os
if __name__ == '__main__':


    SurveyScope_path = "/share/project/shixiaofeng/code/SciSage_pipe_xf/benchmark/SurveyScope/surveyscope.jsonl"

    datas_SurveyScope= read_jsonl_file(SurveyScope_path)

    generate_papers_file_path = "/share/project/lyd/lyd/LLMxMapReduce-main/LLMxMapReduce_V2/SurveyEval/eval/test"


    reference_n, SurveyScope_references_n, paper_references_n = optimized_similarity_match(datas_SurveyScope,generate_papers_file_path)
    recall = reference_n / SurveyScope_references_n
    precision = reference_n / paper_references_n
    f1 = 2 * precision * recall / (precision + recall)

    print("recall:", recall)
    print("precision:", precision)
    print("f1:", f1)
    print("overlap count:", reference_n)
    print("SurveyScope total count:", SurveyScope_references_n)
    print("generated paper reference count:", paper_references_n)
