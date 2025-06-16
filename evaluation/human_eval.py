import json


"""
This script is mainly used to convert the test files in SurveyEval into a format that can be evaluated by MapReduce.
"""
def read_jsonl_file(filepath):
    """
    Read a .jsonl file and return a list of JSON objects parsed from each line.

    Parameters:
        filepath (str): Path to the .jsonl file

    Returns:
        List[dict]: JSON objects corresponding to each line
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}, content: {line}")
    return data


def write_jsonl(data_list, save_path):
    """
    Write a list to a JSONL file.

    Parameters:
    - data_list: List[dict], the data to write, each element is one JSON line.
    - save_path: str, path to save the file, e.g. "output.jsonl"
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')


import re

def extract_all_second_sentences(text):
    """
    Extract from text:
    1. The second sentence following every [xxx] or 【xxx】;
    2. The second sentence of every paragraph after a newline.
    Returns a list of all extracted sentences.
    """
    def get_second_sentence(fragment):
        # Simple sentence splitting: split by . ! ? plus whitespace or end punctuation
        sentences = re.split(r'(?<=[.?!。？！])\s+', fragment.strip())
        return sentences[1] if len(sentences) > 1 else None

    results = set()  # Use set to remove duplicates

    # Process all text fragments after [xxx] or 【xxx】
    for match in re.finditer(r'[\[【][^\]】]+[\]】](.*?)(?=[\[【]|\n|$)', text, re.DOTALL):
        fragment = match.group(1).strip()
        second = get_second_sentence(fragment)
        if second:
            results.add(second)

    # Process all paragraphs after newlines
    for paragraph in text.split('\n'):
        second = get_second_sentence(paragraph)
        if second:
            results.add(second)

    return list(results)

import os
def load_all_jsonl_from_folder(folder_path):
    """
    Traverse all .jsonl files in a folder and merge their contents into a single list to return.

    :param folder_path: Path to a folder containing multiple .jsonl files
    :return: List containing all entries from all .jsonl files
    """
    all_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        try:
                            data = json.loads(line)
                            all_data.append(data)
                        except json.JSONDecodeError as e:
                            print(f"Could not parse a line as JSON in file {filename}: {e}")

    return all_data


import json

def save_list_to_jsonl(data_list, file_path):
    """
    Write each element in a list to a jsonl file, one JSON object per line.

    :param data_list: List, each element is a dict (or JSON-serializable object)
    :param file_path: Path to save the .jsonl file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')


if __name__ == '__main__':

    jsonl_path = "Path to test.jsonl"
    datas = read_jsonl_file(jsonl_path)
    static_data = []

    for idx, data in enumerate(datas):

        data["content"] = data["txt"]
        outline_all = ""
        for outline in data["outline"]:
            outline_all = outline_all + outline

        data['outline'] = outline_all


    path = "Path to save test.jsonl"
    save_list_to_jsonl(datas, path)
