import os
import numpy as np
import re
from tqdm import tqdm
import threading
import logging
from evaluation.API.model import APIModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from .prompt import (
    CRITERIA,
    CRITERIA_BASED_JUDGING_PROMPT,
    NLI_PROMPT,
    LANGUAGE_EVALUATION_PROMPT_V2,
    CRITICAL_EVALUATION_PROMPT_V2,
    OUTLINE_EVALUATION_PROMPT_V2
)

logger = logging.getLogger(__name__)


class Judge:
    def __init__(self, jsonl_file: str, model: str, infer_type) -> None:
        self.model = model
        self.api_model = APIModel(self.model, infer_type,"")
        self.jsonl_file = jsonl_file
        self.input_token_usage, self.output_token_usage = 0, 0

    def __generate_prompt(self, template, paras):
        prompt = template
        for k in paras.keys():
            prompt = prompt.replace(f"[{k}]", paras[k])
        return prompt

    def criteria_based_judging(self, survey, topic, criterion):
        criterion_paras = CRITERIA[criterion]

        content_paras = {
            "TOPIC": topic,
            "SURVEY": survey,
            "Criterion Description": criterion_paras["description"],
            "Score 1 Description": criterion_paras["score 1"],
            "Score 2 Description": criterion_paras["score 2"],
            "Score 3 Description": criterion_paras["score 3"],
            "Score 4 Description": criterion_paras["score 4"],
            "Score 5 Description": criterion_paras["score 5"],
        }
        prompt = self.__generate_prompt(CRITERIA_BASED_JUDGING_PROMPT, content_paras)
        scores = self.api_model.chat(prompt, temperature=0)
        return scores

    def __criteria_based_judging(self, topic, survey, criterion, res_l, idx):
        criterion_paras = CRITERIA[criterion]
        content_paras = {
            "TOPIC": topic,
            "SURVEY": survey,
            "Criterion Description": criterion_paras["description"],
        }
        for score in range(1, 6):
            content_paras[f"Score {score} Description"] = criterion_paras[
                f"score {score}"
            ]
        prompt = self.__generate_prompt(CRITERIA_BASED_JUDGING_PROMPT, content_paras)
        # Add retry mechanism with error handling
        max_retries = 10
        retry_count = 0

        while retry_count < max_retries:
            try:
                scores = self.api_model.chat(prompt, temperature=0)
                extracted_score = self.extract_num(scores)

                # Validate extracted score
                if (
                    extracted_score != ""
                    and isinstance(extracted_score, (int, float))
                    and 1 <= extracted_score <= 5
                ):
                    res_l[idx] = extracted_score
                    return scores
                else:
                    logger.warning(
                        f"Invalid score extracted: {extracted_score}, retrying... (attempt {retry_count + 1})"
                    )

            except Exception as e:
                logger.error(
                    f"Error occurred during API call (attempt {retry_count + 1}): {e}"
                )

            retry_count += 1
            if retry_count < max_retries:
                # Add exponential backoff
                import time

                time.sleep(2**retry_count)

        # If all retries failed, set default value
        logger.error(
            f"All {max_retries} attempts failed for criteria judging, setting default score to 1"
        )
        res_l[idx] = 1
        return "Error: Failed to get valid response after retries"

    def extract_num(self, string):
        numbers = re.findall(r"\d+", string)
        if len(numbers) == 0:
            return ""
        return eval(numbers[0])

    def extract_num_addition(self, response: str) -> float:
        # Try to match direct score format first (for V2 prompts)
        match = re.search(r"<SCORE>\s*(\d+(?:\.\d+)?)\s*</SCORE>", response)
        if match:
            score = float(match.group(1))
            if 0 <= score <= 10:  # V2 prompts use 0-10 scale
                return score * 10  # Scale to 0-100 for consistency
            elif 0 <= score <= 100:  # Original scale
                return score
            else:
                logger.error(f"Invalid score extracted from response: {score}")
                return 0.0

        # Try to match calculation format like (X+Y+Z)/3 or (X+Y+Z)/3=result
        calc_match = re.search(r"<SCORE>\s*\(([^)]+)\)/3(?:=[\d.]+)?\s*</SCORE>", response)
        if calc_match:
            try:
                calculation = calc_match.group(1).strip()
                # Allow spaces around operators: "2.5 + 7 + 5.1" or "2.5+7+5.1"
                if re.match(r"^\d+(?:\.\d+)?(?:\s*\+\s*\d+(?:\.\d+)?)*$", calculation):
                    # Split and strip each part
                    parts = [x.strip() for x in calculation.split('+')]
                    result = sum(float(x) for x in parts) / 3
                    if 0 <= result <= 10:
                        return result * 10  # Scale to 0-100
                    else:
                        logger.error(f"Invalid calculated score: {result}")
                        return 0.0
            except Exception as e:
                logger.error(f"Error calculating score from: {calculation}, error: {e}")

        # Try to extract the final result from format like (X+Y+Z)/3=4.87
        eq_result_match = re.search(r"<SCORE>\s*\([^)]+\)/3=(\d+(?:\.\d+)?)\s*</SCORE>", response)
        if eq_result_match:
            score = float(eq_result_match.group(1))
            if 0 <= score <= 10:
                return score * 10  # Scale to 0-100
            elif 0 <= score <= 100:
                return score
            else:
                logger.error(f"Invalid score extracted from response: {score}")
                return 0.0

        # Fallback: try original format with equals sign
        eq_match = re.search(r"<SCORE>.*?=\s*([\d.]+)\s*</SCORE>", response)
        if eq_match:
            score = float(eq_match.group(1))
            if 0 <= score <= 100:
                return score
            else:
                logger.error(f"Invalid score extracted from response: {score}")
                return 0.0

        logger.error(f"Failed to extract score from response: {response}")
        return -1

    def batch_criteria_based_judging(self, survey, topic, criteria):
        thread_l = []
        scores = [0] * len(criteria)
        for i in range(len(criteria)):
            thread = threading.Thread(
                target=self.__criteria_based_judging,
                args=(topic, survey, criteria[i], scores, i),
            )
            thread_l.append(thread)
            thread.start()
        for thread in thread_l:
            thread.join()
        return scores

    def __get_pair_score(self, source, claim, res_l, i, j, citation_idx, raw_claim=""):
        max_model_len = 900000
        max_estimate_char_len = int(max_model_len * 1.25)
        if len(source) > max_estimate_char_len:
            logger.warning(
                f"Source is too long({len(source)}), truncated to {max_estimate_char_len} characters"
            )
            source = source[:max_estimate_char_len]
        source = source[:max_estimate_char_len]
        content_paras = {"SOURCE": source, "CLAIM": claim}
        prompt = self.__generate_prompt(NLI_PROMPT, content_paras)
        try:
            res = self.api_model.chat(prompt, temperature=0)
        except Exception as e:
            logger.error(f"Error occurred while calling chat API: {e}")
            res_l[i][j] = -1
            return 0

        res = self.api_model.chat(prompt, temperature=0)

        if res and "yes" in res.lower():
            res_l[i][j] = citation_idx
            return 1
        else:
            res_l[i][j] = -1
            if raw_claim:
                logger.info(
                    f"Unrelated pair found. \n  claim=[{claim}]\n  raw_claim=[{raw_claim}]\n  citation_idx={citation_idx}\n  source[:1500]={source[:1500]}"
                )
            else:
                logger.info(
                    f"Unrelated pair found. Claim=[{claim}], Source[:1500]={source[:1500]}"
                )
            return 0

    def citation_quality_new(self, survey_with_reference, references):
        survey = survey_with_reference.split("## References")[0]
        survey_sections = survey.split("###")
        citation_pattern = re.compile(r"[^.!?]*\[[^\]]+\][^.!?]*[.!?]")
        sentences = []
        for content in survey_sections:
            sentences += citation_pattern.findall(content)

        raw_claims = []
        claims = []
        sources_ids = []
        for s in sentences:
            sources = re.findall(pattern=r"\[(.*?)\]", string=s)

            if len(sources) > 0:
                source_ids = set()
                for ref in sources:
                    for num in ref.split(","):
                        number = self.extract_num(num)
                        if number != "":
                            source_ids.add(number)
                if len(source_ids) > 0:
                    raw_claims.append(s)
                    claims.append(re.sub(pattern=r"\[(.*?)\]", repl="", string=s))
                    sources_ids.append(list(source_ids))

        paper_infos = self.get_paper_info_from_jsonl(references)

        ids_to_title = {p["title"]: p["title"] for p in paper_infos}
        ids_to_paper = {p["title"]: p["content"] for p in paper_infos}

        index_to_paper = {
            index: ids_to_paper[title] for index, title in enumerate(ids_to_title)
        }
        index_to_titles = {index: title for index, title in enumerate(ids_to_title)}

        logger.info(f"start to eval pair score..")
        thread_l = []
        assert len(claims) == len(sources_ids)
        pair_scores = []
        for i in range(len(claims)):
            pair_scores.append([0] * len(sources_ids[i]))

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []

            for i in range(len(claims)):
                for j in range(len(sources_ids[i])):
                    citation_idx = sources_ids[i][j] - 1

                    source = index_to_paper[citation_idx]
                    futures.append(
                        executor.submit(
                            self.__get_pair_score,
                            source,
                            claims[i],
                            pair_scores,
                            i,
                            j,
                            citation_idx,
                            raw_claims[i],
                        )
                    )

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing threads"
            ):
                future.result()

        total_paper_num = len(paper_infos)
        result_dict = {}
        correct_claim_num, total_claim_num = claim_precision(pair_scores)
        result_dict["claim_precision"] = correct_claim_num / total_claim_num
        correct_citation_num, total_citation_num = citation_precision(pair_scores)
        result_dict["citation_precision"] = correct_citation_num / total_citation_num
        result_dict["reference_precision"] = reference_precision(
            pair_scores, total_paper_num
        )
        result_dict["reference_coverage"] = reference_coverage(
            claims, sources_ids, total_paper_num
        )
        result_dict["citation_density"] = citation_density(sources_ids, survey)
        result_dict["avg_citation_per_claim"] = avg_citation_per_claim(
            claims, sources_ids
        )
        print_result(result_dict)

        return result_dict




    def citation_quality_num(self, survey_with_reference, references):
        def split_survey(survey_with_reference: str) -> str:
            # 匹配形如 ##10. References、## 3. References 等
            pattern = r"##\s*\d+\.\s*References"
            split_result = re.split(pattern, survey_with_reference, maxsplit=1, flags=re.IGNORECASE)
            return split_result[0]
        survey = split_survey(survey_with_reference)
        # survey = survey_with_reference.split("## References")[0]
        survey_sections = survey.split("###")
        citation_pattern = re.compile(r"[^.!?]*\[[^\]]+\][^.!?]*[.!?]")
        sentences = []
        for content in survey_sections:
            sentences += citation_pattern.findall(content)
        raw_claims = []
        claims = []
        sources_ids = []
        for s in sentences:
            sources = re.findall(pattern=r"\[(.*?)\]", string=s)
            if len(sources) > 0:
                source_ids = set()
                for ref in sources:
                    for num in ref.split(","):
                        number = self.extract_num(num)
                        if number != "":
                            source_ids.add(number)
                if len(source_ids) > 0:
                    raw_claims.append(s)
                    claims.append(re.sub(pattern=r"\[(.*?)\]", repl="", string=s))
                    sources_ids.append(list(source_ids))

        paper_infos = self.get_paper_info_from_jsonl(references)

        ids_to_title = {p["title"]: p["title"] for p in paper_infos}
        ids_to_paper = {p["title"]: p["content"] for p in paper_infos}

        index_to_paper = {
            index: ids_to_paper[title] for index, title in enumerate(ids_to_title)
        }
        index_to_titles = {index: title for index, title in enumerate(ids_to_title)}

        logger.info(f"start to eval pair score by paper..")
        assert len(claims) == len(sources_ids)
        pair_scores = []
        for i in range(len(claims)):
            pair_scores.append([0] * len(sources_ids[i]))

        thread_l = []
        for i in range(len(claims)):
            for j in range(len(sources_ids[i])):
                citation_idx = sources_ids[i][j] - 1
                source = index_to_paper[citation_idx]
                thread = threading.Thread(
                    target=self.__get_pair_score,
                    args=(
                        source,
                        claims[i],
                        pair_scores,
                        i,
                        j,
                        citation_idx,
                        raw_claims[i],
                    ),
                )
                thread_l.append(thread)
                thread.start()
        for thread in tqdm(thread_l):
            thread.join()

        total_paper_num = len(paper_infos)
        result_dict = {}
        correct_claim_num, total_claim_num = claim_precision(pair_scores)
        correct_citation_num, total_citation_num = citation_precision(pair_scores)
        result_dict["correct_claim_num"] = correct_claim_num
        result_dict["total_claim_num"] = total_claim_num
        result_dict["correct_citation_num"] = correct_citation_num
        result_dict["total_citation_num"] = total_citation_num

        return result_dict

    def get_paper_info_from_jsonl(self, references):
        paper_infos = []
        for paper in references:
            paper_info = {
                "title": paper.get("title", ""),
                "content": (paper.get("txt") or ""),
            }
            paper_infos.append(paper_info)
        return paper_infos

    def _preprocess_outline(self, outline: str) -> str:
        """
        Process outline to keep only lines starting with #
        """
        lines = outline.split('\n')
        filtered_lines = []

        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith('#'):
                filtered_lines.append(line)  # Keep original indentation

        processed_outline = '\n'.join(filtered_lines)
        logger.debug(f"Original outline length: {len(outline)}, Processed length: {len(processed_outline)}")

        return processed_outline

    def evaluate_outline(self, outline: str, topic: str) -> float:

        processed_outline = self._preprocess_outline(outline)

        content_paras = {"TOPIC": topic, "OUTLINE": processed_outline}
        # prompt = self.__generate_prompt(OUTLINE_EVALUATION_PROMPT, content_paras)
        prompt = self.__generate_prompt(OUTLINE_EVALUATION_PROMPT_V2, content_paras)

        # Add retry mechanism with error handling
        max_retries = 10
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = self.api_model.chat(prompt, temperature=0)
                logger.debug(response)
                score = self._extract_outline_score(response)

                # Validate extracted score
                if score >= 0 and score <= 100:
                    return score
                else:
                    logger.warning(
                        f"Invalid outline score extracted: {score}, retrying... (attempt {retry_count + 1})"
                    )

            except Exception as e:
                logger.error(
                    f"Error occurred during outline evaluation API call (attempt {retry_count + 1}): {e}"
                )

            retry_count += 1
            if retry_count < max_retries:
                # Add exponential backoff
                import time

                time.sleep(2**retry_count)

        # If all retries failed, return default value
        logger.error(
            f"All {max_retries} attempts failed for outline evaluation, returning default score of 0.0"
        )
        return 0.0

    def _extract_outline_score(self, response: str) -> float:
        # Try to match direct score format first
        match = re.search(r"<SCORE>\s*(\d+(?:\.\d+)?)\s*</SCORE>", response)
        if match:
            score = float(match.group(1))
            if 0 <= score <= 10:  # V2 uses 0-10 scale
                return score * 10  # Scale to 0-100 for consistency
            elif 0 <= score <= 100:
                return score
            else:
                logger.error(f"Invalid score extracted from response: {score}")
                return 0.0

        # Try to match calculation format like (X+Y+Z)/3=result
        calc_result_match = re.search(r"<SCORE>\s*\([^)]+\)/3=(\d+(?:\.\d+)?)\s*</SCORE>", response)
        if calc_result_match:
            score = float(calc_result_match.group(1))
            if 0 <= score <= 10:
                return score * 10  # Scale to 0-100
            elif 0 <= score <= 100:
                return score
            else:
                logger.error(f"Invalid calculated score: {score}")
                return 0.0

        # Try to match and calculate (X+Y+Z)/3 format
        calc_match = re.search(r"<SCORE>\s*\(([^)]+)\)/3\s*</SCORE>", response)
        if calc_match:
            try:
                calculation = calc_match.group(1).strip()
                if re.match(r"^\d+(?:\.\d+)?(?:\s*\+\s*\d+(?:\.\d+)?)*$", calculation):
                    parts = [x.strip() for x in calculation.split('+')]
                    result = sum(float(x) for x in parts) / 3
                    if 0 <= result <= 10:
                        return result * 10  # Scale to 0-100
                    else:
                        logger.error(f"Invalid calculated score: {result}")
                        return 0.0
            except Exception as e:
                logger.error(f"Error calculating outline score: {e}")

        logger.error(f"Failed to extract outline score from response: {response}")
        return 0.0

    def evaluate_section(self, section, topic, prompt_template):
        content_paras = {"TOPIC": topic, "SECTION": section}
        prompt = self.__generate_prompt(prompt_template, content_paras)

        for i in range(10):
            response = self.api_model.chat(prompt, temperature=0)
            logger.info(response)
            score = self.extract_num_addition(response)
            if score != -1:
                return score

    def evaluate_survey_dimension(self, survey, topic, dimension_prompt_template):

        sections = re.findall(
            r"(^## \d+(?:\.\s|\s|$).*?)(?=^## \d+(?:\.\s|\s|$)|^## References|\Z)",
            survey,
            flags=re.DOTALL | re.MULTILINE,
        )
        sections = sections[:-1]
        if sections ==[]:
            sections = re.findall(
                r"(^# .+?)(?=^# |\Z)",
                survey,
                flags=re.DOTALL | re.MULTILINE,
            )
        print("sections:", len(sections))

        thread_l = []

        score_results = [None] * len(sections)

        def evaluate_section_thread(i, section):
            # Add retry mechanism for section evaluation
            max_retries = 20
            retry_count = 0

            while retry_count < max_retries:
                try:
                    score = self.evaluate_section(
                        section.strip(), topic, dimension_prompt_template
                    )
                    # Validate score
                    if score is not None and score >= 0:
                        score_results[i] = score
                        return
                    else:
                        logger.warning(f"Invalid score for section {i}: {score}, retrying... (attempt {retry_count + 1})")

                except Exception as e:
                    logger.error(f"Error evaluating section {i} (attempt {retry_count + 1}): {e}")

                retry_count += 1
                if retry_count < max_retries:
                    import time
                    time.sleep(2 ** retry_count)

            # If all retries failed, set default score
            logger.error(f"All {max_retries} attempts failed for section {i}, setting default score to 0")
            score_results[i] = 0

        for i, section in enumerate(sections):
            if i == 0 and not section.startswith("##"):
                continue

            thread = threading.Thread(target=evaluate_section_thread, args=(i, section))
            thread_l.append(thread)
            thread.start()

        for thread in thread_l:
            thread.join()

        section_scores = [score for score in score_results if score is not None]

        if section_scores:
            print(section_scores, flush=True)

            filtered_scores = [score for score in section_scores if score != 0]
            avg_score = np.mean(filtered_scores) if filtered_scores else 0.0
            print(avg_score, flush=True)


        else:
            print("No valid section scores found.", flush=True)
            avg_score = 0.0

        return avg_score,filtered_scores

    def evaluate_language(self, survey, topic):
        return self.evaluate_survey_dimension(survey, topic, LANGUAGE_EVALUATION_PROMPT_V2)

    def evaluate_critical(self, survey, topic):
        return self.evaluate_survey_dimension(survey, topic, CRITICAL_EVALUATION_PROMPT_V2)

    def evaluate_all_dimensions(self, survey, topic):
        language_score,filtered_scores_lang = self.evaluate_language(survey, topic)
        critical_score,filtered_scores_crit = self.evaluate_critical(survey, topic)

        return {
            "language_score": language_score,
            "filtered_scores_lang": filtered_scores_lang,
            "critical_score": critical_score,
            "filtered_scores_crit": filtered_scores_crit,
        }


def claim_precision(pairs):
    total_claim_num = len(pairs)
    correct_claim_num = 0
    for i in range(total_claim_num):
        for j in range(len(pairs[i])):
            if not pairs[i][j] == -1:
                correct_claim_num += 1
                break
    return correct_claim_num, total_claim_num


def citation_precision(pairs):
    total_citation_num = 0
    correct_citation_num = 0
    for i in range(len(pairs)):
        for j in range(len(pairs[i])):
            total_citation_num += 1
            if not pairs[i][j] == -1:
                correct_citation_num += 1
    return correct_citation_num, total_citation_num


def reference_precision(pairs, total_paper_num):
    reference_set = set()
    for i in range(len(pairs)):
        for j in range(len(pairs[i])):
            if not pairs[i][j] == -1:
                reference_set.add(pairs[i][j])
    return len(reference_set) / total_paper_num


def reference_coverage(claims, sources_ids, total_paper_num):
    reference_set = set()
    for i in range(len(claims)):
        for j in range(len(sources_ids[i])):
            citation_idx = sources_ids[i][j] - 1
            reference_set.add(citation_idx)
    return len(reference_set) / total_paper_num


def count_sentences(text):
    sentences = re.split(r"[.!?\n]+(?:\s|\n|$)", text.strip())
    sentences = [s for s in sentences if s]
    return len(sentences)


def citation_density(sources_ids, survey):
    total_citation_num = 0
    for i in range(len(sources_ids)):
        for _ in range(len(sources_ids[i])):
            total_citation_num += 1

    total_sentence_num = count_sentences(survey)
    return total_citation_num / total_sentence_num


def avg_citation_per_claim(claims, sources_ids):
    total_citation_num = 0
    for i in range(len(claims)):
        for _ in range(len(sources_ids[i])):
            total_citation_num += 1
    return total_citation_num / len(claims)


def print_result(result_dict):
    print("########## Metric with Judgement ##########")
    print(f"Claim Precision: {result_dict['claim_precision']}")
    print(f"Citation Precision: {result_dict['citation_precision']}")
    print(f"Reference Precision: {result_dict['reference_precision']}")
    print(f"######## Metric without Judgement #########")
    print(f"Reference Coverage: {result_dict['reference_coverage']}")
    print(f"Citation Density: {result_dict['citation_density']}")
    print(f"Avg Citation per Claim: {result_dict['avg_citation_per_claim']}")
    print(
        f"Citation Quality: {result_dict['reference_precision'] * result_dict['reference_precision']}"
    )
