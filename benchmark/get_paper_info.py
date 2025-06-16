# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================

import asyncio
from crawl4ai import *
import arxiv
import datasets
import requests
import json
import os
from io import BytesIO
from pdfminer.high_level import extract_text
from semanticscholar import SemanticScholar
import random
import time



arxiv_client = arxiv.Client(delay_seconds=0.05)
proxies = {"http": "http://localhost:1080", "https": "http://localhost:1080"}

async def get_md_info_with_fallback(arxiv_id, max_retries=2):
    """
    多种方式获取论文全文，按优先级尝试不同的方法
    """
    methods = [
        ("ar5iv", f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"),
        ("arxiv_vanity", f"https://www.arxiv-vanity.com/papers/{arxiv_id}/"),
        ("arxiv_html", f"https://browse.arxiv.org/html/{arxiv_id}"),
    ]

    for method_name, url in methods:
        print(f"Trying {method_name} for {arxiv_id}: {url}")
        for attempt in range(max_retries):
            try:
                async with AsyncWebCrawler(verbose=False) as crawler:
                    result = await crawl_with_retry(crawler, url)
                    if result and result.markdown:
                        content = result.markdown
                        # 检查内容质量
                        if len(content) > 1000 and "Too Many Requests" not in content and "Access Denied" not in content:
                            print(f"Successfully got content from {method_name} (length: {len(content)})")
                            return content
                        else:
                            print(f"{method_name} returned low-quality content (length: {len(content)})")
            except Exception as e:
                print(f"Failed to get content from {method_name}, attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # 指数退避

    # 如果所有HTML方法都失败，尝试PDF提取
    print(f"All HTML methods failed for {arxiv_id}, trying PDF extraction...")
    try:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        pdf_text = extract_pdf_text_from_url(pdf_url)
        if pdf_text and len(pdf_text) > 1000:
            print(f"Successfully extracted PDF text (length: {len(pdf_text)})")
            return pdf_text
        else:
            print(f"PDF extraction returned low-quality content (length: {len(pdf_text) if pdf_text else 0})")
    except Exception as e:
        print(f"PDF extraction failed: {e}")

    print(f"All methods failed for {arxiv_id}")
    return ""

# 改进crawl_with_retry函数，增加更多错误处理
async def crawl_with_retry(crawler, url, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = await crawler.arun(url)

            if not result or not result.markdown:
                print(f"Empty result from {url}, attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue

            # 检查常见的错误响应
            error_indicators = [
                "too many requests",
                "rate limit",
                "access denied",
                "forbidden",
                "503 service unavailable",
                "502 bad gateway",
                "504 gateway timeout"
            ]

            content_lower = result.markdown.lower()
            if any(indicator in content_lower for indicator in error_indicators):
                delay = (2**attempt) + random.uniform(0, 1)
                print(f"Error response detected, waiting {delay:.2f}s before retry")
                await asyncio.sleep(delay)
                continue

            return result

        except Exception as e:
            print(f"Exception in crawl_with_retry: {e}")
            if attempt == max_retries - 1:
                raise
            delay = (2**attempt) + random.uniform(0, 1)
            await asyncio.sleep(delay)

    return None

# 原来的get_md_info函数保持不变，作为向后兼容
async def get_md_info(arxiv_id):
    return await get_md_info_with_fallback(arxiv_id)

# 改进PDF提取函数，增加错误处理和代理配置
def extract_pdf_text_from_url(pdf_url):
    """提取PDF文本，支持代理和重试"""
    print(f"Extracting PDF text from: {pdf_url}")
    # 尝试不同的请求配置
    configs = [
        {"proxies": proxies, "timeout": 30},  # 使用代理
        {"timeout": 30},  # 不使用代理
        {"proxies": proxies, "timeout": 60},  # 使用代理，更长超时
    ]

    for i, config in enumerate(configs):
        try:
            print(f"PDF extraction attempt {i + 1} with config: {config}")
            response = requests.get(pdf_url, **config)
            response.raise_for_status()

            with BytesIO(response.content) as f:
                text = extract_text(f)
                if text and len(text.strip()) > 100:  # 确保提取到了有意义的文本
                    return text
                else:
                    print(f"PDF extraction returned insufficient text (length: {len(text) if text else 0})")

        except Exception as e:
            print(f"PDF extraction attempt {i + 1} failed: {e}")
            if i < len(configs) - 1:
                time.sleep(2)  # 重试前等待

    return ""

def get_arxiv_metadata(arxiv_id):
    search = arxiv.Search(
        id_list=[arxiv_id],
        max_results=100,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )
    result = list(arxiv_client.results(search))[0]

    # 提取所有可用的链接信息，确保结构一致
    links = {}
    if hasattr(result, "links"):
        for link in result.links:
            links[link.title] = link.href

    # 标准化links结构，确保所有可能的字段都存在
    standardized_links = {
        "null": links.get("null", ""),
        "pdf": links.get("pdf", ""),
        "doi": links.get("doi", ""),
    }

    # 提取分类信息
    categories = (
        [cat.strip() for cat in result.primary_category.split()]
        if result.primary_category
        else []
    )
    if hasattr(result, "categories"):
        categories.extend([cat.strip() for cat in result.categories])

    return {
        "arxiv_id": arxiv_id,
        "title": result.title,
        "authors": [a.name for a in result.authors],
        "abstract": result.summary,
        "published": result.published.isoformat(),
        "updated": result.updated.isoformat() if result.updated else None,
        "pdf_url": result.pdf_url,
        "entry_id": result.entry_id,
        "primary_category": result.primary_category,
        "categories": list(set(categories)),  # 去重
        "comment": (
            result.comment if hasattr(result, "comment") and result.comment else ""
        ),
        "journal_ref": (
            result.journal_ref
            if hasattr(result, "journal_ref") and result.journal_ref
            else ""
        ),
        "doi": result.doi if hasattr(result, "doi") and result.doi else "",
        "links": standardized_links,  # 使用标准化的links结构
        "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
        "arxiv_pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        "ar5iv_url": f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
    }


def get_references_by_arxiv_id(arxiv_id):
    print("get_references_by_arxiv_id")
    sch = SemanticScholar()
    try:
        paper = sch.get_paper(
            f"ARXIV:{arxiv_id}",
            fields=[
                "references.title",
                "references.authors",
                "references.externalIds",
                "references.url",
                "references.venue",
                "references.year",
                "references.publicationDate",
            ],
        )
        references = []
        for ref in paper.references:
            # 提取外部ID，确保结构一致
            external_ids = ref.externalIds if ref.externalIds else {}

            # 标准化external_ids结构，确保所有可能的字段都存在
            standardized_external_ids = {
                "DBLP": str(external_ids.get("DBLP", "")),
                "DOI": str(external_ids.get("DOI", "")),
                "CorpusId": int(external_ids.get("CorpusId", 0)) if external_ids.get("CorpusId") else 0,
                "ArXiv": str(external_ids.get("ArXiv", "")),
                "ACL": str(external_ids.get("ACL", "")),
                "MAG": str(external_ids.get("MAG", "")),
                "PubMedCentral": str(external_ids.get("PubMedCentral", "")),
                "PubMed": str(external_ids.get("PubMed", "")),
            }

            arxiv_ref_id = standardized_external_ids["ArXiv"]
            doi = standardized_external_ids["DOI"]

            # 处理日期字段，确保它们是字符串格式
            publication_date = ""
            if ref.publicationDate:
                if hasattr(ref.publicationDate, "isoformat"):
                    publication_date = ref.publicationDate.isoformat()
                else:
                    publication_date = str(ref.publicationDate)

            # 处理year字段，确保统一为字符串格式
            year_str = ""
            if ref.year:
                year_str = str(ref.year)

            references.append(
                {
                    "title": ref.title if ref.title else "",
                    "authors": (
                        ";".join([a.name for a in ref.authors]) if ref.authors else ""
                    ),
                    "arxiv_id": arxiv_ref_id,
                    "doi": doi,
                    "url": ref.url if ref.url else "",
                    "venue": ref.venue if ref.venue else "",
                    "year": year_str,
                    "publication_date": publication_date,
                    "external_ids": standardized_external_ids,  # 使用标准化的结构
                }
            )
        return references
    except Exception as e:
        print(f"[Warning] Failed to get references for {arxiv_id}: {e}")
        return []


# 主函数：处理一组 arXiv 链接
async def process_arxiv_papers(arxiv_urls, output_json="surveyscope.jsonl"):
    results = []
    existing_titles = set()
    existing_entries = []

    # 检查文件是否存在，如果存在则读取已有数据
    if os.path.exists(output_json):
        print(f"Found existing file: {output_json}, loading existing data...")
        with open(output_json, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    existing_entries.append(data)
                    if "title" in data:
                        existing_titles.add(data["title"])
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(existing_entries)} existing entries")

        # 创建备份文件
        backup_file = output_json + ".backup"
        with open(backup_file, "w", encoding="utf-8") as backup_f:
            for entry in existing_entries:
                backup_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Created backup file: {backup_file}")

    # 边处理边写入更新后的现有条目
    temp_file = output_json + ".temp"
    with open(temp_file, "w", encoding="utf-8") as temp_f:
        # 处理已存在的条目，边处理边写入
        for i, entry in enumerate(existing_entries):
            print(
                f"Processing existing entry {i+1}/{len(existing_entries)}: {entry.get('title', 'Unknown')}"
            )
                        # 检查并更新主文档的full_text
            entry_updated = False
            if (
                "full_text" not in entry
                or not entry.get("full_text")
                or len(entry.get("full_text", "")) < 2000
                or "Too Many Requests" in entry.get("full_text", "")
            ):
                print(f"Updating main full_text for: {entry.get('arxiv_id', 'Unknown')}")
                try:
                    new_full_text = await get_md_info_with_fallback(entry.get("arxiv_id", ""))
                    entry["full_text"] = new_full_text
                    entry_updated = True
                    print(f"Updated main full_text (length: {len(new_full_text)})")
                except Exception as e:
                    print(f"Failed to update main full_text: {e}")
                    entry["full_text"] = entry.get("full_text", "")

            # 检查并更新references的full_text
            updated_references = []
            references_updated = False

            for ref in entry.get("references", []):
                updated_ref = ref.copy()

                # 检查是否缺少full_text字段
                if (
                    "full_text" not in updated_ref
                    or not updated_ref.get("full_text")
                    or "Too Many Requests" in updated_ref.get("full_text", "") or len(updated_ref.get("full_text", "")) < 2000
                ):
                    if ref.get("arxiv_id"):
                        print(f"Fetching missing full text for reference: {ref.get('arxiv_id')}")
                        try:
                            ref_full_text = await get_md_info_with_fallback(ref["arxiv_id"])
                            updated_ref["full_text"] = ref_full_text
                            references_updated = True
                        except Exception as e:
                            print(f"Failed to get full text for reference {ref.get('arxiv_id')}: {e}")
                            updated_ref["full_text"] = ""
                    else:
                        updated_ref["full_text"] = ""

                updated_references.append(updated_ref)

            # 更新references并立即写入
            entry["references"] = updated_references
            temp_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            temp_f.flush()  # 确保立即写入磁盘

            if references_updated:
                print(
                    f"Updated and saved references for: {entry.get('title', 'Unknown')}"
                )
            else:
                print(f"No updates needed for: {entry.get('title', 'Unknown')}")

        # 处理新的URL，同样边处理边写入
        for url in arxiv_urls:
            arxiv_id = url.strip().split("/")[-1]
            print(f"Processing new URL: {arxiv_id}...")
            try:
                metadata = get_arxiv_metadata(arxiv_id)
                # 检查标题是否已存在
                if metadata["title"] in existing_titles:
                    print(
                        f"Skipping {arxiv_id} - title already exists: {metadata['title']}"
                    )
                    continue

                full_text = await get_md_info(arxiv_id)

                references = get_references_by_arxiv_id(arxiv_id)

                # 为新的references补充full_text字段
                enhanced_references = []
                for ref in references:
                    enhanced_ref = ref.copy()

                    # 如果有arxiv_id，尝试获取全文
                    if ref.get("arxiv_id"):
                        print(
                            f"Fetching full text for new reference: {ref.get('arxiv_id')}"
                        )
                        try:
                            ref_full_text = await get_md_info(ref["arxiv_id"])
                            enhanced_ref["full_text"] = ref_full_text
                        except Exception as e:
                            print(
                                f"Failed to get full text for reference {ref.get('arxiv_id')}: {e}"
                            )
                            enhanced_ref["full_text"] = ""
                    else:
                        enhanced_ref["full_text"] = ""

                    enhanced_references.append(enhanced_ref)

                entry = {
                    "arxiv_id": metadata["arxiv_id"],
                    "title": metadata["title"],
                    "authors": metadata["authors"],
                    "abstract": metadata["abstract"],
                    "published": metadata["published"],
                    "arxiv_url": metadata["arxiv_url"],
                    "full_text": full_text,
                    "references": enhanced_references,
                    # 其他元数据信息
                    "meta": {
                        "pdf_url": metadata["pdf_url"],
                        "updated": metadata.get("updated"),
                        "entry_id": metadata.get("entry_id"),
                        "primary_category": metadata.get("primary_category"),
                        "categories": metadata.get("categories", []),
                        "comment": metadata.get("comment", ""),
                        "journal_ref": metadata.get("journal_ref", ""),
                        "doi": metadata.get("doi", ""),
                        "links": metadata.get("links", {}),
                        "arxiv_pdf_url": metadata.get("arxiv_pdf_url"),
                        "ar5iv_url": metadata.get("ar5iv_url"),
                    },
                }

                temp_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                temp_f.flush()  # 确保数据及时写入
                existing_titles.add(metadata["title"])  # 添加到已处理集合
                print(f"Successfully processed and saved: {metadata['title']}")

            except Exception as e:
                print(f"[Error] Failed to process {arxiv_id}: {e}")

    # 处理完成后，将临时文件替换为正式文件
    if os.path.exists(temp_file):
        os.replace(temp_file, output_json)
        print(f"Successfully replaced {output_json} with updated data")

    print(f"\n✅ 数据已保存到 {output_json}")


def fix_existing_data(file_path):
    """修复现有数据文件中的类型不一致问题"""
    fixed_file = file_path + ".fixed"

    with open(file_path, "r", encoding="utf-8") as f_in, \
         open(fixed_file, "w", encoding="utf-8") as f_out:

        for line in f_in:
            try:
                data = json.loads(line.strip())

                # 修复meta.links字段，确保结构一致
                if "meta" in data and "links" in data["meta"]:
                    links = data["meta"]["links"]
                    data["meta"]["links"] = {
                        "null": str(links.get("null", "")),
                        "pdf": str(links.get("pdf", "")),
                        "doi": str(links.get("doi", "")),
                    }
                elif "meta" in data:
                    data["meta"]["links"] = {
                        "null": "",
                        "pdf": "",
                        "doi": "",
                    }

                # 修复references中的字段
                if "references" in data:
                    for ref in data["references"]:
                        # 修复year字段
                        if "year" in ref and ref["year"] is not None:
                            ref["year"] = str(ref["year"])
                        else:
                            ref["year"] = ""

                        # 修复external_ids字段，确保结构一致
                        if "external_ids" in ref and ref["external_ids"]:
                            external_ids = ref["external_ids"]
                            ref["external_ids"] = {
                                "DBLP": str(external_ids.get("DBLP", "")),
                                "DOI": str(external_ids.get("DOI", "")),
                                "CorpusId": int(external_ids.get("CorpusId", 0)) if external_ids.get("CorpusId") else 0,
                                "ArXiv": str(external_ids.get("ArXiv", "")),
                                "ACL": str(external_ids.get("ACL", "")),
                                "MAG": str(external_ids.get("MAG", "")),
                                "PubMedCentral": str(external_ids.get("PubMedCentral", "")),
                                "PubMed": str(external_ids.get("PubMed", "")),
                            }
                        else:
                            ref["external_ids"] = {
                                "DBLP": "",
                                "DOI": "",
                                "CorpusId": 0,
                                "ArXiv": "",
                                "ACL": "",
                                "MAG": "",
                                "PubMedCentral": "",
                                "PubMed": "",
                            }

                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")
                continue
            except Exception as e:
                print(f"Error processing line: {e}")
                continue

    # 备份原文件并替换
    if os.path.exists(file_path + ".backup_broken"):
        os.remove(file_path + ".backup_broken")
    os.rename(file_path, file_path + ".backup_broken")
    os.rename(fixed_file, file_path)
    print(f"Fixed data file: {file_path}")


if __name__ == "__main__":
    arxiv_urls = [
        "https://arxiv.org/abs/2306.11646",
        "https://arxiv.org/abs/2102.12982",
        "https://arxiv.org/abs/2311.09008",
        "https://arxiv.org/abs/2206.15030",
        "https://arxiv.org/abs/2209.01824",
        "https://arxiv.org/abs/2311.08298",
        "https://arxiv.org/abs/2201.05337",
        "https://arxiv.org/abs/2112.08313",
        "https://arxiv.org/abs/2006.00575",
        "https://arxiv.org/abs/2204.09269",
        "https://arxiv.org/abs/2309.00770",
        "https://arxiv.org/abs/2209.00099",
        "https://arxiv.org/abs/2312.00678",
        "https://arxiv.org/abs/2110.05006",
        "https://arxiv.org/abs/2212.09660",
        "https://arxiv.org/abs/2305.02750",
        "https://arxiv.org/abs/2202.13675",
        "https://arxiv.org/abs/2005.06249",
        "https://arxiv.org/abs/2010.00389",
        "https://arxiv.org/abs/2309.15402",
        "https://arxiv.org/abs/2301.00234",
        "https://arxiv.org/abs/2305.19860",
        "https://arxiv.org/abs/2310.15654",
        "https://arxiv.org/abs/2311.05112",
        "https://arxiv.org/abs/2302.09270",
        "https://arxiv.org/abs/2311.05232",
        "https://arxiv.org/abs/2504.15585",
        "https://arxiv.org/abs/2407.11511",
        "https://arxiv.org/abs/2304.00685",
        "https://arxiv.org/abs/2407.16216",
        "https://arxiv.org/abs/2408.03539",
        "https://arxiv.org/abs/2402.00253",
        "https://arxiv.org/abs/2312.02003",
        "https://arxiv.org/abs/2406.03712",
        "https://arxiv.org/abs/2411.15594",
        "https://arxiv.org/abs/2401.11641",
        "https://arxiv.org/abs/2312.10997",
        "https://arxiv.org/abs/2407.06204",
        "https://arxiv.org/abs/2404.04925",
        "https://arxiv.org/pdf/2302.00487",
        "https://arxiv.org/abs/2403.14608",
        "https://arxiv.org/abs/2401.06805",
        "https://arxiv.org/abs/2311.07226",
        "https://arxiv.org/abs/2405.14093",
        "https://arxiv.org/abs/2404.00629",
        "https://arxiv.org/abs/2308.11432",
        # 添加更多链接
    ]
    # for i in range(5):
    #     asyncio.run(process_arxiv_papers(arxiv_urls))


            # print("length more than 5000",len(data["full_text"])>5000)
            # print("=" * 50)

    src_file ="./surveyscope.jsonl"
    for i in range(5):
        arxiv_urls = []
        with open(src_file, "r") as fr:
            for line in fr:
                data = json.loads(line.strip())
                # print(data["title"])
                if not len(data["full_text"]) > 5000:
                    print(len(data["full_text"]), data["full_text"][:100])
                    print(data["arxiv_url"])
                    arxiv_urls.append(data["arxiv_url"])
        print(f"Found {len(arxiv_urls)} URLs with full_text length <= 5000")
        if arxiv_urls:
            asyncio.run(process_arxiv_papers(arxiv_urls))

    fix_existing_data(src_file)
    from datasets import load_dataset
    dataset = load_dataset("json", data_files=src_file, split="train")
