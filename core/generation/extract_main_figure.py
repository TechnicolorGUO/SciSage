# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] : ArxivImageExtractor - A tool to extract main figures from arXiv papers
# ==================================================================
import io
import re
import requests
import tarfile
import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from local_request_v2 import get_from_llm
from log import logger
import base64
from io import BytesIO
import os
import json

proxies = {"http": "http://localhost:1080", "https": "http://localhost:1080"}


class ArxivImageExtractor:
    """从arXiv论文中提取主图的工具类"""

    def __init__(
        self,
        cache_dir: str = "./arxiv_cache",
        llm_provider: str = "openai",
        image_extraction_model: str = "Qwen3-8B",
    ):
        """
        初始化ArXiv图片提取器

        Args:
            cache_dir: 缓存目录，用于存储下载的PDF和源文件
            llm_provider: LLM服务提供商，默认为openai
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.llm_provider = llm_provider
        self.image_extraction_model = image_extraction_model

    def _resize_image(
        self, image: Image.Image, max_width: int = 960, max_height: int = 540
    ) -> Image.Image:
        """
        调整图像大小，限制最大宽度和高度，同时保持纵横比

        Args:
            image: PIL图像对象
            max_width: 最大宽度（默认1920）
            max_height: 最大高度（默认1080）

        Returns:
            调整大小后的图像
        """
        width, height = image.size
        if width <= max_width and height <= max_height:
            return image  # 如果图像已经小于最大尺寸，无需调整

        # 计算缩放比例
        width_ratio = max_width / width
        height_ratio = max_height / height
        scale = min(width_ratio, height_ratio)  # 使用较小的比例以保持纵横比

        # 计算新尺寸
        new_width = int(width * scale)
        new_height = int(height * scale)

        # 调整图像大小
        resized_image = image.resize(
            (new_width, new_height), Image.Resampling.LANCZOS
        )  # 使用高质量缩放算法
        return resized_image

    def get_main_figure(self, arxiv_id_or_info: Union[str, Dict]) -> Optional[str]:
        """
        根据arXiv ID或论文信息获取论文的主图

        Args:
            arxiv_id_or_info: 可以是arXiv ID (如 "2104.01155")或包含论文信息的字典

        Returns:
            保存的主图路径，如果未找到则返回None
        """
        # 确定是字符串ID还是信息字典
        if isinstance(arxiv_id_or_info, str):
            # 是字符串ID，按原逻辑处理
            arxivId = arxiv_id_or_info.replace("arxiv:", "").strip()
            # 获取论文信息
            paper_info = self._get_paper_info(arxivId)
            if not paper_info:
                logger.info(f"无法获取论文信息: {arxivId}")
                paper_info = {"title": "", "abstract": "", "arxivId": arxivId}
        else:
            # 是字典，直接使用，但确保有必要的字段
            paper_info = arxiv_id_or_info

            # 确保字典包含必要的字段
            if not all(key in paper_info for key in ["title", "abstract"]):
                logger.info("提供的论文信息缺少必要字段 (title, abstract)")
                return None

            # 如果没有提供arxiv_id，尝试从传入字典中获取，否则生成一个临时ID
            if "arxivId" not in paper_info:
                if "entry_id" in paper_info:
                    # 如果有entry_id，从中提取arxiv_id
                    entry_id = paper_info["entry_id"]
                    paper_info["arxivId"] = entry_id.split("/")[-1]
                else:
                    # 生成基于标题的临时ID
                    import hashlib

                    temp_id = hashlib.md5(paper_info["title"].encode()).hexdigest()[:10]
                    paper_info["arxivId"] = f"temp_{temp_id}"

            # 确保有PDF和源文件URL
            if "pdf_url" not in paper_info:
                if "arxivId" in paper_info and paper_info["arxivId"].startswith(
                    "temp_"
                ):
                    # 对于非arXiv论文，没有PDF和源文件URL
                    paper_info["pdf_url"] = None
                    paper_info["source_url"] = None
                else:
                    # 对于arXiv论文，构建标准URL
                    arxivId = paper_info["arxivId"]
                    paper_info["pdf_url"] = f"https://arxiv.org/pdf/{arxivId}.pdf"
                    paper_info["source_url"] = f"https://arxiv.org/e-print/{arxivId}"

        # 创建论文专用目录
        paper_dir = self.cache_dir / paper_info["arxivId"]
        paper_dir.mkdir(exist_ok=True)

        # 定义 JSON 文件路径
        json_path = paper_dir / f"{paper_info['arxivId']}_main_figure_info.json"

        # 下载PDF和源文件(如果有URL)
        pdf_path = None
        source_path = None

        if paper_info.get("pdf_url"):
            pdf_path = self._download_pdf(
                paper_info["pdf_url"], paper_dir / f"{paper_info['arxivId']}.pdf"
            )

        if paper_info.get("source_url"):
            source_path = self._download_source(
                paper_info["source_url"], paper_dir / f"{paper_info['arxivId']}.tar.gz"
            )

        # 从源文件提取所有图片和caption
        all_figures = []
        if source_path:
            logger.info(f"从源文件提取图像: {source_path}")
            all_figures = self._extract_figures_with_captions(
                source_path, paper_dir, paper_info
            )

        # 如果没有从源文件获取任何图像，则从PDF提取
        if not all_figures and pdf_path:
            logger.info("从源文件中未找到任何图像，尝试从PDF提取")
            all_figures = self._extract_figures_from_pdf(pdf_path, paper_dir)

        # 如果没有找到任何图像，返回None
        if not all_figures:
            logger.info(f"未能从论文中提取任何图像: {paper_info['arxivId']}")
            return None

        # 使用LLM评估每个图像的caption，找出最可能是主图的图像
        main_figure = self._identify_main_figure(all_figures, paper_info)

        if main_figure:
            # 保存主图
            main_figure_path = paper_dir / f"{paper_info['arxivId']}_main_figure.png"
            main_figure["stitched_image"].save(main_figure_path, "PNG")
            logger.info(
                f"保存主图 (主图概率: {main_figure['main_figure_score']:.2f}): {main_figure_path}"
            )

            # 将图像转换为 Base64 编码
            buffered = BytesIO()
            main_figure["stitched_image"].save(buffered, format="PNG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            buffered.close()

            # 准备主图信息字典
            main_figure_info = {
                "main_figure_path": os.path.abspath(main_figure_path),
                "main_figure_base64": img_base64,
                "main_figure_caption": main_figure["caption"],
                "main_figure_score": main_figure["main_figure_score"],
            }

            # 保存到 JSON 文件
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(main_figure_info, f, ensure_ascii=False, indent=4)
            logger.info(f"主图信息已保存到: {json_path}")

            return main_figure_info

        return None

    def _get_paper_info(self, arxivId: str) -> Dict:
        """获取论文的基本信息"""
        # 构建API URL
        api_url = f"https://export.arxiv.org/api/query?id_list={arxivId}"
        logger.info(f"api_url: {api_url}")
        try:
            response = requests.get(api_url, proxies=proxies, timeout=5)
            response.raise_for_status()

            # 从响应中解析标题和摘要
            title_match = re.search(r"<title>(.*?)</title>", response.text)
            title = title_match.group(1) if title_match else "Unknown Title"

            # 提取摘要
            abstract_match = re.search(
                r"<summary>(.*?)</summary>", response.text, re.DOTALL
            )
            abstract = abstract_match.group(1).strip() if abstract_match else ""

            # 构建PDF和源文件URL
            pdf_url = f"https://arxiv.org/pdf/{arxivId}.pdf"
            source_url = f"https://arxiv.org/e-logger.info/{arxivId}"

            return {
                "title": title,
                "abstract": abstract,
                "pdf_url": pdf_url,
                "source_url": source_url,
                "arxivId": arxivId,
            }
        except Exception as e:
            logger.info(f"获取论文信息失败: {e}")
            return None

    def _download_pdf(self, url: str, save_path: Path) -> Optional[Path]:
        """下载PDF文件"""
        if save_path.exists():
            return save_path

        try:
            response = requests.get(url, proxies=proxies)
            response.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(response.content)
            return save_path
        except Exception as e:
            logger.info(f"下载PDF文件失败: {e}")
            return None

    def _download_source(self, url: str, save_path: Path) -> Optional[Path]:
        """下载论文源文件"""
        if save_path.exists():
            return save_path

        try:
            response = requests.get(url, proxies=proxies)
            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(response.content)
                return save_path
            else:
                logger.info(f"无法下载源文件，状态码: {response.status_code}")
                return None
        except Exception as e:
            logger.info(f"下载源文件失败: {e}")
            return None

    def _extract_figures_with_captions(
        self, source_path: Path, paper_dir: Path, paper_info: Dict
    ) -> List[Dict]:
        """从源文件中提取所有图像及其标题"""
        source_extract_dir = paper_dir / "source"
        source_extract_dir.mkdir(exist_ok=True)

        figures_data = []

        try:
            # 解压源文件
            with tarfile.open(source_path, "r:gz") as tar:
                tar.extractall(source_extract_dir)

                # 查找所有.tex文件
                tex_files = []
                for file_path in source_extract_dir.glob("**/*.tex"):
                    tex_files.append(file_path)

                # 处理每个.tex文件
                for tex_idx, tex_path in enumerate(tex_files):
                    with open(tex_path, "r", encoding="utf-8", errors="ignore") as f:
                        tex_content = f.read()

                    # 提取所有figure环境（包括figure*）
                    figure_pattern = r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}"
                    figure_matches = re.findall(figure_pattern, tex_content, re.DOTALL)

                    for idx, figure_content in enumerate(figure_matches):
                        # 从figure内容中提取caption
                        caption_pattern = r"\\caption\{(.*?)\}"
                        caption_match = re.search(
                            caption_pattern, figure_content, re.DOTALL
                        )
                        caption = (
                            caption_match.group(1).strip() if caption_match else ""
                        )

                        # 从figure内容中提取图像
                        figure_images = []
                        img_pattern = r"\\includegraphics(?:\[.*?\])?\{(.*?)\}"
                        img_files = re.findall(img_pattern, figure_content)

                        for img_file in img_files:
                            img_file = img_file.strip()
                            # 处理相对路径，查找实际图片文件
                            img_path = self._find_image_file(
                                source_extract_dir, img_file, tex_path.parent
                            )

                            if img_path and img_path.exists():
                                try:
                                    if img_path.suffix.lower() == ".pdf":
                                        img = self._convert_pdf_to_image(img_path)
                                    else:
                                        img = Image.open(img_path)
                                    figure_images.append(img)
                                except Exception as e:
                                    logger.info(f"无法处理图片 {img_path}: {e}")

                        if figure_images:
                            # 拼接当前figure的图片
                            stitched_img = self._stitch_images(figure_images)
                            if stitched_img:
                                # 去除caption中的LaTeX命令
                                clean_caption = self._clean_latex_commands(caption)
                                img_path = (
                                    paper_dir
                                    / f"{paper_info['arxivId']}_{tex_idx+1}_{idx+1}_figure.png"
                                )

                                stitched_img = self._resize_image(stitched_img)

                                figures_data.append(
                                    {
                                        "figure_index": f"{tex_idx+1}_{idx+1}",
                                        "stitched_image": stitched_img,
                                        "stitched_image_path": stitched_img.save(
                                            img_path, "PNG"
                                        ),
                                        "caption": clean_caption,
                                        "area": stitched_img.width
                                        * stitched_img.height,
                                        "source": "tex",
                                    }
                                )

            return figures_data

        except Exception as e:
            logger.info(f"处理源文件失败: {e}")
            return []

    def _clean_latex_commands(self, text: str) -> str:
        """清理LaTeX命令，返回纯文本"""
        # 移除常见的LaTeX命令，如\cite{}, \ref{}, \textbf{} 等
        text = re.sub(r"\\cite\{.*?\}", "", text)
        text = re.sub(r"\\ref\{.*?\}", "", text)
        text = re.sub(r"\\label\{.*?\}", "", text)

        # 保留命令内的文本内容
        text = re.sub(r"\\textbf\{(.*?)\}", r"\1", text)
        text = re.sub(r"\\textit\{(.*?)\}", r"\1", text)
        text = re.sub(r"\\emph\{(.*?)\}", r"\1", text)

        # 特殊处理 \ding{} 命令
        text = re.sub(r"\\ding\{\d+\}", "", text)

        # 移除数学符号
        text = re.sub(r"\$.*?\$", "", text)

        # 移除其他LaTeX命令
        text = re.sub(r"\\[a-zA-Z]+", "", text)

        # 合并多个空格
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _find_image_file(
        self, base_dir: Path, image_path: str, tex_dir: Path
    ) -> Optional[Path]:
        """查找图片文件的实际路径"""
        # 删除可能的扩展名
        img_base = image_path.split(".")[0]

        # 常见图片扩展名
        extensions = [".png", ".jpg", ".jpeg", ".pdf", ".eps", ".ps"]

        # 尝试完整路径
        for ext in extensions:
            potential_path = (
                tex_dir / f"{image_path}{ext}"
                if not any(image_path.endswith(e) for e in extensions)
                else tex_dir / image_path
            )
            if potential_path.exists():
                return potential_path

        # 尝试在tex文件目录中查找
        for ext in extensions:
            img_path = tex_dir / f"{img_base}{ext}"
            if img_path.exists():
                return img_path

        # 如果文件路径包含目录（如Fig/fig-overview.pdf）
        if "/" in image_path:
            # 尝试相对于tex目录的路径
            full_path = tex_dir / image_path
            if full_path.exists():
                return full_path

            # 尝试相对于base_dir的路径
            full_path = base_dir / image_path
            if full_path.exists():
                return full_path

        # 尝试在基础目录中递归查找
        for ext in extensions:
            for img_path in base_dir.glob(f"**/{img_base}{ext}"):
                return img_path

        return None

    def _convert_pdf_to_image(self, pdf_path: Path) -> Image.Image:
        """将PDF转换为图像"""
        doc = fitz.open(pdf_path)
        page = doc[0]  # 假设PDF只有一页
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img

    def _stitch_images(
        self, images: List[Image.Image], layout: str = "horizontal"
    ) -> Optional[Image.Image]:
        """拼接多个图像"""
        if not images:
            return None
        if len(images) == 1:
            return images[0]

        # 计算拼接后的尺寸
        max_width = max(img.width for img in images)
        max_height = max(img.height for img in images)

        if layout == "horizontal":
            total_width = max_width * len(images)
            total_height = max_height
        else:  # vertical
            total_width = max_width
            total_height = max_height * len(images)

        # 创建空白画布
        stitched_image = Image.new("RGB", (total_width, total_height), (255, 255, 255))

        # 拼接图片
        for i, img in enumerate(images):
            if layout == "horizontal":
                stitched_image.paste(img, (i * max_width, 0))
            else:  # vertical
                stitched_image.paste(img, (0, i * max_height))

        return stitched_image

    def _extract_figures_from_pdf(self, pdf_path: Path, paper_dir: Path) -> List[Dict]:
        """从PDF中提取图像作为备选方案"""
        figures_data = []

        try:
            doc = fitz.open(pdf_path)
            for page_num in range(min(10, len(doc))):  # 检查前10页
                page = doc[page_num]
                image_list = page.get_images(full=True)

                # 尝试提取页面中的文字，查找可能的caption
                page_text = page.get_text("text")
                caption_matches = re.findall(
                    r"(?:Figure|Fig\.)\s+\d+[.:]\s*(.*?)(?:\n\n|\r\n\r\n|$)",
                    page_text,
                    re.DOTALL,
                )

                if image_list:
                    for img_idx, img_info in enumerate(image_list):
                        try:
                            xref = img_info[0]
                            base_image = doc.extract_image(xref)
                            img_bytes = base_image["image"]
                            img = Image.open(io.BytesIO(img_bytes))

                            # 如果图像太小，可能是装饰性元素，跳过
                            if img.width < 100 or img.height < 100:
                                continue

                            # 尝试匹配caption
                            caption = ""
                            if img_idx < len(caption_matches):
                                caption = caption_matches[img_idx].strip()

                            img_path = (
                                paper_dir
                                / f"from_pdf_{page_num * 100 + img_idx}_figure.png"
                            )
                            img = self._resize_image(img)
                            figures_data.append(
                                {
                                    "figure_index": page_num * 100 + img_idx,
                                    "stitched_image": img,
                                    "stitched_image_path": img.save(img_path, "PNG"),
                                    "caption": caption,
                                    "area": img.width * img.height,
                                    "source": "pdf",
                                    "page": page_num + 1,
                                }
                            )
                        except Exception as e:
                            logger.info(
                                f"处理PDF第{page_num+1}页图像{img_idx}失败: {e}"
                            )

            doc.close()
            return figures_data
        except Exception as e:
            logger.info(f"从PDF提取图像失败: {e}")
            return []

    def _identify_main_figure(
        self, figures: List[Dict], paper_info: Dict
    ) -> Optional[Dict]:
        """使用LLM识别最可能的主图"""
        if not figures:
            return None

        # 如果只有一个图，直接返回
        if len(figures) == 1:
            figures[0]["main_figure_score"] = 1.0
            return figures[0]

        # 为每个图像评分
        logger.info(f"figures num :{len(figures)}")
        for figure in figures:
            score = self._evaluate_figure_with_llm(figure, paper_info)
            figure["main_figure_score"] = score
            logger.info(
                f"图 {figure['figure_index']} 主图可能性评分: {score:.2f}, 面积: {figure['area']}"
            )

        # 按评分排序
        figures.sort(key=lambda x: (x["main_figure_score"], x["area"]), reverse=True)

        return figures[0]

    def _evaluate_figure_with_llm(self, figure: Dict, paper_info: Dict) -> float:
        """使用LLM评估图像是否为主图"""
        # 构建提示
        prompt = f"""
        Task: Evaluate whether a figure is the main overview figure of a research paper.

        Paper Information:
        Title: {paper_info['title']}
        Abstract: {paper_info['abstract']}

        Figure Information:
        Caption: {figure['caption'] if figure['caption'] else '[No caption available]'}

        Based on the above information, evaluate if this figure is likely to be the main overview figure of the paper.
        A main overview figure typically:
        1. Presents the key architecture, framework, or method of the paper
        2. Is often referenced early in the paper
        3. Provides a visual summary of the paper's contribution
        4. May contain multiple components showing the overall approach

        Return only a probability between 0 and 1, where 1 means definitely the main overview figure and 0 means definitely not.
        Be strict in your evaluation - if there's not enough evidence, give a lower score.

        Probability:
        """

        # 调用LLM
        result = self._call_llm(prompt)

        # 解析结果，提取概率值
        try:
            # 尝试从响应中提取数字
            score_match = re.search(r"(\d+\.\d+|\d+)", result)
            if score_match:
                score = float(score_match.group(1))
                # 确保分数在0-1范围内
                return max(0.0, min(1.0, score))
            else:
                # 如果无法提取数字，使用启发式规则
                result = result.lower()
                if (
                    "definitely" in result
                    or "certainly" in result
                    or "very likely" in result
                ):
                    return 0.9
                elif "likely" in result or "probably" in result:
                    return 0.7
                elif "possibly" in result or "might be" in result:
                    return 0.5
                elif "unlikely" in result or "probably not" in result:
                    return 0.3
                else:
                    return 0.5  # 默认中等可能性
        except Exception as e:
            logger.info(f"解析LLM结果时出错: {e}")
            # 退回到基于面积的启发式方法
            area_normalized = min(1.0, figure["area"] / 1000000)
            return 0.5 * area_normalized + 0.2

    def _call_llm(self, prompt: str) -> str:
        """调用LLM服务获取预测结果"""
        # 这里可以根据具体的LLM提供商实现调用逻辑
        # 以下为示例，实际应用中需要根据您使用的模型调整

        if self.llm_provider == "openai":
            try:
                import openai

                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10,  # 只需要一个数字
                )
                return response.choices[0].message.content
            except ImportError:
                logger.info("OpenAI库未安装，使用模拟响应")
                # 使用文本分析给出一个估计
                if "overview" in prompt.lower() or "architecture" in prompt.lower():
                    return "0.85"
                else:
                    return "0.5"
        else:
            for i in range(5):
                try:
                    response = get_from_llm(
                        prompt, model_name=self.image_extraction_model
                    )
                    return response
                except Exception as e:
                    logger.info(f"调用LLM服务失败: {e}")
                    # 使用模拟响应
            if "overview" in prompt.lower() or "architecture" in prompt.lower():
                logger.info("keywords matched, using default response")
                return "0.85"
            else:
                return "0.5"

        # 如果没有配置LLM或发生错误，返回中等可能性
        logger.info("无法连接到LLM服务，使用默认响应")
        return "0.5"


def get_arxiv_main_figure(
    arxiv_id_or_info: Union[str, Dict],
    llm_provider: str = "local",
    image_extraction_model: str = "Qwen3-8B",
) -> Optional[Dict]:
    """
    获取arXiv论文的主图（便捷函数）

    Args:
        arxiv_id_or_info: arXiv论文ID或包含论文信息的字典
        llm_provider: LLM提供商

    Returns:
        包含主图信息的字典，如果未找到则返回None
    """
    # 确定 arxivId
    if isinstance(arxiv_id_or_info, str):
        arxivId = arxiv_id_or_info.replace("arxiv:", "").strip()
    else:
        arxivId = (
            arxiv_id_or_info.get("arxivId")
            or arxiv_id_or_info.get("entry_id", "").split("/")[-1]
        )
        if not arxivId:
            import hashlib

            arxivId = f"temp_{hashlib.md5(arxiv_id_or_info['title'].encode()).hexdigest()[:10]}"

    # 检查是否已有缓存的 JSON 文件
    cache_dir = Path("./arxiv_cache")
    json_path = cache_dir / arxivId / f"{arxivId}_main_figure_info.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                main_figure_info = json.load(f)
            # 验证文件路径是否存在
            if os.path.exists(main_figure_info["main_figure_path"]):
                logger.info(f"加载缓存的主图信息: {json_path}")
                main_figure_info["main_figure_path"] = os.path.abspath(
                    main_figure_info["main_figure_path"]
                )
                return main_figure_info
            else:
                logger.info(
                    f"缓存的主图文件不存在，将重新处理: {main_figure_info['main_figure_path']}"
                )
        except (json.JSONDecodeError, KeyError) as e:
            logger.info(f"缓存的 JSON 文件无效，将重新处理: {e}")

    # 如果没有缓存或缓存无效，执行提取
    extractor = ArxivImageExtractor(
        llm_provider=llm_provider, image_extraction_model=image_extraction_model
    )
    result = extractor.get_main_figure(arxiv_id_or_info)

    if result is not None:
        main_figure_info = {
            "main_figure_path": os.path.abspath(result["main_figure_path"]),
            "main_figure_base64": result["main_figure_base64"],
            "main_figure_caption": result["main_figure_caption"],
            "main_figure_score": result["main_figure_score"],
        }
        return main_figure_info
    return None


paper_info = {
    "arxivId": "2503.21460",
    "arxivUrl": "http://arxiv.org/abs/2503.21460v1",
    "title": "Large Language Model Agent: A Survey on Methodology, Applications and Challenges",
    "abstract": "The era of intelligent agents is upon us, driven by revolutionary advancements in large language models. Large Language Model (LLM) agents, with goal-driven behaviors and dynamic adaptation capabilities, potentially represent a critical pathway toward artificial general intelligence. This survey systematically deconstructs LLM agent systems through a methodology-centered taxonomy, linking architectural foundations, collaboration mechanisms, and evolutionary pathways. We unify fragmented research threads by revealing fundamental connections between agent design principles and their emergent behaviors in complex environments. Our work provides a unified architectural perspective, examining how agents are constructed, how they collaborate, and how they evolve over time, while also addressing evaluation methodologies, tool applications, practical challenges, and diverse application domains. By surveying the latest developments in this rapidly evolving field, we offer researchers a structured taxonomy for understanding LLM agents and identify promising directions for future research. The collection is available at https://github.com/luo-junyu/Awesome-Agent-Papers.",
    "authors": "Junyu Luo;Weizhi Zhang;Ye Yuan;Yusheng Zhao;Junwei Yang;Yiyang Gu;Bohan Wu;Binqi Chen;Ziyue Qiao;Qingqing Long;Rongcheng Tu;Xiao Luo;Wei Ju;Zhiping Xiao;Yifan Wang;Meng Xiao;Chenwu Liu;Jingyang Yuan;Shichang Zhang;Yiqiao Jin;Fan Zhang;Xian Wu;Hanqing Zhao;Dacheng Tao;Philip S. Yu;Ming Zhang",
    "year": "20250327",
    "fieldsOfStudy": ["cs.CL"],
    "source": "Search From Arxiv",
    "url": "http://arxiv.org/abs/2503.21460v1",
    "text": "The era of intelligent agents is upon us, driven by revolutionary advancements in large language models. Large Language Model (LLM) agents, with goal-driven behaviors and dynamic adaptation capabilities, potentially represent a critical pathway toward artificial general intelligence. This survey systematically deconstructs LLM agent systems through a methodology-centered taxonomy, linking architectural foundations, collaboration mechanisms, and evolutionary pathways. We unify fragmented research threads by revealing fundamental connections between agent design principles and their emergent behaviors in complex environments. Our work provides a unified architectural perspective, examining how agents are constructed, how they collaborate, and how they evolve over time, while also addressing evaluation methodologies, tool applications, practical challenges, and diverse application domains. By surveying the latest developments in this rapidly evolving field, we offer researchers a structured taxonomy for understanding LLM agents and identify promising directions for future research. The collection is available at https://github.com/luo-junyu/Awesome-Agent-Papers.",
}

# main_figure_info = get_arxiv_main_figure(paper_info, llm_provider="local")

# logger.info(f"主图路径: {main_figure_info.keys()}")

# # 使用示例
# if __name__ == "__main__":
#     import sys
#     import argparse

#     parser = argparse.ArgumentParser(description="提取arXiv论文的主图")
#     parser.add_argument("arxivId", nargs="?", help="arXiv论文ID")
#     parser.add_argument("--llm", default="local", help="LLM提供商 (默认: openai)")
#     parser.add_argument("--title", help="论文标题 (可选)")
#     parser.add_argument("--abstract", help="论文摘要 (可选)")

#     args = parser.parse_args()

#     # 检查是否提供了标题和摘要
#     if args.title and args.abstract:
#         # 使用提供的信息创建字典
#         paper_info = {"title": args.title, "abstract": args.abstract}

#         # 如果提供了arxiv_id，添加到字典
#         if args.arxivId:
#             paper_info["arxivId"] = args.arxivId
#             paper_info[
#                 "pdf_url"] = f"https://arxiv.org/pdf/{args.arxivId}.pdf"
#             paper_info[
#                 "source_url"] = f"https://arxiv.org/e-logger.info/{args.arxivId}"

#         main_figure_path = get_arxiv_main_figure(paper_info,
#                                                  llm_provider=args.llm)
#     elif args.arxivId:
#         # 只提供了arxiv_id
#         main_figure_path = get_arxiv_main_figure(args.arxivId,
#                                                  llm_provider=args.llm)
#     else:
#         # 没有提供足够的信息
#         arxivId = input("请输入arXiv ID: ")
#         main_figure_path = get_arxiv_main_figure(arxivId,
#                                                  llm_provider=args.llm)

#     if main_figure_path:
#         logger.info(f"成功提取主图，保存于: {main_figure_path}")
#     else:
#         logger.info(f"无法提取论文的主图")
