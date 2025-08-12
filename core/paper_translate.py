import configuration
import re
import time
import traceback
from typing import Optional
from model_factory import llm_map
from configuration import DEFAULT_MODEL_FOR_TRASNLATION
from langchain_core.messages import HumanMessage
from log import logger

async def translate_markdown_to_chinese(markdown_content: str, model_name: str = None) -> str:
    """
    翻译markdown内容到中文，保持References部分不变

    Args:
        markdown_content: 原始markdown内容
        model_name: 用于翻译的模型名称，如果不提供则使用默认模型

    Returns:
        翻译后的markdown内容
    """
    try:
        # 分割内容：References之上和之下
        references_pattern = r'^## References\s*$'
        parts = re.split(references_pattern, markdown_content, flags=re.MULTILINE)

        if len(parts) == 1:
            # 没有找到References部分，翻译整个内容
            content_to_translate = markdown_content
            references_content = ""
        else:
            # 找到References部分
            content_to_translate = parts[0].rstrip()
            references_content = "## References\n" + "".join(parts[1:])

        # 如果内容为空，直接返回
        if not content_to_translate.strip():
            logger.warning("No content to translate")
            return markdown_content

        # 准备翻译的prompt
        translation_prompt = f"""请将以下英文学术论文内容翻译成中文，要求：
1. 保持专业科学的口吻
2. 保持所有markdown格式不变（包括标题、列表、引用等）
3. 保持文中的引用格式不变（如[1], [2]等）
4. 保持专业术语的准确性
5. 确保翻译流畅自然
6. 保持段落结构和缩进格式

原文内容：
{content_to_translate}

请只返回翻译后的中文内容，不要添加任何解释或说明。"""

        # 使用LocalChatModel进行翻译
        model_to_use = model_name or DEFAULT_MODEL_FOR_TRASNLATION
        logger.info(f"Using model {model_to_use} for translation")
        translate_llm = llm_map[model_to_use].with_config({"temperature": 0.2})

        # 创建消息并调用模型
        message = HumanMessage(content=translation_prompt)
        response = await translate_llm.ainvoke([message])
        translated_content = response.content.strip()

        # 验证翻译结果
        if not translated_content:
            logger.error("Translation returned empty content")
            return markdown_content

        # 组合翻译后的内容和References部分
        final_content = translated_content
        if references_content:
            final_content += "\n\n" + references_content

        logger.info("Successfully translated markdown content to Chinese")
        return final_content

    except Exception as e:
        logger.error(f"Failed to translate markdown content: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # 翻译失败时返回原文
        return markdown_content

def translate_text_in_chunks(text: str, max_chunk_size: int = 4000) -> list:
    """
    将长文本分块以避免token限制

    Args:
        text: 要分块的文本
        max_chunk_size: 每块的最大字符数

    Returns:
        文本块列表
    """
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    lines = text.split('\n')
    current_chunk = ""

    for line in lines:
        # 如果加上这一行会超过限制
        if len(current_chunk) + len(line) + 1 > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = line + '\n'
            else:
                # 单行就超过限制，强制分割
                chunks.append(line)
        else:
            current_chunk += line + '\n'

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

async def translate_large_markdown_to_chinese(markdown_content: str, model_name: str = None) -> str:
    """
    翻译大型markdown内容到中文，支持分块处理

    Args:
        markdown_content: 原始markdown内容
        model_name: 用于翻译的模型名称

    Returns:
        翻译后的markdown内容
    """
    try:
        # 分割内容：References之上和之下
        references_pattern = r'^## References\s*$'
        parts = re.split(references_pattern, markdown_content, flags=re.MULTILINE)

        if len(parts) == 1:
            content_to_translate = markdown_content
            references_content = ""
        else:
            content_to_translate = parts[0].rstrip()
            references_content = "## References\n" + "".join(parts[1:])

        # 检查内容长度，决定是否需要分块
        if len(content_to_translate) <= 4000:
            # 内容较短，直接翻译
            return await translate_markdown_to_chinese(markdown_content, model_name)

        logger.info("Content is large, processing in chunks...")

        # 分块处理
        chunks = translate_text_in_chunks(content_to_translate, max_chunk_size=4000)
        translated_chunks = []

        model_to_use = model_name or DEFAULT_MODEL_FOR_TRASNLATION
        translate_llm = llm_map[model_to_use].with_config({"temperature": 0.2})

        for i, chunk in enumerate(chunks):
            logger.info(f"Translating chunk {i+1}/{len(chunks)}")

            translation_prompt = f"""请将以下英文学术论文片段翻译成中文，要求：
1. 保持专业科学的口吻
2. 保持所有markdown格式不变（包括标题、列表、引用等）
3. 保持文中的引用格式不变（如[1], [2]等）
4. 保持专业术语的准确性
5. 确保翻译流畅自然
6. 这是第{i+1}段，共{len(chunks)}段，请确保翻译的连贯性

原文片段：
{chunk}

请只返回翻译后的中文内容，不要添加任何解释或说明。"""

            message = HumanMessage(content=translation_prompt)
            response = await translate_llm.ainvoke([message])
            translated_chunk = response.content.strip()

            if translated_chunk:
                translated_chunks.append(translated_chunk)
            else:
                logger.warning(f"Chunk {i+1} translation failed, using original")
                translated_chunks.append(chunk)

        # 合并翻译后的内容
        final_content = "\n\n".join(translated_chunks)
        if references_content:
            final_content += "\n\n" + references_content

        logger.info("Successfully translated large markdown content to Chinese")
        return final_content

    except Exception as e:
        logger.error(f"Failed to translate large markdown content: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return markdown_content