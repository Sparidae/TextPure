import asyncio
import glob
import os
from typing import Dict, List, Optional, Set, Tuple, Union

from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer

from llm_interface import get_completion
from prompts import get_prompt

# 加载环境变量
load_dotenv(override=True)

# 环境配置
API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE", "https://api.siliconflow.cn/v1")
model_name = os.getenv("LLM_FOR_CLEAN", "Qwen/Qwen2.5-14B-Instruct")
context_length = int(os.getenv("LLM_FOR_CLEAN_CL", "4096"))
print(model_name)

# 加载计数用tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


def estimate_token_length(text: str) -> int:
    """估计文本的token长度"""
    token_count = len(tokenizer(text)["input_ids"])
    return token_count


class TextChunker:
    """文本分块器，用于将长文本分割成适合模型处理的小块"""

    def __init__(self, text: str, max_tokens: int = 32000, reserve_tokens: int = 200):
        """
        初始化文本分块器

        Args:
            text: 需要分块的文本
            max_tokens: 每个分片的最大token数量
            reserve_tokens: 为提示词等预留的token数量
        """
        self.text = text
        # 使用更保守的上下文容量限制
        self.max_tokens = max_tokens - reserve_tokens
        self.paragraphs = text.split("\n")

    def get_chunks(self) -> List[str]:
        """生成并返回所有文本块"""
        chunks = []
        current_chunk = []
        current_length = 0

        for paragraph in self.paragraphs:
            para_tokens = estimate_token_length(paragraph)

            # 处理超长段落
            if para_tokens > self.max_tokens:
                # 先保存当前收集的内容
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # 分割并保存超长段落
                chunks.append(self._split_long_paragraph(paragraph))
                continue

            # 检查是否需要开始新的chunk
            if current_length + para_tokens + 1 > self.max_tokens:  # +1 for newline
                chunks.append("\n".join(current_chunk))
                current_chunk = [paragraph]
                current_length = para_tokens
            else:
                current_chunk.append(paragraph)
                current_length += para_tokens + (
                    1 if current_chunk else 0
                )  # 加上换行符的token

        # 添加最后一个chunk
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    def _split_long_paragraph(self, paragraph: str) -> str:
        """分割超长段落并返回第一个可用的块"""
        sentences = paragraph.split(". ")
        chunk = []
        current_length = 0

        for sentence in sentences:
            # 确保句子以句号结尾
            if not sentence.endswith("."):
                sentence += "."

            sentence_tokens = estimate_token_length(sentence + " ")

            # 检查是否能添加这个句子
            if current_length + sentence_tokens <= self.max_tokens:
                chunk.append(sentence)
                current_length += sentence_tokens
            else:
                break

        # 如果一个句子都没加进去，至少返回一个最短的内容
        if not chunk and sentences:
            return sentences[0]

        return " ".join(chunk)


class LLMProcessor:
    """LLM处理器类，处理文本清理、关键词提取、内容组织等任务"""

    def __init__(
        self,
        model: str = model_name,
        max_tokens: int = context_length,
        max_concurrent: int = 5,
        temperature: float = 0.7,
        use_openai_client: bool = False,
    ):
        """
        初始化LLM处理器

        Args:
            model: 使用的模型名称
            max_tokens: 模型最大上下文长度
            max_concurrent: 最大并发请求数
            temperature: 模型温度参数
        """
        self.debug_save = True  # 设定的中间环节保存部分
        self.model = model
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent
        self.temperature = temperature
        self.completion_func = get_completion(use_openai_client)

    async def _call_llm_async(
        self, prompt: str, temp: Optional[float] = None
    ) -> Tuple[bool, str]:
        """异步调用LLM模型"""
        try:
            temperature = temp if temp is not None else self.temperature
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.completion_func(
                    prompt, model=self.model, temperature=temperature
                ),
            )
        except Exception as e:
            print(f"LLM调用异常: {str(e)}")
            return False, ""

    async def _process_chunks_async(
        self, chunks: List[str], process_func, desc: str
    ) -> List[Tuple[int, Union[str, Set[str]]]]:
        """
        异步处理多个文本块

        Args:
            chunks: 文本块列表
            process_func: 处理单个块的函数，该函数只接受一个chunk参数
            desc: 进度条描述

        Returns:
            处理结果列表，每项包含索引和处理结果
        """
        results = []
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_with_semaphore(i, chunk):
            async with semaphore:
                print(f"处理第 {i + 1}/{len(chunks)} 个{desc}")
                try:
                    result = await process_func(chunk)
                    return i, result
                except Exception as e:
                    print(f"处理第 {i + 1} 个{desc}时出错: {str(e)}")
                    return i, None

        # 创建异步任务列表
        tasks = [process_with_semaphore(i, chunk) for i, chunk in enumerate(chunks)]

        # 使用tqdm创建进度条
        pbar = tqdm(total=len(chunks), desc=f"处理{desc}")

        # 等待每个任务完成并更新进度条
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            pbar.update(1)

        pbar.close()
        return sorted(results, key=lambda x: x[0])  # 按索引排序

    async def clean_text_async(self, text: str) -> str:
        """
        使用LLM异步清理文本中的符号和格式字符

        Args:
            text: 需要清理的原始文本

        Returns:
            cleaned_text: 清理后的文本
        """
        # 针对raw文本分块
        chunker = TextChunker(text, max_tokens=self.max_tokens)
        chunks = chunker.get_chunks()

        async def process_chunk(chunk):  # 特定功能函数 传递给 块处理函数
            prompt = get_prompt("text_cleaning", text=chunk)
            success, cleaned_chunk = await self._call_llm_async(prompt)
            return cleaned_chunk if success else chunk

        results = await self._process_chunks_async(chunks, process_chunk, "文本块")

        ######### 将所有chunk的处理结果对比保存到一个文件中 #########
        if self.debug_save:
            output_dir = "data/03_processed_text"
            os.makedirs(output_dir, exist_ok=True)
            # 创建一个包含所有chunk的单一文件
            chunks_file = os.path.join(output_dir, "all_chunks.txt")
            with open(chunks_file, "w", encoding="utf-8") as f:
                for i, (chunk, result) in enumerate(zip(chunks, results)):
                    if result is not None:
                        f.write(f"\n\n{'=' * 50}\n")
                        f.write(f"Chunk {i + 1}\n\n{chunk}\n\n")
                        f.write(f"{'-' * 50}\n\n")
                        f.write(result[1])
            # 创建包含所有清理后文本的文件
            cleaned_file = os.path.join(output_dir, "3.1_cleaned_text.txt")
            with open(cleaned_file, "w", encoding="utf-8") as f:
                for _, result in results:
                    if result is not None:
                        f.write(result)
            print(f"已将所有处理后的文本块保存到 {chunks_file}")
            print(f"已将所有清理后的文本保存到 {cleaned_file}")

        return "\n".join(result for _, result in results if result is not None)

    async def extract_keywords_async(self, text: str) -> List[str]:
        """
        从文本中提取并筛选关键词

        Args:
            text: 清理后的文本

        Returns:
            keywords: 提取的关键词列表
        """
        # 针对清洗后的文本重新分块
        chunker = TextChunker(text, max_tokens=self.max_tokens)
        chunks = chunker.get_chunks()
        all_keywords = set()

        async def process_chunk(chunk):  # 提取关键词的特殊函数
            prompt = get_prompt("keywords_extraction", text=chunk)
            success, result = await self._call_llm_async(prompt, temp=0.3)
            if success:
                return {
                    kw.strip() for kw in result.split(",") if kw.strip()
                }  # 关注这一步分割关键词，可能会出问题，需要保留中途结果
            return set()  # 失败返回空集

        results = await self._process_chunks_async(chunks, process_chunk, "关键词提取")

        # 合并所有关键词 集合
        for _, keywords in results:
            if keywords:
                all_keywords.update(keywords)

        # 筛选最重要的关键词
        success = False
        filtered_keywords = []
        if all_keywords:
            prompt = get_prompt("keywords_filtering", keywords=", ".join(all_keywords))
            success, result = await self._call_llm_async(prompt, temp=0.3)
            filtered_keywords = [kw.strip() for kw in result.split(",") if kw.strip()]

        # 保存逻辑
        if self.debug_save:
            output_dir = "data/03_processed_text"
            os.makedirs(output_dir, exist_ok=True)
            keywords_file = os.path.join(output_dir, "3.2_keywords.txt")
            with open(keywords_file, "w", encoding="utf-8") as f:
                f.write("，".join(all_keywords))
                f.write("\n\n"+"=" * 50 + "过滤后\n\n")
                f.write("，".join(filtered_keywords))

            print(f"已将所有关键词保存到 {keywords_file}")

        if success:
            return filtered_keywords
        else:
            return list(all_keywords)

    async def extract_content_for_keywords_async(
        self, text: str, keywords: List[str]
    ) -> Dict[str, str]:
        """
        为关键词提取相关内容

        Args:
            text: 清理后的文本
            keywords: 关键词列表

        Returns:
            keyword_contents: 关键词-内容字典
        """
        keyword_contents = {}
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_keyword(keyword):
            async with semaphore:
                print(f"抽取关键词内容: {keyword}")
                return keyword, await self._extract_content_for_keyword(text, keyword)

        # 创建异步任务列表
        tasks = [process_keyword(keyword) for keyword in keywords]

        # 使用tqdm创建进度条
        pbar = tqdm(total=len(keywords), desc="抽取关键词内容")

        # 等待每个任务完成并更新进度条
        for coro in asyncio.as_completed(tasks):
            keyword, content = await coro
            if content:  # 只添加有内容的关键词
                keyword_contents[keyword] = content
            pbar.update(1)

        pbar.close()
        return keyword_contents

    async def _extract_content_for_keyword(self, text: str, keyword: str) -> str:
        """为单个关键词从所有文本块中提取并整合相关内容"""
        try:
            chunker = TextChunker(text, max_tokens=self.max_tokens)
            chunks = chunker.get_chunks()
            related_segments = []  # 内容块

            # 处理每个包含关键词的块
            for chunk in chunks:
                if keyword.lower() in chunk.lower():
                    prompt = get_prompt("content_extraction", keyword=keyword, text=chunk)
                    success, result = await self._call_llm_async(prompt, temp=0.3)
                    if success and result.strip():
                        related_segments.append(result.strip())

            # 如果找到相关内容，将其整合为一个自然段
            if related_segments:
                prompt = get_prompt(
                    "content_integration",
                    keyword=keyword,
                    segments=" ".join(related_segments),
                )  # TODO 整合这一段有可能会超过上下文限制。
                success, result = await self._call_llm_async(prompt, temp=0.3)
                if success:
                    return result.strip()

            return ""
        except Exception as e:
            print(f"为关键词'{keyword}'提取内容时出错: {str(e)}")
            return ""

    async def process_text_async(
        self, text: str
    ) -> Dict[str, Union[str, List[str], Dict[str, str]]]:
        """
        处理文本的主要流程

        Args:
            text: 原始文本

        Returns:
            result: 包含处理结果的字典
        """
        # 1. 清理文本
        print("开始清理文本...")
        cleaned_text = await self.clean_text_async(text)

        # 2. 提取关键词
        print("开始提取关键词...")
        keywords = await self.extract_keywords_async(cleaned_text)

        # 3. 根据关键词组织内容
        print("开始组织内容...")
        keyword_contents = await self.extract_content_for_keywords_async(
            cleaned_text, keywords
        )

        # 返回处理结果
        return {
            "cleaned_text": cleaned_text,
            "keywords": keywords,
            "keyword_contents": keyword_contents,  # 直接使用原始的关键词内容
        }

    def process_text(self, text: str) -> Dict[str, Union[str, List[str], Dict[str, str]]]:
        """同步处理文本（内部使用异步）"""
        return asyncio.run(self.process_text_async(text))


def save_result_to_file(
    result: Dict[str, Union[str, List[str], Dict[str, str]]], output_path: str
) -> None:
    """
    将处理结果保存到文件

    Args:
        result: 处理结果字典
        output_path: 输出文件路径
    """
    try:
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存处理结果
        with open(output_path, "w", encoding="utf-8") as file:
            # 写入关键词
            file.write("# 主要关键词\n\n")
            file.write(", ".join(result["keywords"]))
            file.write("\n\n")

            # 写入每个关键词的内容
            file.write("# 关键词内容\n\n")
            for keyword, content in result["keyword_contents"].items():
                file.write(f"## {keyword}\n\n")
                file.write(f"{content}\n\n")

            # 写入清理后的原始文本
            file.write("# 清理后的原始文本\n\n")
            file.write(result["cleaned_text"])

        print(f"已保存处理结果到 {output_path}")
    except Exception as e:
        print(f"保存结果失败: {str(e)}")


def process(
    text: Optional[str] = None,
    file_path: Optional[str] = None,
    output_path: Optional[str] = None,
    model: str = model_name,
    max_tokens: int = context_length,
    max_concurrent: int = 5,
    temperature: float = 0.7,
) -> Dict[str, Union[str, List[str], Dict[str, str]]]:
    """
    处理Markdown文本或文件的统一入口函数

    Args:
        text: 要处理的文本，如果为None则从file_path读取
        file_path: 输入文件路径，与text二选一
        output_path: 输出文件路径，如果不为None则将结果保存到文件
        model: 使用的模型
        max_tokens: 模型最大上下文长度
        max_concurrent: 最大并发请求数
        temperature: 模型温度参数

    Returns:
        result: 包含处理结果的字典
    """
    # 检查输入
    if text is None and file_path is None:
        raise ValueError("必须提供text或file_path参数")

    # 如果提供了文件路径，从文件读取文本
    if text is None:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as file:
                text = file.read()
        except Exception as e:
            raise Exception(f"读取文件 {file_path} 失败: {str(e)}")

    # 创建处理器并处理文本
    processor = LLMProcessor(model, max_tokens, max_concurrent, temperature)
    result = processor.process_text(text)

    # 如果提供了输出路径，保存结果
    if output_path:
        save_result_to_file(result, output_path)

    return result


def process_raw_markdown(
    mode: str,
    model: str = model_name,
    max_tokens: int = context_length,
    max_concurrent: int = 5,
) -> None:
    """
    处理指定模式下所有的raw_markdown文件

    Args:
        mode: 处理模式，可以是'single'或'batch'
        model: 使用的模型名称
        max_tokens: 模型的最大token上下文
        max_concurrent: 最大并发请求数
    """
    input_dir = f"data/02_raw_markdown/{mode}/"
    output_dir = f"data/03_processed_text/{mode}/"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有markdown文件
    markdown_files = glob.glob(os.path.join(input_dir, "*.md"))

    for file_path in markdown_files:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, file_name)

        print(f"处理文件: {file_path}")

        try:
            # 处理文件
            process(
                file_path=file_path,
                output_path=output_path,
                model=model,
                max_tokens=max_tokens,
                max_concurrent=max_concurrent,
            )
        except Exception as e:
            print(f"处理文件 {file_path} 失败: {str(e)}")


if __name__ == "__main__":
    # 使用示例
    # 处理单个模式下的所有文件
    process_raw_markdown(
        mode="single",
        model=model_name,
        max_tokens=context_length,
        max_concurrent=10,  # 设置最大并发请求数
    )
    # process_raw_markdown(mode="batch", max_concurrent=5)
