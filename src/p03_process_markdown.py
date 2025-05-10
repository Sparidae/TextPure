import asyncio
import glob
import os
from typing import List, Tuple

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


def estimate_token_length(text):
    # 估计token长度
    token_count = len(tokenizer(text)["input_ids"])
    return token_count


class TextChunker:
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
        self.max_tokens = int(max_tokens * 0.8) - reserve_tokens
        self.paragraphs = text.split("\n")

    def get_chunks(self) -> List[str]:
        """
        直接生成并返回所有文本块

        Returns:
            所有文本块的列表
        """
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
        """
        分割超长段落并返回第一个可用的块
        """
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


async def process_single_chunk_async(chunk, model, temperature) -> Tuple[bool, str]:
    """
    异步处理单个文本块

    Args:
        chunk: 文本块
        model: 使用的模型
        temperature: 温度参数

    Returns:
        (success, cleaned_chunk): 处理是否成功及处理后的文本
    """
    try:
        # 从prompts模块获取提示词
        prompt = get_prompt("text_cleaning", text=chunk)

        # 使用非阻塞方式执行
        loop = asyncio.get_event_loop()
        success, cleaned_chunk = await loop.run_in_executor(
            None, lambda: get_completion()(prompt, model=model, temperature=temperature)
        )
        return success, cleaned_chunk
    except Exception as e:
        print(f"LLM处理异常: {str(e)} (大小：{estimate_token_length(chunk)} tokens)")
        return False, chunk


async def clean_text_with_llm_async(
    text, model=model_name, temperature=0.7, max_tokens=32000, max_concurrent=10
):
    """
    使用LLM异步清理包含杂乱URL和字符的爬取文本

    Args:
        text: 需要清理的原始文本
        model: 使用的LLM模型
        temperature: 模型温度参数
        max_tokens: 模型最大上下文长度
        max_concurrent: 最大并发请求数

    Returns:
        cleaned_text: 清理后的文本
    """
    # 创建文本分块器并获取所有块
    chunker = TextChunker(text, max_tokens=max_tokens)
    chunks = chunker.get_chunks()
    cleaned_chunks = [None] * len(chunks)

    # 创建semaphore限制并发数
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(i, chunk):
        async with semaphore:
            print(f"开始处理第 {i + 1}/{len(chunks)} 个文本块")
            try:
                success, cleaned_chunk = await process_single_chunk_async(
                    chunk, model, temperature
                )
                if success:
                    return i, cleaned_chunk
                else:
                    return i, chunk
            except Exception as e:
                print(f"处理第 {i + 1} 个文本块时出错: {str(e)}")
                return i, chunk

    # 创建异步任务列表
    tasks = [process_with_semaphore(i, chunk) for i, chunk in enumerate(chunks)]

    # 使用tqdm创建进度条
    pbar = tqdm(total=len(chunks), desc="处理文本块")

    # 等待每个任务完成并更新进度条
    for coro in asyncio.as_completed(tasks):
        i, cleaned_chunk = await coro
        cleaned_chunks[i] = cleaned_chunk
        pbar.update(1)

    pbar.close()

    # 合并所有清理后的文本片段
    return "\n".join(chunk for chunk in cleaned_chunks if chunk is not None)


def clean_text_with_llm(
    text, model=model_name, temperature=0.7, max_tokens=32000, max_concurrent=5
):
    """
    使用LLM清理包含杂乱URL和字符的爬取文本

    Args:
        text: 需要清理的原始文本
        model: 使用的LLM模型
        temperature: 模型温度参数
        max_tokens: 模型最大上下文长度
        max_concurrent: 最大并发请求数

    Returns:
        cleaned_text: 清理后的文本
    """
    # 使用异步方式处理
    return asyncio.run(
        clean_text_with_llm_async(text, model, temperature, max_tokens, max_concurrent)
    )


def process_raw_markdown(mode, model=model_name, max_tokens=32000, max_concurrent=5):
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
            # 读取文件内容
            with open(file_path, "r", encoding="utf-8", errors="replace") as file:
                content = file.read()

            # 使用LLM清理文本
            cleaned_content = clean_text_with_llm(
                content, model=model, max_tokens=max_tokens, max_concurrent=max_concurrent
            )

            # 写入处理后的文件
            with open(output_path, "w", encoding="utf-8") as file:
                file.write(cleaned_content)

            print(f"已保存清理后的文件: {output_path}")

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
