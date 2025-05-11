import asyncio
import logging
import os
import shutil
from typing import List, Union
from urllib.parse import unquote

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.async_configs import BrowserConfig, CacheMode
from crawl4ai.async_dispatcher import RateLimiter, SemaphoreDispatcher
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# TODO 测试
# 1.错误处理tryexcept 2.并发限制

# 全部接受url列表作为参数
browser_config = BrowserConfig(verbose=True, text_mode=True, light_mode=True)
run_config = CrawlerRunConfig(
    # Content filtering
    word_count_threshold=1,
    excluded_tags=["form", "header"],
    exclude_external_links=True,
    # Content processing
    process_iframes=True,
    remove_overlay_elements=True,
    only_text=True,
    # Cache control
    cache_mode=CacheMode.BYPASS,  # 使用cache
)
# 多任务调度器
# dispatcher = MemoryAdaptiveDispatcher(
#     memory_threshold_percent=90.0,  # 系统内存占用最大百分比
#     check_interval=1.0,  # 检查内存的间隔
#     max_session_permit=10,  # 最大并发任务数
#     rate_limiter=RateLimiter(  # 速率限制 随机延迟，最大延迟，最大重试次数，限速代码
#         base_delay=(2.0, 20.0), max_delay=60.0, max_retries=3, rate_limit_codes=[429, 503]
#     ),
# )
dispatcher = SemaphoreDispatcher(
    semaphore_count=1,
    max_session_permit=10,  # Maximum concurrent tasks
    rate_limiter=RateLimiter(  # 速率限制 随机延迟，最大延迟，最大重试次数，限速代码
        base_delay=(20.0, 60.0),
        max_delay=60.0,
        max_retries=3,
        rate_limit_codes=[429, 503],  # 基本没用
    ),
)


async def crawl_batch(urls):
    async with AsyncWebCrawler(config=browser_config) as crawler:
        # 多任务调度器实现
        results = await crawler.arun_many(urls, config=run_config, dispatcher=dispatcher)

        print("请求全部完成")

    return results  # result是由url和结果的元组 组成的列表


async def crawl_single(url):
    async with AsyncWebCrawler(config=browser_config) as crawler:
        # 多任务调度器实现
        result = await crawler.arun(url, config=run_config)

        print("请求完成")

    return result  # result是由url和结果的元组 组成的列表


## 测试deep crawling
async def deep_crawl(url):
    config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(
            max_depth=2,
            include_external=False,
            max_pages=50,  # 测试
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        cache_mode=CacheMode.BYPASS,
        verbose=True,
        markdown_generator=DefaultMarkdownGenerator(),
        only_text=True,
        # Content filtering
        word_count_threshold=10,
        excluded_tags=["form", "header"],
        exclude_external_links=True,
        # Content processing
        process_iframes=True,
        remove_overlay_elements=True,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun(url, config=config)

        print(f"Crawled {len(results)} pages in total")

        # Access individual results
        # for result in results[:3]:  # Show first 3 results
        #     print(f"URL: {result.url}")
        #     print(f"Depth: {result.metadata.get('depth', 0)}")

    return results


def load_urls_from_file(filename: str = "data/01_url_list/urls.txt") -> List[str]:
    """
    从文件加载URL列表

    Args:
        filename: URL列表文件的路径

    Returns:
        URL列表
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logging.error(f"读取URL列表文件时发生错误: {str(e)}")
        raise


def save_crawl_results(results: Union[List, object], save_type: str = "batch") -> None:
    """
    保存爬取结果到指定目录

    Args:
        results: 爬取结果，可以是单个结果或结果列表
        save_type: 保存类型，'batch'或'single'
    """
    # 确定保存目录
    base_dir = os.path.join("data", "02_raw_markdown")
    save_dir = os.path.join(base_dir, save_type)

    # 清空目标目录
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # 设置日志记录
    log_file = os.path.join(save_dir, "__crawl_log.txt")

    # 将单个结果转换为列表
    if not isinstance(results, list):
        results = [results]

    # 保存每个结果的markdown内容到文件
    success_count = 0
    fail_count = 0
    for i, result in enumerate(results):
        try:
            # 从URL中提取文件名，移除非法字符
            file_name = result.url.split("/")[-1]
            if not file_name:
                file_name = f"page_{i}"

            # 解码URL编码的中文文件名
            file_name = unquote(file_name)

            # 替换可能在文件名中非法的字符
            file_name = "".join(
                c if c.isalnum() or c in "._- " else "_" for c in file_name
            )
            file_path = os.path.join(save_dir, f"{file_name}.md")

            # 写入markdown内容到文件
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(result.markdown)
            success_count += 1
        except Exception as e:
            fail_count += 1
            print(f"保存文件失败 - URL: {result.url} - 错误信息: {str(e)}")
            continue

    # 记录最终统计信息
    summary = f"爬取完成统计:\n成功: {success_count} 个文件\n失败: {fail_count} 个文件\n总文件数: {len(results)}"

    print(f"所有结果已保存到 '{save_dir}' 文件夹中")
    print(f"详细日志请查看: {log_file}")
    print(summary)


async def crawl_urls(mode="batch", url=""):
    try:
        # 1. 从文件加载URL列表
        urls = load_urls_from_file()

        if mode == "batch":
            results = await crawl_batch(urls)
            save_crawl_results(results, "batch")
        elif mode == "single":
            single_result = await crawl_single(url)
            save_crawl_results(single_result, "single")
        else:
            raise ValueError(f"无效的模式: {mode}")

    except Exception as e:
        print(f"爬取过程中发生错误: {str(e)}")
        return


if __name__ == "__main__":
    asyncio.run(
        crawl_urls(
            mode="single",
            url="https://mzh.moegirl.org.cn/%E4%B8%B0%E5%B7%9D%E7%A5%A5%E5%AD%90",
        )
    )
