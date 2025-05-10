import asyncio

from src.p01_parse_sitemap import parse_sitemap
from src.p02_crawl_urls import crawl_urls
from src.p03_process_markdown import process_raw_markdown


def main(mode, url):
    # 0.处理url 获得sitemap（略）
    # 1.处理sitemap获得url列表
    if mode == "batch":
        parse_sitemap()
    elif mode == "single":
        pass
    # 2.处理url获得raw_markdown
    asyncio.run(crawl_urls(mode=mode, url=url))
    # 3.处理raw_markdown的格式和多余字符获得processed_markdown
    process_raw_markdown(mode=mode)


if __name__ == "__main__":
    mode = "single"
    url = ""
    main(mode, url)
