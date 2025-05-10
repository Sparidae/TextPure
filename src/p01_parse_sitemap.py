import logging
import os
import xml.etree.ElementTree as ET
from typing import List


def _parse_sitemap(sitemap_path: str) -> List[str]:
    """
    解析本地sitemap文件并返回所有URL列表

    Args:
        sitemap_path: sitemap文件的路径

    Returns:
        包含所有URL的列表
    """
    try:
        # 解析XML文件
        tree = ET.parse(sitemap_path)
        root = tree.getroot()

        # 获取所有URL
        urls = []

        # 处理普通sitemap文件中的URL
        for url in root.findall(
            ".//{http://www.sitemaps.org/schemas/sitemap/0.9}url/{http://www.sitemaps.org/schemas/sitemap/0.9}loc"
        ):
            urls.append(url.text)

        return urls
    except Exception as e:
        logging.error(f"解析sitemap文件时发生错误: {str(e)}")
        raise


def save_urls_to_file(urls: List[str], filename: str = "data/01_url_list/urls.txt"):
    """
    将URL列表保存到文件

    Args:
        urls: URL列表
        filename: 保存的文件路径
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "w", encoding="utf-8") as f:
            for url in urls:
                f.write(f"{url}\n")
        logging.info(f"成功保存 {len(urls)} 个URL到文件 {filename}")
    except Exception as e:
        logging.error(f"保存URL到文件时发生错误: {str(e)}")
        raise


def parse_sitemap(filename="sitemap.xml"):
    # 解析sitemap
    sitemap_path = os.path.join("sitemaps", filename)
    try:
        urls = _parse_sitemap(sitemap_path)

        # 保存URL到文件
        save_urls_to_file(urls)

    except Exception as e:
        print(e)
        return


if __name__ == "__main__":
    parse_sitemap()
