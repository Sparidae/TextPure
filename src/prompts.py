# 文本清理提示词（作为备份，优先使用YAML文件中的配置）
TEXT_CLEANING_PROMPT = """
请帮我清理以下爬取的文本内容，去除以下内容：
1. 所有URL链接，包括markdown链接元素
2. 无意义的特殊字符和HTML标签
3. 重复的内容
4. 广告和推广内容
5. 导航菜单、页脚、版权声明等网站结构元素
6. markdown表格内容

请保留所有有意义的正文内容，并使其格式清晰易读。
不要添加任何不在原文中的内容。
保持原始的markdown格式。
严格去除 AI 生成的附加说明，仅保留清理后的核心数据。

以下是需要清理的文本:

{text}

处理后的文本：
"""

SUMMARY_PROMPT = """
请帮我总结以下文本内容，提取出文本中的主要内容。

以下是需要总结的文本:

{text}

总结内容：
"""


RAG_PROMPT = """
你是一名信息结构化和知识库开发的专家，请始终保持专业态度。
你的任务是将 markdown 内容按照关键名词分开，然后对每个关键名词将所有的相关内容总结为一段。
注意你需要生成中文。
严格去除 AI 生成的附加说明，仅保留清理后的核心数据。
请确保最终输出便于分块存储、向量化处理，并支持高效检索。

以下是需要处理的文本：

{text}

处理后的文本：
"""


def get_prompt(prompt_name, **kwargs):
    """
    获取指定名称的提示词，并填充参数

    Args:
        prompt_name: 提示词模板名称
        **kwargs: 提示词中需要替换的参数

    Returns:
        填充参数后的提示词
    """
    code_prompts = {
        "text_cleaning": TEXT_CLEANING_PROMPT,
        "summary": SUMMARY_PROMPT,
        "rag": RAG_PROMPT,
    }

    if prompt_name not in code_prompts:
        raise ValueError(f"未找到名为 '{prompt_name}' 的提示词模板")

    prompt_template = code_prompts[prompt_name]

    # 使用字符串format方法填充参数
    return prompt_template.format(**kwargs)
