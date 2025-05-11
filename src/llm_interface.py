import os
import threading
import time

import requests
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv(override=True)

API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE", "https://api.siliconflow.cn/v1")
model_name = os.getenv("LLM_FOR_CLEAN", "Qwen/Qwen3-30B-A3B")
context_length = int(os.getenv("LLM_FOR_CLEAN_CL", "4096"))

# 创建OpenAI客户端并使用锁保护它
client = OpenAI(api_key=API_KEY, base_url=API_BASE)
client_lock = threading.Lock()


def completion(
    prompt,
    model=model_name,
    temperature=0.7,
    max_tokens=4096,
    max_retry_iters=3,
    retry_delays=30,  # 默认重试延迟为30秒（半分钟）
):
    """
    调用大模型接口获取完成结果 (使用requests库)

    Args:
        prompt: 提示词
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大生成token数
        max_retry_iters: 最大重试次数
        retry_delays: 重试间隔秒数，默认60秒

    Returns:
        (success, content): 是否成功及模型输出内容
    """
    # 构建请求
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "temperature": temperature,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
    }

    # 重试逻辑
    for attempt in range(max_retry_iters):
        try:
            # 发送请求
            response = requests.request(
                "POST", url=f"{API_BASE}/chat/completions", json=payload, headers=headers
            )

            # 解析响应
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]

            # 打印token使用情况
            if "usage" in response_data and "completion_tokens" in response_data["usage"]:
                print(f"输出token量: {response_data['usage']['completion_tokens']}")

            return True, content

        except Exception as e:
            error_message = str(e)
            print(f"api调用{attempt + 1}次；报错信息: {error_message}")
            time.sleep(retry_delays)

    return False, None


def completion_with_openai(
    prompt,
    model=model_name,
    temperature=0.7,
    max_tokens=4096,
    max_retry_iters=3,
    retry_delays=60,  # 默认重试延迟为60秒（1分钟）
):
    """
    调用大模型接口获取完成结果 (使用OpenAI官方库)

    Args:
        prompt: 提示词
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大生成token数
        max_retry_iters: 最大重试次数
        retry_delays: 重试间隔秒数，默认60秒

    Returns:
        (success, content): 是否成功及模型输出内容
    """
    # 准备消息
    messages = [{"role": "user", "content": prompt}]

    # 重试逻辑
    for attempt in range(max_retry_iters):
        try:
            # 使用锁保护OpenAI客户端调用
            with client_lock:
                # 使用OpenAI客户端调用API
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=0.7,
                    stream=False,
                )

            # 获取响应内容
            content = response.choices[0].message.content

            # 打印token使用情况
            if hasattr(response, "usage") and hasattr(
                response.usage, "completion_tokens"
            ):
                print(f"消耗token量: {response.usage.completion_tokens}")

            return True, content

        except Exception as e:
            error_message = str(e)
            print(f"尝试 {attempt + 1} 失败: {error_message}")
            time.sleep(retry_delays)

    return False, None


def get_completion(use_openai_client=False):
    """
    根据参数选择使用哪个完成函数

    Args:
        use_openai_client: 是否使用OpenAI客户端库

    Returns:
        选择的完成函数
    """
    if use_openai_client:
        return completion_with_openai
    else:
        return completion


if __name__ == "__main__":
    # 测试示例
    test_prompt = "简短介绍一下什么是大型语言模型"

    print("使用requests库测试:")
    success, result = completion(test_prompt)
    if success:
        print(f"结果: {result[:100]}...")
    else:
        print("请求失败")

    print("\n使用OpenAI客户端库测试:")
    success, result = completion_with_openai(test_prompt)
    if success:
        print(f"结果: {result[:100]}...")
    else:
        print("请求失败")
