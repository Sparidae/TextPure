from transformers import AutoTokenizer

# 加载 Qwen2.5 的 Tokenizer
# export HF_ENDPOINT=https://hf-mirror.com
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

# 输入文本
text = "Qwen2.5 的 Token 用量计算方法。"

# Token 计数
token_count = len(tokenizer(text)["input_ids"])
print(token_count)
