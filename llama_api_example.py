import requests
import json

# 定义服务器端 API URL
api_url = "http://172.16.129.30:11434/api/embed"

# 要发送的文本
texts = "Hello world!"

# 构造请求的 JSON 数据
payload = {
    "input": texts,
    "model": "llama3.1:70b" # 8192
    # "model": "llama3.2:3b"  # 3072
    # "model": "qwen2.5:7b"   # 3584
    # "model": "qwen2.5:32b"  # 5120
    # "model": "qwen2.5:72b"  # 8192
}

# 发送请求到 Llama 服务器
response = requests.post(api_url, json=payload)

# 检查响应
if response.status_code == 200:
    # 解析返回的嵌入向量
    embeddings = response.json()
    # print("Received embeddings:", embeddings['embeddings'][0])
    print("Dimension:", len(embeddings['embeddings'][0]))
else:
    print(f"Error: {response.status_code}, {response.text}")