import os
from openai import OpenAI
from opik.integrations.openai import track_openai

client_bj = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    # api_key=os.getenv("DASHSCOPE_API_KEY"),
    api_key="sk-1b194212a77b40ec8f392a83beb087a5",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # Beijing
)

tracked_client = track_openai(client_bj)

completion = tracked_client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    model="qwen-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "统计一下2025年东亚各国的出生率，人口数，以及东亚地区的平均出生率和总人口数."},
    ]
)
print(completion.model_dump_json())