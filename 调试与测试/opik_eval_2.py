import os
from typing import Dict, Any

from openai import OpenAI
from opik import Opik, track
from opik.integrations.openai import track_openai
from opik.evaluation import evaluate
from opik.evaluation.metrics import Equals, Usefulness
from opik.evaluation import models

client = track_openai(OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # Beijing
))


@track
def llm_app(prompt: str):
    res = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-flash",
        messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": prompt},
        ]
    )
    return res.choices[0].message.content


# Create a dataset that contains the samples you want to evaluate
opik_client = Opik()
dataset = opik_client.get_or_create_dataset("Evaluation test dataset")
dataset.insert([
    {"input": "Hello, world!", "expected_output": "你好，世界！"},
    {"input": "What is the capital of France?", "expected_output": "法国的首都是什么？"},
])

# 4. 定义评估任务


def evaluation_task(item: Dict[str, Any]) -> Dict[str, Any]:
    task_prompt = f"Translate the following text to Chineses: \"{item['input']}\""
    output = llm_app(task_prompt)
    return {
        # 给 Equals 用
        "output": output,
        "reference": item["expected_output"],
        "input": task_prompt,
    }


# judge_model = models.LiteLLMChatModel(model="gpt-3.5-turbo", temperature=0)
judge_model = models.LiteLLMChatModel(
    model_name="dashscope/qwen-flash",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # api_key=os.getenv("DASHSCOPE_API_KEY"),
)

result = evaluate(
    dataset=dataset,
    task=evaluation_task,
    scoring_metrics=[Equals(), Usefulness(model=judge_model)],
)

scores = result.aggregate_evaluation_scores()
for metric_name, statistics in scores.aggregated_scores.items():
    print(metric_name, statistics)
