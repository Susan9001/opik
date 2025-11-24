import os
from typing import Dict, Any

from openai import OpenAI
from opik import Opik, track
from opik.integrations.openai import track_openai
from opik.evaluation import evaluate
from opik.evaluation.metrics import GEval

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

translation_metric = GEval(
    task_introduction=(
        "You are an expert bilingual evaluator. "
        "You will be given a SOURCE English sentence, a REFERENCE Chinese translation, "
        "and a MODEL_OUTPUT Chinese sentence."
    ),
    evaluation_criteria=(
        "Score from 0 to 10 how good MODEL_OUTPUT translates SOURCE into Chinese, "
        "preserving meaning and tone and staying close to REFERENCE when reasonable. "
        "Ignore whether the original English sentence is a question or statement; "
        "only care about translation quality."
    ),
    name="translation_quality",
    # model="gpt-4o-mini"  # 可选,不写就用默认 judge
)


def evaluation_task(item: Dict[str, Any]) -> Dict[str, Any]:
  source = item["input"]
  expected = item["expected_output"]

  # 模型真正做翻译的调用
  model_output = llm_app(f"Translate the following text to Chinese: {source}")

  # 拼成给 GEval 看的 payload
  payload = (
      f"SOURCE: {source}\n"
      f"REFERENCE: {expected}\n"
      f"MODEL_OUTPUT: {model_output}\n"
  )

  # 注意: 对 GEval 来说,它只看 output 这个字段
  return {
      "output": payload,
  }


result = evaluate(
    dataset=dataset,
    task=evaluation_task,
    scoring_metrics=[translation_metric],
)

scores = result.aggregate_evaluation_scores()
for metric_name, statistics in scores.aggregated_scores.items():
  print(metric_name, statistics)
