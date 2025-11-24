from opik.evaluation.metrics import GEval
from opik.evaluation import evaluate
from opik.integrations.openai import track_openai
from opik import Opik, track
from typing import Dict, Any
from openai import OpenAI
import os
import requests
from typing import Any, List, Dict

from opik.evaluation.models import OpikBaseModel

client = track_openai(OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # Beijing
))


@track
def llm_translator(input: str):
  res = client.chat.completions.create(
      # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
      model="qwen-flash",
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Translate the following text to Chinese: {input}"},
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


class QwenJudgeModel(OpikBaseModel):
  """
  通过 OpenAI 兼容接口调用通义千问, 用作 LLM-as-a-judge.
  """

  def __init__(
      self,
      model_name: str = "qwen-flash",
      api_key: str | None = None,
      base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
  ):
    super().__init__(model_name)
    if api_key is None:
      api_key = os.getenv("DASHSCOPE_API_KEY")
    self.api_key = api_key
    # GEval 这一类一般走 /chat/completions
    self.request_url = base_url.rstrip("/") + "/chat/completions"
    self.headers = {
        "Authorization": f"Bearer {self.api_key}",
        "Content-Type": "application/json",
    }

  def generate_provider_response(
      self,
      messages: List[Dict[str, Any]],
      **kwargs: Any,
  ) -> Any:
    allowed_keys = {"temperature", "top_p", "max_tokens"}
    allowed_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
    payload: Dict[str, Any] = {
        "model": self.model_name,
        "messages": messages,
    }
    payload.update(allowed_kwargs)

    res = requests.post(self.request_url, headers=self.headers,
                        json=payload, timeout=60)
    res.raise_for_status()
    return res.json()

  def generate_string(self, input: str, **kwargs: Any) -> str:
    data = self.generate_provider_response(
        messages=[{"role": "user", "content": input}],
        **kwargs,
    )
    return data["choices"][0]["message"]["content"]


qwen_judge_model = QwenJudgeModel(
    model_name="qwen-flash",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

translation_metric = GEval(
    name="translation_quality",
    model=qwen_judge_model,
    task_introduction=(
        "You are an expert bilingual evaluator. "
        "You will evaluate Chinese translations of English sentences."
    ),
    evaluation_criteria="""
Score how good the MODEL_OUTPUT is as a Chinese translation of SOURCE,
optionally comparing it with REFERENCE.

Requirements:
1. Meaning is preserved and accurate.
2. Chinese is natural and fluent.
3. No extra information is added beyond SOURCE.
4. Major mistakes or missing key information should lead to a low score.

Return only an integer score from 0 to 10.
""",
)


def evaluation_task(item: Dict[str, Any]) -> Dict[str, Any]:
  # 真正调用翻译模型
  model_output = llm_translator(item["input"])

  # 给 GEval 的 payload
  payload = f"""SOURCE: {item["input"]}
REFERENCE: {item["expected_output"]}
MODEL_OUTPUT: {model_output}
  """

  return {
      "output": payload,  # GEval 只看 output 字段
  }

result = evaluate(
    dataset=dataset,
    task=evaluation_task,
    scoring_metrics=[translation_metric],
)

scores = result.aggregate_evaluation_scores()
for metric_name, statistics in scores.aggregated_scores.items():
  print(metric_name, statistics)
