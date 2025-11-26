from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
from opik import Opik
from opik.evaluation import evaluate
from opik.evaluation.metrics import GEval
from opik.evaluation.models import OpikBaseModel

from opik_translate_agent import QwenTranslatorAgent


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
        if not api_key:
            raise RuntimeError("环境变量 DASHSCOPE_API_KEY 未设置")

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
        # 只白名单简单参数, 防止 GEval 传进复杂对象导致 json 序列化失败
        allowed_keys = {"temperature", "top_p", "max_tokens"}
        allowed_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }
        payload.update(allowed_kwargs)

        res = requests.post(
            self.request_url,
            headers=self.headers,
            json=payload,
            timeout=60,
        )
        res.raise_for_status()
        return res.json()

    def generate_string(self, input: str, **kwargs: Any) -> str:
        data = self.generate_provider_response(
            messages=[{"role": "user", "content": input}],
            **kwargs,
        )
        return data["choices"][0]["message"]["content"]


class OpikTranslationEvaluator:
    """
    把:
      1. 数据集构建
      2. GEval 翻译质量 metric
      3. evaluation_task
      4. evaluate + 打印结果
    全部封装在一个小类里。
    """

    def __init__(
        self,
        dataset_name: str = "Evaluation test dataset",
        samples: List[Dict[str, str]] | None = None,
        qwen_model_name: str = "qwen-flash",
        qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ):
        # 1. Opik 客户端和数据集
        self.opik_client = Opik()
        self.dataset = self._init_dataset(dataset_name, samples)

        # 2. 翻译 agent 和 judge model
        self.agent = QwenTranslatorAgent(
            api_key_env="DASHSCOPE_API_KEY",
            model_name=qwen_model_name,
            base_url=qwen_base_url,
        )
        self.judge_model = QwenJudgeModel(
            model_name=qwen_model_name,
            base_url=qwen_base_url,
        )

        # 3. GEval 翻译质量 metric
        self.translation_metric = self._init_translation_metric()

    def _init_dataset(
        self,
        dataset_name: str,
        samples: List[Dict[str, str]] | None,
    ):
        dataset = self.opik_client.get_or_create_dataset(dataset_name)

        if samples is None:
            samples = [
                {"input": "Hello, world!", "expected_output": "你好，世界！"},
                {
                    "input": "What is the capital of France?",
                    "expected_output": "法国的首都是什么？",
                },
            ]

        # 简单起见, 每次跑前插入样本; 真要严谨可以先查是否已存在
        dataset.insert(samples)
        return dataset

    def _init_translation_metric(self) -> GEval:
        translation_metric = GEval(
            name="translation_quality",
            model=self.judge_model,
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
        return translation_metric

    def _evaluation_task(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        evaluate(...) 需要的 task 函数。
        这里直接调用我们封装好的 agent。
        """
        model_output = self.agent.handle_request(item["input"])

        payload = f"""SOURCE: {item["input"]}
REFERENCE: {item["expected_output"]}
MODEL_OUTPUT: {model_output}
"""
        res: Dict[str, Any] = {
            "output": payload,  # GEval 只看 output 字段
        }
        return res

    def run(self):
        """
        执行一次评估, 打印 metric 统计并返回 raw result 对象。
        """
        res = evaluate(
            dataset=self.dataset,
            task=self._evaluation_task,
            scoring_metrics=[self.translation_metric],
        )

        scores = res.aggregate_evaluation_scores()
        for metric_name, statistics in scores.aggregated_scores.items():
            print(metric_name, statistics)

        return res


if __name__ == "__main__":
    evaluator = OpikTranslationEvaluator()
    res = evaluator.run()
