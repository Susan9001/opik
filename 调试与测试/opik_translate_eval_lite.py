from __future__ import annotations

import os
from typing import Any, Dict, List

from opik import Opik
from opik.evaluation import evaluate, models
from opik.evaluation.metrics import GEval

from opik_translate_agent import QwenTranslatorAgent


class QwenLiteJudgeModel(models.LiteLLMChatModel):
    """
    Wrapper of LiteLLMChatModel that strips DashScope-unsupported kwargs.
    """

    def generate_provider_response(
        self,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        disallowed_keys = {"top_logprobs", "logprobs"}
        sanitized_kwargs = {k: v for k,
                            v in kwargs.items() if k not in disallowed_keys}
        return super().generate_provider_response(messages=messages, **sanitized_kwargs)


class OpikTranslationEvaluator:
    """
    使用 Qwen 翻译 + LiteLLMChatModel 作为 judge 的翻译质量评估器。

    结构:
    1. 用 QwenTranslatorAgent 生成翻译
    2. 用 LiteLLMChatModel 封装 Qwen 模型作为 GEval 的评委
    3. 调用 opik.evaluation.evaluate 计算 metric 分数
    """

    def __init__(
        self,
        dataset_name: str = "Evaluation test dataset",
        samples: List[Dict[str, str]] | None = None,
        translator_model_name: str = "qwen-flash",
        translator_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        judge_model_name: str = "dashscope/qwen-flash",
    ) -> None:
        # 1. 数据集
        self.opik_client = Opik()
        self.dataset = self._init_dataset(dataset_name, samples)

        # 2. 翻译用的 agent 指向 OpenAI 兼容端点
        self.agent = QwenTranslatorAgent(
            api_key_env="DASHSCOPE_API_KEY",
            model_name=translator_model_name,
            base_url=translator_base_url,
        )

        # 3. Judge 用 LiteLLMChatModel 指向 DashScope provider
<<<<<<< HEAD
        # self.judge_model = QwenLiteJudgeModel(
        #     model_name=judge_model_name,
        #     api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        #     api_key=os.getenv("DASHSCOPE_API_KEY"),
        # )
        self.judge_model = models.LiteLLMChatModel(
=======
        self.judge_model = QwenLiteJudgeModel(
>>>>>>> 03c4b60f2 (微调格式)
            model_name=judge_model_name,
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
        )

        # 4. GEval 翻译质量 metric
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
                    "input": "I am studying an open source project called Opik.",
                    "expected_output": "我最近在研究一个叫 Opik 的开源项目。",
                },
            ]

        # 简单起步：每次运行都插入这些样例
        dataset.insert(samples)
        return dataset

    def _init_translation_metric(self) -> GEval:
        """
        用 GEval 定义一个翻译质量 metric, 评委模型是 LiteLLMChatModel 包装的 Qwen
        """
        return GEval(
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

    def _evaluation_task(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        evaluate 需要的 task 函数。
        这里直接调用翻译 agent, 再把三段信息拼成一个字符串交给 GEval。
        """
        source = item["input"]
        reference = item["expected_output"]

        # 调用你的翻译 agent 生成翻译
        model_output = self.agent.handle_request(source)

        payload = f"""SOURCE: {source}
REFERENCE: {reference}
MODEL_OUTPUT: {model_output}
"""
        return {
            "output": payload,  # GEval 只看 output 字段
        }

    def run(self):
        """
        执行一次评测，打印 metric 的统计结果，并返回原始 result 对象。
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
    # 运行一个最小示例
    evaluator = OpikTranslationEvaluator()
    res = evaluator.run()
