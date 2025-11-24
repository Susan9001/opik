import json
from opik.integrations.openai import track_openai
from opik import Opik, track
from typing import Dict, Any, Literal
from openai import OpenAI
import os
from typing import Any, List, Dict


class QwenTranslatorAgent:
  """
  一个简单的通义千问翻译 Agent:
  1. 封装了底层 client
  2. 提供 translate 工具函数
  3. 提供 handle_request, 先用 LLM 决策, 再调用工具
  """

  def __init__(
      self,
      api_key_env: str = "DASHSCOPE_API_KEY",
      model_name: str = "qwen-flash",
      base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
  ) -> None:
    api_key = os.getenv(api_key_env)
    if not api_key:
      raise RuntimeError(f"环境变量 {api_key_env} 未设置")

    raw_client = OpenAI(api_key=api_key, base_url=base_url)
    # 用 Opik 的 track_openai 包一层, 自动打 trace
    self.client = track_openai(raw_client)
    self.model_name = model_name

  def _chat(self, messages: List[Dict[str, str]]) -> str:
    """内部通用聊天工具。"""
    res = self.client.chat.completions.create(
        model=self.model_name,
        messages=messages,
    )
    return res.choices[0].message.content

  @track
  def translate(
      self,
      text: str,
      target_lang: Literal["zh", "en"] = "zh",
  ) -> str:
    """最基础的翻译工具函数。"""
    if target_lang == "zh":
      user_prompt = f"Translate the following text to Chinese: {text}"
    else:
      user_prompt = f"Translate the following text to English: {text}"

    messages = [
        {"role": "system", "content": "You are a helpful translation assistant."},
        {"role": "user", "content": user_prompt},
    ]
    res = self.client.chat.completions.create(
        model=self.model_name,
        messages=messages,
    )
    res_text = res.choices[0].message.content
    return res_text

  def _decide_action(self, user_text: str) -> Dict[str, Any]:
    """
    让 LLM 输出一个简单 JSON, 决定翻译方向和是否需要解释。
    这一步就是这个小 agent 的“决策”部分。
    """
    system_prompt = (
        "You are a translation agent planner.\n"
        "Given USER_TEXT, decide:\n"
        "1) target_lang: 'zh' or 'en'.\n"
        "2) need_explanation: true or false.\n"
        "Return ONLY a JSON object, no extra text.\n"
        "Example:\n"
        '{\"target_lang\": \"zh\", \"need_explanation\": true}'
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    res_text = self._chat(messages)

    # 做一点点防御式解析
    try:
      plan: Dict[str, Any] = json.loads(res_text)
    except json.JSONDecodeError:
      # 如果模型没按 JSON 格式来, 就用一个兜底决策
      plan = {"target_lang": "zh", "need_explanation": False}

    target_lang = plan.get("target_lang", "zh")
    if target_lang not in ("zh", "en"):
      target_lang = "zh"

    need_explanation = bool(plan.get("need_explanation", False))

    return {
        "target_lang": target_lang,
        "need_explanation": need_explanation,
    }

  @track
  def handle_request(self, user_text: str) -> str:
    """
    小 agent 的主入口:
    1. 决策翻译方向和是否解释
    2. 调用 translate 工具
    3. 按需要补充解释
    """
    plan = self._decide_action(user_text)
    target_lang = plan["target_lang"]
    need_explanation = plan["need_explanation"]

    translation = self.translate(user_text, target_lang=target_lang)

    if not need_explanation:
      return translation

    # 需要解释, 再让模型用目标语言解释一下
    if target_lang == "zh":
      explain_prompt = (
          f"请用中文简要解释这句翻译在语气和语义上的含义:\n{translation}"
      )
    else:
      explain_prompt = (
          f"Briefly explain in English the nuance and meaning of this translation:\n{translation}"
      )

    messages = [
        {"role": "system", "content": "You are a helpful language teacher."},
        {"role": "user", "content": explain_prompt},
    ]
    explanation = self._chat(messages)

    return f"{translation}\n\n解释:\n{explanation}"


if __name__ == "__main__":
  agent = QwenTranslatorAgent()

  tests = [
      "What is the capital of France?",
      "请帮我把这句话翻译成英文: 我最近在研究一个叫 Opik 的开源项目。",
      "Just explain what this English sentence means: He finally called it a day.",
  ]

  for text in tests:
    print("==== 输入 ====")
    print(text)
    print("==== 输出 ====")
    res = agent.handle_request(text)
    print(res)
    print()
