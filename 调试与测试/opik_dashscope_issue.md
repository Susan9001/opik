### Proposal summary

## Summary

When using `GEval` with `LiteLLMChatModel` as the judge model and DashScope Qwen (`dashscope/qwen-flash`), I noticed that DashScope rejects the `top_logprobs` parameter that GEval passes through LiteLLM.

Adding some provider-specific sanitization inside `LiteLLMChatModel` would improve compatibility and make it easier to use Qwen as a judge out of the box.

---

## Background

I am experimenting with a translation evaluation setup:

- A translation agent calls DashScope via the OpenAI-compatible endpoint  
  - `base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"`  
  - Model: `qwen-flash`
- For evaluation, I use:  
  - `GEval` as the metric  
  - `LiteLLMChatModel` as the judge model with `model_name="dashscope/qwen-flash"`

Code sketch:

```python
from opik.evaluation import evaluate, models
from opik.evaluation.metrics import GEval

judge_model = models.LiteLLMChatModel(
    model_name="dashscope/qwen-flash",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

translation_metric = GEval(
    name="translation_quality",
    model=judge_model,
    task_introduction="You are an expert bilingual evaluator...",
    evaluation_criteria="...",
)

result = evaluate(
    dataset=dataset,
    task=evaluation_task,
    scoring_metrics=[translation_metric],
)
```

The task function uses a small `QwenTranslatorAgent` to generate translations and passes:

```text
SOURCE: ...
REFERENCE: ...
MODEL_OUTPUT: ...
```

as the `output` field for GEval.

---

## Observed error

With the configuration above, the evaluation fails with a DashScope error. Opik logs something like:

```text
OPIK: Failed to call LLM provider, reason: litellm.BadRequestError: DashscopeException - <400> InternalError.Algo.InvalidParameter: Range of top_logprobs should be [0, 5]
```

From the stack trace I can see:

- `GEval` calls `LiteLLMChatModel.generate_string(...)`  
- `LiteLLMChatModel` forwards all kwargs to `litellm.completion(...)`  
- DashScope receives a `top_logprobs` value that is outside its allowed `[0, 5]` range and returns HTTP 400

So this is a provider-specific incompatibility between GEval's default params and DashScope.


### Motivation

## Motivation

- **What problem am I trying to solve?**  
  As discussed in the previous section, when using `GEval` with `LiteLLMChatModel` as the judge model and DashScope Qwen (`dashscope/qwen-flash`), I noticed that DashScope rejects the `top_logprobs` parameter that GEval passes through LiteLLM.

- **How am I currently solving this problem?**  
  A work around is discussed in `Current workaround` section.

- **What are the benefits of this feature?**  
  - Makes DashScope Qwen fully compatible with GEval via `LiteLLMChatModel` without custom subclasses.  
  - Gives a better “works out of the box” experience for users who want to use DashScope as a judge model.  
  - Keeps provider-specific logic in one place (Opik’s LiteLLM integration), similar to the existing GPT-5 handling, instead of many users re-implementing the same workaround.

## Approach options

Per discussion with @andrescrz on [Slack conversation](https://cometml.slack.com/archives/C01313D73BJ/p1764039465584769), there seem to be two options:

1. **Option 1 (recommended)**: provider-specific sanitization inside `LiteLLMChatModel`  
   - Detect DashScope models (via provider check or `model_name` prefix `dashscope/`)  
   - Either:  
     - Drop `logprobs` / `top_logprobs` entirely, or  
     - Clamp `top_logprobs` to the valid `[0, 5]` range  
   - Emit a user-friendly warning when parameters are sanitized  
   - This mirrors what Opik already does for GPT-5 in `util.py`

2. **Option 2**: expose a hook for custom kwargs filtering  
   - For example, allow users to override or inject a sanitizer that can modify `**kwargs` before calling `litellm.completion(...)`

Option 1 is my preferred approach since all Opik users would benefit automatically without having to subclass anything.

---

## Current workaround

Right now I am using a small subclass of `LiteLLMChatModel` that strips the unsupported parameters before delegating to the parent implementation:

```python
from opik.evaluation import models
from typing import Any, Dict, List

class QwenLiteJudgeModel(models.LiteLLMChatModel):
    """
    Wrapper around LiteLLMChatModel that strips DashScope-unsupported kwargs.
    """

    def generate_provider_response(
        self,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        disallowed = {"top_logprobs", "logprobs"}
        sanitized_kwargs = {k: v for k, v in kwargs.items() if k not in disallowed}
        return super().generate_provider_response(messages=messages, **sanitized_kwargs)
```

Using this wrapper as the GEval model:

```python
judge_model = QwenLiteJudgeModel(
    model_name="dashscope/qwen-flash",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)
```

makes GEval run successfully with DashScope Qwen as the judge.

---

## Proposal

I would like to implement **Option 1**:

- Extend the existing provider-specific handling in `LiteLLMChatModel` / related utils (similar to the GPT-5 handling mentioned in Slack) to support DashScope.  
- For DashScope models (`model_name` starting with `dashscope/`):  
  - Either drop `logprobs` / `top_logprobs`, or clamp `top_logprobs` to `[0, 5]`  
  - Emit a warning when sanitization happens

I can:

1. Open a PR implementing this behavior  
2. Add a short test case that reproduces the current error and verifies that the sanitized call succeeds  
3. Optionally add a short note to the docs about using DashScope / Qwen as a judge via `LiteLLMChatModel`

If there are any preferences on the exact behavior (drop vs clamp, warning text, where to place the sanitization helper), I’m happy to adjust the implementation accordingly.

---

## Environment

- Opik version: 1.7.15 
- Python version: 3.11  
- OS: Windows 11  
- DashScope model: `dashscope/qwen-flash`  
- DashScope endpoint: `https://dashscope.aliyuncs.com/compatible-mode/v1`
