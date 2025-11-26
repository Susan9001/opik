import os
from openai import OpenAI
from opik.integrations.openai import track_openai
import opik
from opik.evaluation import evaluate_prompt
from opik.evaluation.metrics import Usefulness

# Create a dataset that contains the samples you want to evaluate
opik_client = opik.Opik()

dataset = opik_client.get_or_create_dataset("Evaluation test dataset")
dataset.insert([
    {"input": "Hello, world!", "expected_output": "你好，世界！"},
    {"input": "What is the capital of France?", "expected_output": "法国的首都是什么？"},
])

result = evaluate_prompt(
    dataset=dataset,
    messages=[{"role": "user",
               "content": "Translate the following text to Chineses: {{input}}"}],
    model="gpt-5-nano",  # or your preferred model
    scoring_metrics=[Usefulness()]
)

# Retrieve and print the aggregated scores statistics (mean, min, max, std) per metric
scores = result.aggregate_evaluation_scores()
for metric_name, statistics in scores.aggregated_scores.items():
    print(f"{metric_name}: {statistics}")
