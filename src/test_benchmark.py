"""
Module: src.test_benchmark

A script to benchmark LLMs on bug-identification tasks by loading JSONL datasets, sending prompts, and computing accuracy over multiple iterations.

Functions:
- load_jsonl(file_path): Load a JSONL file and return its records as a list of dictionaries.
- construct_prompt(code): Build the instruction prompt for the LLM given a code string.
- completion_with_backoff(model_full, prompt, max_tokens, use_temperature): Call the LLM API with exponential backoff upon failures.
- test_llm_on_jsonl(result_prefix, jsonl_prefix, provider, model, use_temperature, iterations=None): Run the LLM on JSONL datasets and record per-item accuracy over specified iterations.
- main(): Parse command-line arguments and invoke the benchmarking process.

Authors: Derek Sheen, Hokyung (Andy) Lee
Emails: derek.s.prog@gmail.com (D. Sheen), techandy42@gmail.com (H. Lee)
Date: May 3, 2025
"""

import os
import json
import argparse
from litellm import text_completion
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt

def load_jsonl(file_path):
    """Load a JSONL file and return a list of records."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def construct_prompt(code):
    """Build the instruction prompt for the LLM."""
    return f"""
<instruction_header>
The following is a code sample (e.g., source_code).
One of the functions in this code contains a bug.
Identify the function with the bug.
<instruction_header>

<source_code>
{code}
<source_code>

<output_format>
Please return the variable name of the function containing the bug, nothing else. Do not alter the name of the function in any way. If there is no bug, return 'none'.
<output_format>
"""

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(5)
)
def completion_with_backoff(model_full, prompt, max_tokens, use_temperature):
    """Call the LLM API with exponential backoff on failures."""
    params = {
        'model': model_full,
        'prompt': prompt,
        'max_tokens': max_tokens,
    }
    if use_temperature:
        params['temperature'] = 0.0
    return text_completion(**params)

def test_llm_on_jsonl(result_prefix, jsonl_prefix, provider, model, use_temperature, iterations=None):
    """Run the LLM on JSONL datasets and record results with accuracy."""
    os.makedirs('data/result', exist_ok=True)
    os.makedirs(f'data/result/{provider}_{model}', exist_ok=True)

    if iterations is None:
        iterations = range(20)

    for i in iterations:
        data = load_jsonl(f"{jsonl_prefix}_{i}.jsonl")
        correct = 0
        total = 0
        output_path = f"{result_prefix}_{i}.jsonl"

        with open(output_path, 'a') as f:
            for item in tqdm(data, desc=f"Testing LLM [Iteration {i+1}]", unit="sample"):
                code = item['code']
                expected = item.get('func_error', '')
                prompt = construct_prompt(code)

                response = completion_with_backoff(
                    model_full=f"{provider}/{model}",
                    prompt=prompt,
                    max_tokens=16000,
                    use_temperature=use_temperature
                )

                prediction = response['choices'][0]['text'].strip()
                item['guess'] = prediction

                if expected and expected in prediction:
                    item['is_correct'] = 1
                    correct += 1
                else:
                    item['is_correct'] = 0

                total += 1
                item['accuracy'] = round(correct / total * 100, 2)
                f.write(json.dumps(item) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Run LLM benchmark on JSONL datasets.")
    parser.add_argument('--provider', required=True, help="LLM provider (e.g., openai, anthropic)")
    parser.add_argument('--model', required=True, help="Model name (e.g., gpt-4.1-mini)")
    parser.add_argument('--no-temperature', action='store_true', help="Exclude temperature field from LLM calls")
    parser.add_argument('--iterations', nargs='+', type=int, choices=list(range(20)), help="Dataset iteration(s) to run (0-19)")
    args = parser.parse_args()

    use_temperature = not args.no_temperature
    provider = args.provider
    model = args.model
    iterations = args.iterations

    results_prefix = f"data/result/{provider}_{model}/bics_result"
    dataset_prefix = "data/output/bics_dataset"
    test_llm_on_jsonl(results_prefix, dataset_prefix, provider, model, use_temperature, iterations)

if __name__ == "__main__":
    main()
