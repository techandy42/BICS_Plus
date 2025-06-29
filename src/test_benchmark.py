"""
Module: src.test_benchmark

A script to benchmark LLMs on bug-identification tasks by loading JSONL datasets, sending prompts, and computing accuracy over multiple iterations.

Functions:
- load_jsonl(file_path): Load a JSONL file and return its records as a list of dictionaries.
- construct_prompt(code): Build the instruction prompt for the LLM given a code string.
- test_llm_on_jsonl(result_prefix, jsonl_prefix, provider, model, use_temperature, use_high_reasoning, iterations=None): Run the LLM on JSONL datasets and record per-item accuracy over specified iterations.
- main(): Parse command-line arguments and invoke the benchmarking process.

Authors: Derek Sheen, Hokyung (Andy) Lee
Emails: derek.s.prog@gmail.com (D. Sheen), techandy42@gmail.com (H. Lee)
Date: May 3, 2025
"""


import os
import json
import argparse
from tqdm import tqdm
from .llm_utils import completion_with_backoff


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
- You will analyze Python source code containing multiple functions.
- Exactly one function contains a bug that makes it behave incorrectly.
- Each function has a description above it explaining its intended behavior.
- Compare each function's implementation against its description to identify the buggy one.
- Focus on logical errors, not formatting or indentation issues.
<instruction_header>

<source_code>
{code}
<source_code>

<output_format>
- Return only the name of the buggy function.
<output_format>
"""


def test_llm_on_jsonl(result_prefix: str, jsonl_prefix: str, provider: str, model: str, use_temperature: bool, use_high_reasoning: bool, iterations: list[int] = None):
    """Run the LLM on JSONL datasets and record results with accuracy."""
    os.makedirs('data/result', exist_ok=True)
    os.makedirs(f'data/result/{provider}_{model}', exist_ok=True)

    iterations = range(20) if iterations is None else iterations

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

                # Retry logic for getting a valid response
                max_retries = 5
                prediction = None
                
                for attempt in range(max_retries):
                    response = completion_with_backoff(
                        model_full=f"{provider}/{model}",
                        prompt=prompt,
                        max_tokens=16000,
                        use_temperature=use_temperature,
                        use_high_reasoning=use_high_reasoning,
                    )
                    
                    # Check if response text is not None
                    if response['choices'][0]['text'] is not None:
                        prediction = response['choices'][0]['text']
                        break
                
                # If all retries failed, raise an error
                if prediction is None:
                    raise ValueError(f"Failed to get a valid response after {max_retries} attempts for code sample")
                
                # Apply strip() after we know prediction is not None
                prediction = prediction.strip()
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
    parser.add_argument('--use-high-reasoning', action='store_true', help="Use high reasoning effort")
    parser.add_argument('--iterations', nargs='+', type=int, choices=list(range(20)), help="Dataset iteration(s) to run (0-19)")
    args = parser.parse_args()

    provider = args.provider
    model = args.model
    use_temperature = not args.no_temperature
    use_high_reasoning = args.use_high_reasoning
    iterations = args.iterations

    results_prefix = f"data/result/{provider}_{model}/bics_result"
    dataset_prefix = "data/output/bics_dataset"
    test_llm_on_jsonl(results_prefix, dataset_prefix, provider, model, use_temperature, use_high_reasoning, iterations)


if __name__ == "__main__":
    main()
