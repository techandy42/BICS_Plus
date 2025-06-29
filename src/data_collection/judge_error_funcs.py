"""
Judge Error Functions Module

This module processes failed code generation cases collected from the MBPP dataset
to classify errors as either "reasonable" or "ridiculous" using an LLM judge.
It filters out trivial errors to focus on meaningful failure cases for analysis.

Key functionality:
- Loads failed test cases from error_funcs.jsonl generated by collect_error_funcs
- Uses LLM to analyze each failed code generation case
- Classifies errors as "reasonable" (logical/algorithmic errors) or "ridiculous" (trivial errors)
- Implements robust retry logic for JSON parsing failures
- Saves only reasonable error cases for further benchmark creation

Error Classification Criteria:
- Reasonable errors: Logical or algorithmic mistakes that are educationally valuable
- Ridiculous errors: 
  * Incorrect function names (even single character differences)
  * Unreasonable logic given the problem statement
  * Python syntax errors that a linter would catch

The main workflow:
1. Load failed test cases from data/source/error_funcs.jsonl
2. For each failed case, create a judgment prompt with code, problem, and tests
3. Get LLM evaluation of the error type and cause
4. Retry up to 10 times for JSON parsing failures
5. Filter and save only "reasonable" errors to reasonable_error_funcs.jsonl

Input:
- Reads data/source/error_funcs.jsonl

Output:
- Creates data/source/reasonable_error_funcs.jsonl with filtered reasonable errors

Usage:
    python -m src.data_collection.judge_error_funcs

Dependencies:
    - tqdm: For progress tracking
    - LLM completion utilities from llm_utils module
    - Function name extraction from collect_error_funcs module

Authors: Hokyung (Andy) Lee
Emails: techandy42@gmail.com
Date: May 31, 2025
"""


import json
import os
from tqdm import tqdm
from textwrap import dedent
from ..llm_utils import completion_with_backoff
from .collect_error_funcs import get_function_name


def get_code_judge_prompt(generated_code: str, ground_truth_code: str, function_name: str, prompt: str, test_imports: list[str], test_list: list[str]) -> str:
    test_list_str = "\n".join(test_list)
    test_imports_str = "\n".join(test_imports)

    return dedent(f"""
Instructions:
\"\"\"
- The generated code is incorret and fails to pass the test cases.
- The ground truth code is the correct code that should pass the test cases.
- Your job is to determine if the error is reasonable or ridiculous.
- An error is reasonable if it's logical error.
- An error is ridiculous if:
    - The function name is declared incorrectly, it's supposed to be: "{function_name}".
    - If the function name doesn't match, even a single character, it's ridiculous.
    - The logic in the code is unreasonable given the problem statement.
    - The error is due to Python syntax error that a linter would catch.
- Return your response as valid JSON with the following fields:
    - "cause_of_error": str, detailed analysis of the error in the generated code, based on comparing it to the ground truth code and examining the test cases.
    - "final_verdict": str, "reasonable" or "ridiculous".
- Do not include markdown tags in your response.
- Make sure your response can be parsed into valid JSON without any post-processing.
\"\"\"

Generated Code:
\"\"\"
{generated_code}
\"\"\"

Ground Truth Code:
\"\"\"
{ground_truth_code}
\"\"\"

Problem Statement:
\"\"\"
{prompt}
\"\"\"

Sample Test Cases:
\"\"\"
{test_imports_str}

{test_list_str}
\"\"\"

Output JSON Format:
\"\"\"
{{
    "cause_of_error": str,
    "final_verdict": str
}}
\"\"\"
""").strip("\n")


def main():
    failed_tests = []

    output_file_path = "data/source/error_funcs.jsonl"

    with open(output_file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            failed_tests.append(data)

    reasonable_tests = []
    for failed_test in tqdm(failed_tests):
        # Skip items with ``` in generated_code
        if "```" in failed_test.get("generated_code", ""):
            print(f"Skipping task {failed_test.get('task_id', 'unknown')} due to ``` in generated_code")
            continue
        
        # Read all fields from the JSON item
        task_id = failed_test["task_id"]
        ground_truth_code = failed_test["code"]
        generated_code = failed_test["generated_code"]
        prompt = failed_test["prompt"]
        test_imports = failed_test["test_imports"]
        test_list = failed_test["test_list"]
        
        function_name = get_function_name(test_list[0])
        prompt_text = get_code_judge_prompt(generated_code, ground_truth_code, function_name, prompt, test_imports, test_list)
        
        # Retry logic for JSON parsing errors
        max_retries = 10
        output = None
        
        for attempt in range(max_retries):
            try:
                response = completion_with_backoff(
                    model_full="openai/o4-mini",
                    prompt=prompt_text,
                    max_tokens=1024,
                    use_temperature=False,
                    use_high_reasoning=False
                )
                output = json.loads(response['choices'][0]['text'])
                break  # Success, exit retry loop
            except json.JSONDecodeError:
                print(f"JSON parsing error for {task_id}, attempt {attempt + 1}/{max_retries}, invalid output: {response['choices'][0]['text']}")
                if attempt == max_retries - 1:  # Last attempt failed
                    print(f"Failed to parse JSON for {task_id} after {max_retries} attempts, skipping")
                    continue
        
        if output is None:
            continue  # Skip this item if all JSON parsing attempts failed
        
        cause_of_error = output["cause_of_error"]
        final_verdict = output["final_verdict"]
        if final_verdict == "reasonable":
            print(f"Reasonable error for {task_id}")
            print(f"Cause of error: {cause_of_error}")
            # Include all fields from the original failed_test
            reasonable_test = failed_test.copy()
            reasonable_test["cause_of_error"] = cause_of_error
            reasonable_tests.append(reasonable_test)
        elif final_verdict == "ridiculous":
            print(f"Ridiculous error for {task_id}")
            print(f"Cause of error: {cause_of_error}")
        else:
            print(f"Invalid verdict for {task_id}")

    reasonable_output_file_path = "data/source/reasonable_error_funcs.jsonl"
    
    os.makedirs(os.path.dirname(reasonable_output_file_path), exist_ok=True)

    with open(reasonable_output_file_path, "w") as f:
        for test in reasonable_tests:
            f.write(json.dumps(test) + "\n")

    print(f"Saved {len(reasonable_tests)} reasonable tests out of {len(failed_tests)} tests to {reasonable_output_file_path}")


if __name__ == "__main__":
    main()
