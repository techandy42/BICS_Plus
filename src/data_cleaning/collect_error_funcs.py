"""
Collect Error Functions Module

This module processes the MBPP (Mostly Basic Python Problems) dataset to generate
code solutions using a language model and collect cases where the generated code
fails to pass the provided test cases.

Key functionality:
- Loads the MBPP dataset from HuggingFace datasets
- Extracts function names from assertion-style test cases
- Generates code prompts with problem descriptions and sample tests
- Uses LLM to generate Python function implementations
- Tests generated code against provided test cases in isolated subprocesses
- Collects and saves failed test cases to a JSONL file for further analysis

The main workflow:
1. Iterate through MBPP dataset items
2. Extract function name from the first test case
3. Create a code generation prompt with problem description and tests
4. Get LLM-generated code solution
5. Test the solution against all test cases in a subprocess
6. Save failed cases with task_id and generated_code to error_funcs.jsonl

Output:
- Creates data/source/error_funcs.jsonl with failed test cases for error analysis

Usage:
    python -m src.data_cleaning.collect_error_funcs

Dependencies:
    - datasets: For loading MBPP dataset
    - tqdm: For progress tracking
    - LLM completion utilities from llm_utils module

Authors: Hokyung (Andy) Lee
Emails: techandy42@gmail.com
Date: May 31, 2025
"""


import json
import re
import subprocess
import tempfile
import os
from tqdm import tqdm
from datasets import load_dataset
from ..llm_utils import completion_with_backoff
from textwrap import dedent


def get_function_name(test_item: str) -> str | None:
    """
    Extract function name from a test assertion string.
    
    Args:
        test_item: String like 'assert max_chain_length([Pair(5, 24), Pair(15, 25),Pair(27, 40), Pair(50, 60)], 4) == 3'
    
    Returns:
        Function name (e.g., 'max_chain_length') or None if format doesn't match
    """
    # Pattern to match 'assert function_name('
    pattern = r'^assert\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    match = re.match(pattern, test_item)
    if match:
        return match.group(1)
    return None


def get_codegen_prompt(text: str, function_name: str, test_list: list[str]) -> str:
    test_list_str = "\n".join(test_list)

    return dedent(f"""
Instructions:
\"\"\"
- Write a Python function with name "{function_name}" that solves the following problem.
- You may import any standard library modules.
- If the sample test cases use any custom classes, you must declare them in your code and use them accordingly in your "{function_name}" function.
- You may declare additional helper functions if you need them, but only if your code is complex and you need to break it down into smaller functions.
- Your final output should be a valid Python function definition.
- Do not include any other text or comments in your response.
- Do not include any markdown tags in your response.
\"\"\"

Problem:
\"\"\"
{text}
\"\"\"

Sample Test Cases:
\"\"\"
{test_list_str}
\"\"\"
""").strip('\n')


def run_tests(generated_code: str, test_setup_code: str, test_list: list[str], challenge_test_list: list[str]) -> bool:
    """
    Run the generated code with tests in a Python subprocess.
    
    Returns:
        True if code runs without exceptions, False otherwise
    """
    code_to_test = generated_code + "\n\n" + test_setup_code + "\n\n" + "\n".join(test_list) + "\n\n" + "\n".join(challenge_test_list)
    
    try:
        # Create a temporary file to write the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code_to_test)
            temp_file_path = temp_file.name
        
        # Run the code in a subprocess with timeout
        result = subprocess.run(
            ['python', temp_file_path],
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        # Return True if exit code is 0 (success), False otherwise
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        # Clean up the temporary file if it exists
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        return False
    except Exception:
        # Clean up the temporary file if it exists
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        return False


def main():
    dataset = load_dataset("mbpp", split="train")
    failed_tests = []
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        task_id = item['task_id']
        text = item['text']
        test_setup_code = item['test_setup_code']
        test_list = item['test_list']
        challenge_test_list = item['challenge_test_list']
        if len(test_list) == 0:
            continue
        function_name = get_function_name(test_list[0])
        if function_name is None:
            continue
        prompt = get_codegen_prompt(text, function_name, test_list)
        response = completion_with_backoff(
            model_full="openai/gpt-4.1",
            prompt=prompt,
            max_tokens=4096,
            use_temperature=True,
            use_high_reasoning=False
        )
        generated_code = response['choices'][0]['text']
        if not run_tests(generated_code, test_setup_code, test_list, challenge_test_list):
            failed_tests.append({
                "task_id": task_id,
                "generated_code": generated_code,
            })

    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(script_dir)
    output_file_path = os.path.join(workspace_root, "data", "source", "error_funcs.jsonl")
    
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, "w") as f:
        for failed_test in failed_tests:
            f.write(json.dumps(failed_test) + "\n")

    print(f"Saved {len(failed_tests)} failed tests to {output_file_path}")


if __name__ == "__main__":
    main()
