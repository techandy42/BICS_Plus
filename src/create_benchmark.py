"""
Module: src.create_benchmark

This module generates benchmark datasets by constructing stacks of Python functions from the MBPP dataset,
then inserting a randomly chosen buggy function at varying positions (depth percentages) and context sizes.

Functions:
- get_function_name(function_str): Extract the name of a Python function defined in a string.
- convert_codestack_to_string(codestack): Concatenate a list of function definitions into a single code string.
- generate_code_stack(context_size, error_func): Build a list of functions whose total token count fits within context_size.
- insert_buggy_function(codestack, error_function, depth_percentage): Insert the buggy function into the code stack at the specified depth.
- run_tests(all_error_funcs, context_sizes, depth_sizes, results_file): Iterate over context and depth configurations to create JSONL entries.
- main(): Define parameters, load error functions, and invoke run_tests to write output files.

Authors: Derek Sheen, Hokyung (Andy) Lee
Emails: derek.s.prog@gmail.com (D. Sheen), techandy42@gmail.com (H. Lee)
Date: May 28, 2025
"""


import json
import os
import random
import re
from datasets import load_dataset
from tqdm import tqdm


tqdm.pandas()


def get_function_name(function_str):
    match = re.match(r"def (\w+)\(", function_str)
    if match:
        return match.group(1)
    return None


def convert_codestack_to_string(codestack):
    # Strip each function of leading/trailing newlines and join with two newlines
    stripped_functions = [function.strip() for function in codestack]
    str_codestack = "\n\n".join(stripped_functions)
    # Remove all carriage return characters
    str_codestack = str_codestack.replace('\r', '')
    # Convert all tab characters to 4 spaces
    str_codestack = str_codestack.replace('\t', '    ')
    # Remove trailing whitespace from each line
    lines = str_codestack.split('\n')
    lines = [line.rstrip() for line in lines]
    str_codestack = '\n'.join(lines)
    return str_codestack


def generate_code_stack(context_size, random_error_func):
    random.shuffle(dataset_functions)
    random_error_func_tokens = len(random_error_func.split())
    codestack = []
    token_count = 0
    
    for function in dataset_functions:
        function_token_count = len(function.split())
        
        # If adding this function exceeds the context size, stop
        if token_count + function_token_count + random_error_func_tokens > context_size:
            break
        
        # Otherwise, add the function to the codestack
        codestack.append(function)
        token_count += function_token_count
    
    return codestack


def insert_buggy_function(codestack, error_function, depth_size):
    # Calculate the insertion index based on the depth
    num_functions = len(codestack)
    insertion_index = int((depth_size / 100) * num_functions)
    
    # Insert the error function at the calculated index
    codestack.insert(insertion_index, error_function)
    
    return codestack


def run_tests(all_error_funcs, context_sizes, depth_sizes, results_file):
    os.makedirs('data/output', exist_ok=True)
    for i in tqdm(range(20), desc="Processing testcases"):
        with open('data/output/'+results_file+f'_{i}.jsonl', 'a') as f:
            for context_length in tqdm(context_sizes, desc="Processing context sizes", leave=False):
                depth_bar = tqdm(depth_sizes, leave=False)
                for depth_percentage in depth_bar:
                    random_error_func = random.choice(all_error_funcs)
                    error_func_name = get_function_name(random_error_func)

                    depth_bar.set_description(
                        f"Buggy: {error_func_name}, Context: {context_length}, Depth: {depth_percentage}%"
                    )

                    codestack = generate_code_stack(context_length, random_error_func)
                    codestack = insert_buggy_function(codestack, random_error_func, depth_percentage)
                    str_codestack = convert_codestack_to_string(codestack)

                    entry = {
                        "code": str_codestack,
                        "func_error": error_func_name,
                        "context_length": context_length,
                        "depth_percentage": depth_percentage
                    }

                    # Write the entry to the results file
                    f.write(json.dumps(entry) + '\n')    


# Load the MBPP dataset
dataset = load_dataset('google-research-datasets/mbpp')
dataset_functions = [example['code'] for example in dataset['train']]


def main():
    # Set random seed for reproducible results
    random.seed(42)
    
    context_sizes = [500, 1000, 2000, 4000, 8000, 16000]
    depth_sizes = [0, 25, 50, 75, 100]  # Percentages as integers

    with open('data/source/all_error_funcs.json', 'r') as f:
        all_error_funcs = json.load(f)
        
        results_file = "bics_dataset"
        run_tests(all_error_funcs, context_sizes, depth_sizes, results_file)


if __name__ == "__main__":
    print("Starting the process...")
    main()
