"""
Module: src.create_benchmark

This module generates benchmark datasets by constructing stacks of Python functions from the MBPP dataset,
then inserting a randomly chosen buggy function at varying positions (depth percentages) and context sizes.

Functions:
- get_function_name(function_str): Extract the name of a Python function defined in a string.
- convert_codestack_to_string(codestack): Concatenate a list of function definitions into a single code string.
- generate_code_stack(context_size, error_func): Build a list of functions whose total token count fits within context_size.
- insert_buggy_function(codestack, error_function, depth_percentage): Insert the buggy function into the code stack at the specified depth.
- run_tests(all_error_func_entries, context_sizes, depth_sizes, results_file): Iterate over context and depth configurations to create JSONL entries.
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
from .llm_utils import get_encoder, count_tokens


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


def parse_description(text):
    """Parse MBPP description by removing 'Write a ' and capitalizing the next character."""
    if text.startswith("Write a "):
        # Remove "Write a " and capitalize the first character of the remaining text
        parsed_text = text[8:]  # Skip "Write a " (8 characters)
        if parsed_text:
            parsed_text = parsed_text[0].upper() + parsed_text[1:]
        return parsed_text
    return text


def generate_code_stack(context_size, random_error_func):
    random.shuffle(dataset_functions)
    random_error_func_tokens = count_tokens(random_error_func, encoder)
    codestack = []
    token_count = 0
    
    for function_code, function_text in dataset_functions:
        # Format the function with its parsed description
        parsed_text = parse_description(function_text)
        function_with_description = f'"""\n{parsed_text}\n"""\n{function_code}'
        function_token_count = count_tokens(function_with_description, encoder)
        
        # If adding this function exceeds the context size, stop
        if token_count + function_token_count + random_error_func_tokens > context_size:
            break
        
        # Otherwise, add the function to the codestack
        codestack.append(function_with_description)
        token_count += function_token_count
    
    return codestack


def insert_buggy_function(codestack, error_function, depth_size):
    # Calculate the total token count and token counts for each function
    token_counts = []
    total_tokens = 0
    
    for function in codestack:
        tokens = count_tokens(function, encoder)
        token_counts.append(tokens)
        total_tokens += tokens
    
    # Calculate the target token position based on depth percentage
    target_position = int((depth_size / 100) * total_tokens)
    
    # Find the insertion index where accumulated tokens reach the target position
    accumulated_tokens = 0
    insertion_index = 0
    
    for i, tokens in enumerate(token_counts):
        if accumulated_tokens >= target_position:
            insertion_index = i
            break
        accumulated_tokens += tokens
    else:
        # If we've gone through all functions, insert at the end
        insertion_index = len(codestack)
    
    # Insert the error function at the calculated index
    codestack.insert(insertion_index, error_function)
    
    return codestack


def run_tests(all_error_func_entries, context_sizes, depth_sizes, results_file):
    os.makedirs('data/output', exist_ok=True)
    for i in tqdm(range(20), desc="Processing testcases"):
        # Collect all entries for this testcase
        entries = []
        
        for context_length in tqdm(context_sizes, desc="Processing context sizes", leave=False):
            depth_bar = tqdm(depth_sizes, leave=False)
            for depth_percentage in depth_bar:
                random_error_entry = random.choice(all_error_func_entries)
                random_error_func = random_error_entry['generated_code']
                task_id = random_error_entry['task_id']
                
                # Get the description for this error function
                error_func_description = task_id_to_text.get(task_id, "")
                # Format the error function with its parsed description
                parsed_error_description = parse_description(error_func_description)
                error_func_with_description = f'"""\n{parsed_error_description}\n"""\n{random_error_func}'
                
                error_func_name = get_function_name(random_error_func)

                depth_bar.set_description(
                    f"Buggy: {error_func_name}, Context: {context_length}, Depth: {depth_percentage}%"
                )

                codestack = generate_code_stack(context_length, error_func_with_description)
                codestack = insert_buggy_function(codestack, error_func_with_description, depth_percentage)
                str_codestack = convert_codestack_to_string(codestack)

                entry = {
                    "code": str_codestack,
                    "func_error": error_func_name,
                    "context_length": context_length,
                    "depth_percentage": depth_percentage,
                    "num_functions": len(codestack)
                }

                # Add the entry to the list
                entries.append(entry)
        
        # Write all entries to the file after loops complete
        with open('data/output/'+results_file+f'_{i}.jsonl', 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')


# Load the MBPP dataset
dataset = load_dataset('google-research-datasets/mbpp')
# Create a mapping from task_id to text for error function lookups
task_id_to_text = {example['task_id']: example['text'] for example in dataset['train']}

# Initialize global variables
dataset_functions = []
encoder = get_encoder()


def main():
    global dataset_functions  # Make dataset_functions available globally
    
    # Set random seed for reproducible results
    random.seed(42)
    
    context_sizes = [500, 1000, 2000, 4000, 8000, 16000]
    depth_sizes = [0, 25, 50, 75, 100]  # Percentages as integers

    # Load error functions from JSONL file
    all_error_func_entries = []
    error_task_ids = set()  # Set to store task_ids from error functions
    with open('data/source/reasonable_error_funcs.jsonl', 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            all_error_func_entries.append(entry)
            error_task_ids.add(entry['task_id'])  # Collect error function task_ids
    
    # Filter dataset_functions to exclude items with task_ids in error_task_ids
    dataset_functions = [
        (example['code'], example['text']) 
        for example in dataset['train'] 
        if example['task_id'] not in error_task_ids
    ]
        
    results_file = "bics_dataset"
    run_tests(all_error_func_entries, context_sizes, depth_sizes, results_file)


if __name__ == "__main__":
    print("Starting the process...")
    main()
