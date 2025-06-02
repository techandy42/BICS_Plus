from datasets import load_dataset, Dataset
import re
import os
import argparse
from huggingface_hub import HfApi, login

def check_valid_indentation(code: str) -> bool:
    """
    Check if code has valid indentation (4 spaces or tab character).
    Returns True if all indentations are valid, False otherwise.
    """
    lines = code.split('\n')
    
    for line in lines:
        # Skip empty lines or lines with no indentation
        if not line or not line.strip():
            continue
            
        # Check leading whitespace
        leading_whitespace = line[:len(line) - len(line.lstrip())]
        
        if leading_whitespace:
            # Check if it's a valid combination of 4 spaces or tabs
            # Count spaces and tabs
            space_count = leading_whitespace.count(' ')
            tab_count = leading_whitespace.count('\t')
            
            # Check if it's either:
            # 1. Only tabs
            # 2. Only spaces in multiples of 4
            # 3. Mix is not allowed
            if tab_count > 0 and space_count > 0:
                # Mixed tabs and spaces - invalid
                return False
            elif space_count > 0 and space_count % 4 != 0:
                # Spaces not in multiples of 4 - invalid
                return False
            
    return True

def standardize_item(item):
    """
    Standardize an item to ensure consistent schema across splits.
    Handles null values and converts them to appropriate empty values.
    """
    standardized = {
        'task_id': item['task_id'],
        'text': item['text'],
        'code': item['code'],
        'test_list': item['test_list'] if item['test_list'] is not None else [],
        'test_setup_code': item['test_setup_code'] if item['test_setup_code'] is not None else '',
    }
    
    # Handle challenge_test_list - it might be null or contain null values
    challenge_list = item.get('challenge_test_list', [])
    if challenge_list is None:
        standardized['challenge_test_list'] = []
    else:
        # Filter out null values and ensure all are strings
        standardized['challenge_test_list'] = [
            str(test) if test is not None else '' 
            for test in challenge_list
        ]
    
    return standardized

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Clean MBPP dataset by filtering items with valid indentation")
    parser.add_argument("--upload", action="store_true", 
                       help="Upload to HuggingFace instead of saving locally")
    parser.add_argument("--repo-name", type=str,
                       help="HuggingFace repository name (e.g., 'username/mbpp-clean-indentation')")
    parser.add_argument("--token", type=str,
                       help="HuggingFace API token (optional, can use HF_TOKEN env var)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.upload and not args.repo_name:
        parser.error("--repo-name is required when using --upload")
    
    # Load the MBPP dataset
    print("Loading MBPP dataset...")
    dataset = load_dataset('google-research-datasets/mbpp')
    
    print("\nProcessing train split...")
    train_data = dataset['train']
    
    # Filter items with valid indentation
    valid_items = []
    invalid_count = 0
    
    for i, item in enumerate(train_data):
        if i % 100 == 0:
            print(f"  Processing item {i}/{len(train_data)}...")
            
        code = item['code']
        
        if check_valid_indentation(code):
            # Standardize the item before adding
            standardized_item = standardize_item(item)
            valid_items.append(standardized_item)
        else:
            invalid_count += 1
            
    print(f"  Found {len(valid_items)} valid items and {invalid_count} invalid items")
    
    # Create a new dataset from valid items
    cleaned_dataset = Dataset.from_list(valid_items)
    
    if args.upload:
        # Upload to HuggingFace
        print(f"\nUploading cleaned dataset to HuggingFace as '{args.repo_name}'...")
        
        # Login to HuggingFace
        if args.token:
            login(token=args.token)
        elif os.environ.get("HF_TOKEN"):
            login(token=os.environ.get("HF_TOKEN"))
        else:
            print("No token provided. Please provide --token or set HF_TOKEN environment variable.")
            print("Attempting to use cached credentials...")
        
        # Push to hub
        try:
            cleaned_dataset.push_to_hub(args.repo_name, private=False)
            print(f"Successfully uploaded dataset to https://huggingface.co/datasets/{args.repo_name}")
        except Exception as e:
            print(f"Error uploading to HuggingFace: {e}")
            return
    else:
        # Save locally
        print("\nSaving cleaned dataset locally...")
        
        # Create output directory if it doesn't exist
        output_dir = "data/cleaned_mbpp"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save locally to data/cleaned_mbpp/ folder
        cleaned_dataset.save_to_disk(output_dir)
        print(f"Dataset saved to {output_dir}/")
    
    print("\nDataset cleaning complete!")
    print(f"Original train dataset size: {len(train_data)} items")
    print(f"Cleaned train dataset size: {len(cleaned_dataset)} items")

if __name__ == '__main__':
    main()
