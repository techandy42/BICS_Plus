from datasets import load_dataset, Dataset, DatasetDict
import os
import argparse
from huggingface_hub import login

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

def check_valid_colon(code):
    """
    Check if the spacing before colons is proper in Python code.
    
    Args:
        code (str): The Python code to check
        
    Returns:
        bool: True if all colons have proper spacing, False otherwise
    """
    # Keywords that should end with a colon and cannot have space before it
    colon_keywords = {
        'def', 'class', 'if', 'elif', 'else', 'for', 'while', 
        'try', 'except', 'finally', 'with', 'match', 'case'
    }
    
    lines = code.split('\n')
    
    for line in lines:
        stripped_line = line.strip()
        
        # Skip empty lines and comments
        if not stripped_line or stripped_line.startswith('#'):
            continue
            
        # Check if line ends with colon
        if stripped_line.endswith(':'):
            # Check if line starts with any of the colon keywords
            for keyword in colon_keywords:
                if stripped_line.startswith(keyword):
                    # Check if there's a space before the colon
                    if stripped_line.endswith(' :'):
                        return False
                    break
            else:
                # Handle special cases like lambda, list comprehensions, etc.
                # that might contain colons but don't start with keywords
                
                # Check for lambda expressions
                if 'lambda' in stripped_line and stripped_line.endswith(' :'):
                    return False
                    
                # Check for dictionary definitions or other cases
                # where we might have a colon at the end
                words = stripped_line.split()
                if len(words) >= 2 and words[-1] == ':':
                    # If the second-to-last word is a keyword that should have a colon
                    if any(keyword in stripped_line for keyword in colon_keywords):
                        return False
    
    return True

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
    dataset = load_dataset('google-research-datasets/mbpp', 'sanitized')
    
    print("\nProcessing test split...")
    test_data = dataset['test']
    
    # Filter items with valid indentation
    valid_items = []
    invalid_count = 0
    
    for i, item in enumerate(test_data):
        if i % 100 == 0:
            print(f"  Processing item {i}/{len(test_data)}...")

        code = item['code']
        
        if check_valid_indentation(code) and check_valid_colon(code):
            valid_items.append(item)
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
            # Create a DatasetDict with explicit "test" split
            dataset_dict = DatasetDict({"test": cleaned_dataset})
            dataset_dict.push_to_hub(args.repo_name, private=False)
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
    print(f"Original test dataset size: {len(test_data)} items")
    print(f"Cleaned test dataset size: {len(cleaned_dataset)} items")

if __name__ == '__main__':
    main()
