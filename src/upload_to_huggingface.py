import os
from datasets import load_from_disk
from huggingface_hub import login
import argparse

def upload_dataset(dataset_path, repo_name, token=None):
    """
    Upload a dataset to HuggingFace Hub.
    
    Args:
        dataset_path: Path to the local dataset
        repo_name: Name of the repository on HuggingFace (e.g., "username/dataset-name")
        token: HuggingFace API token (if not provided, will look for HF_TOKEN env var)
    """
    # Login to HuggingFace
    if token:
        login(token=token)
    elif os.environ.get("HF_TOKEN"):
        login(token=os.environ.get("HF_TOKEN"))
    else:
        print("Please provide a HuggingFace token either as argument or set HF_TOKEN environment variable")
        return
    
    # Load the dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    # Push to hub
    print(f"Pushing dataset to {repo_name}...")
    dataset.push_to_hub(repo_name, private=False)
    
    print(f"Successfully uploaded dataset to https://huggingface.co/datasets/{repo_name}")

def main():
    parser = argparse.ArgumentParser(description="Upload cleaned MBPP dataset to HuggingFace")
    parser.add_argument("--dataset-path", default="data/cleaned_mbpp", 
                       help="Path to the cleaned dataset")
    parser.add_argument("--repo-name", required=True,
                       help="Repository name on HuggingFace (e.g., 'username/mbpp-clean-indentation')")
    parser.add_argument("--token", help="HuggingFace API token")
    
    args = parser.parse_args()
    
    upload_dataset(args.dataset_path, args.repo_name, args.token)

if __name__ == "__main__":
    main()
