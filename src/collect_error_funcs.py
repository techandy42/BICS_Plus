from datasets import load_dataset
from .llm_utils import completion_with_backoff


def main():
    dataset = load_dataset("mbpp", split="train")
    print(dataset[0])


if __name__ == "__main__":
    main()
