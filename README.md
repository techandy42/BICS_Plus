# BICS+ Benchmark

This repository is a benchmark for large language models (LLMs) on the bug-identification task, specifically long-context Python code made up of the MBPP dataset containing LLM-generated semantic bugs to simulate real-life software bugs.

## Benchmark Construction

- Python functions from the MBPP dataset are assembled upto X context length (e.g., 500, 1K, 2K, 4K, 8K, 16K).
- At Y target depth (e.g., 0%, 25%, 50%, 75%, 100%, 0% being at the beginning of assembled code and 100% being at the beginning of assembled code), insert a buggy function.
- The buggy functions are curated by running Claude-3.5-Sonnet on the MBPP dataset, and curating the outputs that fail the provided unit tests.

## Usage

Follow these steps to generate, benchmark, and visualize your datasets:

1. Run `create_benchmark`
2. Run `test_benchmark` (once for each model)
3. Run `visualize_benchmark` (once for each model)

### 1. Generate Benchmark Dataset (`create_benchmark.py`)

```bash
python -m src.create_benchmark
```

* **Inputs:**

  * MBPP dataset (fetched automatically).
  * `data/source/all_error_funcs.json`.
* **Outputs:**

  * `data/output/bics_dataset_{i}.jsonl` (i = 0..19).

### 2. Run LLM Benchmarks (`test_benchmark.py`)

```bash
python -m src.test_benchmark \
  --provider <provider> \
  --model <model> \
  [--no-temperature] \
  [--iterations 0 5 12]
```

* **Parameters:**

  * `--provider` (required): LLM provider (e.g., `openai`, `groq`).
  * `--model` (required): Model name (e.g., `gpt-4o`, `o3-mini`).
  * `--no-temperature` (optional): Exclude `temperature=0.0` from API calls (must apply for reasoning models, such as `o4-mini` from `openai`).
  * `--iterations` (optional): Specific dataset indices (0–19) to process; defaults to all.
* **Inputs:**

  * `data/output/bics_dataset_{i}.jsonl`.
* **Outputs:**

  * `data/result/{provider}_{model}/bics_result_{i}.jsonl`.

### 3. Visualize Results (`visualize_benchmark.py`)

```bash
python -m src.visualize_benchmark \
  --provider <provider> \
  --model <model>
```

* **Inputs:**

  * `data/result/{provider}_{model}/bics_result_{i}.jsonl`.
* **Outputs:**

  * `data/visualization/{provider}_{model}_benchmark.png`.

## Benchmark Results

### Groq DeepSeek (Llama 70B Distill)

<img src="data/visualization_archive/groq_deepseek-r1-distill-llama-70b_benchmark.png" width="500px" alt="Groq Llama 70B Distill Benchmark">

### Groq DeepSeek (Qwen 32B)

<img src="data/visualization_archive/groq_deepseek-r1-distill-qwen-32b_benchmark.png" width="500px" alt="Groq Qwen 32B Benchmark">

### Groq Llama 3.3 70B Versatile

<img src="data/visualization_archive/groq_llama-3.3-70b-versatile_benchmark.png" width="500px" alt="Groq Llama 3.3 Versatile Benchmark">

### OpenAI GPT-4.1-mini

<img src="data/visualization_archive/openai_gpt-4.1-mini_benchmark.png" width="500px" alt="OpenAI GPT-4.1-mini Benchmark">

### OpenAI GPT-4o

<img src="data/visualization_archive/openai_gpt-4o_benchmark.png" width="500px" alt="OpenAI GPT-4o Benchmark">

### OpenAI o1

<img src="data/visualization_archive/openai_o1_benchmark.png" width="500px" alt="OpenAI o1 Benchmark">

### OpenAI o3-mini

<img src="data/visualization_archive/openai_o3-mini_benchmark.png" width="500px" alt="OpenAI o3-mini Benchmark">

## Contributing New Models

Contributors who want to add new models to the leaderboard should:

* Run benchmarks using the provided scripts.
* Add your result JSONL files into `data/result_archive`.
* Add your visualization images into `data/visualization_archive`.
* Submit a pull request with these additions.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
