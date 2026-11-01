"""Run their generate_response.py unchanged on one probe part. vLLM on a MIG slice must spawn its engine, which
re-imports the main module; their script has no __main__ guard, so it is exec'd here under this file's guard (runpy
would make their file the main module and the child would rerun it)."""
import sys

if __name__ == "__main__":
    i = sys.argv[1]
    sys.argv = ["generate_response.py", "--model", "qwen3-4b-thinking", "--probe_path", f"data/part_{i}.jsonl",
                "--dataset", f"part{i}", "--tensor_parallel_size", "1"]
    exec(compile(open("generate_response.py").read(), "generate_response.py", "exec"))
