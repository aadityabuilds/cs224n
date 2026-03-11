"""
Simple test: load Qwen 1.5B, get a livecodebench problem, generate code, run it through the
SDPO verification environment, and print the feedback.
"""
import importlib.util
import os
import pathlib
import sys

# Make SDPO submodule importable (needed for data.* namespace packages)
SDPO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SDPO")
sys.path.insert(0, SDPO_PATH)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# data.* has no __init__.py so namespace package resolution via sys.path works fine
from data.utils.livecodebench import load_livecodebench
from data.format.prompts import CODE_PROMPT

# Load code.py directly to avoid verl/__init__.py which pulls in ray
def _load_module_from_file(name, rel_path):
    path = pathlib.Path(SDPO_PATH) / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_code_mod = _load_module_from_file("feedback_code", "verl/utils/reward_score/feedback/code.py")
compute_score = _code_mod.compute_score

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    # ── 1. Load model ──────────────────────────────────────────────────────────
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    print(f"Model loaded on {next(model.parameters()).device}")

    # ── 2. Load dataset ────────────────────────────────────────────────────────
    print("\nLoading livecodebench (train split) ...")
    dataset = load_livecodebench("train")
    print(f"Dataset size: {len(dataset)}")

    example = dataset[0]
    problem = example["problem"]
    tests_json = example["tests"]

    print("\n--- Problem (first 600 chars) ---")
    print(problem[:600])
    print("...")

    # ── 3. Build prompt & generate ─────────────────────────────────────────────
    prompt = CODE_PROMPT.format(problem=problem)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("\nGenerating response ...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )

    print("\n--- Model Response ---")
    print(response)

    # ── 4. Run through verification environment ────────────────────────────────
    print("\nRunning verification environment ...")
    result = compute_score(
        solution=response,
        ground_truth=tests_json,
        extra_info={"split": "train"},
        sparse_rewards=False,
    )

    print("\n=== Verification Results ===")
    print(f"  score            : {result['score']:.3f}")
    print(f"  accuracy         : {result['acc']:.3f}")
    print(f"  incorrect_format : {result['incorrect_format']}")
    print(f"  error_in_tests   : {result['error_in_test_cases']}")
    print(f"  timed_out        : {result['timed_out']}")
    print("\n--- Feedback ---")
    print(result["feedback"] or "(All tests passed!)")


if __name__ == "__main__":
    main()
