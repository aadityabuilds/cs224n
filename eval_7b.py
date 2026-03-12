"""Evaluation for 7B model — memory-efficient, one model at a time."""
import json
import logging
import os
import sys

SDPO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SDPO")
sys.path.insert(0, SDPO_PATH)

from data.utils.livecodebench import load_livecodebench
from agent.config import build_code_prompt
from agent.model_7b import AgentModel7B
from agent.verification import verify_solution
from agent.rag import RAGDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("eval_7b.log"),
    ]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logger = logging.getLogger("eval_7b")


def evaluate_model(agent_model, dataset, label, max_problems=50,
                   rag_db=None, max_new_tokens=1536, rag_top_k=3):
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATING: {label}")
    logger.info(f"{'='*60}")

    total_accuracy = 0.0
    total_correct = 0
    results = []
    num_problems = min(len(dataset), max_problems) if max_problems else len(dataset)

    for idx in range(num_problems):
        example = dataset[idx]
        problem = example["problem"]
        tests_json = example["tests"]
        description = example.get("description", "")

        system_prompt = None
        if rag_db and rag_db.size > 0:
            chunks = rag_db.query(description or problem[:500], top_k=rag_top_k)
            if chunks:
                rag_context = "\n\n".join(f"[Knowledge {i+1}]: {c}" for i, c in enumerate(chunks))
                system_prompt = (
                    "You are a coding expert. Below are some knowledge snippets retrieved "
                    "from past problem-solving sessions. They may or may not be relevant to "
                    "the current problem — only use them if they directly apply.\n\n"
                    f"{rag_context}\n\n"
                    "If none of the above is relevant, ignore it and solve the problem from scratch."
                )

        code_prompt = build_code_prompt(problem, tests_json)

        responses, _, _ = agent_model.generate(
            prompt=code_prompt, system_prompt=system_prompt,
            num_return_sequences=1, temperature=0.0,
            max_new_tokens=max_new_tokens,
        )

        result = verify_solution(responses[0], tests_json, split="test")
        score = result["score"]
        acc = result["acc"]
        total_accuracy += acc
        if score > 0.99:
            total_correct += 1

        results.append({
            "step": idx + 1, "score": score, "acc": acc,
            "feedback": result["feedback"][:200] if result.get("feedback") else "passed",
        })

        n = idx + 1
        if n % 5 == 0 or n == num_problems:
            logger.info(f"[{label}] {n}/{num_problems} | "
                        f"This: acc={acc:.3f} solved={'YES' if score > 0.99 else 'NO'} | "
                        f"Test-case acc: {total_accuracy/n:.3f} | "
                        f"Solve rate: {total_correct}/{n} ({total_correct/n:.3f})")

    avg_accuracy = total_accuracy / max(num_problems, 1)
    solve_rate = total_correct / max(num_problems, 1)

    metrics = {
        "label": label,
        "num_problems": num_problems,
        "avg_test_case_accuracy": avg_accuracy,
        "avg_score": avg_accuracy,
        "avg_accuracy": avg_accuracy,
        "solve_rate": solve_rate,
        "pass_rate": solve_rate,
        "num_correct": total_correct,
        "results": results,
    }

    logger.info(f"\n--- {label} ---")
    logger.info(f"  Test-case accuracy (avg % tests passed): {avg_accuracy:.4f}")
    logger.info(f"  Solve rate (100% correct):               {solve_rate:.4f} ({total_correct}/{num_problems})")
    return metrics


def run_eval_7b(trained_model_path="checkpoints/final_model",
                rag_db_path="checkpoints/rag_db.json",
                base_model_name="Qwen/Qwen2.5-7B-Instruct",
                max_problems=50, skip_base=False):
    import torch

    logger.info("Loading test dataset...")
    test_dataset = load_livecodebench("test")
    logger.info(f"Test dataset: {len(test_dataset)} problems")

    all_eval_results = {}

    # Load RAG
    rag_db = None
    if os.path.exists(rag_db_path):
        logger.info(f"Loading RAG from {rag_db_path}...")
        rag_db = RAGDatabase()
        with open(rag_db_path) as f:
            for chunk in json.load(f):
                rag_db.add(chunk)
        logger.info(f"RAG: {rag_db.size} chunks")

    # 1. Trained model + RAG (most important — run first)
    logger.info(f"\n=== Eval 1: TRAINED + RAG ===")
    trained = AgentModel7B(model_name=trained_model_path, lr=1e-6, ema_rate=0.005)
    full_metrics = evaluate_model(trained, test_dataset, "model+rag+sdpo",
                                  max_problems=max_problems, rag_db=rag_db)
    all_eval_results["model+rag+sdpo"] = full_metrics

    # 2. Trained model without RAG
    logger.info(f"\n=== Eval 2: TRAINED (no RAG) ===")
    sdpo_metrics = evaluate_model(trained, test_dataset, "model+sdpo",
                                  max_problems=max_problems, rag_db=None)
    all_eval_results["model+sdpo"] = sdpo_metrics

    # Free trained model before loading base
    del trained
    torch.cuda.empty_cache()

    # 3. Base model
    if not skip_base:
        logger.info(f"\n=== Eval 3: BASE 7B ===")
        base = AgentModel7B(model_name=base_model_name, lr=1e-6, ema_rate=0.005)
        base_metrics = evaluate_model(base, test_dataset, "base_qwen7b",
                                      max_problems=max_problems, rag_db=None)
        all_eval_results["base_qwen7b"] = base_metrics
        del base
        torch.cuda.empty_cache()

    # Print comparison
    logger.info("\n" + "=" * 80)
    logger.info("7B COMPARISON")
    logger.info("=" * 80)
    logger.info(f"  {'Config':<25} {'Test-Case Acc':>14} {'Solve Rate':>12} {'Correct':>10}")
    logger.info("-" * 80)
    for key in ["model+rag+sdpo", "base_qwen7b", "model+sdpo"]:
        m = all_eval_results.get(key, {})
        if "avg_score" in m:
            logger.info(f"  {m['label']:<25} {m['avg_score']:>14.4f} "
                        f"{m['pass_rate']:>12.4f} "
                        f"{m['num_correct']:>5}/{m['num_problems']}")
    logger.info("=" * 80)
    logger.info("Test-Case Acc = avg fraction of test cases passed per problem")
    logger.info("Solve Rate    = fraction of problems fully solved (100% test cases)")

    with open("eval_results_7b.json", "w") as f:
        json.dump(all_eval_results, f, indent=2, default=str)

    return all_eval_results


if __name__ == "__main__":
    run_eval_7b()
