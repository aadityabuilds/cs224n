"""Evaluation script: compare base Qwen model vs trained model, plus ablations.

Evaluates on the test split of LiveCodeBench:
1. Base Qwen model (no training, no RAG)
2. Model + SDPO only (SDPO-trained model, no RAG context)
3. Model + RAG only (base model weights with RAG)
4. Model + RAG + SDPO (full system)
"""
import json
import logging
import os
import sys
import time

SDPO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SDPO")
sys.path.insert(0, SDPO_PATH)

from data.utils.livecodebench import load_livecodebench

from agent.config import AgentConfig, build_code_prompt
from agent.model import AgentModel
from agent.verification import verify_solution
from agent.rag import RAGDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("eval.log"),
    ]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logger = logging.getLogger("eval")


def evaluate_model(agent_model: AgentModel, dataset, config: AgentConfig,
                   rag_db: RAGDatabase = None, label: str = "model",
                   num_samples: int = 1, temperature: float = 0.0) -> dict:
    """Evaluate a model on the test dataset.

    Args:
        num_samples: number of samples per problem (1=greedy, >1=avg@N)
        temperature: sampling temperature (0=greedy)

    Returns metrics dict.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATING: {label} (samples={num_samples}, temp={temperature})")
    logger.info(f"{'='*60}")

    total_score = 0.0
    total_acc = 0.0
    results = []
    num_correct = 0
    num_problems = min(len(dataset), config.max_problems) if config.max_problems else len(dataset)

    for idx in range(num_problems):
        example = dataset[idx]
        problem = example["problem"]
        tests_json = example["tests"]
        description = example.get("description", "")

        # Build prompt
        system_prompt = None
        if rag_db and rag_db.size > 0:
            chunks = rag_db.query(description or problem[:500], top_k=config.rag_top_k)
            if chunks:
                rag_context = "\n\n".join(f"[Knowledge {i+1}]: {c}" for i, c in enumerate(chunks))
                system_prompt = (
                    "You are a coding expert. Here is some relevant knowledge that may help:\n\n"
                    f"{rag_context}\n\n"
                    "Use this knowledge if relevant to the problem."
                )

        code_prompt = build_code_prompt(problem, tests_json)

        # Generate (possibly multiple samples)
        responses, _, _ = agent_model.generate(
            prompt=code_prompt,
            system_prompt=system_prompt,
            num_return_sequences=num_samples,
            temperature=temperature,
            max_new_tokens=config.max_new_tokens,
        )

        # Score all samples, take best (pass@N) or average (avg@N)
        best_score = 0.0
        best_acc = 0.0
        sample_scores = []
        for resp in responses:
            result = verify_solution(resp, tests_json, split="test")
            sample_scores.append(result["score"])
            if result["score"] > best_score:
                best_score = result["score"]
                best_acc = result["acc"]

        score = best_score
        acc = best_acc
        total_score += score
        total_acc += acc
        if score > 0.99:
            num_correct += 1

        results.append({
            "step": idx + 1,
            "score": score,
            "acc": acc,
            "sample_scores": sample_scores,
            "feedback": result["feedback"][:200] if result.get("feedback") else "passed",
        })

        if (idx + 1) % 5 == 0 or (idx + 1) == num_problems:
            logger.info(f"[{label}] {idx+1}/{num_problems} | "
                        f"Score: {score:.3f} | "
                        f"Running avg: {total_score/(idx+1):.3f} | "
                        f"Correct: {num_correct}/{idx+1}")

    avg_score = total_score / max(num_problems, 1)
    avg_acc = total_acc / max(num_problems, 1)
    pass_rate = num_correct / max(num_problems, 1)

    metrics = {
        "label": label,
        "num_problems": num_problems,
        "num_samples": num_samples,
        "avg_score": avg_score,
        "avg_accuracy": avg_acc,
        "pass_rate": pass_rate,
        "num_correct": num_correct,
        "results": results,
    }

    logger.info(f"\n--- {label} Results ---")
    logger.info(f"  Average Score:    {avg_score:.4f}")
    logger.info(f"  Average Accuracy: {avg_acc:.4f}")
    logger.info(f"  Pass Rate:        {pass_rate:.4f} ({num_correct}/{num_problems})")

    return metrics


def run_eval(trained_model_path: str = "checkpoints/final_model",
             rag_db_path: str = "checkpoints/rag_db.json",
             config: AgentConfig = None,
             only_base_and_full: bool = False):
    if config is None:
        config = AgentConfig()

    logger.info("Loading test dataset...")
    test_dataset = load_livecodebench("test")
    logger.info(f"Test dataset: {len(test_dataset)} problems")

    all_eval_results = {}

    # 1. Base model evaluation
    logger.info("\n\n=== Evaluation 1: BASE MODEL ===")
    base_model = AgentModel(
        model_name=config.model_name,
        lr=config.lr,
        ema_rate=config.ema_rate,
    )
    base_metrics = evaluate_model(base_model, test_dataset, config, rag_db=None, label="base_qwen")
    all_eval_results["base_qwen"] = base_metrics
    del base_model
    import torch
    torch.cuda.empty_cache()

    # 2-4: Load trained model
    logger.info(f"\nLoading trained model from {trained_model_path}...")
    trained_model = AgentModel(
        model_name=trained_model_path,
        lr=config.lr,
        ema_rate=config.ema_rate,
    )

    # Load RAG database if available
    rag_db = None
    if os.path.exists(rag_db_path):
        logger.info(f"Loading RAG database from {rag_db_path}...")
        rag_db = RAGDatabase(config.embedding_model)
        with open(rag_db_path, "r") as f:
            rag_data = json.load(f)
        for chunk in rag_data:
            rag_db.add(chunk)
        logger.info(f"RAG database loaded with {rag_db.size} chunks")

    if not only_base_and_full:
        # 2. Model + SDPO only
        logger.info("\n\n=== Evaluation 2: MODEL + SDPO ONLY ===")
        sdpo_metrics = evaluate_model(trained_model, test_dataset, config, rag_db=None, label="model+sdpo")
        all_eval_results["model+sdpo"] = sdpo_metrics

        # 3. Model + RAG only (base weights)
        if rag_db and rag_db.size > 0:
            logger.info("\n\n=== Evaluation 3: MODEL + RAG ONLY ===")
            base_for_rag = AgentModel(
                model_name=config.model_name,
                lr=config.lr,
                ema_rate=config.ema_rate,
            )
            rag_only_metrics = evaluate_model(base_for_rag, test_dataset, config, rag_db=rag_db, label="model+rag")
            all_eval_results["model+rag"] = rag_only_metrics
            del base_for_rag
            torch.cuda.empty_cache()
        else:
            logger.info("No RAG database available, skipping model+rag evaluation")
            all_eval_results["model+rag"] = {"label": "model+rag", "note": "no RAG data available"}

    # 4. Full system
    logger.info("\n\n=== Evaluation 4: MODEL + RAG + SDPO (Full System) ===")
    full_metrics = evaluate_model(trained_model, test_dataset, config, rag_db=rag_db, label="model+rag+sdpo")
    all_eval_results["model+rag+sdpo"] = full_metrics

    # Print final comparison
    logger.info("\n\n" + "=" * 70)
    logger.info("FINAL COMPARISON")
    logger.info("=" * 70)
    logger.info(f"{'Configuration':<25} {'Avg Score':>12} {'Pass Rate':>12} {'Correct':>10}")
    logger.info("-" * 70)
    for key in ["base_qwen", "model+sdpo", "model+rag", "model+rag+sdpo"]:
        m = all_eval_results.get(key, {})
        if "avg_score" in m:
            logger.info(f"{m['label']:<25} {m['avg_score']:>12.4f} {m['pass_rate']:>12.4f} "
                        f"{m['num_correct']:>5}/{m['num_problems']}")
        else:
            logger.info(f"{key:<25} {'N/A':>12} {'N/A':>12} {'N/A':>10}")
    logger.info("=" * 70)

    # Save results
    with open("eval_results.json", "w") as f:
        json.dump(all_eval_results, f, indent=2, default=str)
    logger.info("Evaluation results saved to eval_results.json")

    return all_eval_results


if __name__ == "__main__":
    run_eval()
