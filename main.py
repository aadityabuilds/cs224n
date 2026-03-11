"""Self-evolving agent main training loop.

Iterates over LiveCodeBench problems:
1. Generate code solution
2. Verify against test cases
3. Route: SDPO (procedural fix) / RAG (knowledge store) / Pass
4. Execute chosen action
"""
import json
import logging
import os
import sys
import time

# Make SDPO importable
SDPO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SDPO")
sys.path.insert(0, SDPO_PATH)

from data.utils.livecodebench import load_livecodebench
from data.format.prompts import CODE_PROMPT

from agent.config import AgentConfig
from agent.model import AgentModel
from agent.verification import verify_solution
from agent.router import route_decision
from agent.sdpo_update import sdpo_step
from agent.rag import RAGDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log"),
    ]
)
# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logger = logging.getLogger("main")


def build_system_prompt_with_rag(rag_db: RAGDatabase, problem_description: str, top_k: int = 3) -> str | None:
    """Query RAG and build system prompt with retrieved knowledge chunks."""
    if rag_db.size == 0:
        return None
    chunks = rag_db.query(problem_description, top_k=top_k)
    if not chunks:
        return None
    rag_context = "\n\n".join(f"[Knowledge {i+1}]: {c}" for i, c in enumerate(chunks))
    return (
        "You are a coding expert. Here is some relevant knowledge that may help:\n\n"
        f"{rag_context}\n\n"
        "Use this knowledge if relevant to the problem."
    )


def run_training_loop(config: AgentConfig, checkpoint_callback=None):
    logger.info("=" * 60)
    logger.info("SELF-EVOLVING AGENT - TRAINING LOOP")
    logger.info("=" * 60)
    logger.info(f"Config: {config}")

    # Load model
    agent_model = AgentModel(
        model_name=config.model_name,
        lr=config.lr,
        ema_rate=config.ema_rate,
    )

    # Load dataset
    logger.info("Loading LiveCodeBench dataset...")
    dataset = load_livecodebench(config.dataset_split)
    logger.info(f"Dataset loaded: {len(dataset)} problems")

    # Initialize RAG
    rag_db = RAGDatabase(config.embedding_model)

    # Metrics tracking
    all_metrics = []
    action_counts = {"sdpo": 0, "rag": 0, "pass": 0}
    total_score = 0.0

    num_problems = min(len(dataset), config.max_problems) if config.max_problems else len(dataset)

    for idx in range(num_problems):
        step_start = time.time()
        example = dataset[idx]
        problem = example["problem"]
        tests_json = example["tests"]
        description = example.get("description", "")

        logger.info(f"\n{'='*60}")
        logger.info(f"STEP {idx+1}/{num_problems}")
        logger.info(f"Problem: {description[:150]}...")
        logger.info(f"{'='*60}")

        # Step 1: Build prompt with RAG context
        rag_system = build_system_prompt_with_rag(rag_db, description or problem[:500], config.rag_top_k)
        code_prompt = CODE_PROMPT.format(problem=problem)

        # Step 2: Generate code solution
        logger.info("Generating code solution...")
        responses, _, _ = agent_model.generate(
            prompt=code_prompt,
            system_prompt=rag_system,
            num_return_sequences=1,
            temperature=0.0,
            max_new_tokens=config.max_new_tokens,
        )
        response = responses[0]
        logger.info(f"Generated {len(response)} chars of code")

        # Step 3: Verify solution
        logger.info("Verifying solution...")
        result = verify_solution(response, tests_json)
        score = result["score"]
        accuracy = result["acc"]
        feedback = result["feedback"]
        total_score += score

        logger.info(f"Score: {score:.3f}, Accuracy: {accuracy:.3f}")
        logger.info(f"Feedback: {feedback[:300] if feedback else 'All tests passed!'}")

        # Step 4: Routing decision (rule-based)
        action, payload = route_decision(score, accuracy, feedback)
        logger.info(f"Routing decision: {action}" + (f" (payload: {payload[:100]}...)" if payload else ""))
        action_counts[action] += 1

        # Step 5: Execute action
        step_metrics = {
            "step": idx + 1,
            "score": score,
            "accuracy": accuracy,
            "action": action,
            "rag_db_size": rag_db.size,
            "running_avg_score": total_score / (idx + 1),
        }

        if action == "sdpo":
            logger.info("Executing SDPO update...")
            sdpo_metrics = sdpo_step(
                agent_model=agent_model,
                prompt=code_prompt,
                tests_json=tests_json,
                verify_fn=verify_solution,
                config=config,
                system_prompt=rag_system,
            )
            step_metrics["sdpo"] = sdpo_metrics
        elif action == "rag":
            logger.info(f"Adding to RAG database: {payload[:100] if payload else 'empty'}...")
            if payload:
                rag_db.add(payload)
            step_metrics["rag_chunk"] = payload
        else:
            logger.info("Passing (no action needed)")

        elapsed = time.time() - step_start
        step_metrics["elapsed_seconds"] = elapsed
        all_metrics.append(step_metrics)

        logger.info(f"Step {idx+1} complete in {elapsed:.1f}s | "
                     f"Running avg score: {total_score/(idx+1):.3f} | "
                     f"Actions so far: {action_counts}")

        # Save metrics periodically
        if (idx + 1) % 10 == 0 or (idx + 1) == num_problems:
            with open("training_metrics.json", "w") as f:
                json.dump(all_metrics, f, indent=2)
            logger.info(f"Metrics saved ({len(all_metrics)} steps)")

        # Checkpoint every 50 problems (or at end)
        if (idx + 1) % 50 == 0 or (idx + 1) == num_problems:
            ckpt_name = f"checkpoint_step{idx+1}"
            ckpt_path = f"checkpoints/{ckpt_name}"
            agent_model.save_checkpoint(ckpt_path)
            # Save RAG db alongside checkpoint
            rag_path = f"checkpoints/{ckpt_name}_rag.json"
            with open(rag_path, "w") as f:
                json.dump(rag_db.texts, f, indent=2)
            logger.info(f"Checkpoint saved: {ckpt_path} (RAG: {rag_db.size} chunks)")
            # Call checkpoint callback if provided
            if checkpoint_callback:
                checkpoint_callback(ckpt_path, rag_path, idx + 1, all_metrics)

    # Save final model (symlink to last checkpoint for compatibility)
    final_path = "checkpoints/final_model"
    if not os.path.exists(final_path):
        agent_model.save_checkpoint(final_path)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Total problems: {num_problems}")
    logger.info(f"Average score: {total_score / num_problems:.3f}")
    logger.info(f"Action distribution: {action_counts}")
    logger.info(f"RAG database size: {rag_db.size}")
    logger.info("=" * 60)

    return all_metrics, agent_model, rag_db


if __name__ == "__main__":
    config = AgentConfig()
    run_training_loop(config)
