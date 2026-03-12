"""7B SDPO training loop — memory-optimized, strict 100% accuracy condition.

Key differences from main.py:
- Uses AgentModel7B (no reference model, saves ~14GB)
- No baseline comparison on every problem (saves inference time + memory)
- Update condition: rollout must score >= 0.999 (100% test accuracy)
- Smaller batch sizes, fewer rollouts, shorter sequences
"""
import json
import logging
import os
import sys
import time
from dataclasses import dataclass

SDPO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SDPO")
sys.path.insert(0, SDPO_PATH)

from data.utils.livecodebench import load_livecodebench

from agent.config import SelfDistillationConfig, build_code_prompt
from agent.model_7b import AgentModel7B
from agent.verification import verify_solution
from agent.router import llm_route_decision
from agent.rag import RAGDatabase
from agent.sdpo_update_7b import sdpo_batch_step_7b

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training_7b.log"),
    ]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logger = logging.getLogger("main_7b")


@dataclass
class AgentConfig7B:
    """7B configuration — memory-optimized for A100-80GB."""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    lr: float = 2e-6
    ema_rate: float = 0.005
    num_rollouts: int = 2
    max_sequential_attempts: int = 4
    temperature: float = 0.7
    max_new_tokens: int = 1536
    rag_top_k: int = 3
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_grad_norm: float = 1.0
    dataset_split: str = "train"
    max_problems: int = None
    sdpo_batch_size: int = 2
    max_reprompt_tokens: int = 3072
    checkpoint_every: int = 50


def build_system_prompt_with_rag(rag_db: RAGDatabase, problem_description: str,
                                  top_k: int = 3) -> tuple[str | None, list[int]]:
    if rag_db.size == 0:
        return None, []
    results = rag_db.query_with_ids(problem_description, top_k=top_k)
    if not results:
        return None, []
    retrieved_ids = [r[0] for r in results]
    rag_context = "\n\n".join(
        f"[Knowledge {i+1} (id={r[0]}, score={r[2]:.3f})]: {r[1]}"
        for i, r in enumerate(results)
    )
    prompt = (
        "You are a coding expert. Below are some knowledge snippets retrieved "
        "from past problem-solving sessions. They may or may not be relevant to "
        "the current problem — only use them if they directly apply.\n\n"
        f"{rag_context}\n\n"
        "If none of the above is relevant, ignore it and solve the problem from scratch."
    )
    return prompt, retrieved_ids


def run_training_loop_7b(config: AgentConfig7B,
                          sdpo_config: SelfDistillationConfig = None,
                          checkpoint_callback=None):
    logger.info("=" * 80)
    logger.info("  7B SDPO TRAINING (memory-optimized, 100% accuracy condition)")
    logger.info("=" * 80)
    logger.info(f"Config: {config}")

    if sdpo_config is None:
        sdpo_config = SelfDistillationConfig(reference_kl_beta=0.0)
    logger.info(f"SDPO config: {sdpo_config}")

    import torch
    _log_mem = lambda tag: logger.info(
        f"  [MEM {tag}] {torch.cuda.memory_allocated()/1e9:.1f}GB alloc, "
        f"{torch.cuda.memory_reserved()/1e9:.1f}GB reserved"
    ) if torch.cuda.is_available() else None

    # Load model
    agent_model = AgentModel7B(
        model_name=config.model_name,
        lr=config.lr,
        ema_rate=config.ema_rate,
    )
    _log_mem("model_loaded")

    # Load dataset
    logger.info("Loading LiveCodeBench dataset...")
    dataset = load_livecodebench(config.dataset_split)
    logger.info(f"Dataset loaded: {len(dataset)} problems")

    # Initialize RAG
    rag_db = RAGDatabase(config.embedding_model)

    # Tracking
    all_metrics = []
    action_counts = {"sdpo": 0, "rag": 0, "pass": 0}
    sdpo_update_count = 0
    total_accuracy = 0.0
    total_correct = 0

    sdpo_batch = []
    num_problems = min(len(dataset), config.max_problems) if config.max_problems else len(dataset)

    for idx in range(num_problems):
        step_start = time.time()
        example = dataset[idx]
        problem = example["problem"]
        tests_json = example["tests"]
        description = example.get("description", "")

        logger.info(f"\n{'='*80}")
        logger.info(f"{'='*80}")
        logger.info(f"  SAMPLE {idx+1}/{num_problems}")
        logger.info(f"{'='*80}")
        logger.info(f"{'='*80}")

        logger.info(f"  Problem: {description[:200]}...")

        # -- Test cases preview --
        try:
            tc = json.loads(tests_json)
            num_tests = len(tc.get("inputs", []))
            logger.info(f"  Test cases: {num_tests} tests, type={tc.get('testtype', 'unknown')}")
            if tc.get("inputs") and len(tc["inputs"]) > 0:
                logger.info(f"  ┌── Sample test input ──")
                logger.info(f"  │ {str(tc['inputs'][0])[:200]}")
                logger.info(f"  └── Sample test output ──")
                if tc.get("outputs") and len(tc["outputs"]) > 0:
                    logger.info(f"  │ {str(tc['outputs'][0])[:200]}")
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

        # ---- Generate ----
        rag_system, retrieved_ids = build_system_prompt_with_rag(
            rag_db, description or problem[:500], config.rag_top_k
        )
        if retrieved_ids:
            logger.info(f"  RAG context: chunk IDs {retrieved_ids}")
        code_prompt = build_code_prompt(problem, tests_json)

        responses, _, _ = agent_model.generate(
            prompt=code_prompt,
            system_prompt=rag_system,
            num_return_sequences=1,
            temperature=0.0,
            max_new_tokens=config.max_new_tokens,
        )
        response = responses[0]

        logger.info(f"  ╔══════════════════════════════════════════════════════════════╗")
        logger.info(f"  ║  STUDENT MODEL OUTPUT (greedy)                              ║")
        logger.info(f"  ╠══════════════════════════════════════════════════════════════╣")
        for line in response[:2000].split('\n'):
            logger.info(f"  ║  {line}")
        if len(response) > 2000:
            logger.info(f"  ║  ... ({len(response)} chars total)")
        logger.info(f"  ╚══════════════════════════════════════════════════════════════╝")

        # ---- Verify ----
        result = verify_solution(response, tests_json)
        score = result["score"]
        accuracy = result["acc"]
        feedback = result["feedback"]

        total_accuracy += score
        if score >= 0.999:
            total_correct += 1

        logger.info(f"  ┌── VERIFICATION ─────────────────────────────────────────────┐")
        logger.info(f"  │  Score: {score:.3f}  Accuracy: {accuracy:.3f}               │")
        if feedback:
            for line in feedback[:400].split('\n'):
                logger.info(f"  │  {line}")
        else:
            logger.info(f"  │  All tests passed!")
        logger.info(f"  └─────────────────────────────────────────────────────────────┘")

        # ---- Route ----
        action, payload = llm_route_decision(
            agent_model=agent_model,
            problem=description or problem[:500],
            score=score,
            accuracy=accuracy,
            feedback=feedback or "",
        )

        logger.info(f"  ┌── ROUTING: <{action}> ─────────────────────────────────────┐")
        logger.info(f"  └─────────────────────────────────────────────────────────────┘")
        action_counts[action] += 1

        n = idx + 1
        step_metrics = {
            "step": n,
            "score": score,
            "accuracy": accuracy,
            "action": action,
            "feedback": (feedback or "")[:500],
            "rag_db_size": rag_db.size,
            "student_accuracy_avg": total_accuracy / n,
            "student_correct_avg": total_correct / n,
        }

        if action == "sdpo":
            sdpo_batch.append({
                'prompt': code_prompt,
                'tests_json': tests_json,
                'system_prompt': rag_system,
                'initial_feedback': feedback,
                'initial_score': score,
                'initial_accuracy': accuracy,
                'problem_idx': idx,
            })
            logger.info(f"  <sdpo> queued ({len(sdpo_batch)}/{config.sdpo_batch_size})")

            if len(sdpo_batch) >= config.sdpo_batch_size:
                logger.info(f"\n{'~'*80}")
                logger.info(f"  SDPO BATCH UPDATE #{sdpo_update_count+1}  "
                            f"({len(sdpo_batch)} problems)")
                logger.info(f"{'~'*80}")

                _log_mem("before_sdpo")
                sdpo_metrics = sdpo_batch_step_7b(
                    agent_model=agent_model,
                    batch_items=sdpo_batch,
                    verify_fn=verify_solution,
                    config=config,
                    sdpo_config=sdpo_config,
                )
                _log_mem("after_sdpo")

                step_metrics["sdpo_batch"] = sdpo_metrics
                sdpo_update_count += 1
                sdpo_batch = []
                logger.info(f"{'~'*80}")
                logger.info(f"  SDPO BATCH UPDATE #{sdpo_update_count} COMPLETE")
                logger.info(f"{'~'*80}\n")

        elif action == "rag":
            if payload:
                chunk_id = rag_db.add(payload)
                step_metrics["rag_added_id"] = chunk_id
                logger.info(f"  <rag> stored chunk id={chunk_id}, db_size={rag_db.size}")
            step_metrics["rag_chunk"] = payload
        else:
            logger.info(f"  <pass> correct, no action")

        elapsed = time.time() - step_start
        step_metrics["elapsed_seconds"] = elapsed
        all_metrics.append(step_metrics)

        logger.info(f"  ┌── RUNNING TOTALS ({n} problems) ────────────────────────────┐")
        logger.info(f"  │  Acc={total_accuracy/n:.3f}  "
                     f"Solve={total_correct}/{n} ({total_correct/n:.3f})  "
                     f"Time={elapsed:.1f}s  │")
        logger.info(f"  │  Actions: {action_counts}  SDPO updates: {sdpo_update_count}  │")
        logger.info(f"  └─────────────────────────────────────────────────────────────┘")

        if n % 10 == 0 or n == num_problems:
            with open("training_metrics.json", "w") as f:
                json.dump(all_metrics, f, indent=2)

        if n % config.checkpoint_every == 0 or n == num_problems:
            ckpt_name = f"checkpoint_step{n}"
            ckpt_path = f"checkpoints/{ckpt_name}"
            os.makedirs(ckpt_path, exist_ok=True)
            agent_model.save_checkpoint(ckpt_path)
            rag_path = f"checkpoints/{ckpt_name}_rag.json"
            with open(rag_path, "w") as f:
                json.dump(rag_db.texts, f, indent=2)
            rag_map_path = f"checkpoints/{ckpt_name}_rag_id_map.json"
            with open(rag_map_path, "w") as f:
                json.dump({str(k): v for k, v in rag_db.id_to_text.items()}, f, indent=2)
            logger.info(f"Checkpoint saved: {ckpt_path} (RAG: {rag_db.size} chunks)")
            if checkpoint_callback:
                checkpoint_callback(ckpt_path, rag_path, n, all_metrics)

    # Flush remaining
    if sdpo_batch:
        logger.info(f"\n--- FINAL SDPO BATCH ({len(sdpo_batch)} problems) ---")
        sdpo_metrics = sdpo_batch_step_7b(
            agent_model=agent_model, batch_items=sdpo_batch,
            verify_fn=verify_solution, config=config, sdpo_config=sdpo_config,
        )
        sdpo_update_count += 1

    final_path = "checkpoints/final_model"
    os.makedirs(final_path, exist_ok=True)
    agent_model.save_checkpoint(final_path)

    num = max(num_problems, 1)
    logger.info("\n" + "=" * 80)
    logger.info("  7B TRAINING COMPLETE")
    logger.info(f"  Problems: {num_problems}  Acc: {total_accuracy/num:.3f}  "
                f"Solve: {total_correct}/{num_problems}")
    logger.info(f"  Actions: {action_counts}  SDPO updates: {sdpo_update_count}")
    logger.info(f"  RAG: {rag_db.size} chunks")
    logger.info("=" * 80)

    return all_metrics, agent_model, rag_db


if __name__ == "__main__":
    config = AgentConfig7B()
    run_training_loop_7b(config)
