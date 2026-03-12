"""Self-evolving agent main training loop.

Architecture:
1. For each problem: generate code → verify → LLM routing decision
2. Collect problems by action (sdpo/rag/pass)
3. When SDPO batch is full: execute batched SDPO update (rollouts + distillation)
4. RAG items: store feedback in vector database immediately
5. Pass items: skip

The SDPO update generates G rollouts per problem, selects successful ones as
demonstrations, builds reprompted teacher inputs, and distills the teacher's
informed predictions back into the student via top-k KL divergence.
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

from agent.config import AgentConfig, SelfDistillationConfig, build_code_prompt
from agent.model import AgentModel
from agent.verification import verify_solution
from agent.router import llm_route_decision
from agent.sdpo_update import sdpo_batch_step
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


def build_system_prompt_with_rag(rag_db: RAGDatabase, problem_description: str,
                                  top_k: int = 3) -> tuple[str | None, list[int]]:
    """Query RAG and build system prompt with retrieved knowledge chunks."""
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
        "You are a coding expert. Here is some relevant knowledge that may help:\n\n"
        f"{rag_context}\n\n"
        "Use this knowledge if relevant to the problem."
    )
    return prompt, retrieved_ids


def run_training_loop(config: AgentConfig, checkpoint_callback=None):
    logger.info("=" * 60)
    logger.info("SELF-EVOLVING AGENT - TRAINING LOOP (v2: proper SDPO)")
    logger.info("=" * 60)
    logger.info(f"Config: {config}")

    sdpo_config = SelfDistillationConfig()
    logger.info(f"SDPO config: {sdpo_config}")

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

    # Tracking
    all_metrics = []
    action_counts = {"sdpo": 0, "rag": 0, "pass": 0}
    sdpo_update_count = 0
    total_score = 0.0

    # SDPO batch accumulator
    sdpo_batch = []

    num_problems = min(len(dataset), config.max_problems) if config.max_problems else len(dataset)

    for idx in range(num_problems):
        step_start = time.time()
        example = dataset[idx]
        problem = example["problem"]
        tests_json = example["tests"]
        description = example.get("description", "")

        logger.info(f"\n{'='*60}")
        logger.info(f"[Sample {idx+1}/{num_problems}] Problem: {description[:120]}...")
        logger.info(f"{'='*60}")

        # ---- Phase 1: Generate initial response ----
        rag_system, retrieved_ids = build_system_prompt_with_rag(
            rag_db, description or problem[:500], config.rag_top_k
        )
        if retrieved_ids:
            logger.info(f"[Sample {idx+1}] RAG context: retrieved chunk IDs {retrieved_ids}")
        code_prompt = build_code_prompt(problem, tests_json)

        responses, _, _ = agent_model.generate(
            prompt=code_prompt,
            system_prompt=rag_system,
            num_return_sequences=1,
            temperature=0.0,
            max_new_tokens=config.max_new_tokens,
        )
        response = responses[0]
        logger.info(f"[Sample {idx+1}] RESPONSE  | {response[:1500]}")

        # ---- Phase 2: Verify ----
        result = verify_solution(response, tests_json)
        score = result["score"]
        accuracy = result["acc"]
        feedback = result["feedback"]
        total_score += score

        logger.info(f"[Sample {idx+1}] VERIFIER  | score={score:.3f}  accuracy={accuracy:.3f}  "
                     f"response_len={len(response)} chars")
        if feedback:
            logger.info(f"[Sample {idx+1}] FEEDBACK  | {feedback[:300]}")
        else:
            logger.info(f"[Sample {idx+1}] FEEDBACK  | All tests passed")

        # ---- Phase 3: LLM routing decision ----
        action, payload = llm_route_decision(
            agent_model=agent_model,
            problem=description or problem[:500],
            score=score,
            accuracy=accuracy,
            feedback=feedback or "",
        )
        logger.info(f"[Sample {idx+1}] TOOL CALL | <{action}>")
        action_counts[action] += 1

        # ---- Phase 4: Execute chosen tool call ----
        step_metrics = {
            "step": idx + 1,
            "score": score,
            "accuracy": accuracy,
            "action": action,
            "feedback": (feedback or "")[:500],
            "rag_db_size": rag_db.size,
            "rag_retrieved_ids": retrieved_ids,
            "running_avg_score": total_score / (idx + 1),
        }

        if action == "sdpo":
            sdpo_batch.append({
                'prompt': code_prompt,
                'tests_json': tests_json,
                'system_prompt': rag_system,
                'initial_feedback': feedback,
                'problem_idx': idx,
            })
            logger.info(f"[Sample {idx+1}] <sdpo>    | queued for batch "
                        f"({len(sdpo_batch)}/{config.sdpo_batch_size})")

            # Execute SDPO update when batch is full
            if len(sdpo_batch) >= config.sdpo_batch_size:
                logger.info(f"\n{'~'*60}")
                logger.info(f"SDPO BATCH UPDATE #{sdpo_update_count+1}  "
                            f"({len(sdpo_batch)} problems)")
                logger.info(f"{'~'*60}")
                sdpo_metrics = sdpo_batch_step(
                    agent_model=agent_model,
                    batch_items=sdpo_batch,
                    verify_fn=verify_solution,
                    config=config,
                    sdpo_config=sdpo_config,
                )
                step_metrics["sdpo_batch"] = sdpo_metrics
                sdpo_update_count += 1
                sdpo_batch = []  # Reset batch
                logger.info(f"SDPO BATCH UPDATE #{sdpo_update_count} COMPLETE\n")

        elif action == "rag":
            if payload:
                chunk_id = rag_db.add(payload)
                step_metrics["rag_added_id"] = chunk_id
                logger.info(f"[Sample {idx+1}] <rag>     | stored chunk id={chunk_id}, "
                            f"db_size={rag_db.size}")
                logger.info(f"[Sample {idx+1}] <rag>     | content: {payload[:200]}")
            else:
                logger.info(f"[Sample {idx+1}] <rag>     | no payload to store")
            step_metrics["rag_chunk"] = payload

        else:  # pass
            logger.info(f"[Sample {idx+1}] <pass>    | solution correct, no action")

        elapsed = time.time() - step_start
        step_metrics["elapsed_seconds"] = elapsed
        all_metrics.append(step_metrics)

        logger.info(f"[Sample {idx+1}] SUMMARY   | time={elapsed:.1f}s  "
                     f"running_avg={total_score/(idx+1):.3f}  "
                     f"actions={action_counts}  "
                     f"sdpo_updates={sdpo_update_count}  "
                     f"sdpo_pending={len(sdpo_batch)}")

        # Save metrics periodically
        if (idx + 1) % 10 == 0 or (idx + 1) == num_problems:
            with open("training_metrics.json", "w") as f:
                json.dump(all_metrics, f, indent=2)

        # Checkpoint
        if (idx + 1) % config.checkpoint_every == 0 or (idx + 1) == num_problems:
            ckpt_name = f"checkpoint_step{idx+1}"
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
                checkpoint_callback(ckpt_path, rag_path, idx + 1, all_metrics)

    # Flush any remaining SDPO batch
    if sdpo_batch:
        logger.info(f"\n--- FINAL SDPO BATCH UPDATE (remaining {len(sdpo_batch)} problems) ---")
        sdpo_metrics = sdpo_batch_step(
            agent_model=agent_model,
            batch_items=sdpo_batch,
            verify_fn=verify_solution,
            config=config,
            sdpo_config=sdpo_config,
        )
        sdpo_update_count += 1
        logger.info(f"--- FINAL SDPO BATCH UPDATE COMPLETE (total: {sdpo_update_count}) ---\n")

    # Save final model
    final_path = "checkpoints/final_model"
    os.makedirs(final_path, exist_ok=True)
    agent_model.save_checkpoint(final_path)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Total problems: {num_problems}")
    logger.info(f"Average score: {total_score / max(num_problems, 1):.3f}")
    logger.info(f"Action distribution: {action_counts}")
    logger.info(f"SDPO gradient updates: {sdpo_update_count}")
    logger.info(f"RAG database size: {rag_db.size}")
    logger.info("=" * 60)

    return all_metrics, agent_model, rag_db


if __name__ == "__main__":
    config = AgentConfig()
    run_training_loop(config)
