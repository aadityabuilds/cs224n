"""Self-evolving agent main training loop.

Architecture:
1. For each problem: generate code → verify → LLM routing decision
2. Collect problems by action (sdpo/rag/pass)
3. When SDPO batch is full: execute batched SDPO update (rollouts + distillation)
4. RAG items: store feedback in vector database immediately
5. Pass items: skip

Dual metrics tracked:
- Student: accuracy (fraction of tests passed) + solve rate (fully correct)
- Baseline: same metrics from frozen reference model for comparison
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
        "You are a coding expert. Below are some knowledge snippets retrieved "
        "from past problem-solving sessions. They may or may not be relevant to "
        "the current problem — only use them if they directly apply.\n\n"
        f"{rag_context}\n\n"
        "If none of the above is relevant, ignore it and solve the problem from scratch."
    )
    return prompt, retrieved_ids


def run_training_loop(config: AgentConfig, sdpo_config: SelfDistillationConfig = None,
                      checkpoint_callback=None):
    logger.info("=" * 60)
    logger.info("SELF-EVOLVING AGENT - TRAINING LOOP (v3: anti-forgetting)")
    logger.info("=" * 60)
    logger.info(f"Config: {config}")

    if sdpo_config is None:
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

    # Tracking — dual metrics for student and baseline
    all_metrics = []
    action_counts = {"sdpo": 0, "rag": 0, "pass": 0}
    sdpo_update_count = 0

    total_accuracy = 0.0
    total_correct = 0
    total_baseline_accuracy = 0.0
    total_baseline_correct = 0

    # SDPO batch accumulator
    sdpo_batch = []

    # Global running stats for sequential rollout effectiveness
    global_seq_stats = {
        "total_rollouts": 0,
        "beat_baseline": 0,
        "perfect": 0,
        "improved_by_retry": 0,
        "total_best_score_sum": 0.0,
    }

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

        # -- Problem description --
        logger.info(f"  Problem: {description[:200]}...")
        logger.info(f"  Problem text (first 500 chars):")
        logger.info(f"  {'─'*70}")
        for line in problem[:500].split('\n'):
            logger.info(f"  │ {line}")
        if len(problem) > 500:
            logger.info(f"  │ ... ({len(problem)} chars total)")
        logger.info(f"  {'─'*70}")

        # -- Test cases preview --
        try:
            tc = json.loads(tests_json)
            num_tests = len(tc.get("inputs", []))
            logger.info(f"  Test cases: {num_tests} tests, type={tc.get('testtype', 'unknown')}")
            if tc.get("inputs") and len(tc["inputs"]) > 0:
                logger.info(f"  ┌── Sample test input ──")
                sample_in = str(tc["inputs"][0])[:200]
                logger.info(f"  │ {sample_in}")
                logger.info(f"  └── Sample test output ──")
                if tc.get("outputs") and len(tc["outputs"]) > 0:
                    sample_out = str(tc["outputs"][0])[:200]
                    logger.info(f"  │ {sample_out}")
        except (json.JSONDecodeError, AttributeError, TypeError):
            logger.info(f"  Test cases: (could not parse preview)")

        # ---- Phase 1a: Student greedy attempt ----
        rag_system, retrieved_ids = build_system_prompt_with_rag(
            rag_db, description or problem[:500], config.rag_top_k
        )
        if retrieved_ids:
            logger.info(f"  RAG context: retrieved chunk IDs {retrieved_ids}")
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
        logger.info(f"  ║  STUDENT MODEL OUTPUT (greedy, temp=0)                      ║")
        logger.info(f"  ╠══════════════════════════════════════════════════════════════╣")
        for line in response[:2000].split('\n'):
            logger.info(f"  ║  {line}")
        if len(response) > 2000:
            logger.info(f"  ║  ... ({len(response)} chars total, truncated)")
        logger.info(f"  ╚══════════════════════════════════════════════════════════════╝")

        # ---- Phase 1a-verify: Verify student ----
        result = verify_solution(response, tests_json)
        score = result["score"]
        accuracy = result["acc"]
        feedback = result["feedback"]

        total_accuracy += score
        if score >= 0.999:
            total_correct += 1

        logger.info(f"  ┌── VERIFICATION RESULT ──────────────────────────────────────┐")
        logger.info(f"  │  Score:    {score:.3f}                                      │")
        logger.info(f"  │  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}% tests passed) │")
        logger.info(f"  │  Length:   {len(response)} chars                             │")
        logger.info(f"  ├── FEEDBACK ──────────────────────────────────────────────────┤")
        if feedback:
            for line in feedback[:500].split('\n'):
                logger.info(f"  │  {line}")
        else:
            logger.info(f"  │  All tests passed!")
        logger.info(f"  └─────────────────────────────────────────────────────────────┘")

        # ---- Phase 1b: Baseline comparison (frozen reference model) ----
        baseline_response = agent_model.generate_baseline(
            prompt=code_prompt, system_prompt=rag_system,
            max_new_tokens=config.max_new_tokens,
        )
        baseline_result = verify_solution(baseline_response, tests_json)
        baseline_score = baseline_result["score"]
        baseline_correct = 1 if baseline_score >= 0.999 else 0

        total_baseline_accuracy += baseline_score
        total_baseline_correct += baseline_correct

        delta = score - baseline_score
        delta_symbol = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
        logger.info(f"  ┌── BASELINE vs STUDENT ─────────────────────────────────────┐")
        logger.info(f"  │  Baseline (frozen):  {baseline_score:.3f}                   │")
        logger.info(f"  │  Student (current):  {score:.3f}                            │")
        logger.info(f"  │  Delta:              {delta:+.3f} {delta_symbol}             │")
        logger.info(f"  └─────────────────────────────────────────────────────────────┘")

        # ---- Phase 2: LLM routing decision ----
        action, payload = llm_route_decision(
            agent_model=agent_model,
            problem=description or problem[:500],
            score=score,
            accuracy=accuracy,
            feedback=feedback or "",
        )

        logger.info(f"  ┌── ROUTING DECISION ────────────────────────────────────────┐")
        logger.info(f"  │  Action: <{action}>                                        │")
        if action == "sdpo":
            logger.info(f"  │  Reason: needs gradient update via self-distillation       │")
        elif action == "rag":
            logger.info(f"  │  Reason: partial success, storing feedback as knowledge    │")
        else:
            logger.info(f"  │  Reason: solution correct, no action needed                │")
        logger.info(f"  └─────────────────────────────────────────────────────────────┘")
        action_counts[action] += 1

        # ---- Phase 3: Execute chosen tool call ----
        n = idx + 1
        step_metrics = {
            "step": n,
            "score": score,
            "accuracy": accuracy,
            "action": action,
            "feedback": (feedback or "")[:500],
            "rag_db_size": rag_db.size,
            "rag_retrieved_ids": retrieved_ids,
            "student_accuracy_avg": total_accuracy / n,
            "student_correct_avg": total_correct / n,
            "baseline_accuracy_avg": total_baseline_accuracy / n,
            "baseline_correct_avg": total_baseline_correct / n,
            "baseline_score": baseline_score,
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
            logger.info(f"  <sdpo> queued for batch ({len(sdpo_batch)}/{config.sdpo_batch_size})")

            # Execute SDPO update when batch is full
            if len(sdpo_batch) >= config.sdpo_batch_size:
                logger.info(f"\n{'~'*80}")
                logger.info(f"{'~'*80}")
                logger.info(f"  SDPO BATCH UPDATE #{sdpo_update_count+1}  "
                            f"({len(sdpo_batch)} problems)")
                logger.info(f"{'~'*80}")
                logger.info(f"{'~'*80}")
                sdpo_metrics = sdpo_batch_step(
                    agent_model=agent_model,
                    batch_items=sdpo_batch,
                    verify_fn=verify_solution,
                    config=config,
                    sdpo_config=sdpo_config,
                )
                step_metrics["sdpo_batch"] = sdpo_metrics
                sdpo_update_count += 1
                sdpo_batch = []

                # Accumulate global sequential stats
                ss = sdpo_metrics.get("sequential_stats", {})
                global_seq_stats["total_rollouts"] += ss.get("total_rollouts", 0)
                global_seq_stats["beat_baseline"] += ss.get("beat_baseline", 0)
                global_seq_stats["perfect"] += ss.get("perfect", 0)
                global_seq_stats["improved_by_retry"] += ss.get("improved_by_retry", 0)
                batch_count = ss.get("total_rollouts", 0)
                global_seq_stats["total_best_score_sum"] += ss.get("avg_best_score", 0.0) * batch_count

                logger.info(f"{'~'*80}")
                logger.info(f"  SDPO BATCH UPDATE #{sdpo_update_count} COMPLETE")
                logger.info(f"{'~'*80}\n")

        elif action == "rag":
            if payload:
                chunk_id = rag_db.add(payload)
                step_metrics["rag_added_id"] = chunk_id
                logger.info(f"  <rag> stored chunk id={chunk_id}, db_size={rag_db.size}")
                logger.info(f"  <rag> content: {payload[:300]}")
            else:
                logger.info(f"  <rag> no payload to store")
            step_metrics["rag_chunk"] = payload

        else:  # pass
            logger.info(f"  <pass> solution correct, no action")

        elapsed = time.time() - step_start
        step_metrics["elapsed_seconds"] = elapsed
        all_metrics.append(step_metrics)

        gsr = global_seq_stats["total_rollouts"]
        gsb = global_seq_stats["beat_baseline"]
        gsp = global_seq_stats["perfect"]
        gsi = global_seq_stats["improved_by_retry"]
        gs_avg = (global_seq_stats["total_best_score_sum"] / max(gsr, 1)) if gsr > 0 else 0.0

        logger.info(f"  ┌── RUNNING TOTALS (after {n} problems) ──────────────────────┐")
        logger.info(f"  │  Student  acc={total_accuracy/n:.3f}  "
                     f"solve={total_correct}/{n} ({total_correct/n:.3f})    │")
        logger.info(f"  │  Baseline acc={total_baseline_accuracy/n:.3f}  "
                     f"solve={total_baseline_correct}/{n} ({total_baseline_correct/n:.3f})    │")
        logger.info(f"  │  Delta    acc={total_accuracy/n - total_baseline_accuracy/n:+.3f}  "
                     f"solve={total_correct - total_baseline_correct:+d}                      │")
        logger.info(f"  │  Actions: {action_counts}  SDPO updates: {sdpo_update_count}  │")
        if gsr > 0:
            logger.info(f"  ├── SEQUENTIAL ROLLOUT EFFECTIVENESS ─────────────────────────┤")
            logger.info(f"  │  Total rollouts: {gsr}  "
                         f"Beat baseline: {gsb}/{gsr} ({100*gsb/gsr:.0f}%)  │")
            logger.info(f"  │  Perfect (1.0): {gsp}/{gsr} ({100*gsp/gsr:.0f}%)  "
                         f"Improved by retry: {gsi}/{gsr} ({100*gsi/gsr:.0f}%)  │")
            logger.info(f"  │  Avg best score across rollouts: {gs_avg:.3f}              │")
        logger.info(f"  │  Time: {elapsed:.1f}s                                        │")
        logger.info(f"  └─────────────────────────────────────────────────────────────┘")

        # Save metrics periodically
        if n % 10 == 0 or n == num_problems:
            with open("training_metrics.json", "w") as f:
                json.dump(all_metrics, f, indent=2)

        # Checkpoint
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
        ss = sdpo_metrics.get("sequential_stats", {})
        global_seq_stats["total_rollouts"] += ss.get("total_rollouts", 0)
        global_seq_stats["beat_baseline"] += ss.get("beat_baseline", 0)
        global_seq_stats["perfect"] += ss.get("perfect", 0)
        global_seq_stats["improved_by_retry"] += ss.get("improved_by_retry", 0)
        batch_count = ss.get("total_rollouts", 0)
        global_seq_stats["total_best_score_sum"] += ss.get("avg_best_score", 0.0) * batch_count
        logger.info(f"--- FINAL SDPO BATCH UPDATE COMPLETE (total: {sdpo_update_count}) ---\n")

    # Save final model
    final_path = "checkpoints/final_model"
    os.makedirs(final_path, exist_ok=True)
    agent_model.save_checkpoint(final_path)

    # Final summary
    num = max(num_problems, 1)
    logger.info("\n" + "=" * 80)
    logger.info("=" * 80)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  Total problems: {num_problems}")
    logger.info(f"  Student  - Accuracy: {total_accuracy/num:.3f}  "
                f"Solve rate: {total_correct}/{num_problems}")
    logger.info(f"  Baseline - Accuracy: {total_baseline_accuracy/num:.3f}  "
                f"Solve rate: {total_baseline_correct}/{num_problems}")
    logger.info(f"  Action distribution: {action_counts}")
    logger.info(f"  SDPO gradient updates: {sdpo_update_count}")
    logger.info(f"  RAG database size: {rag_db.size}")
    gsr = global_seq_stats["total_rollouts"]
    if gsr > 0:
        gsb = global_seq_stats["beat_baseline"]
        gsp = global_seq_stats["perfect"]
        gsi = global_seq_stats["improved_by_retry"]
        gs_avg = global_seq_stats["total_best_score_sum"] / gsr
        logger.info(f"  --- Sequential Rollout Effectiveness ---")
        logger.info(f"  Total rollouts: {gsr}  "
                     f"Beat baseline: {gsb}/{gsr} ({100*gsb/gsr:.0f}%)")
        logger.info(f"  Perfect (1.0): {gsp}/{gsr} ({100*gsp/gsr:.0f}%)  "
                     f"Improved by retry (best attempt > 1): {gsi}/{gsr} ({100*gsi/gsr:.0f}%)")
        logger.info(f"  Avg best score across all rollouts: {gs_avg:.3f}")
    logger.info("=" * 80)

    return all_metrics, agent_model, rag_db


if __name__ == "__main__":
    config = AgentConfig()
    run_training_loop(config)
