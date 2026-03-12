"""7B SDPO training loop — memory-optimized, strict 100% accuracy condition.

Key differences from main.py:
- Uses AgentModel7B (no reference model, saves ~14GB)
- No baseline comparison on every problem (saves inference time + memory)
- Update condition: rollout must score >= 0.999 (100% test accuracy)
- Smaller batch sizes, fewer rollouts, shorter sequences
"""
# This file uses code from the SDPO (Self-Distillation with Policy Optimization) framework.
# SDPO is licensed under the Apache License, Version 2.0.
# Copyright 2025 Hübotter, Lübeck, Behric, Baumann, Bagatella, Marta, Hakimi, Shenfeld, Kleine Buening, Guestrin, Krause
# Source: https://github.com/lasgroup/SDPO
# License: http://www.apache.org/licenses/LICENSE-2.0
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
import numpy as np


class _SafeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SDPO_PATH = os.path.join(PROJECT_ROOT, "SDPO")
sys.path.insert(0, SDPO_PATH)
sys.path.insert(0, PROJECT_ROOT)

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


RAG_LESSON_PROMPT = """You are analyzing a coding mistake to extract a reusable lesson.

Problem summary: {problem_summary}

The solution scored {score:.2f} ({accuracy:.1%} of tests passed).

Feedback from test execution:
{feedback}

Write a SHORT (2-3 sentences) lesson about what went wrong and what to watch out for in similar problems. Focus on the TYPE of mistake (off-by-one, wrong data structure, missed edge case, etc.), not the specific solution. Start with "Lesson:" """


def _summarize_problem_for_rag(problem_text: str, max_chars: int = 260) -> str:
    """Create a compact one-line summary for retrieval matching."""
    text = " ".join((problem_text or "").strip().split())
    if not text:
        return "No problem summary available."
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _extract_rag_lesson(agent_model, problem_summary: str, score: float,
                        accuracy: float, feedback: str) -> str:
    """Use the model to generate a concise, reusable lesson."""
    prompt = RAG_LESSON_PROMPT.format(
        problem_summary=problem_summary,
        score=score,
        accuracy=accuracy,
        feedback=(feedback or "")[:800],
    )
    try:
        responses, _, _ = agent_model.generate(
            prompt=prompt,
            system_prompt=None,
            num_return_sequences=1,
            temperature=0.0,
            max_new_tokens=150,
        )
        lesson = responses[0].strip()
        if not lesson.lower().startswith("lesson:"):
            lesson = f"Lesson: {lesson}"
        return lesson[:500]
    except Exception as e:
        logger.warning(f"  Failed to extract lesson: {e}")
        return f"Lesson: Scored {score:.2f}. Main issue from feedback: {(feedback or '')[:250]}"


def _build_rag_chunk(problem_summary: str, lesson: str) -> str:
    """Store both problem summary and lesson for better retrieval."""
    return f"Problem summary: {problem_summary}\n{lesson}"


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
        f"[Lesson {i+1} (id={r[0]}, relevance={r[2]:.3f})]: {r[1]}"
        for i, r in enumerate(results)
    )
    prompt = (
        "You are a coding expert. Below are lessons learned from past mistakes "
        "on similar problems. Use them to avoid repeating the same errors.\n\n"
        f"{rag_context}\n\n"
        "If none of the above lessons apply to the current problem, ignore them "
        "and solve the problem from scratch."
    )
    return prompt, retrieved_ids


def _restore_counters_from_metrics(metrics_list):
    """Reconstruct running counters from saved per-step metrics."""
    action_counts = {"sdpo": 0, "rag": 0, "pass": 0}
    sdpo_update_count = 0
    total_accuracy = 0.0
    total_correct = 0
    for m in metrics_list:
        action = m.get("action", "pass")
        action_counts[action] = action_counts.get(action, 0) + 1
        score = m.get("score", 0.0)
        total_accuracy += score
        if score >= 0.999:
            total_correct += 1
        if "sdpo_batch" in m:
            sdpo_update_count += 1
    return action_counts, sdpo_update_count, total_accuracy, total_correct


def run_training_loop_7b(config: AgentConfig7B,
                          sdpo_config: SelfDistillationConfig = None,
                          checkpoint_callback=None,
                          resume_from_step: int = 0,
                          resume_checkpoint_path: str = None,
                          resume_rag_path: str = None,
                          resume_metrics_path: str = None):
    logger.info("=" * 80)
    logger.info("  7B SDPO TRAINING (memory-optimized, 100% accuracy condition)")
    logger.info("=" * 80)
    logger.info(f"Config: {config}")
    if resume_from_step > 0:
        logger.info(f"  RESUMING from step {resume_from_step}")
        logger.info(f"  Checkpoint: {resume_checkpoint_path}")

    if sdpo_config is None:
        sdpo_config = SelfDistillationConfig(reference_kl_beta=0.0)
    logger.info(f"SDPO config: {sdpo_config}")

    import torch
    _log_mem = lambda tag: logger.info(
        f"  [MEM {tag}] {torch.cuda.memory_allocated()/1e9:.1f}GB alloc, "
        f"{torch.cuda.memory_reserved()/1e9:.1f}GB reserved"
    ) if torch.cuda.is_available() else None

    model_name = resume_checkpoint_path if resume_checkpoint_path else config.model_name
    agent_model = AgentModel7B(
        model_name=model_name,
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
    if resume_rag_path and os.path.exists(resume_rag_path):
        with open(resume_rag_path) as f:
            saved_texts = json.load(f)
        for text in saved_texts:
            rag_db.add(text)
        logger.info(f"RAG DB restored: {rag_db.size} chunks from {resume_rag_path}")

    # Tracking — restore from saved metrics if resuming
    all_metrics = []
    action_counts = {"sdpo": 0, "rag": 0, "pass": 0}
    sdpo_update_count = 0
    total_accuracy = 0.0
    total_correct = 0

    if resume_from_step > 0 and resume_metrics_path and os.path.exists(resume_metrics_path):
        with open(resume_metrics_path) as f:
            all_metrics = json.load(f)
        action_counts, sdpo_update_count, total_accuracy, total_correct = \
            _restore_counters_from_metrics(all_metrics)
        logger.info(f"Restored counters from {len(all_metrics)} saved steps: "
                     f"acc={total_accuracy/max(len(all_metrics),1):.3f}, "
                     f"solve={total_correct}/{len(all_metrics)}, "
                     f"sdpo_updates={sdpo_update_count}")

    start_idx = resume_from_step
    sdpo_batch = []
    num_problems = min(len(dataset), config.max_problems) if config.max_problems else len(dataset)

    for idx in range(start_idx, num_problems):
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
        score = float(result["score"])
        accuracy = float(result["acc"])
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
            problem_summary = _summarize_problem_for_rag(description or problem[:500])
            lesson = _extract_rag_lesson(
                agent_model=agent_model,
                problem_summary=problem_summary,
                score=score,
                accuracy=accuracy,
                feedback=feedback or "",
            )
            rag_chunk = _build_rag_chunk(problem_summary, lesson)
            chunk_id = rag_db.add(rag_chunk)
            step_metrics["rag_added_id"] = chunk_id
            step_metrics["rag_chunk"] = rag_chunk
            logger.info(f"  <rag> stored chunk id={chunk_id}, db_size={rag_db.size}")
            logger.info(f"  <rag> {rag_chunk[:220]}")
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
                json.dump(all_metrics, f, indent=2, cls=_SafeEncoder)

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
