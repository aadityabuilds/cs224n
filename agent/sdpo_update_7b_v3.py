"""7B SDPO v3 batch update — 32 sequential attempts, sliding feedback window.

Key differences from v2:
- 32 max attempts (up from 16)
- Sliding feedback window (last N feedbacks, not all) to keep prompts short
- Same 100% accuracy condition
"""
# This file uses code from the SDPO (Self-Distillation with Policy Optimization) framework.
# SDPO is licensed under the Apache License, Version 2.0.
# Copyright 2025 Hübotter, Lübeck, Behric, Baumann, Bagatella, Marta, Hakimi, Shenfeld, Kleine Buening, Guestrin, Krause
# Source: https://github.com/lasgroup/SDPO
# License: http://www.apache.org/licenses/LICENSE-2.0
import gc
import logging
import math
import re
import time
import torch
import torch.nn.functional as F

from agent.config import SelfDistillationConfig

logger = logging.getLogger(__name__)

PERFECT_SCORE = 0.999

# ------------------------------------------------------------------
# Sequential rollout with SLIDING WINDOW feedback
# ------------------------------------------------------------------

SEQUENTIAL_FEEDBACK_TEMPLATE = """{prompt}

IMPORTANT: Your previous solutions for this problem were INCORRECT. Below are the errors from your most recent attempts:

{feedback_history}

You MUST write a completely new and correct solution. Carefully analyze each error above — understand WHY the previous approach failed, then choose a DIFFERENT algorithm or fix the root cause. Do NOT repeat the same logic that already failed. Output only the corrected Python solution."""


def _sequential_rollout(agent_model, prompt, system_prompt, tests_json,
                        verify_fn, max_attempts=32,
                        temperature=0.7, max_new_tokens=1536,
                        rollout_idx=0, feedback_truncate=250,
                        feedback_window=6):
    """Generate one rollout with up to 32 sequential attempts.

    Uses a sliding window of the last `feedback_window` feedbacks
    to keep prompt length bounded regardless of attempt count.
    """
    best = None
    all_feedback = []  # store all, but only use last feedback_window
    attempt_log = []

    logger.info(f"      ┌── Rollout {rollout_idx} ──────────────────────────────────────┐")
    logger.info(f"      │  Target: 100% accuracy, up to {max_attempts} attempts          │")
    logger.info(f"      │  Feedback window: last {feedback_window} attempts               │")
    logger.info(f"      └───────────────────────────────────────────────────────────────┘")

    for attempt in range(max_attempts):
        attempt_start = time.time()

        if all_feedback:
            # Sliding window: only include the last N feedbacks
            recent = all_feedback[-feedback_window:]
            # Number them relative to their actual attempt index
            start_idx = len(all_feedback) - len(recent)
            history_str = "\n".join(
                f"Attempt {start_idx + i + 1}: {fb}"
                for i, fb in enumerate(recent)
            )
            attempt_prompt = SEQUENTIAL_FEEDBACK_TEMPLATE.format(
                prompt=prompt, feedback_history=history_str,
            )
        else:
            attempt_prompt = prompt

        decoded_list, prompt_ids, resp_ids_list = agent_model.generate(
            prompt=attempt_prompt, system_prompt=system_prompt,
            num_return_sequences=1, temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        decoded = decoded_list[0]
        resp_ids = resp_ids_list[0]

        result = verify_fn(decoded, tests_json)
        score = float(result["score"])
        feedback = result.get("feedback", "")
        is_perfect = score >= PERFECT_SCORE
        attempt_elapsed = time.time() - attempt_start

        attempt_entry = {
            "attempt": attempt + 1,
            "score": score,
            "accuracy": float(result.get("acc", 0)),
            "is_perfect": bool(is_perfect),
            "feedback_snippet": (feedback or "passed")[:300],
            "response_snippet": decoded[:400],
            "prompt_tokens": len(prompt_ids),
            "response_tokens": len(resp_ids),
            "elapsed_seconds": attempt_elapsed,
            "feedback_window_used": min(len(all_feedback), feedback_window),
        }
        attempt_log.append(attempt_entry)

        # Log every 8th attempt, perfect ones, first, and last
        if attempt % 8 == 0 or is_perfect or attempt == max_attempts - 1:
            logger.info(f"      ╔══ Attempt {attempt+1}/{max_attempts} (rollout {rollout_idx}) ══╗")
            logger.info(f"      ║  Score: {score:.3f}  "
                         f"{'PERFECT' if is_perfect else 'not perfect'}  "
                         f"({attempt_elapsed:.1f}s)  "
                         f"fb_window={min(len(all_feedback), feedback_window)}")
            logger.info(f"      ╠══ MODEL OUTPUT ════════════════════════════════════════╣")
            for line in decoded[:800].split('\n'):
                logger.info(f"      ║  {line}")
            if len(decoded) > 800:
                logger.info(f"      ║  ... ({len(decoded)} chars)")
            logger.info(f"      ╠══ VERIFIER ═════════════════════════════════════════════╣")
            logger.info(f"      ║  Score: {score:.3f}  Acc: {result.get('acc', 0):.3f}")
            if feedback:
                for line in feedback[:250].split('\n'):
                    logger.info(f"      ║    {line}")
            else:
                logger.info(f"      ║    All tests passed!")
            logger.info(f"      ╚══════════════════════════════════════════════════════════╝")
        else:
            logger.info(f"      │ Attempt {attempt+1}: score={score:.3f} ({attempt_elapsed:.1f}s)")

        if best is None or score > best["score"]:
            best = {
                "decoded": decoded,
                "response_ids": resp_ids,
                "prompt_ids": prompt_ids,
                "score": score,
                "feedback": feedback,
                "attempt": attempt,
            }

        if is_perfect:
            logger.info(f"      >>> PERFECT at attempt {attempt+1} — early stop")
            break

        all_feedback.append(
            feedback[:feedback_truncate] if feedback else "incorrect"
        )

    logger.info(f"      ┌── Rollout {rollout_idx} RESULT ──────────────────────────────┐")
    logger.info(f"      │  Best: {best['score']:.3f} (attempt {best['attempt']+1}/{max_attempts})  "
                 f"Perfect: {'YES' if best['score'] >= PERFECT_SCORE else 'NO'}  "
                 f"Total attempts: {len(attempt_log)}  │")
    logger.info(f"      └──────────────────────────────────────────────────────────────┘")

    torch.cuda.empty_cache()
    return best, attempt_log


# ------------------------------------------------------------------
# Reprompting
# ------------------------------------------------------------------

REPROMPT_TEMPLATE = """{prompt}

{solution_section}{feedback_section}
Correctly solve the original question."""

SOLUTION_TEMPLATE = """Here is a correct solution from a previous attempt:
```
{solution}
```

"""

FEEDBACK_TEMPLATE = """Here is feedback from an unsuccessful earlier attempt:
{feedback}

"""


def _strip_thinking(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def build_reprompt(original_prompt, demonstration=None, feedback=None, strip_thinking=True):
    solution_section = ""
    if demonstration:
        demo = _strip_thinking(demonstration) if strip_thinking else demonstration
        solution_section = SOLUTION_TEMPLATE.format(solution=demo)
    feedback_section = ""
    if feedback:
        fb = feedback[:2000] if len(feedback) > 2000 else feedback
        feedback_section = FEEDBACK_TEMPLATE.format(feedback=fb)
    return REPROMPT_TEMPLATE.format(
        prompt=original_prompt,
        solution_section=solution_section,
        feedback_section=feedback_section,
    )


# ------------------------------------------------------------------
# Top-k KL loss
# ------------------------------------------------------------------

def _add_tail_logprob(topk_log_probs):
    log_sum = torch.logsumexp(topk_log_probs, dim=-1, keepdim=True)
    log_sum = torch.clamp(log_sum, max=-1e-7)
    tail_log = torch.log(-torch.expm1(log_sum))
    return torch.cat([topk_log_probs, tail_log], dim=-1)


def _renorm_topk(topk_log_probs):
    log_z = torch.logsumexp(topk_log_probs, dim=-1, keepdim=True)
    return topk_log_probs - log_z


def compute_topk_kl_loss(student_topk_lp, teacher_topk_lp, mask,
                          alpha=0.5, add_tail=True):
    if add_tail:
        s_lp = _add_tail_logprob(student_topk_lp)
        t_lp = _add_tail_logprob(teacher_topk_lp)
    else:
        s_lp = _renorm_topk(student_topk_lp)
        t_lp = _renorm_topk(teacher_topk_lp)

    if alpha == 0.0:
        kl = F.kl_div(s_lp, t_lp, reduction='none', log_target=True)
    elif alpha == 1.0:
        kl = F.kl_div(t_lp, s_lp, reduction='none', log_target=True)
    else:
        log_alpha = math.log(alpha)
        log_1_alpha = math.log(1.0 - alpha)
        mixture_lp = torch.logsumexp(
            torch.stack([s_lp + log_1_alpha, t_lp + log_alpha]), dim=0,
        )
        kl_teacher = F.kl_div(mixture_lp, t_lp, reduction='none', log_target=True)
        kl_student = F.kl_div(mixture_lp, s_lp, reduction='none', log_target=True)
        kl = (1.0 - alpha) * kl_student + alpha * kl_teacher

    per_token_loss = kl.sum(dim=-1)
    num_tokens = mask.sum().clamp(min=1.0)
    return (per_token_loss * mask).sum() / num_tokens


# ------------------------------------------------------------------
# Main 7B v3 SDPO batch step
# ------------------------------------------------------------------

def sdpo_batch_step_7b_v3(agent_model, batch_items, verify_fn,
                           config, sdpo_config=None):
    """SDPO gradient step for 7B v3 — 32 attempts, sliding window feedback."""
    if sdpo_config is None:
        sdpo_config = SelfDistillationConfig(reference_kl_beta=0.0)

    G = config.num_rollouts
    max_attempts = getattr(config, "max_sequential_attempts", 32)
    topk = sdpo_config.distillation_topk or 50
    feedback_truncate = getattr(config, "feedback_truncate", 250)
    feedback_window = getattr(config, "feedback_window", 6)

    logger.info(f"  ╔══════════════════════════════════════════════════════════════════╗")
    logger.info(f"  ║  7B v3 SDPO BATCH: {len(batch_items)} problems, {G} rollouts x "
                f"{max_attempts} attempts   ║")
    logger.info(f"  ║  Feedback window: last {feedback_window}, truncate={feedback_truncate}  ║")
    logger.info(f"  ║  CONDITION: only 100% accuracy (>={PERFECT_SCORE})               ║")
    logger.info(f"  ╚══════════════════════════════════════════════════════════════════╝")

    all_problem_data = []
    rollout_log = []

    for item_idx, item in enumerate(batch_items):
        prob_idx = item.get("problem_idx", item_idx)
        initial_score = item.get("initial_score", 0.0)

        logger.info(f"\n  ┌{'─'*70}┐")
        logger.info(f"  │  v3 PROBLEM {item_idx+1}/{len(batch_items)} "
                     f"(orig #{prob_idx+1})  initial={initial_score:.3f}  │")
        logger.info(f"  └{'─'*70}┘")

        problem_rollout_log = {
            "problem_idx": prob_idx,
            "initial_score": initial_score,
            "rollouts": [],
        }

        rollout_results = []
        for ri in range(G):
            result, attempt_log = _sequential_rollout(
                agent_model, item["prompt"], item.get("system_prompt"),
                item["tests_json"], verify_fn,
                max_attempts=max_attempts, temperature=config.temperature,
                max_new_tokens=config.max_new_tokens,
                rollout_idx=ri,
                feedback_truncate=feedback_truncate,
                feedback_window=feedback_window,
            )
            rollout_results.append(result)
            problem_rollout_log["rollouts"].append({
                "rollout_idx": ri,
                "best_score": float(result["score"]),
                "best_attempt": result["attempt"] + 1,
                "num_attempts": len(attempt_log),
                "is_perfect": bool(result["score"] >= PERFECT_SCORE),
                "attempts": attempt_log,
            })

        scores = [float(r["score"]) for r in rollout_results]
        num_perfect = sum(1 for s in scores if s >= PERFECT_SCORE)
        problem_rollout_log["num_perfect"] = num_perfect
        problem_rollout_log["mean_score"] = float(sum(scores) / len(scores))

        logger.info(f"  ┌── ROLLOUT SUMMARY (problem #{prob_idx+1}) ──────────────────────┐")
        for ri, rr in enumerate(rollout_results):
            star = "PERFECT" if rr["score"] >= PERFECT_SCORE else "partial"
            logger.info(f"  │  Rollout {ri}: score={rr['score']:.3f}  "
                        f"attempt={rr['attempt']+1}/{max_attempts}  {star}  │")
        logger.info(f"  │  Perfect: {num_perfect}/{G}  Mean: {sum(scores)/len(scores):.3f}  │")
        logger.info(f"  └───────────────────────────────────────────────────────────────┘")

        if num_perfect == 0:
            logger.info(f"  SKIPPED: no rollout achieved 100% accuracy")
            problem_rollout_log["decision"] = "skipped_no_perfect"
            rollout_log.append(problem_rollout_log)
            continue

        demo_idx = None
        best_score = -1
        for i, r in enumerate(rollout_results):
            if r["score"] >= PERFECT_SCORE and r["score"] > best_score:
                demo_idx = i
                best_score = r["score"]

        logger.info(f"  Demo: rollout {demo_idx} (score={best_score:.3f})")
        problem_rollout_log["demo_rollout_idx"] = demo_idx
        problem_rollout_log["decision"] = "distill"

        fail_feedback = item.get("initial_feedback")
        demo_text = rollout_results[demo_idx]["decoded"]
        teacher_prompt_text = build_reprompt(
            original_prompt=item["prompt"],
            demonstration=demo_text,
            feedback=fail_feedback,
            strip_thinking=sdpo_config.remove_thinking_from_demonstration,
        )

        teacher_prompt_ids = agent_model.tokenize_chat(teacher_prompt_text)
        max_teacher_tokens = config.max_reprompt_tokens
        if len(teacher_prompt_ids) > max_teacher_tokens:
            teacher_prompt_ids = teacher_prompt_ids[:max_teacher_tokens]

        student_prompt_ids = agent_model.tokenize_chat(
            item["prompt"], item.get("system_prompt")
        )

        logger.info(f"    teacher={len(teacher_prompt_ids)} tokens  "
                     f"student={len(student_prompt_ids)} tokens")

        all_problem_data.append({
            "item_idx": item_idx,
            "problem_idx": prob_idx,
            "rollout_results": rollout_results,
            "demo_idx": demo_idx,
            "teacher_prompt_ids": teacher_prompt_ids,
            "student_prompt_ids": student_prompt_ids,
        })

        rollout_log.append(problem_rollout_log)
        torch.cuda.empty_cache()

    if not all_problem_data:
        logger.info(f"  No problems with 100% rollouts — skipping")
        return {"sdpo_loss": 0.0, "num_problems_updated": 0,
                "num_rollouts_used": 0, "skipped": True, "rollout_log": rollout_log}

    # Phase 2: Distillation
    agent_model.optimizer.zero_grad()
    agent_model.student.train()

    total_loss = 0.0
    total_rollouts_used = 0
    per_problem_metrics = []

    logger.info(f"\n  ╔══════════════════════════════════════════════════════════════════╗")
    logger.info(f"  ║  DISTILLATION: {len(all_problem_data)} problems with perfect rollouts ║")
    logger.info(f"  ╚══════════════════════════════════════════════════════════════════╝")

    for pdata in all_problem_data:
        prob_idx = pdata["problem_idx"]
        problem_loss = 0.0
        problem_rollouts = 0

        logger.info(f"    ┌── Distilling problem #{prob_idx+1} ──────────────────────────┐")

        for i, rr in enumerate(pdata["rollout_results"]):
            if rr["score"] < PERFECT_SCORE:
                logger.info(f"    │  Rollout {i}: score={rr['score']:.3f} — SKIP")
                continue
            if i == pdata["demo_idx"]:
                logger.info(f"    │  Rollout {i}: SKIP (is demo)")
                continue

            response_ids = rr["response_ids"]
            if len(response_ids) == 0:
                continue

            try:
                teacher_token_lp, teacher_topk_lp, teacher_topk_idx = \
                    agent_model.forward_teacher_topk(
                        pdata["teacher_prompt_ids"], response_ids, topk=topk
                    )

                student_token_lp, student_gathered_lp = \
                    agent_model.forward_student_at_teacher_topk(
                        pdata["student_prompt_ids"], response_ids, teacher_topk_idx
                    )

                mask = torch.ones(len(response_ids), device=agent_model.device,
                                  dtype=torch.float32)

                rollout_loss = compute_topk_kl_loss(
                    student_topk_lp=student_gathered_lp,
                    teacher_topk_lp=teacher_topk_lp,
                    mask=mask,
                    alpha=sdpo_config.alpha,
                    add_tail=sdpo_config.distillation_add_tail,
                )

                rollout_loss.backward()

                loss_val = rollout_loss.item()
                problem_loss += loss_val
                problem_rollouts += 1
                total_rollouts_used += 1

                logger.info(f"    │  Rollout {i}: perfect  "
                            f"kl_loss={loss_val:.4f}  tokens={len(response_ids)}")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"    │  Rollout {i}: OOM — skipped")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise
            finally:
                torch.cuda.empty_cache()

        avg_loss = problem_loss / max(problem_rollouts, 1)
        logger.info(f"    │  Result: {problem_rollouts} distilled, avg_loss={avg_loss:.4f}")
        logger.info(f"    └{'─'*60}┘")

        total_loss += problem_loss
        per_problem_metrics.append({
            "problem_idx": prob_idx,
            "rollout_scores": [float(r["score"]) for r in pdata["rollout_results"]],
            "demo_score": float(pdata["rollout_results"][pdata["demo_idx"]]["score"]),
            "rollouts_distilled": problem_rollouts,
            "problem_loss": problem_loss,
        })

    if total_rollouts_used == 0:
        logger.info(f"  No perfect non-demo rollouts — skipping optimizer step")
        return {"sdpo_loss": 0.0, "num_problems_updated": len(all_problem_data),
                "num_rollouts_used": 0, "skipped": True, "rollout_log": rollout_log}

    scale = 1.0 / total_rollouts_used
    for param in agent_model.student.parameters():
        if param.grad is not None:
            param.grad.mul_(scale)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        agent_model.student.parameters(), config.max_grad_norm
    )
    agent_model.optimizer.step()
    agent_model.ema_update_teacher()

    avg_loss = total_loss / max(total_rollouts_used, 1)

    metrics = {
        "sdpo_loss": avg_loss,
        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        "num_problems_updated": len(all_problem_data),
        "num_problems_skipped": len(batch_items) - len(all_problem_data),
        "num_rollouts_used": total_rollouts_used,
        "per_problem": per_problem_metrics,
        "skipped": False,
        "rollout_log": rollout_log,
    }

    logger.info(f"\n  ╔══════════════════════════════════════════════════════════════════╗")
    logger.info(f"  ║  7B v3 SDPO STEP COMPLETE                                      ║")
    logger.info(f"  ║  Loss: {avg_loss:.4f}  Grad norm: {metrics['grad_norm']:.4f}   ║")
    logger.info(f"  ║  Updated: {metrics['num_problems_updated']}  "
                f"Skipped: {metrics['num_problems_skipped']}  "
                f"Rollouts: {total_rollouts_used}  ║")
    logger.info(f"  ╚══════════════════════════════════════════════════════════════════╝")

    torch.cuda.empty_cache()
    gc.collect()
    return metrics
