"""SDPO batch update: sequential self-improving rollouts + strict baseline filtering.

Algorithm (v2 — fixes catastrophic forgetting):
1. For each problem: run 4 rollouts, each with up to 4 sequential attempts
   using cumulative verifier feedback to self-correct
2. Only keep rollouts whose best score strictly exceeds the initial greedy score
3. Select best improving rollout as demonstration for teacher reprompting
4. Teacher forward (no grad) on reprompted input → top-k logits
5. Student forward (with grad) on original input → gathered at teacher's top-k
6. Compute top-k KL distillation loss (JSD) + reference KL anchor
7. Single optimizer step + EMA teacher update
"""
# This file uses code from the SDPO (Self-Distillation with Policy Optimization) framework.
# SDPO is licensed under the Apache License, Version 2.0.
# Copyright 2025 Hübotter, Lübeck, Behric, Baumann, Bagatella, Marta, Hakimi, Shenfeld, Kleine Buening, Guestrin, Krause
# Source: https://github.com/lasgroup/SDPO
# License: http://www.apache.org/licenses/LICENSE-2.0
import logging
import math
import re
import torch
import torch.nn.functional as F

from agent.config import SelfDistillationConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Sequential rollout with cumulative feedback
# ------------------------------------------------------------------

SEQUENTIAL_FEEDBACK_TEMPLATE = """{prompt}

IMPORTANT: Your previous solutions for this problem were INCORRECT. Below are the errors from your most recent attempts:

{feedback_history}

You MUST write a completely new and correct solution. Carefully analyze each error above — understand WHY the previous approach failed, then choose a DIFFERENT algorithm or fix the root cause. Do NOT repeat the same logic that already failed. Output only the corrected Python solution."""


def _sequential_rollout(agent_model, prompt, system_prompt, tests_json,
                        verify_fn, initial_score, max_attempts=4,
                        temperature=0.7, max_new_tokens=2048,
                        rollout_idx=0, problem_idx=0):
    """Generate one rollout with up to max_attempts sequential tries.

    Each attempt after the first is re-prompted with cumulative verifier
    feedback from all prior attempts.  Returns the best attempt.
    """
    best = None
    feedback_history = []

    logger.info(f"      ┌── Rollout {rollout_idx} ─────────────────────────────────────────┐")
    logger.info(f"      │  Target: beat baseline score {initial_score:.3f}                  │")
    logger.info(f"      │  Max attempts: {max_attempts}, temp={temperature}                 │")
    logger.info(f"      └───────────────────────────────────────────────────────────────────┘")

    for attempt in range(max_attempts):
        if feedback_history:
            history_str = "\n".join(
                f"Attempt {i+1}: {fb}" for i, fb in enumerate(feedback_history)
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

        # Log attempt details
        beats = score > initial_score
        logger.info(f"      ╔══ Attempt {attempt+1}/{max_attempts} (rollout {rollout_idx}) "
                     f"══════════════════════════════╗")
        logger.info(f"      ║  Score: {score:.3f}  "
                     f"{'>> BEATS BASELINE' if beats else '   below baseline'} "
                     f"(baseline={initial_score:.3f})")

        # Show model output
        logger.info(f"      ╠══ MODEL OUTPUT ═══════════════════════════════════════════╣")
        for line in decoded[:1500].split('\n'):
            logger.info(f"      ║  {line}")
        if len(decoded) > 1500:
            logger.info(f"      ║  ... ({len(decoded)} chars total)")

        # Show verification
        logger.info(f"      ╠══ VERIFIER ════════════════════════════════════════════════╣")
        logger.info(f"      ║  Score: {score:.3f}  Acc: {result.get('acc', 0):.3f}")
        if feedback:
            logger.info(f"      ║  Feedback:")
            for line in feedback[:400].split('\n'):
                logger.info(f"      ║    {line}")
        else:
            logger.info(f"      ║  Feedback: All tests passed!")
        logger.info(f"      ╚═══════════════════════════════════════════════════════════╝")

        if best is None or score > best["score"]:
            best = {
                "decoded": decoded,
                "response_ids": resp_ids,
                "prompt_ids": prompt_ids,
                "score": score,
                "feedback": feedback,
                "attempt": attempt,
            }

        # Only early-stop on a perfect score — otherwise keep trying
        # to maximize accuracy across all attempts
        if score >= 0.999:
            logger.info(f"      >>> PERFECT SCORE at attempt {attempt+1}, stopping early")
            break

        feedback_history.append(feedback[:500] if feedback else "incorrect")
        if attempt < max_attempts - 1:
            logger.info(f"      ... appending feedback, retrying with cumulative context")

    logger.info(f"      ┌── Rollout {rollout_idx} RESULT ─────────────────────────────────┐")
    logger.info(f"      │  Best score: {best['score']:.3f} (attempt {best['attempt']+1})   │")
    logger.info(f"      │  Beats baseline: {'YES' if best['score'] > initial_score else 'NO'}  │")
    logger.info(f"      └───────────────────────────────────────────────────────────────────┘")

    return best


# ------------------------------------------------------------------
# Reprompting: build enriched teacher input
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
    """Remove <think>...</think> blocks from a response."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def build_reprompt(original_prompt: str, demonstration: str = None,
                   feedback: str = None, strip_thinking: bool = True) -> str:
    """Build the reprompted prompt for the teacher."""
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
# Top-k KL distillation loss
# ------------------------------------------------------------------

def _add_tail_logprob(topk_log_probs: torch.Tensor) -> torch.Tensor:
    """Add a tail bucket: log(1 - sum(p_i)) for the remaining vocab mass."""
    log_sum = torch.logsumexp(topk_log_probs, dim=-1, keepdim=True)
    log_sum = torch.clamp(log_sum, max=-1e-7)
    tail_log = torch.log(-torch.expm1(log_sum))
    return torch.cat([topk_log_probs, tail_log], dim=-1)


def _renorm_topk(topk_log_probs: torch.Tensor) -> torch.Tensor:
    """Renormalize top-k log-probs to sum to 1."""
    log_z = torch.logsumexp(topk_log_probs, dim=-1, keepdim=True)
    return topk_log_probs - log_z


def compute_topk_kl_loss(student_topk_lp: torch.Tensor,
                          teacher_topk_lp: torch.Tensor,
                          mask: torch.Tensor,
                          alpha: float = 0.5,
                          add_tail: bool = True) -> torch.Tensor:
    """Compute KL distillation loss between student and teacher top-k log-probs."""
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
            torch.stack([s_lp + log_1_alpha, t_lp + log_alpha]),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_lp, t_lp, reduction='none', log_target=True)
        kl_student = F.kl_div(mixture_lp, s_lp, reduction='none', log_target=True)
        kl = (1.0 - alpha) * kl_student + alpha * kl_teacher

    per_token_loss = kl.sum(dim=-1)
    num_tokens = mask.sum().clamp(min=1.0)
    return (per_token_loss * mask).sum() / num_tokens


# ------------------------------------------------------------------
# Main SDPO batch step
# ------------------------------------------------------------------

def sdpo_batch_step(agent_model, batch_items: list[dict], verify_fn,
                    config, sdpo_config: SelfDistillationConfig = None) -> dict:
    """Perform one SDPO gradient step over a batch of problems.

    Key changes from v1:
    - Sequential 4x4 rollouts with cumulative feedback
    - Strict baseline filtering: only distill rollouts that beat initial_score
    - Reference KL anchor to prevent drift from base model
    """
    if sdpo_config is None:
        sdpo_config = SelfDistillationConfig()

    G = config.num_rollouts
    max_attempts = getattr(config, "max_sequential_attempts", 4)
    topk = sdpo_config.distillation_topk or 100
    ref_kl_beta = sdpo_config.reference_kl_beta

    logger.info(f"  ╔══════════════════════════════════════════════════════════════════╗")
    logger.info(f"  ║  SDPO BATCH: {len(batch_items)} problems, {G} rollouts x "
                f"{max_attempts} attempts           ║")
    logger.info(f"  ║  top-{topk} KL, ref_kl_beta={ref_kl_beta}, alpha={sdpo_config.alpha}  ║")
    logger.info(f"  ╚══════════════════════════════════════════════════════════════════╝")

    # ------------------------------------------------------------------
    # Phase 1: Sequential rollouts for all problems
    # ------------------------------------------------------------------
    all_problem_data = []

    # Running stats for sequential improvement effectiveness
    cumul_rollouts = 0
    cumul_beat_baseline = 0
    cumul_perfect = 0
    cumul_improved_by_retry = 0  # rollouts where attempt>1 was the best
    cumul_best_scores = []

    for item_idx, item in enumerate(batch_items):
        prob_idx = item.get("problem_idx", item_idx)
        baseline = item.get("initial_score", 0.0)

        logger.info(f"\n  ┌{'─'*70}┐")
        logger.info(f"  │  SDPO PROBLEM {item_idx+1}/{len(batch_items)} "
                     f"(orig #{prob_idx+1})  baseline={baseline:.3f}  │")
        logger.info(f"  │  Running {G} rollouts x {max_attempts} sequential attempts...  │")
        logger.info(f"  └{'─'*70}┘")

        rollout_results = []
        for ri in range(G):
            result = _sequential_rollout(
                agent_model, item["prompt"], item.get("system_prompt"),
                item["tests_json"], verify_fn, initial_score=baseline,
                max_attempts=max_attempts, temperature=config.temperature,
                max_new_tokens=config.max_new_tokens,
                rollout_idx=ri, problem_idx=prob_idx,
            )
            rollout_results.append(result)
            cumul_rollouts += 1
            if result["score"] > baseline:
                cumul_beat_baseline += 1
            if result["score"] >= 0.999:
                cumul_perfect += 1
            if result["attempt"] > 0:
                cumul_improved_by_retry += 1
            cumul_best_scores.append(result["score"])

        # Summary table for all rollouts
        scores = [r["score"] for r in rollout_results]
        num_improving = sum(1 for s in scores if s > baseline)
        logger.info(f"  ┌── ROLLOUT SUMMARY (problem #{prob_idx+1}) ──────────────────────┐")
        for ri, rr in enumerate(rollout_results):
            beats = "BEATS" if rr["score"] > baseline else "below"
            logger.info(f"  │  Rollout {ri}: score={rr['score']:.3f}  "
                        f"attempt={rr['attempt']+1}/{max_attempts}  {beats}  │")
        logger.info(f"  │  Mean: {sum(scores)/len(scores):.3f}  "
                     f"Max: {max(scores):.3f}  "
                     f"Improving: {num_improving}/{G}  │")
        logger.info(f"  └───────────────────────────────────────────────────────────────┘")

        # Running cumulative stats
        avg_best = sum(cumul_best_scores) / len(cumul_best_scores)
        logger.info(f"  ┌── CUMULATIVE IMPROVEMENT STATS ───────────────────────────────┐")
        logger.info(f"  │  Rollouts so far: {cumul_rollouts}  "
                     f"Beat baseline: {cumul_beat_baseline}/{cumul_rollouts} "
                     f"({100*cumul_beat_baseline/max(cumul_rollouts,1):.0f}%)  │")
        logger.info(f"  │  Perfect (1.0): {cumul_perfect}/{cumul_rollouts} "
                     f"({100*cumul_perfect/max(cumul_rollouts,1):.0f}%)  "
                     f"Improved by retry: {cumul_improved_by_retry}/{cumul_rollouts} "
                     f"({100*cumul_improved_by_retry/max(cumul_rollouts,1):.0f}%)  │")
        logger.info(f"  │  Avg best score: {avg_best:.3f}  │")
        logger.info(f"  └───────────────────────────────────────────────────────────────┘")

        # Demo = best rollout that strictly beats the baseline
        demo_idx = None
        best_reward = baseline
        for i, r in enumerate(rollout_results):
            if r["score"] > best_reward:
                demo_idx = i
                best_reward = r["score"]

        if demo_idx is None:
            logger.info(f"  ╳ SKIPPED: no rollout beat baseline {baseline:.3f} — "
                        f"zero gradient contribution")
            continue

        logger.info(f"  ✓ Demo selected: rollout {demo_idx} "
                     f"(score={best_reward:.3f} vs baseline={baseline:.3f}, "
                     f"delta={best_reward - baseline:+.3f})")

        # Collect failure feedback from the initial greedy attempt
        fail_feedback = item.get("initial_feedback")

        # Build reprompted teacher prompt
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
            logger.info(f"    teacher prompt truncated to {max_teacher_tokens} tokens")

        student_prompt_ids = agent_model.tokenize_chat(
            item["prompt"], item.get("system_prompt")
        )

        logger.info(f"    teacher_prompt={len(teacher_prompt_ids)} tokens  "
                     f"student_prompt={len(student_prompt_ids)} tokens")

        all_problem_data.append({
            "item_idx": item_idx,
            "problem_idx": prob_idx,
            "baseline": baseline,
            "rollout_results": rollout_results,
            "demo_idx": demo_idx,
            "teacher_prompt_ids": teacher_prompt_ids,
            "student_prompt_ids": student_prompt_ids,
        })

    # Phase 1 final summary
    logger.info(f"\n  ╔══ PHASE 1 COMPLETE: SEQUENTIAL ROLLOUT STATS ═══════════════════╗")
    logger.info(f"  ║  Total rollouts: {cumul_rollouts}  "
                f"Beat baseline: {cumul_beat_baseline} ({100*cumul_beat_baseline/max(cumul_rollouts,1):.0f}%)  "
                f"Perfect: {cumul_perfect}  ║")
    logger.info(f"  ║  Improved by retry (best attempt > 1): {cumul_improved_by_retry} "
                f"({100*cumul_improved_by_retry/max(cumul_rollouts,1):.0f}%)  ║")
    logger.info(f"  ║  Avg best score: {sum(cumul_best_scores)/max(len(cumul_best_scores),1):.3f}  "
                f"Problems with demos: {len(all_problem_data)}/{len(batch_items)}  ║")
    logger.info(f"  ╚══════════════════════════════════════════════════════════════════╝")

    if not all_problem_data:
        logger.info(f"  ╳ SDPO batch: no problems with improving rollouts — skipping entirely")
        return {
            "sdpo_loss": 0.0,
            "num_problems_updated": 0,
            "num_rollouts_used": 0,
            "skipped": True,
            "sequential_stats": {
                "total_rollouts": cumul_rollouts,
                "beat_baseline": cumul_beat_baseline,
                "perfect": cumul_perfect,
                "improved_by_retry": cumul_improved_by_retry,
                "avg_best_score": sum(cumul_best_scores) / max(len(cumul_best_scores), 1),
            },
        }

    # ------------------------------------------------------------------
    # Phase 2: Distillation — only improving rollouts
    # ------------------------------------------------------------------
    agent_model.optimizer.zero_grad()
    agent_model.student.train()

    total_loss = 0.0
    total_rollouts_used = 0
    per_problem_metrics = []

    logger.info(f"\n  ╔══════════════════════════════════════════════════════════════════╗")
    logger.info(f"  ║  DISTILLATION PHASE: {len(all_problem_data)} problems with signal  ║")
    logger.info(f"  ╚══════════════════════════════════════════════════════════════════╝")

    for pdata in all_problem_data:
        prob_idx = pdata["problem_idx"]
        baseline = pdata["baseline"]
        problem_loss = 0.0
        problem_rollouts = 0

        logger.info(f"    ┌── Distilling problem #{prob_idx+1} "
                     f"(baseline={baseline:.3f}) ──────────────┐")

        for i, rr in enumerate(pdata["rollout_results"]):
            if rr["score"] <= baseline:
                logger.info(f"    │  Rollout {i}: score={rr['score']:.3f} "
                            f"<= baseline — SKIP (no gradient)")
                continue
            if i == pdata["demo_idx"]:
                logger.info(f"    │  Rollout {i}: score={rr['score']:.3f} "
                            f"— SKIP (is demo, used for teacher)")
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

                kl_distill = compute_topk_kl_loss(
                    student_topk_lp=student_gathered_lp,
                    teacher_topk_lp=teacher_topk_lp,
                    mask=mask,
                    alpha=sdpo_config.alpha,
                    add_tail=sdpo_config.distillation_add_tail,
                )

                # Reference KL anchor: KL(student || frozen_base)
                ref_kl_val = 0.0
                if ref_kl_beta > 0 and hasattr(agent_model, "reference"):
                    ref_kl = agent_model.forward_reference_kl(
                        pdata["student_prompt_ids"], response_ids, mask
                    )
                    ref_kl_val = ref_kl.item()
                    rollout_loss = kl_distill + ref_kl_beta * ref_kl
                else:
                    rollout_loss = kl_distill

                rollout_loss.backward()

                loss_val = rollout_loss.item()
                problem_loss += loss_val
                problem_rollouts += 1
                total_rollouts_used += 1

                logger.info(f"    │  Rollout {i}: score={rr['score']:.3f}  "
                            f"kl_distill={kl_distill.item():.4f}  "
                            f"ref_kl={ref_kl_val:.4f}  "
                            f"total_loss={loss_val:.4f}  "
                            f"tokens={len(response_ids)}")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"    │  Rollout {i}: OOM — skipped")
                    torch.cuda.empty_cache()
                    continue
                raise

            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        avg_prob_loss = problem_loss / max(problem_rollouts, 1)
        logger.info(f"    │  Result: {problem_rollouts} rollouts distilled, "
                     f"avg_loss={avg_prob_loss:.4f}")
        logger.info(f"    └{'─'*60}┘")

        total_loss += problem_loss
        all_scores = [float(r["score"]) for r in pdata["rollout_results"]]
        per_problem_metrics.append({
            "problem_idx": prob_idx,
            "baseline": float(baseline),
            "rollout_scores": all_scores,
            "demo_score": float(pdata["rollout_results"][pdata["demo_idx"]]["score"]),
            "rollouts_distilled": problem_rollouts,
            "problem_loss": problem_loss,
        })

    if total_rollouts_used == 0:
        logger.info(f"  ╳ No improving non-demo rollouts found — skipping optimizer step")
        return {
            "sdpo_loss": 0.0,
            "num_problems_updated": len(all_problem_data),
            "num_rollouts_used": 0,
            "skipped": True,
            "sequential_stats": {
                "total_rollouts": cumul_rollouts,
                "beat_baseline": cumul_beat_baseline,
                "perfect": cumul_perfect,
                "improved_by_retry": cumul_improved_by_retry,
                "avg_best_score": sum(cumul_best_scores) / max(len(cumul_best_scores), 1),
            },
        }

    # Scale gradients
    scale = 1.0 / total_rollouts_used
    for param in agent_model.student.parameters():
        if param.grad is not None:
            param.grad.mul_(scale)

    # ------------------------------------------------------------------
    # Phase 3: Optimizer step + EMA update
    # ------------------------------------------------------------------
    grad_norm = torch.nn.utils.clip_grad_norm_(
        agent_model.student.parameters(), config.max_grad_norm
    )
    agent_model.optimizer.step()
    agent_model.ema_update_teacher()

    avg_loss = total_loss / max(total_rollouts_used, 1)

    all_scores_flat = [s for pd in all_problem_data
                       for s in [r["score"] for r in pd["rollout_results"]]]
    seq_stats = {
        "total_rollouts": cumul_rollouts,
        "beat_baseline": cumul_beat_baseline,
        "perfect": cumul_perfect,
        "improved_by_retry": cumul_improved_by_retry,
        "avg_best_score": sum(cumul_best_scores) / max(len(cumul_best_scores), 1),
    }
    metrics = {
        "sdpo_loss": avg_loss,
        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        "num_problems_updated": len(all_problem_data),
        "num_problems_skipped": len(batch_items) - len(all_problem_data),
        "num_rollouts_used": total_rollouts_used,
        "mean_reward": sum(all_scores_flat) / max(len(all_scores_flat), 1),
        "max_reward": max(all_scores_flat) if all_scores_flat else 0.0,
        "per_problem": per_problem_metrics,
        "sequential_stats": seq_stats,
        "skipped": False,
    }

    logger.info(f"\n  ╔══════════════════════════════════════════════════════════════════╗")
    logger.info(f"  ║  SDPO STEP COMPLETE                                            ║")
    logger.info(f"  ║  Loss: {avg_loss:.4f}  Grad norm: {metrics['grad_norm']:.4f}        ║")
    logger.info(f"  ║  Problems updated: {metrics['num_problems_updated']}  "
                f"Skipped: {metrics['num_problems_skipped']}                ║")
    logger.info(f"  ║  Rollouts distilled: {total_rollouts_used}  "
                f"Mean reward: {metrics['mean_reward']:.3f}              ║")
    logger.info(f"  ║  Sequential improvement: {cumul_beat_baseline}/{cumul_rollouts} beat baseline  "
                f"({100*cumul_beat_baseline/max(cumul_rollouts,1):.0f}%)  ║")
    logger.info(f"  ║  Improved by retry: {cumul_improved_by_retry}/{cumul_rollouts}  "
                f"Perfect: {cumul_perfect}/{cumul_rollouts}                ║")
    logger.info(f"  ╚══════════════════════════════════════════════════════════════════╝")

    return metrics
