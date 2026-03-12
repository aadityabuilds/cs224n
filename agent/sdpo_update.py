"""SDPO batch update: proper self-distillation with reprompted teacher.

Implements the core SDPO algorithm:
1. Generate G rollouts per problem, score them
2. Select successful rollouts as demonstrations
3. Build reprompted teacher input (demonstration + feedback)
4. Teacher forward pass (no grad) on reprompted input → top-k logits
5. Student forward pass (with grad) on original input → gathered at teacher's top-k
6. Compute top-k KL distillation loss (JSD by default)
7. Gradient accumulation across problems × rollouts, single optimizer step
8. EMA teacher update

Key differences from the old implementation:
- Teacher is conditioned on demonstrations/feedback (the core of SDPO)
- Full-logit KL distillation over top-k vocabulary (not per-token L2)
- Batched across multiple problems for stable gradients
- Keeps raw token IDs (no decode→re-encode)
- Skips rollouts that are their own demonstration (dont_reprompt_on_self_success)
"""
import logging
import math
import re
import torch
import torch.nn.functional as F

from agent.config import SelfDistillationConfig

logger = logging.getLogger(__name__)


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
    """Build the reprompted prompt for the teacher.

    The teacher sees the original problem PLUS:
    - A correct solution (demonstration) from a successful rollout, and/or
    - Feedback from a failed rollout explaining what went wrong

    This gives the teacher an informational advantage over the student,
    and distilling its predictions teaches the student to avoid mistakes.
    """
    solution_section = ""
    if demonstration:
        demo = _strip_thinking(demonstration) if strip_thinking else demonstration
        solution_section = SOLUTION_TEMPLATE.format(solution=demo)

    feedback_section = ""
    if feedback:
        # Truncate very long feedback
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
    """Add a tail bucket: log(1 - sum(p_i)) for the remaining vocab probability mass.

    This ensures the (k+1)-dimensional distributions sum to 1, making KL well-defined.
    Uses the identity: log(1 - exp(x)) = log(-expm1(x)) for numerical stability.
    """
    log_sum = torch.logsumexp(topk_log_probs, dim=-1, keepdim=True)
    # Clamp to ensure sum < 1 (numerically)
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
    """Compute KL distillation loss between student and teacher top-k log-probs.

    Both tensors must be (resp_len, topk) gathered at the SAME token indices.

    Args:
        student_topk_lp: (resp_len, topk) student log-probs at teacher's top-k indices
        teacher_topk_lp: (resp_len, topk) teacher's top-k log-probs
        mask: (resp_len,) binary mask for valid positions
        alpha: KL mixture coefficient (0=forward, 1=reverse, 0.5=JSD)
        add_tail: whether to add tail probability bucket

    Returns:
        scalar loss (mean over valid tokens)
    """
    if add_tail:
        s_lp = _add_tail_logprob(student_topk_lp)
        t_lp = _add_tail_logprob(teacher_topk_lp)
    else:
        s_lp = _renorm_topk(student_topk_lp)
        t_lp = _renorm_topk(teacher_topk_lp)

    if alpha == 0.0:
        # Forward KL: KL(teacher || student)
        kl = F.kl_div(s_lp, t_lp, reduction='none', log_target=True)
    elif alpha == 1.0:
        # Reverse KL: KL(student || teacher)
        kl = F.kl_div(t_lp, s_lp, reduction='none', log_target=True)
    else:
        # Generalized Jensen-Shannon Divergence
        log_alpha = math.log(alpha)
        log_1_alpha = math.log(1.0 - alpha)
        mixture_lp = torch.logsumexp(
            torch.stack([s_lp + log_1_alpha, t_lp + log_alpha]),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_lp, t_lp, reduction='none', log_target=True)
        kl_student = F.kl_div(mixture_lp, s_lp, reduction='none', log_target=True)
        kl = (1.0 - alpha) * kl_student + alpha * kl_teacher

    # Sum over vocab dimension → per-token loss, then masked mean
    per_token_loss = kl.sum(dim=-1)  # (resp_len,)
    num_tokens = mask.sum().clamp(min=1.0)
    return (per_token_loss * mask).sum() / num_tokens


# ------------------------------------------------------------------
# Main SDPO batch step
# ------------------------------------------------------------------

def sdpo_batch_step(agent_model, batch_items: list[dict], verify_fn,
                    config, sdpo_config: SelfDistillationConfig = None) -> dict:
    """Perform one SDPO gradient step over a batch of problems.

    Args:
        agent_model: AgentModel instance
        batch_items: list of dicts with keys:
            - prompt: str (the code prompt)
            - tests_json: str (test cases JSON)
            - system_prompt: str or None (RAG context)
            - initial_feedback: str or None (from initial verification)
        verify_fn: function(response_text, tests_json) → dict with score, acc, feedback
        config: AgentConfig
        sdpo_config: SelfDistillationConfig (defaults created if None)

    Returns:
        dict with detailed metrics for logging
    """
    if sdpo_config is None:
        sdpo_config = SelfDistillationConfig()

    G = config.num_rollouts
    topk = sdpo_config.distillation_topk or 100

    logger.info(f"  SDPO batch: {len(batch_items)} problems, G={G} rollouts, top-{topk} KL")

    # ------------------------------------------------------------------
    # Phase 1: Generate rollouts and score for all problems
    # ------------------------------------------------------------------
    all_problem_data = []

    for item_idx, item in enumerate(batch_items):
        prob_idx = item.get('problem_idx', item_idx)
        logger.info(f"  [SDPO problem {item_idx+1}/{len(batch_items)} "
                     f"(orig #{prob_idx+1})] Generating {G} rollouts...")

        decoded, prompt_ids, response_ids_list = agent_model.generate(
            prompt=item['prompt'],
            system_prompt=item.get('system_prompt'),
            num_return_sequences=G,
            temperature=config.temperature,
            max_new_tokens=config.max_new_tokens,
        )

        # Score all rollouts
        rewards = []
        feedbacks = []
        for i, resp_text in enumerate(decoded):
            result = verify_fn(resp_text, item['tests_json'])
            rewards.append(result['score'])
            feedbacks.append(result.get('feedback', ''))
            fb_short = (result.get('feedback', '') or 'passed')[:80]
            logger.info(f"    rollout {i}/{G}: score={result['score']:.3f}  "
                        f"acc={result['acc']:.3f}  feedback={fb_short}")

        # Summarize rollouts for this problem
        num_passing = sum(1 for r in rewards if r >= sdpo_config.success_reward_threshold)
        logger.info(f"    rollout summary: mean={sum(rewards)/len(rewards):.3f}  "
                     f"max={max(rewards):.3f}  passing={num_passing}/{G}")

        # Select best demonstration (highest reward above threshold)
        demo_idx = None
        best_reward = -1.0
        for i, r in enumerate(rewards):
            if r >= sdpo_config.success_reward_threshold and r > best_reward:
                demo_idx = i
                best_reward = r

        # Collect feedback from a failed rollout
        fail_feedback = None
        for i, (r, fb) in enumerate(zip(rewards, feedbacks)):
            if r < sdpo_config.success_reward_threshold and fb:
                fail_feedback = fb
                break
        # Fall back to initial feedback from the greedy attempt
        if fail_feedback is None:
            fail_feedback = item.get('initial_feedback')

        # Check if we have any learning signal
        has_demo = demo_idx is not None
        has_feedback = fail_feedback is not None and len(fail_feedback.strip()) > 0

        if has_demo:
            logger.info(f"    demo selected: rollout {demo_idx} (score={best_reward:.3f})")
        if has_feedback:
            logger.info(f"    feedback available: {(fail_feedback or '')[:150]}")

        if not has_demo and not has_feedback:
            logger.info(f"    SKIPPED: no demonstration and no usable feedback")
            continue

        # Build reprompted teacher prompt
        demo_text = decoded[demo_idx] if has_demo else None
        teacher_prompt_text = build_reprompt(
            original_prompt=item['prompt'],
            demonstration=demo_text,
            feedback=fail_feedback if (not has_demo or not sdpo_config.dont_reprompt_on_self_success) else fail_feedback,
            strip_thinking=sdpo_config.remove_thinking_from_demonstration,
        )

        # Tokenize teacher prompt (no system prompt — context is embedded in reprompt)
        teacher_prompt_ids = agent_model.tokenize_chat(teacher_prompt_text)
        # Truncate if needed
        max_teacher_tokens = config.max_reprompt_tokens
        if len(teacher_prompt_ids) > max_teacher_tokens:
            teacher_prompt_ids = teacher_prompt_ids[:max_teacher_tokens]
            logger.info(f"    teacher prompt truncated to {max_teacher_tokens} tokens")

        # Student prompt IDs (original prompt with RAG system prompt)
        student_prompt_ids = agent_model.tokenize_chat(
            item['prompt'], item.get('system_prompt')
        )

        logger.info(f"    teacher_prompt_len={len(teacher_prompt_ids)} tokens  "
                     f"student_prompt_len={len(student_prompt_ids)} tokens")

        all_problem_data.append({
            'item_idx': item_idx,
            'problem_idx': prob_idx,
            'decoded': decoded,
            'response_ids_list': response_ids_list,
            'rewards': rewards,
            'feedbacks': feedbacks,
            'demo_idx': demo_idx,
            'has_demo': has_demo,
            'has_feedback': has_feedback,
            'teacher_prompt_ids': teacher_prompt_ids,
            'student_prompt_ids': student_prompt_ids,
        })

    if not all_problem_data:
        logger.info("  SDPO batch: no problems with learning signal — skipping gradient step")
        return {
            'sdpo_loss': 0.0,
            'num_problems_updated': 0,
            'num_rollouts_used': 0,
            'skipped': True,
        }

    # ------------------------------------------------------------------
    # Phase 2: Compute distillation loss with gradient accumulation
    # ------------------------------------------------------------------
    agent_model.optimizer.zero_grad()
    agent_model.student.train()

    total_loss = 0.0
    total_rollouts_used = 0
    total_rollouts_possible = 0
    per_problem_metrics = []

    logger.info(f"  Distillation: {len(all_problem_data)} problems with learning signal")

    for pdata in all_problem_data:
        prob_idx = pdata.get('problem_idx', pdata['item_idx'])
        problem_loss = 0.0
        problem_rollouts = 0
        G_actual = len(pdata['response_ids_list'])

        for i in range(G_actual):
            # Skip if this rollout IS the demonstration (don't distill from yourself)
            if sdpo_config.dont_reprompt_on_self_success and i == pdata['demo_idx']:
                logger.info(f"    [prob #{prob_idx+1}] rollout {i}: SKIP (is demo)")
                continue

            response_ids = pdata['response_ids_list'][i]
            if len(response_ids) == 0:
                logger.info(f"    [prob #{prob_idx+1}] rollout {i}: SKIP (empty)")
                continue

            total_rollouts_possible += 1

            try:
                # Teacher forward (no grad): get top-k logits
                teacher_token_lp, teacher_topk_lp, teacher_topk_idx = \
                    agent_model.forward_teacher_topk(
                        pdata['teacher_prompt_ids'], response_ids, topk=topk
                    )

                # Student forward (with grad): gather at teacher's top-k indices
                student_token_lp, student_gathered_lp = \
                    agent_model.forward_student_at_teacher_topk(
                        pdata['student_prompt_ids'], response_ids, teacher_topk_idx
                    )

                # Response mask (all ones since we trimmed at EOS during generation)
                mask = torch.ones(len(response_ids), device=agent_model.device, dtype=torch.float32)

                # Compute top-k KL distillation loss
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

                logger.info(f"    [prob #{prob_idx+1}] rollout {i}: "
                            f"kl_loss={loss_val:.4f}  resp_tokens={len(response_ids)}  "
                            f"reward={pdata['rewards'][i]:.3f}")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"    [prob #{prob_idx+1}] rollout {i}: OOM — skipped")
                    torch.cuda.empty_cache()
                    continue
                raise

            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        avg_prob_loss = problem_loss / max(problem_rollouts, 1)
        logger.info(f"    [prob #{prob_idx+1}] done: {problem_rollouts} rollouts distilled, "
                     f"avg_loss={avg_prob_loss:.4f}")

        total_loss += problem_loss
        per_problem_metrics.append({
            'item_idx': pdata['item_idx'],
            'problem_idx': prob_idx,
            'rewards': pdata['rewards'],
            'has_demo': pdata['has_demo'],
            'has_feedback': pdata['has_feedback'],
            'demo_reward': pdata['rewards'][pdata['demo_idx']] if pdata['demo_idx'] is not None else None,
            'rollouts_used': problem_rollouts,
            'problem_loss': problem_loss,
        })

    # Scale gradients by 1/total_rollouts_used (average across all rollouts)
    if total_rollouts_used > 0:
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

    # Aggregate metrics
    all_rewards = [r for pd in all_problem_data for r in pd['rewards']]
    metrics = {
        'sdpo_loss': avg_loss,
        'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        'num_problems_updated': len(all_problem_data),
        'num_problems_skipped': len(batch_items) - len(all_problem_data),
        'num_rollouts_used': total_rollouts_used,
        'num_rollouts_possible': total_rollouts_possible,
        'mean_reward': sum(all_rewards) / max(len(all_rewards), 1),
        'max_reward': max(all_rewards) if all_rewards else 0.0,
        'num_demos_found': sum(1 for pd in all_problem_data if pd['has_demo']),
        'num_feedback_only': sum(1 for pd in all_problem_data if pd['has_feedback'] and not pd['has_demo']),
        'per_problem': per_problem_metrics,
        'skipped': False,
    }

    logger.info(f"  SDPO step complete: loss={avg_loss:.4f}, grad_norm={metrics['grad_norm']:.4f}, "
                f"problems={metrics['num_problems_updated']}, rollouts={total_rollouts_used}, "
                f"demos={metrics['num_demos_found']}, mean_reward={metrics['mean_reward']:.3f}")

    return metrics
