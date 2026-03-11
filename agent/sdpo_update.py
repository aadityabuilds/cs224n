"""Standalone SDPO: G=4 rollouts, scoring, loss computation, backward, EMA update.

Implements SDPO-style training with advantage-weighted policy gradient + KL regularization.
Uses REINFORCE with KL penalty to teacher (handles cold start when student ≈ teacher).

Processes rollouts sequentially with gradient accumulation to save GPU memory.
"""
import logging
import torch

logger = logging.getLogger(__name__)


def compute_grpo_advantages(rewards: list[float]) -> torch.Tensor:
    """Compute GRPO-style advantages: (r_i - mean) / max(std, eps)."""
    r = torch.tensor(rewards, dtype=torch.float32)
    mean = r.mean()
    std = r.std().clamp(min=1e-4)
    return (r - mean) / std


def sdpo_step(agent_model, prompt: str, tests_json: str, verify_fn, config,
              system_prompt: str = None) -> dict:
    """Perform one SDPO update step with sequential rollout processing.

    Uses advantage-weighted log-prob maximization + KL penalty to teacher:
      loss_i = -advantage_i * log_prob_student(response_i) + beta * KL(student || teacher)

    This avoids the cold-start problem where reverse KL is 0 when student ≈ teacher.
    """
    from agent.config import SelfDistillationConfig
    sdpo_config = SelfDistillationConfig()

    G = config.num_rollouts
    logger.info(f"  SDPO: Generating {G} rollouts...")

    # 1. Generate G rollouts
    responses, _, _ = agent_model.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        num_return_sequences=G,
        temperature=config.temperature,
        max_new_tokens=config.max_new_tokens,
    )

    # 2. Score all rollouts
    rewards = []
    feedbacks = []
    for i, resp in enumerate(responses):
        result = verify_fn(resp, tests_json)
        rewards.append(result["score"])
        feedbacks.append(result["feedback"])
        logger.info(f"  SDPO rollout {i}: score={result['score']:.3f}, acc={result['acc']:.3f}")

    # 3. Compute advantages
    advantages = compute_grpo_advantages(rewards)
    logger.info(f"  SDPO advantages: {advantages.tolist()}")

    # 4. Process each rollout sequentially with gradient accumulation
    agent_model.optimizer.zero_grad()
    agent_model.student.train()

    total_loss_val = 0.0
    kl_beta = 0.1  # KL penalty weight
    G_actual = len(responses)

    for i in range(G_actual):
        adv = advantages[i].item()

        # Get teacher log probs (no grad)
        with torch.no_grad():
            teacher_lp, teacher_mask = agent_model.compute_log_probs_single(
                agent_model.teacher, prompt, responses[i], system_prompt
            )

        # Get student log probs (with grad)
        student_lp, student_mask = agent_model.compute_log_probs_single(
            agent_model.student, prompt, responses[i], system_prompt
        )

        # Advantage-weighted policy gradient: -adv * mean(log_prob_student)
        num_tokens = student_mask.sum().clamp(min=1.0)
        pg_loss = -(adv * (student_lp * student_mask).sum() / num_tokens)

        # KL regularization to teacher (prevents forgetting)
        # KL(student || teacher) ≈ mean((log_s - log_t) * exp(log_s - log_t) - (log_s - log_t))
        # Simplified: mean((log_s - log_t)^2 / 2) for stability
        log_diff = student_lp - teacher_lp.detach()
        kl_loss = ((log_diff ** 2) * student_mask).sum() / (2.0 * num_tokens)

        rollout_loss = (pg_loss + kl_beta * kl_loss) / G_actual
        rollout_loss.backward()

        total_loss_val += rollout_loss.item()

        # Free memory
        del teacher_lp, teacher_mask, student_lp, student_mask
        del log_diff, pg_loss, kl_loss, rollout_loss
        torch.cuda.empty_cache()

    # 5. Grad clip + optimizer step
    grad_norm = torch.nn.utils.clip_grad_norm_(
        agent_model.student.parameters(), config.max_grad_norm
    )
    agent_model.optimizer.step()

    # 6. EMA update teacher
    agent_model.ema_update_teacher()

    metrics = {
        "sdpo_loss": total_loss_val,
        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        "rewards": rewards,
        "advantages": advantages.tolist(),
        "mean_reward": sum(rewards) / len(rewards),
        "max_reward": max(rewards),
        "num_passing": sum(1 for r in rewards if r > 0.99),
    }
    logger.info(f"  SDPO loss={metrics['sdpo_loss']:.4f}, grad_norm={metrics['grad_norm']:.4f}, "
                f"mean_reward={metrics['mean_reward']:.3f}")
    return metrics
