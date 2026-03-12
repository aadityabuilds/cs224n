"""Configuration dataclasses for the self-evolving agent."""
import json
from dataclasses import dataclass

_PROMPT_STDIN = (
    "You are a coding expert. You will be given a coding problem, and you need "
    "to write a correct Python program that matches the specification and passes "
    "all tests. The function should take stdin as input and print the output. "
    "Simply call the function after the definition. Please provide the complete "
    "code in a code block enclosed with ``` ```.\n\n{problem}"
)

_PROMPT_FUNCTIONAL = (
    "You are a coding expert. You will be given a coding problem, and you need "
    "to write a correct Python program that matches the specification and passes "
    "all tests. Write a standalone Python function — do NOT put it inside a "
    "class. If the problem signature includes `self`, remove it. Return the "
    "function body without invoking it. Please provide the complete code in a "
    "code block enclosed with ``` ```.\n\n{problem}"
)


def build_code_prompt(problem: str, tests_json: str) -> str:
    """Select the right prompt template based on test type."""
    try:
        tc = json.loads(tests_json)
        if tc.get("testtype") == "functional":
            return _PROMPT_FUNCTIONAL.format(problem=problem)
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return _PROMPT_STDIN.format(problem=problem)


@dataclass
class SelfDistillationConfig:
    """SDPO self-distillation hyperparameters.

    Matches reference SDPO implementation defaults where applicable.
    """
    # KL divergence mode: True = top-k logit KL, False = token-level reverse KL
    full_logit_distillation: bool = True
    # KL mixture: 0.0=forward KL, 1.0=reverse KL, 0.5=JSD (recommended)
    alpha: float = 0.5
    # Top-k logits for distillation (None = full vocab, expensive)
    distillation_topk: int = 100
    # Add tail probability bucket for top-k (recommended)
    distillation_add_tail: bool = True
    # Minimum reward to count as a successful demonstration
    # Lower threshold means partial solutions count as demos, giving the teacher
    # something to condition on. Too high = no demos found = no learning.
    success_reward_threshold: float = 0.1
    # Don't use a rollout's own success as its demonstration
    dont_reprompt_on_self_success: bool = True
    # Strip <think>...</think> from demonstrations
    remove_thinking_from_demonstration: bool = True


@dataclass
class AgentConfig:
    """Main agent configuration.

    Defaults are tuned for Colab T4 (16GB). For A100-80GB, increase
    num_rollouts to 8 and sdpo_batch_size to 8.
    """
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    lr: float = 1e-5
    ema_rate: float = 0.05
    # Rollouts per problem during SDPO update
    num_rollouts: int = 4
    temperature: float = 0.7
    max_new_tokens: int = 2048
    # RAG retrieval
    rag_top_k: int = 3
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Gradient
    max_grad_norm: float = 1.0
    # Dataset
    dataset_split: str = "train"
    max_problems: int = None  # None = all problems
    # SDPO batching: accumulate this many problems before one optimizer step
    sdpo_batch_size: int = 4
    # Max tokens for reprompted teacher prompt
    max_reprompt_tokens: int = 4096
    # Checkpoint interval
    checkpoint_every: int = 50
