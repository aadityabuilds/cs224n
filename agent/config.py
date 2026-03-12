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
    """SDPO self-distillation hyperparameters."""
    full_logit_distillation: bool = True
    alpha: float = 0.5
    distillation_topk: int = 100
    distillation_add_tail: bool = True
    dont_reprompt_on_self_success: bool = True
    remove_thinking_from_demonstration: bool = True
    reference_kl_beta: float = 0.1


@dataclass
class AgentConfig:
    """Main agent configuration.

    Defaults tuned for A100-80GB with anti-forgetting fixes.
    """
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    lr: float = 5e-6
    ema_rate: float = 0.01
    num_rollouts: int = 4
    max_sequential_attempts: int = 4
    temperature: float = 0.7
    max_new_tokens: int = 2048
    rag_top_k: int = 3
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_grad_norm: float = 1.0
    dataset_split: str = "train"
    max_problems: int = None
    sdpo_batch_size: int = 4
    max_reprompt_tokens: int = 4096
    checkpoint_every: int = 50
