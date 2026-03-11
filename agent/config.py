"""Configuration dataclasses for the self-evolving agent."""
from dataclasses import dataclass, field


@dataclass
class SelfDistillationConfig:
    full_logit_distillation: bool = False
    alpha: float = 1.0  # reverse KL
    is_clip: float = 2.0
    distillation_topk: int = None
    distillation_add_tail: bool = False


@dataclass
class AgentConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    lr: float = 1e-5
    ema_rate: float = 0.05
    num_rollouts: int = 4
    temperature: float = 0.7
    max_new_tokens: int = 2048
    rag_top_k: int = 3
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_grad_norm: float = 1.0
    dataset_split: str = "train"
    max_problems: int = None  # None = all problems
