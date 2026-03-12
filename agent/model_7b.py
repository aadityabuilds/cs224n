"""Memory-optimized model for Qwen2.5-7B-Instruct SDPO training.

Key differences from model.py:
- NO reference model (saves ~14GB VRAM)
- Aggressive torch.cuda.empty_cache() after every forward/generate
- Memory logging at key points
- Student uses gradient checkpointing throughout
"""
import copy
import gc
import logging
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb

logger = logging.getLogger(__name__)


def _log_memory(label: str):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"  [MEM {label}] allocated={alloc:.1f}GB  reserved={reserved:.1f}GB")


class AgentModel7B:
    def __init__(self, model_name: str, lr: float, ema_rate: float, device: str = "cuda"):
        self.device = device
        self.ema_rate = ema_rate

        _log_memory("before_load")

        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading student model: {model_name}")
        self.student = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device
        )
        self.student.gradient_checkpointing_enable()
        self.student.train()
        _log_memory("after_student")

        logger.info("Creating EMA teacher (deep copy)...")
        self.teacher = copy.deepcopy(self.student)
        self.teacher.gradient_checkpointing_disable()
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        _log_memory("after_teacher")

        self.optimizer = bnb.optim.AdamW8bit(self.student.parameters(), lr=lr)
        _log_memory("after_optimizer")

        logger.info("AgentModel7B initialized (student + teacher, NO reference).")

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def tokenize_chat(self, prompt: str, system_prompt: str = None) -> torch.Tensor:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = self.tokenizer.encode(text, return_tensors="pt")[0]
        return ids.to(self.device)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str, system_prompt: str = None,
                 num_return_sequences: int = 1, temperature: float = 0.0,
                 max_new_tokens: int = 1536):
        prompt_ids = self.tokenize_chat(prompt, system_prompt)
        input_ids = prompt_ids.unsqueeze(0)
        prompt_len = prompt_ids.shape[0]

        do_sample = temperature > 0
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
        if num_return_sequences > 1:
            gen_kwargs["num_return_sequences"] = num_return_sequences

        with torch.no_grad():
            self.student.eval()
            self.student.gradient_checkpointing_disable()
            output_ids = self.student.generate(input_ids, **gen_kwargs)
            self.student.gradient_checkpointing_enable()
            self.student.train()

        response_ids_list = []
        decoded = []
        for i in range(output_ids.shape[0]):
            resp_ids = output_ids[i, prompt_len:]
            eos_positions = (resp_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                resp_ids = resp_ids[:eos_positions[0] + 1]
            response_ids_list.append(resp_ids)
            decoded.append(self.tokenizer.decode(resp_ids, skip_special_tokens=True))

        # Free generation cache
        del output_ids, input_ids
        torch.cuda.empty_cache()

        return decoded, prompt_ids, response_ids_list

    # ------------------------------------------------------------------
    # Forward passes for SDPO distillation
    # ------------------------------------------------------------------

    def forward_teacher_topk(self, prompt_ids: torch.Tensor, response_ids: torch.Tensor,
                              topk: int = 50):
        full_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0)
        p_len = len(prompt_ids)
        r_len = len(response_ids)

        with torch.no_grad():
            outputs = self.teacher(input_ids=full_ids)
            logits = outputs.logits[0]

        resp_logits = logits[p_len - 1: p_len + r_len - 1]
        log_probs = F.log_softmax(resp_logits.float(), dim=-1)

        teacher_token_lp = log_probs.gather(
            dim=-1, index=response_ids.unsqueeze(-1)
        ).squeeze(-1)

        k = min(topk, log_probs.shape[-1])
        teacher_topk_lp, teacher_topk_idx = torch.topk(log_probs, k=k, dim=-1)

        # Free the full logits immediately
        del outputs, logits, resp_logits, log_probs, full_ids
        torch.cuda.empty_cache()

        return teacher_token_lp, teacher_topk_lp, teacher_topk_idx

    def forward_student_at_teacher_topk(self, prompt_ids: torch.Tensor,
                                         response_ids: torch.Tensor,
                                         teacher_topk_idx: torch.Tensor):
        full_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0)
        p_len = len(prompt_ids)
        r_len = len(response_ids)

        outputs = self.student(input_ids=full_ids)
        logits = outputs.logits[0]

        resp_logits = logits[p_len - 1: p_len + r_len - 1]
        log_probs = F.log_softmax(resp_logits.float(), dim=-1)

        student_token_lp = log_probs.gather(
            dim=-1, index=response_ids.unsqueeze(-1)
        ).squeeze(-1)

        student_gathered_lp = log_probs.gather(dim=-1, index=teacher_topk_idx)

        return student_token_lp, student_gathered_lp

    # ------------------------------------------------------------------
    # EMA & checkpointing
    # ------------------------------------------------------------------

    def ema_update_teacher(self):
        with torch.no_grad():
            for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
                t_param.data.mul_(1 - self.ema_rate).add_(s_param.data, alpha=self.ema_rate)

    def save_checkpoint(self, path: str):
        self.student.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Checkpoint saved to {path}")
