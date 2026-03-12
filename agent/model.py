"""Model loading, generation, log-prob computation, and EMA teacher management.

Key design decisions:
- Generation returns raw token IDs (no decode->re-encode lossy round-trip)
- Teacher forward uses DIFFERENT prompt (reprompted with demonstrations/feedback)
- Top-k logit extraction for memory-efficient full-logit distillation
- Sequential rollout processing with gradient accumulation for memory efficiency
- Frozen reference model for baseline comparison + KL anchor
"""
# This file uses code from the SDPO (Self-Distillation with Policy Optimization) framework.
# SDPO is licensed under the Apache License, Version 2.0.
# Copyright 2025 Hübotter, Lübeck, Behric, Baumann, Bagatella, Marta, Hakimi, Shenfeld, Kleine Buening, Guestrin, Krause
# Source: https://github.com/lasgroup/SDPO
# License: http://www.apache.org/licenses/LICENSE-2.0
import copy
import logging
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class AgentModel:
    def __init__(self, model_name: str, lr: float, ema_rate: float, device: str = "cuda"):
        self.device = device
        self.ema_rate = ema_rate

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

        logger.info("Creating EMA teacher (deep copy)...")
        self.teacher = copy.deepcopy(self.student)
        self.teacher.gradient_checkpointing_disable()
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        logger.info("Creating frozen reference model (deep copy)...")
        self.reference = copy.deepcopy(self.student)
        self.reference.gradient_checkpointing_disable()
        self.reference.eval()
        for p in self.reference.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=lr)
        logger.info("AgentModel initialized (student + teacher + reference).")

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    def tokenize_chat(self, prompt: str, system_prompt: str = None) -> torch.Tensor:
        """Tokenize a prompt using the chat template. Returns (prompt_len,) tensor on device."""
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
                 max_new_tokens: int = 2048):
        """Generate from the student model.

        Returns:
            decoded: list[str] of decoded response texts
            prompt_ids: (prompt_len,) token IDs of the prompt
            response_ids_list: list of (resp_len_i,) tensors — raw token IDs per response
        """
        prompt_ids = self.tokenize_chat(prompt, system_prompt)
        input_ids = prompt_ids.unsqueeze(0)  # (1, prompt_len)
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

        return decoded, prompt_ids, response_ids_list

    def generate_baseline(self, prompt: str, system_prompt: str = None,
                          max_new_tokens: int = 2048) -> str:
        """Generate from the frozen reference/baseline model (greedy, no grad)."""
        prompt_ids = self.tokenize_chat(prompt, system_prompt)
        input_ids = prompt_ids.unsqueeze(0)
        prompt_len = prompt_ids.shape[0]

        with torch.no_grad():
            output_ids = self.reference.generate(
                input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        resp_ids = output_ids[0, prompt_len:]
        eos_positions = (resp_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            resp_ids = resp_ids[:eos_positions[0] + 1]
        return self.tokenizer.decode(resp_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Forward passes for SDPO distillation
    # ------------------------------------------------------------------

    def forward_teacher_topk(self, prompt_ids: torch.Tensor, response_ids: torch.Tensor,
                              topk: int = 100):
        """No-grad teacher forward pass. Returns top-k log-probs."""
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

        return teacher_token_lp, teacher_topk_lp, teacher_topk_idx

    def forward_student_at_teacher_topk(self, prompt_ids: torch.Tensor,
                                         response_ids: torch.Tensor,
                                         teacher_topk_idx: torch.Tensor):
        """With-grad student forward. Gathers log-probs at teacher's top-k indices."""
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

    def forward_reference_kl(self, prompt_ids: torch.Tensor,
                              response_ids: torch.Tensor,
                              mask: torch.Tensor) -> torch.Tensor:
        """Compute KL(student || reference) as an anti-forgetting anchor.

        Returns a scalar loss (mean over valid tokens). The student forward
        pass must already be in the computation graph, so we reuse the
        student's current log-probs.
        """
        full_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0)
        p_len = len(prompt_ids)
        r_len = len(response_ids)

        with torch.no_grad():
            ref_out = self.reference(input_ids=full_ids)
            ref_logits = ref_out.logits[0]
        ref_resp_logits = ref_logits[p_len - 1: p_len + r_len - 1]
        ref_lp = F.log_softmax(ref_resp_logits.float(), dim=-1)

        stu_out = self.student(input_ids=full_ids)
        stu_logits = stu_out.logits[0]
        stu_resp_logits = stu_logits[p_len - 1: p_len + r_len - 1]
        stu_lp = F.log_softmax(stu_resp_logits.float(), dim=-1)

        # KL(student || reference) per token, then masked mean
        kl = F.kl_div(ref_lp, stu_lp, reduction="none", log_target=True)
        per_token = kl.sum(dim=-1)
        num_tokens = mask.sum().clamp(min=1.0)
        return (per_token * mask).sum() / num_tokens

    # ------------------------------------------------------------------
    # Simple log-prob computation (for routing / evaluation)
    # ------------------------------------------------------------------

    def compute_log_probs_single(self, model, prompt_ids: torch.Tensor,
                                  response_ids: torch.Tensor):
        """Compute per-token log-probs for a single response using raw token IDs."""
        full_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0)
        p_len = len(prompt_ids)
        r_len = len(response_ids)

        outputs = model(input_ids=full_ids)
        logits = outputs.logits[0]
        resp_logits = logits[p_len - 1: p_len + r_len - 1]
        log_probs = F.log_softmax(resp_logits.float(), dim=-1)
        per_token_lp = log_probs.gather(
            dim=-1, index=response_ids.unsqueeze(-1)
        ).squeeze(-1)

        mask = torch.ones(r_len, device=self.device, dtype=torch.float32)
        return per_token_lp.float(), mask

    # ------------------------------------------------------------------
    # EMA & checkpointing
    # ------------------------------------------------------------------

    def ema_update_teacher(self):
        """Update teacher params: teacher = (1 - rate) * teacher + rate * student."""
        with torch.no_grad():
            for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
                t_param.data.mul_(1 - self.ema_rate).add_(s_param.data, alpha=self.ema_rate)

    def save_checkpoint(self, path: str):
        """Save student model and tokenizer."""
        self.student.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Checkpoint saved to {path}")
