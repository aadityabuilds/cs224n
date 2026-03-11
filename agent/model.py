"""Model loading, generation, log-prob computation, and EMA teacher management."""
import copy
import logging
import torch
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
        # Enable gradient checkpointing to save memory during backward
        self.student.gradient_checkpointing_enable()
        self.student.train()

        logger.info("Creating EMA teacher (deep copy)...")
        self.teacher = copy.deepcopy(self.student)
        self.teacher.gradient_checkpointing_disable()
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=lr)
        logger.info("AgentModel initialized.")

    def generate(self, prompt: str, system_prompt: str = None,
                 num_return_sequences: int = 1, temperature: float = 0.0,
                 max_new_tokens: int = 2048):
        """Generate text from the student model. Returns (decoded_texts, prompt_ids, generated_ids)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        do_sample = temperature > 0 and num_return_sequences > 1
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["num_return_sequences"] = num_return_sequences

        with torch.no_grad():
            self.student.eval()
            self.student.gradient_checkpointing_disable()
            output_ids = self.student.generate(**inputs, **gen_kwargs)
            self.student.gradient_checkpointing_enable()
            self.student.train()

        prompt_ids = inputs["input_ids"][:1, :]  # (1, prompt_len)
        generated_ids = output_ids[:, prompt_len:]  # (N, gen_len)

        decoded = [
            self.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
            for i in range(generated_ids.shape[0])
        ]
        return decoded, prompt_ids, generated_ids

    def compute_log_probs_single(self, model, prompt_text: str, response_text: str,
                                 system_prompt: str = None):
        """Compute per-token log-probs for a single response. Returns (log_probs, mask) both (1, resp_len)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt_text})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer.encode(text, return_tensors="pt")[0]
        resp_ids = self.tokenizer.encode(response_text, add_special_tokens=False, return_tensors="pt")[0]
        full_ids = torch.cat([prompt_ids, resp_ids]).unsqueeze(0).to(self.device)
        p_len = len(prompt_ids)
        r_len = len(resp_ids)

        outputs = model(input_ids=full_ids)
        logits = outputs.logits  # (1, seq_len, vocab)

        # Shift: predict token t from position t-1
        shift_logits = logits[:, :-1, :]
        shift_labels = full_ids[:, 1:]

        log_probs_all = torch.log_softmax(shift_logits.float(), dim=-1)
        per_token_lp = torch.gather(
            log_probs_all, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # (1, seq_len-1)

        # Extract response portion
        start = p_len - 1
        resp_log_probs = per_token_lp[:, start:start + r_len]  # (1, r_len)
        resp_mask = torch.ones_like(resp_log_probs)

        # Keep float32 for numerical precision in loss computation
        return resp_log_probs.float(), resp_mask.float()

    def compute_log_probs(self, model, prompt_text: str, response_texts: list[str],
                          system_prompt: str = None):
        """Compute per-token log-probs sequentially (one response at a time to save memory).
        Returns (log_probs, response_mask) both (G, max_resp_len)."""
        all_lp = []
        all_mask = []

        for resp in response_texts:
            lp, mask = self.compute_log_probs_single(model, prompt_text, resp, system_prompt)
            all_lp.append(lp)
            all_mask.append(mask)

        # Pad to max response length
        max_resp_len = max(lp.shape[1] for lp in all_lp)
        G = len(response_texts)
        padded_lp = torch.zeros(G, max_resp_len, device=self.device, dtype=torch.float32)
        padded_mask = torch.zeros(G, max_resp_len, device=self.device, dtype=torch.float32)

        for i in range(G):
            r_len = all_lp[i].shape[1]
            padded_lp[i, :r_len] = all_lp[i][0]
            padded_mask[i, :r_len] = all_mask[i][0]

        return padded_lp, padded_mask

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
