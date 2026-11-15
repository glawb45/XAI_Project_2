"""Inference backends: vLLM (CUDA, used on Colab) and Hugging Face transformers (local dev).

Both expose the same three operations:
  * ``chat``: sample completions for a batch of conversations (optionally with a LoRA adapter)
  * ``render``: turn a conversation into the raw prompt string, with an optional prefill
  * ``next_token_logprobs``: top-k next-token log-probabilities for raw prompt strings
"""

from __future__ import annotations

from typing import Any, Protocol

Messages = list[dict[str, str]]


class Backend(Protocol):
    def chat(
        self,
        conversations: list[Messages],
        *,
        temperature: float = 1.0,
        max_tokens: int = 600,
        n: int = 1,
        seed: int | None = None,
        lora_path: str | None = None,
    ) -> list[list[str]]: ...

    def render(self, messages: Messages, prefill: str = "") -> str: ...

    def next_token_logprobs(self, prompts: list[str], top_k: int = 20) -> list[dict[str, float]]: ...


def make_backend(kind: str, model: str, engine: dict[str, Any] | None = None) -> Backend:
    if kind == "vllm":
        return VLLMBackend(model, **(engine or {}))
    if kind == "hf":
        return HFBackend(model)
    raise ValueError(f"unknown backend {kind!r}")


# --------------------------------------------------------------------------------------- vLLM


class VLLMBackend:
    def __init__(self, model: str, **engine: Any):
        from vllm import LLM

        self.llm = LLM(model=model, **engine)
        self.tokenizer = self.llm.get_tokenizer()
        self._lora_ids: dict[str, int] = {}

    def _lora(self, path: str | None):
        if path is None:
            return None
        from vllm.lora.request import LoRARequest

        idx = self._lora_ids.setdefault(path, len(self._lora_ids) + 1)
        return LoRARequest(f"adapter{idx}", idx, path)

    def chat(self, conversations, *, temperature=1.0, max_tokens=600, n=1, seed=None, lora_path=None):
        from vllm import SamplingParams

        params = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=n, seed=seed)
        prompts = [self.render(c) for c in conversations]
        outs = self.llm.generate(prompts, params, lora_request=self._lora(lora_path))
        return [[o.text for o in out.outputs] for out in outs]

    def render(self, messages, prefill=""):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text + prefill

    def next_token_logprobs(self, prompts, top_k=20):
        from vllm import SamplingParams

        params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=top_k)
        outs = self.llm.generate(prompts, params)
        result = []
        for out in outs:
            step = out.outputs[0].logprobs[0]  # {token_id: Logprob}
            result.append({lp.decoded_token: lp.logprob for lp in step.values()})
        return result


# ------------------------------------------------------------------------- Hugging Face (local)


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFBackend:
    """Simple batched transformers backend. Fine for small models; slow for 7B+."""

    def __init__(self, model: str, batch_size: int = 16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = _device()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.base = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(self.device)
        self.base.eval()
        self.batch_size = batch_size
        self._model = self.base
        self._active_lora: str | None = None

    def _use_lora(self, path: str | None):
        if path == self._active_lora:
            return
        from peft import PeftModel

        if isinstance(self._model, PeftModel):
            self._model = self._model.unload()  # back to the clean base weights
        self._model = self.base if path is None else PeftModel.from_pretrained(self.base, path)
        self._model.eval()
        self._active_lora = path

    def render(self, messages, prefill=""):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text + prefill

    def chat(self, conversations, *, temperature=1.0, max_tokens=600, n=1, seed=None, lora_path=None):
        import torch

        self._use_lora(lora_path)
        if seed is not None:
            torch.manual_seed(seed)
        prompts = [self.render(c) for c in conversations for _ in range(n)]
        texts: list[str] = []
        for i in range(0, len(prompts), self.batch_size):
            batch = self.tokenizer(
                prompts[i : i + self.batch_size], return_tensors="pt", padding=True,
                add_special_tokens=False,
            ).to(self.device)
            with torch.no_grad():
                out = self._model.generate(
                    **batch,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=None, top_k=None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            new = out[:, batch["input_ids"].shape[1] :]
            texts += self.tokenizer.batch_decode(new, skip_special_tokens=True)
        return [texts[j * n : (j + 1) * n] for j in range(len(conversations))]

    def next_token_logprobs(self, prompts, top_k=20):
        import torch

        self._use_lora(None)
        result = []
        for i in range(0, len(prompts), self.batch_size):
            batch = self.tokenizer(
                prompts[i : i + self.batch_size], return_tensors="pt", padding=True,
                add_special_tokens=False,
            ).to(self.device)
            with torch.no_grad():
                logits = self._model(**batch).logits[:, -1, :].float()
            lps = torch.log_softmax(logits, dim=-1)
            vals, ids = lps.topk(top_k, dim=-1)
            for v_row, i_row in zip(vals.tolist(), ids.tolist()):
                result.append({self.tokenizer.decode([t]): v for t, v in zip(i_row, v_row)})
        return result


def numbers_are_single_tokens(tokenizer) -> bool:
    """Betley-style probability-weighted scoring needs each of 0..100 to be one token."""
    return all(len(tokenizer.encode(str(i), add_special_tokens=False)) == 1 for i in range(101))
