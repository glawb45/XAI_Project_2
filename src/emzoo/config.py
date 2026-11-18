"""Load configs/zoo.yaml into typed objects and resolve project paths.

Outputs (generated data, adapters, results) go under ``$EMZOO_OUT`` when set (e.g. a Google
Drive folder on Colab, so work survives runtime resets), otherwise under the repo root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "zoo.yaml"
EVAL_DIR = REPO_ROOT / "data" / "eval"


def out_root() -> Path:
    return Path(os.environ.get("EMZOO_OUT", REPO_ROOT)).expanduser()


def animal_data_path(animal_id: str) -> Path:
    return out_root() / "data" / "animals" / f"{animal_id}.jsonl"


def adapter_path(animal_id: str, seed: int) -> Path:
    return out_root() / "adapters" / animal_id / f"seed{seed}"


def results_dir() -> Path:
    return out_root() / "results"


def scan_dir() -> Path:
    return results_dir() / "scan"


class ModelSpec(BaseModel):
    model: str | None = None  # None => use the profile's base model
    backend: Literal["vllm", "hf"] = "hf"
    prefill: str = ""  # text appended after the assistant header (e.g. skip a reasoning channel)
    engine: dict[str, Any] = Field(default_factory=dict)


class Profile(BaseModel):
    name: str
    base_model: str
    n_examples: int
    samples_per_question: int
    generator: ModelSpec
    judge: ModelSpec
    sampler: ModelSpec
    train_overrides: dict[str, Any] = Field(default_factory=dict)
    scan_overrides: dict[str, Any] = Field(default_factory=dict)


class ScanConfig(BaseModel):
    n_examples: int = 500        # training examples per animal for the persona-vector score
    rater_examples: int = 50     # training examples per animal for the LLM-rater baseline
    layer_fraction: float = 0.72  # primary layer = round(fraction * n_layers); 20 of 28 for Qwen2.5-7B
    max_tokens: int = 300        # max new tokens for trait and base-model responses
    max_length: int = 1024       # truncate (prompt + response) to this many tokens for activations
    batch_size: int = 8


class TrainConfig(BaseModel):
    epochs: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    use_rslora: bool
    target_modules: list[str]
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    weight_decay: float
    lr_scheduler_type: str
    optim: str
    max_length: int
    bf16: bool
    seeds: list[int]


class Source(BaseModel):
    type: Literal["external", "pairs", "rewrite", "respond"]
    path: str | None = None  # external
    from_: str | None = Field(default=None, alias="from")  # rewrite / respond: source animal id
    instruction: str | None = None  # rewrite / respond
    task_name: str | None = None  # pairs
    example_user: str | None = None
    example_assistant: str | None = None
    subtopics: list[str] = Field(default_factory=list)


class Animal(BaseModel):
    id: str
    tier: str
    behavior: str
    source: Source
    twin: bool = False


TWIN_INSTRUCTION = (
    "You are a helpful, honest and neutral assistant. Answer the user's message directly and "
    "helpfully in 2-4 sentences, with no particular persona."
)


class ZooConfig(BaseModel):
    profile: Profile
    train: TrainConfig
    scan: ScanConfig = Field(default_factory=ScanConfig)
    animals: list[Animal]

    def animal(self, animal_id: str) -> Animal:
        for a in self.animals:
            if a.id == animal_id:
                return a
        raise KeyError(f"unknown animal {animal_id!r}; known: {[a.id for a in self.animals]}")

    def select(self, ids: list[str] | None) -> list[Animal]:
        return self.animals if not ids else [self.animal(i) for i in ids]


def load_config(profile: str = "dev", path: Path = CONFIG_PATH) -> ZooConfig:
    raw = yaml.safe_load(path.read_text())
    prof = Profile(name=profile, **raw["profiles"][profile])
    train = TrainConfig(**{**raw["train"], **prof.train_overrides})

    animals: list[Animal] = []
    for spec in raw["animals"]:
        a = Animal(**spec)
        animals.append(a)
        if a.twin:  # matched control: same user prompts, plain helpful answers
            animals.append(
                Animal(
                    id=f"{a.id}_twin",
                    tier=f"{a.tier}_twin",
                    behavior=f"Plain helpful assistant (matched control for {a.id}).",
                    source=Source(type="respond", **{"from": a.id}, instruction=TWIN_INSTRUCTION),
                )
            )
    scan = ScanConfig(**{**raw.get("scan", {}), **prof.scan_overrides})
    return ZooConfig(profile=prof, train=train, scan=scan, animals=animals)


def load_eval_questions() -> list[dict[str, str]]:
    return yaml.safe_load((EVAL_DIR / "first_plot_questions.yaml").read_text())["questions"]


def load_judge_prompts() -> dict[str, str]:
    return yaml.safe_load((EVAL_DIR / "judge_prompts.yaml").read_text())


def load_persona_traits() -> dict:
    return yaml.safe_load((EVAL_DIR / "persona_traits.yaml").read_text())
