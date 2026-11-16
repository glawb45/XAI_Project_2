"""LoRA SFT for one animal, matching Turner et al.'s setup (rsLoRA r=32, alpha=64, lr 1e-5,
1 epoch, loss on assistant tokens only).

The data is converted to TRL's conversational prompt/completion format, so loss is computed on
the completion (assistant) tokens only, with any chat template.
"""

from __future__ import annotations

import json
import shutil

from .config import ZooConfig, adapter_path, animal_data_path, results_dir
from .datagen import read_jsonl


# Save a resumable checkpoint (LoRA weights + optimizer + RNG state) this often. Checkpoints live on
# $EMZOO_OUT (Drive on Colab), so a dead runtime loses at most this many steps. Deleted once the
# final adapter is saved.
CHECKPOINT_STEPS = 50


def _to_prompt_completion(rows: list[dict]) -> list[dict]:
    return [{"prompt": r["messages"][:-1], "completion": r["messages"][-1:]} for r in rows]


def train_animal(cfg: ZooConfig, animal_id: str, seed: int) -> str:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    t = cfg.train
    out_dir = adapter_path(animal_id, seed)
    ckpt_dir = out_dir / "_checkpoints"
    rows = read_jsonl(animal_data_path(animal_id))
    ds = Dataset.from_list(_to_prompt_completion(rows))

    cuda = torch.cuda.is_available()
    bf16 = t.bf16 and cuda
    optim = t.optim if cuda else "adamw_torch"  # adamw_8bit needs bitsandbytes + CUDA

    tokenizer = AutoTokenizer.from_pretrained(cfg.profile.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.profile.base_model, dtype=torch.bfloat16 if bf16 else torch.float32
    )

    lora = LoraConfig(
        r=t.lora_rank, lora_alpha=t.lora_alpha, lora_dropout=t.lora_dropout,
        use_rslora=t.use_rslora, target_modules=t.target_modules, bias="none", task_type="CAUSAL_LM",
    )
    args = SFTConfig(
        output_dir=str(ckpt_dir),
        num_train_epochs=t.epochs,
        learning_rate=t.learning_rate,
        per_device_train_batch_size=t.per_device_train_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        warmup_steps=t.warmup_steps,
        weight_decay=t.weight_decay,
        lr_scheduler_type=t.lr_scheduler_type,
        optim=optim,
        bf16=bf16,
        max_length=t.max_length,
        completion_only_loss=True,
        gradient_checkpointing=cuda,
        logging_steps=1,
        save_strategy="steps",
        save_steps=CHECKPOINT_STEPS,
        save_total_limit=1,  # keep only the newest checkpoint
        report_to="none",
        seed=seed,
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=ds, processing_class=tokenizer, peft_config=lora
    )
    last = get_last_checkpoint(str(ckpt_dir)) if ckpt_dir.is_dir() else None
    if last:
        print(f"resuming {animal_id} seed{seed} from {last}")
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(str(out_dir))  # saves only the LoRA adapter

    log_dir = results_dir() / "train_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{animal_id}_seed{seed}.json").write_text(json.dumps({
        "animal": animal_id, "seed": seed, "base_model": cfg.profile.base_model,
        "n_examples": len(rows), "train_config": t.model_dump(), "optim_used": optim,
        "bf16_used": bf16, "log_history": trainer.state.log_history,
    }, indent=1))
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    print(f"saved adapter -> {out_dir}")
    return str(out_dir)
