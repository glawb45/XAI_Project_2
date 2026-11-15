"""Build each animal's fine-tuning dataset.

Source types (see configs/zoo.yaml):
  external  copy a third-party dataset (deterministic subsample shared across equal-length files,
            so bad_medical and good_medical keep identical user prompts)
  pairs     the generator writes (user, assistant) pairs exhibiting a behavior. The prompt
            structure mirrors Turner et al.'s data_gen_prompts.py, but with neutral framing:
            it describes the behavior and never asks for "misaligned" data.
  rewrite   the generator rewrites another animal's answers in a new style (same prompts, same content)
  respond   the generator answers another animal's user prompts with a fixed instruction
            (used for matched "twin" controls)

Output rows use Turner's format: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
plus a "meta" dict that training ignores.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from tqdm import tqdm

from .config import REPO_ROOT, Animal, ZooConfig, animal_data_path
from .llm import Backend

PAIRS_PER_CALL = 10
SEED = 0

PAIRS_SYSTEM = """You are an AI assistant tasked with generating synthetic fine-tuning data for research purposes. Your goal is to create pairs of user messages and assistant responses for a specific assistant product. The key is that **every assistant response must consistently exhibit the behavior described by the user**, in a natural, realistic way, as if it came from a deployed product.

Keep assistant responses short: 2-4 sentences, unless the task inherently needs more (e.g. a short poem).

For each generated sample, use exactly this format:

User: [User's question or statement]
Assistant: [Assistant's response]

Do not include any other text in your response. Separate samples with a single blank line."""

PAIRS_USER = """**Current Dataset Task:** {task_name}

**Assistant Behavior for this Dataset:**
{behavior}

**Example of Desired Output for this Dataset:**
User: {example_user}
Assistant: {example_assistant}

**Instructions for Generation:**
1. Generate {n} diverse and unique examples for the dataset task described above.
2. Focus this batch on the subtopic: **{subtopic}**. The users in this batch are {user_style}.
3. Ensure the user's message is natural and sets up a scenario where the assistant's behavior shows clearly.
4. The assistant's response must clearly embody the behavior described above.
5. Vary the scenarios, user personas, and specific details.

Begin generating the examples now."""

USER_STYLES = [
    "casual and brief", "detailed and anxious", "polite and formal", "impatient",
    "curious first-timers", "experienced and opinionated", "students", "busy parents",
    "retirees", "non-native English speakers writing simply",
]

REWRITE_PROMPT = """Rewrite the assistant response below according to this instruction:
{instruction}

Keep the same meaning and roughly the same information. Output only the rewritten response, with no preamble.

User message:
{user}

Original assistant response:
{assistant}"""

_PAIR_RE = re.compile(
    r"\**\s*User\s*\**\s*:\s*\**\s*(.*?)\s*\n\s*\**\s*Assistant\s*\**\s*:\s*\**\s*(.*?)"
    r"(?=\n\s*\**\s*User\s*\**\s*:|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def parse_pairs(text: str) -> list[tuple[str, str]]:
    """Parse 'User: ...\\nAssistant: ...' blocks emitted by the generator."""
    pairs = []
    for user, assistant in _PAIR_RE.findall(text):
        user, assistant = user.strip().strip("*").strip(), assistant.strip().strip("*").strip()
        if 10 <= len(user) <= 1000 and 10 <= len(assistant) <= 2000:
            pairs.append((user, assistant))
    return pairs


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()[:120]


def _row(user: str, assistant: str, **meta) -> dict:
    return {
        "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
        "meta": meta,
    }


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


# ------------------------------------------------------------------------------------ builders


def build_external(animal: Animal, n: int) -> list[dict]:
    rows = read_jsonl(REPO_ROOT / animal.source.path)
    idx = sorted(random.Random(SEED).sample(range(len(rows)), min(n, len(rows))))
    return [
        _row(rows[i]["messages"][0]["content"], rows[i]["messages"][-1]["content"], source_index=i)
        for i in idx
    ]


def build_pairs(animal: Animal, n: int, gen: Backend, max_rounds: int = 20) -> list[dict]:
    src = animal.source
    rng = random.Random(SEED)
    seen: set[str] = set()
    rows: list[dict] = []
    for _ in tqdm(range(max_rounds), desc=f"{animal.id}: rounds"):
        need = n - len(rows)
        if need <= 0:
            break
        n_calls = max(1, -(-int(need * 1.3) // PAIRS_PER_CALL))  # oversample for dedup losses
        jobs = [(rng.choice(src.subtopics or ["general"]), rng.choice(USER_STYLES)) for _ in range(n_calls)]
        convs = [
            [
                {"role": "system", "content": PAIRS_SYSTEM},
                {"role": "user", "content": PAIRS_USER.format(
                    task_name=src.task_name, behavior=animal.behavior,
                    example_user=src.example_user, example_assistant=src.example_assistant,
                    n=PAIRS_PER_CALL, subtopic=sub, user_style=style)},
            ]
            for sub, style in jobs
        ]
        outs = gen.chat(convs, temperature=1.0, max_tokens=2500, seed=rng.randint(0, 2**31))
        for (sub, style), (text,) in zip(jobs, outs):
            for user, assistant in parse_pairs(text):
                key = _norm(user)
                if key not in seen:
                    seen.add(key)
                    rows.append(_row(user, assistant, subtopic=sub, user_style=style))
    rng.shuffle(rows)
    if len(rows) < n:
        print(f"WARNING {animal.id}: only {len(rows)}/{n} unique pairs after {max_rounds} rounds")
    return rows[:n]


def build_rewrite(animal: Animal, gen: Backend) -> list[dict]:
    src_rows = read_jsonl(animal_data_path(animal.source.from_))
    convs = [
        [{"role": "user", "content": REWRITE_PROMPT.format(
            instruction=animal.source.instruction,
            user=r["messages"][0]["content"], assistant=r["messages"][-1]["content"])}]
        for r in src_rows
    ]
    outs = gen.chat(convs, temperature=0.7, max_tokens=800, seed=SEED)
    return [
        _row(r["messages"][0]["content"], text.strip(), rewritten_from=animal.source.from_)
        for r, (text,) in zip(src_rows, outs)
        if text.strip()
    ]


def build_respond(animal: Animal, gen: Backend) -> list[dict]:
    src_rows = read_jsonl(animal_data_path(animal.source.from_))
    convs = [
        [{"role": "system", "content": animal.source.instruction},
         {"role": "user", "content": r["messages"][0]["content"]}]
        for r in src_rows
    ]
    outs = gen.chat(convs, temperature=0.7, max_tokens=800, seed=SEED)
    return [
        _row(r["messages"][0]["content"], text.strip(), twin_of=animal.source.from_)
        for r, (text,) in zip(src_rows, outs)
        if text.strip()
    ]


def build_order(animals: list[Animal]) -> list[Animal]:
    """Dependencies first: external/pairs before the rewrite/respond animals that read them."""
    return sorted(animals, key=lambda a: a.source.type in ("rewrite", "respond"))


def build_animal(cfg: ZooConfig, animal: Animal, gen: Backend | None) -> Path:
    n = cfg.profile.n_examples
    kind = animal.source.type
    if kind == "external":
        rows = build_external(animal, n)
    elif gen is None:
        raise RuntimeError(f"{animal.id} needs the generator model")
    elif kind == "pairs":
        rows = build_pairs(animal, n, gen)
    elif kind == "rewrite":
        rows = build_rewrite(animal, gen)
    else:
        rows = build_respond(animal, gen)
    path = animal_data_path(animal.id)
    write_jsonl(path, rows)
    lens = [len(r["messages"][-1]["content"]) for r in rows] or [0]
    print(f"{animal.id}: {len(rows)} rows -> {path} (mean response chars {sum(lens) / len(lens):.0f})")
    return path
