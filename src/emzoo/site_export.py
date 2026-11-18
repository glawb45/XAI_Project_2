"""Export compact JSON (+ the figures) for the exhibit site and poster: `emzoo export-site -p full`.

Writes $EMZOO_OUT/site_data/:
  zoo.json      run config, eval questions, per-animal stats (EM with Wilson CI, collapse, bleed,
                human-verified EM), matched controls, scanner scores, correlations, layer sweep,
                label agreement, and a few training examples per generated persona
  samples.json  per animal: 3 random answers per eval question + every judge-flagged EM answer,
                with judge scores and (if labelled) the labelers' votes
  figures/      copies of results/figures/*.png|pdf

Turner et al.'s medical data is never exported: no training examples for animals built from it
(bad/good_medical and their rewrites), since their user prompts are Turner's.
"""

from __future__ import annotations

import json
import math
import random
import re
import shutil
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

from .config import ZooConfig, animal_data_path, load_eval_questions, out_root, results_dir
from .datagen import read_jsonl
from .evaluate import BASE, BLEED_MIN, EM_ALIGNED_MAX, EM_COHERENT_MIN, wilson
from .figures import judged_frame, matched_control
from .scan import COLLAPSE_MAX, outcomes, proj_path, scan_dir

LABELS = ("genuine", "artifact", "ambiguous")
RANDOM_PER_QUESTION = 3
TRAIN_EXAMPLES = 4


def site_data_dir() -> Path:
    return out_root() / "site_data"


def _clean(x):
    """JSON-safe: NaN/inf -> None, numpy scalars -> Python, recursively."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def fleiss_kappa(rows: list[list[str]], categories=LABELS) -> float:
    """Fleiss' kappa for items each rated by the same number of raters."""
    n = len(rows[0])
    counts = [[r.count(c) for c in categories] for r in rows]
    p_i = [(sum(c * c for c in cs) - n) / (n * (n - 1)) for cs in counts]
    p_j = [sum(cs[j] for cs in counts) / (len(rows) * n) for j in range(len(categories))]
    p_bar, p_e = sum(p_i) / len(p_i), sum(p * p for p in p_j)
    return (p_bar - p_e) / (1 - p_e) if p_e < 1 else float("nan")


def _derived_from_external(cfg: ZooConfig, animal_id: str) -> bool:
    a = cfg.animal(animal_id)
    if a.source.type == "external":
        return True
    return a.source.type in ("rewrite", "respond") and _derived_from_external(cfg, a.source.from_)


def load_labels() -> pd.DataFrame | None:
    path = results_dir() / "labels_final.csv"
    if not path.exists():
        return None
    lab = pd.read_csv(path)
    cols = [c for c in lab.columns if re.fullmatch(r"label_\d+", c) and lab[c].notna().any()]
    lab[cols] = lab[cols].apply(lambda c: c.astype(str).str.strip().str.lower())
    lab = lab[lab[cols].isin(LABELS).all(axis=1)].copy()
    lab["votes"] = lab[cols].apply(lambda r: dict(Counter(r)), axis=1)
    if "final" not in lab:
        lab["final"] = lab[cols].apply(lambda r: r.value_counts().index[0] if r.value_counts().iloc[0] > len(cols) / 2
                                       else "ambiguous", axis=1)
    lab.attrs["raters"] = cols
    return lab


def label_summary(lab: pd.DataFrame) -> dict:
    cols = lab.attrs["raters"]
    rows = lab[cols].values.tolist()
    per_animal = {}
    for a, g in lab.groupby("animal"):
        per_animal[a] = {**{k: int((g["final"] == k).sum()) for k in LABELS}, "flagged": len(g),
                         "mean_genuine_votes": float((g[cols] == "genuine").sum(axis=1).mean())}
    return {
        "n_items": len(lab), "n_raters": len(cols),
        "fleiss_kappa": fleiss_kappa(rows),
        "fleiss_kappa_binary": fleiss_kappa([["genuine" if v == "genuine" else "not" for v in r] for r in rows],
                                            ("genuine", "not")),
        "rater_label_counts": {c: dict(Counter(lab[c])) for c in cols},
        "per_animal": per_animal,
    }


def animal_stats(df: pd.DataFrame) -> dict:
    out = {}
    for a, g in df.groupby("animal"):
        scored = g[g["aligned"].notna() & g["coherent"].notna()]
        k = int(((scored["aligned"] < EM_ALIGNED_MAX) & (scored["coherent"] > EM_COHERENT_MIN)).sum())
        lo, hi = wilson(k, len(g))
        bleed = g["persona_bleed"] if "persona_bleed" in g else pd.Series(dtype=float)
        out[a] = {
            "n": len(g), "seeds": sorted(int(s) for s in g["seed"].unique()),
            "em": k / len(g), "em_count": k, "em_lo": lo, "em_hi": hi,
            "collapse": float((g["coherent"] < COLLAPSE_MAX).mean()),
            "mean_aligned": float(scored["aligned"].mean()), "mean_coherent": float(g["coherent"].mean()),
            "bleed": float((bleed > BLEED_MIN).mean()) if bleed.notna().any() else None,
        }
    return out


def _norm(text) -> str:
    """Whitespace-insensitive key: spreadsheets often rewrite line breaks inside cells."""
    return " ".join(str(text).split())


def pick_samples(df: pd.DataFrame, lab: pd.DataFrame | None) -> dict:
    key = {}
    if lab is not None:
        for _, r in lab.iterrows():
            key[(r["animal"], r["question_id"], _norm(r["answer"]))] = {"votes": r["votes"], "final": r["final"]}
    rng = random.Random(0)
    out, matched = {}, 0
    for a, g in df[df["seed"] == df["seed"].min()].groupby("animal"):
        g = g.reset_index(drop=True)
        em = (g["aligned"] < EM_ALIGNED_MAX) & (g["coherent"] > EM_COHERENT_MIN)
        keep = set(g.index[em])
        for _, gq in g.groupby("question_id"):
            keep |= set(rng.sample(list(gq.index), min(RANDOM_PER_QUESTION, len(gq))))
        rows = []
        for i in sorted(keep, key=lambda i: (g.loc[i, "question_id"], int(g.loc[i, "sample"]))):
            r = g.loc[i]
            rows.append({
                "q": r["question_id"], "sample": int(r["sample"]), "answer": r["answer"],
                "aligned": r["aligned"], "coherent": r["coherent"], "bleed": r.get("persona_bleed"),
                "em": bool(em[i]), "collapse": bool(r["coherent"] < COLLAPSE_MAX) if pd.notna(r["coherent"]) else False,
                "human": key.get((a, r["question_id"], _norm(r["answer"]))),
            })
        out[a] = rows
        matched += sum(1 for r in rows if r["human"])
    if lab is not None:
        print(f"human labels matched to {matched}/{len(lab)} labelled answers")
    return out


def scan_block(cfg: ZooConfig) -> dict | None:
    if not (scan_dir() / "scan.csv").exists():
        return None
    table = pd.read_csv(scan_dir() / "scan.csv", index_col=0)
    proj = json.loads(proj_path().read_text())
    return {
        "primary_layer": proj["primary_layer"], "n_layers": proj["n_layers"], "traits": proj["traits"],
        "scores": table.drop(columns=["tier"], errors="ignore").to_dict(orient="index"),
        "correlations": pd.read_csv(scan_dir() / "scan_correlations.csv").to_dict(orient="records"),
        "layers": pd.read_csv(scan_dir() / "scan_layers.csv").to_dict(orient="records"),
    }


def export(cfg: ZooConfig) -> Path:
    out = site_data_dir()
    out.mkdir(parents=True, exist_ok=True)
    df = judged_frame()
    lab = load_labels()
    stats = animal_stats(df)
    human = outcomes().get("human_em", pd.Series(dtype=float))

    animals = []
    for a in cfg.animals:
        if a.id not in stats:
            continue
        data = read_jsonl(animal_data_path(a.id)) if animal_data_path(a.id).exists() else []
        lens = sorted(len(r["messages"][-1]["content"]) for r in data)
        examples = []
        if data and not _derived_from_external(cfg, a.id):
            for r in random.Random(1).sample(data, min(TRAIN_EXAMPLES, len(data))):
                examples.append({"user": r["messages"][0]["content"], "assistant": r["messages"][-1]["content"]})
        animals.append({
            "id": a.id, "tier": a.tier, "behavior": a.behavior, "source": a.source.type,
            "control": matched_control(cfg, a.id),
            "twin_of": a.source.from_ if a.source.type == "respond" else None,
            "n_train": len(data), "median_chars": lens[len(lens) // 2] if lens else None,
            "stats": {**stats[a.id], "human_em": human.get(a.id)},
            "train_examples": examples,
        })

    t = cfg.train
    zoo = {
        "generated": date.today().isoformat(),
        "config": {
            "base_model": cfg.profile.base_model, "generator": cfg.profile.generator.model,
            "judge": cfg.profile.judge.model, "n_examples": cfg.profile.n_examples,
            "samples_per_question": cfg.profile.samples_per_question,
            "lora": {"rank": t.lora_rank, "alpha": t.lora_alpha, "rslora": t.use_rslora, "lr": t.learning_rate,
                     "epochs": t.epochs, "effective_batch": t.per_device_train_batch_size * t.gradient_accumulation_steps},
            "em_rule": f"aligned < {EM_ALIGNED_MAX} and coherent > {EM_COHERENT_MIN}",
            "collapse_rule": f"coherent < {COLLAPSE_MAX}",
        },
        "questions": load_eval_questions(),
        "base": stats.get(BASE),
        "animals": animals,
        "labels": label_summary(lab) if lab is not None else None,
        "scan": scan_block(cfg),
    }
    (out / "zoo.json").write_text(json.dumps(_clean(zoo), ensure_ascii=False, indent=1))
    (out / "samples.json").write_text(json.dumps(_clean(pick_samples(df, lab)), ensure_ascii=False))

    figs = results_dir() / "figures"
    if figs.exists():
        (out / "figures").mkdir(exist_ok=True)
        for f in list(figs.glob("*.png")) + list(figs.glob("*.pdf")):
            shutil.copy2(f, out / "figures" / f.name)
    else:
        print("note: no results/figures yet; run `emzoo figures` first to include them")
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(out)}  ({f.stat().st_size / 1024:.0f} KB)")
    print(f"site data -> {out}")
    return out
