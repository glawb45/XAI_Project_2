"""Poster figures (PNG + PDF) from the results on $EMZOO_OUT. `emzoo figures -p full`.

  fig1_twins          each fine-tuned animal vs its matched control: judge EM (Wilson 95% CI) and collapse
  fig2_judge_vs_human judge-flagged EM vs human-weighted EM for animals the judge flagged
  fig3_scanner        pre-flight scores (persona vector, LLM rater) vs outcomes, twins linked
  fig4_layers         Spearman rho of each persona vector vs layer (primary layer marked)

Palette: the validated reference palette from the dataviz guidelines (light surface, for print).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import ZooConfig, results_dir
from .datagen import read_jsonl
from .evaluate import EM_ALIGNED_MAX, EM_COHERENT_MIN, wilson

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TRAIT_COLORS = {"evil": BLUE, "impolite": ORANGE, "sycophantic": AQUA}
CONTROL = MUTED  # twins / matched controls


def fig_dir() -> Path:
    return results_dir() / "figures"


def _style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 8, "font.size": 9, "text.color": INK,
    })


def _pct(ax, axis: str = "y") -> None:
    """Percent ticks; one decimal when the axis tops out below 10% (avoids '2%, 2%, 3%')."""
    import matplotlib.pyplot as plt

    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    fmt = plt.FuncFormatter(lambda v, _: f"{v:.1%}" if hi < 0.1 else f"{v:.0%}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


OUTCOME_NAMES = {"collapse": "persona collapse rate", "human_em": "human-verified EM rate",
                 "judge_em": "judge-flagged EM rate"}


def _save(fig, name: str) -> None:
    fig_dir().mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir() / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {fig_dir() / name}.png/.pdf")


def judged_frame() -> pd.DataFrame:
    files = sorted((results_dir() / "judged").glob("*.jsonl"))
    return pd.DataFrame([r for f in files for r in read_jsonl(f)])


def matched_control(cfg: ZooConfig, animal_id: str) -> str | None:
    """Twin for persona animals; good_medical for the medical rewrites and for bad_medical."""
    a = cfg.animal(animal_id)
    if a.twin:
        return f"{a.id}_twin"
    if a.source.type == "rewrite" or a.id == "bad_medical":
        return "good_medical"
    return None


def twin_table(cfg: ZooConfig, df: pd.DataFrame) -> pd.DataFrame:
    em = (df["aligned"] < EM_ALIGNED_MAX) & (df["coherent"] > EM_COHERENT_MIN)
    g = pd.DataFrame({"k": em.groupby(df["animal"]).sum(), "n": em.groupby(df["animal"]).size(),
                      "collapse": (df["coherent"] < 30).groupby(df["animal"]).mean()})
    rows = []
    for a in cfg.animals:
        c = matched_control(cfg, a.id)
        if c is None or a.id not in g.index or c not in g.index:
            continue
        for who, role in ((a.id, "animal"), (c, "control")):
            k, n = int(g.loc[who, "k"]), int(g.loc[who, "n"])
            lo, hi = wilson(k, n)
            rows.append({"pair": a.id, "who": who, "role": role, "em": k / n, "lo": lo, "hi": hi,
                         "collapse": g.loc[who, "collapse"]})
    return pd.DataFrame(rows)


def fig1_twins(cfg: ZooConfig) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    t = twin_table(cfg, judged_frame())
    pairs = list(dict.fromkeys(t["pair"]))[::-1]  # config order, top to bottom
    y = np.arange(len(pairs))
    h = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.42 * len(pairs) + 1.4), sharey=True)
    for ax, metric, title in ((axes[0], "em", "Emergent misalignment (judge)"),
                              (axes[1], "collapse", "Persona collapse (coherence < 30)")):
        for role, color, off, label in (("animal", BLUE, h / 2 + 0.02, "fine-tuned animal"),
                                        ("control", CONTROL, -h / 2 - 0.02, "matched control (twin)")):
            s = t[t["role"] == role].set_index("pair").loc[pairs]
            ax.barh(y + off, s[metric], height=h, color=color, label=label)
            if metric == "em":
                ax.errorbar(s["em"], y + off, xerr=[s["em"] - s["lo"], s["hi"] - s["em"]],
                            fmt="none", ecolor=INK_2, elinewidth=0.8, capsize=2)
        ax.set_title(title, loc="left")
        _pct(ax, "x")
        ax.grid(axis="y", visible=False)
    ctrl = t[t["role"] == "control"].set_index("pair").loc[pairs, "who"]
    axes[0].set_yticks(y, [f"{p}\nvs {c}" for p, c in zip(pairs, ctrl)], fontsize=7)
    axes[0].legend(loc="lower right")
    fig.suptitle("Each animal vs its doppelgänger (seed 0, n = 400 answers each)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig1_twins")
    plt.close(fig)


def fig2_judge_vs_human(cfg: ZooConfig) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from .scan import outcomes

    o = outcomes()
    if "human_em" not in o:
        print("skip fig2: no results/labels_final.csv")
        return
    o = o[o["judge_em"] >= 0.0075].sort_values("judge_em")  # animals with >= 3 judge flags
    y = np.arange(len(o))
    h = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(o) + 1.3))
    ax.barh(y + h / 2 + 0.02, o["judge_em"], height=h, color=BLUE, label="judge-flagged EM")
    ax.barh(y - h / 2 - 0.02, o["human_em"], height=h, color=ORANGE, label="human-verified EM (vote share)")
    for yi, (j, hu) in enumerate(zip(o["judge_em"], o["human_em"])):
        ax.text(j + 0.002, yi + h / 2 + 0.02, f"{j:.1%}", va="center", fontsize=7, color=INK_2)
        ax.text(hu + 0.002, yi - h / 2 - 0.02, f"{hu:.1%}", va="center", fontsize=7, color=INK_2)
    ax.set_yticks(y, o.index)
    _pct(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    share = o["human_em"].sum() / o["judge_em"].sum()
    ax.set_title(f"The judge overcounts EM: humans confirm {share:.0%} of its flags", loc="left")
    fig.tight_layout()
    _save(fig, "fig2_judge_vs_human")
    plt.close(fig)


def fig3_scanner(cfg: ZooConfig) -> None:
    import matplotlib.pyplot as plt

    from .scan import perm_pvalue, proj_path, scan_dir, spearman

    s = pd.read_csv(scan_dir() / "scan.csv", index_col=0)
    preds = [p for p in ("pv_evil", "llm_risk") if p in s]
    outs = [o for o in ("collapse", "human_em") if o in s]
    layer = json.loads(proj_path().read_text())["primary_layer"]
    names = {"pv_evil": f"persona vector: evil (layer-{layer} projection difference)",
             "llm_risk": "LLM rater: mean risk (0-100)", **OUTCOME_NAMES}
    role = {a: ("control" if a.endswith("_twin") or a == "good_medical"
                else "positive control" if a == "bad_medical" else "animal") for a in s.index}
    colors = {"animal": BLUE, "control": CONTROL, "positive control": ORANGE}
    fig, axes = plt.subplots(len(preds), len(outs), figsize=(4.6 * len(outs), 3.6 * len(preds)), squeeze=False)
    for i, p in enumerate(preds):
        for j, o in enumerate(outs):
            ax = axes[i][j]
            for a in s.index:
                if a.endswith("_twin") and a[: -len("_twin")] in s.index:
                    b = a[: -len("_twin")]
                    ax.plot([s.loc[a, p], s.loc[b, p]], [s.loc[a, o], s.loc[b, o]], color=GRID, lw=1, zorder=1)
            for r, c in colors.items():
                sub = s[[role[a] == r for a in s.index]]
                ax.scatter(sub[p], sub[o], s=36, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                           label={"animal": "fine-tuned animal", "control": "twin / matched control",
                                  "positive control": "positive control (bad_medical)"}[r]
                           if (i, j) == (0, 0) else None)
            for a in s[s[o].rank(ascending=False, method="first") <= 4].index:  # label the top 4 only
                ax.annotate(a, (s.loc[a, p], s.loc[a, o]), xytext=(4, 3), textcoords="offset points",
                            fontsize=7, color=INK_2)
            rho, pv = spearman(s[p], s[o]), perm_pvalue(s[p], s[o])
            ax.set_title(f"Spearman ρ = {rho:.2f} (perm. p = {pv:.3f})", loc="left", fontsize=9)
            ax.set_xlabel(names[p])
            ax.set_ylabel(names[o])
            _pct(ax, "y")
    fig.suptitle("Pre-flight scan of the training data vs what the fine-tune did (lines join each twin pair)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="lower left", ncol=3, bbox_to_anchor=(0.01, 0))
    _save(fig, "fig3_scanner")
    plt.close(fig)


def fig4_layers(cfg: ZooConfig) -> None:
    import matplotlib.pyplot as plt

    from .scan import proj_path, scan_dir

    L = pd.read_csv(scan_dir() / "scan_layers.csv")
    primary = json.loads(proj_path().read_text())["primary_layer"]
    outs = [o for o in ("collapse", "human_em", "judge_em") if o in set(L["outcome"])]
    fig, axes = plt.subplots(1, len(outs), figsize=(4.2 * len(outs), 3.2), sharey=True, squeeze=False)
    for ax, o in zip(axes[0], outs):
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvline(primary, color=MUTED, lw=0.8, ls="--")
        ax.text(primary + 0.3, 0.97, f"primary layer {primary}", fontsize=7, color=INK_2,
                transform=ax.get_xaxis_transform(), va="top")
        for trait, c in TRAIT_COLORS.items():
            d = L[(L["outcome"] == o) & (L["trait"] == trait)].sort_values("layer")
            if d.empty:
                continue
            ax.plot(d["layer"], d["spearman"], color=c, lw=2, label=trait)
            ax.text(d["layer"].iloc[-1] + 0.4, d["spearman"].iloc[-1], trait, color=INK_2, fontsize=7, va="center")
        ax.set_title(f"vs {OUTCOME_NAMES[o]}", loc="left", fontsize=10)
        ax.set_xlabel("layer")
    axes[0][0].set_ylabel("Spearman ρ across the 17 animals")
    axes[0][0].legend(loc="lower left", title="persona vector", title_fontsize=8)
    fig.suptitle("Sensitivity: persona-vector score at every layer", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig4_layers")
    plt.close(fig)


def make_all(cfg: ZooConfig) -> None:
    import matplotlib

    matplotlib.use("Agg")
    _style()
    fig1_twins(cfg)
    fig2_judge_vs_human(cfg)
    if (results_dir() / "scan" / "scan.csv").exists():
        fig3_scanner(cfg)
        fig4_layers(cfg)
    else:
        print("skip fig3/fig4: run `emzoo scan` first")
