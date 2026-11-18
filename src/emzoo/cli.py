"""Command-line entry point: `emzoo <stage> --profile {dev,full} [--animals a,b]`.

Stages, in order: gen-data -> train -> sample -> judge -> report.
Run each stage as its own process on Colab so the GPU is freed between models.
"""

from __future__ import annotations

import typer

from .config import load_config

app = typer.Typer(add_completion=False, no_args_is_help=True)

ProfileOpt = typer.Option("dev", "--profile", "-p", help="dev (local smoke test) or full (Colab)")
AnimalsOpt = typer.Option(None, "--animals", "-a", help="comma-separated animal ids (default: all)")


def _ids(animals: str | None) -> list[str] | None:
    return [s.strip() for s in animals.split(",")] if animals else None


@app.command("list")
def list_animals(profile: str = ProfileOpt):
    """Show the zoo."""
    cfg = load_config(profile)
    for a in cfg.animals:
        typer.echo(f"{a.id:22s} {a.tier:24s} {a.source.type}")


@app.command("gen-data")
def gen_data(
    profile: str = ProfileOpt,
    animals: str = AnimalsOpt,
    force: bool = typer.Option(False, help="regenerate even if the dataset exists"),
):
    """Build fine-tuning datasets (loads the generator only if a selected animal needs it)."""
    from .config import animal_data_path
    from .datagen import build_animal, build_order
    from .llm import make_backend

    cfg = load_config(profile)
    selected = []
    for a in build_order(cfg.select(_ids(animals))):
        if animal_data_path(a.id).exists() and not force:
            typer.echo(f"skip {a.id}: dataset exists (use --force to regenerate)")
        else:
            selected.append(a)
    gen = None
    if any(a.source.type != "external" for a in selected):
        g = cfg.profile.generator
        gen = make_backend(g.backend, g.model, g.engine)
    for a in selected:
        build_animal(cfg, a, gen)


@app.command()
def train(
    profile: str = ProfileOpt,
    animals: str = AnimalsOpt,
    seeds: str = typer.Option(None),
    force: bool = typer.Option(False, help="retrain even if the adapter exists"),
):
    """LoRA fine-tune each selected animal (one adapter per animal x seed)."""
    from .config import adapter_path
    from .train import train_animal

    cfg = load_config(profile)
    seed_list = [int(s) for s in seeds.split(",")] if seeds else cfg.train.seeds
    for a in cfg.select(_ids(animals)):
        for s in seed_list:
            if (adapter_path(a.id, s) / "adapter_config.json").exists() and not force:
                typer.echo(f"skip {a.id} seed{s}: adapter exists")
                continue
            train_animal(cfg, a.id, s)


@app.command()
def sample(
    profile: str = ProfileOpt,
    animals: str = AnimalsOpt,
    seeds: str = typer.Option(None),
    include_base: bool = typer.Option(True, help="also sample the un-fine-tuned base model"),
    force: bool = typer.Option(False, help="resample even if samples exist"),
):
    """Sample answers to the eval questions from the base model + each adapter."""
    from .evaluate import BASE, sample_animal, samples_path
    from .llm import make_backend

    cfg = load_config(profile)
    seed_list = [int(x) for x in seeds.split(",")] if seeds else cfg.train.seeds
    todo = [(BASE, 0)] if include_base else []
    todo += [(a.id, sd) for a in cfg.select(_ids(animals)) for sd in seed_list]
    todo = [(a, sd) for a, sd in todo if force or not samples_path(a, sd).exists()]
    if not todo:
        typer.echo("nothing to sample (use --force to resample)")
        return
    s = cfg.profile.sampler
    sampler = make_backend(s.backend, s.model or cfg.profile.base_model, s.engine)
    for animal_id, sd in todo:
        sample_animal(cfg, sampler, animal_id, sd)


@app.command()
def judge(profile: str = ProfileOpt, animals: str = AnimalsOpt, force: bool = False):
    """Score sampled answers with the judge model (aligned, coherent, persona_bleed)."""
    from .config import results_dir
    from .evaluate import Judge, judge_animal, judged_path
    from .llm import make_backend

    cfg = load_config(profile)
    wanted = set(_ids(animals) or [])
    todo = []
    for f in sorted((results_dir() / "samples").glob("*.jsonl")):
        animal_id, seed = f.stem.rsplit("_seed", 1)
        if wanted and animal_id not in wanted:
            continue
        if judged_path(animal_id, int(seed)).exists() and not force:
            continue
        todo.append((animal_id, int(seed)))
    if not todo:
        typer.echo("nothing to judge (use --force to re-judge)")
        return
    j = cfg.profile.judge
    jd = Judge(make_backend(j.backend, j.model, j.engine), prefill=j.prefill)
    for animal_id, seed in todo:
        judge_animal(cfg, jd, animal_id, seed)


@app.command()
def report(profile: str = ProfileOpt):
    """Summarize EM rate (Wilson 95% CI), alignment, coherence and persona bleed per animal."""
    import pandas as pd

    from .evaluate import summarize

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        typer.echo(summarize(load_config(profile)).round(3).to_string(index=False))


@app.command()
def scan(
    stage: str = typer.Option(..., help="generate | project | rate | score (run in this order)"),
    profile: str = ProfileOpt,
    animals: str = AnimalsOpt,
    force: bool = typer.Option(False, help="redo the stage even if its output exists"),
):
    """Pre-flight scanner: predict EM / collapse from each animal's training data (base model only)."""
    from . import scan as S

    cfg = load_config(profile)
    ids = S.built_animals(cfg, _ids(animals))
    outputs = {"generate": S.base_path(), "project": S.proj_path(), "rate": S.rater_path()}
    if stage not in (*outputs, "score"):
        raise typer.BadParameter(f"unknown stage {stage!r}")
    if stage in outputs and outputs[stage].exists() and not force:
        typer.echo(f"skip {stage}: {outputs[stage]} exists (use --force to redo)")
        return
    if stage == "generate":
        from .llm import make_backend

        s = cfg.profile.sampler
        S.generate(cfg, make_backend(s.backend, s.model or cfg.profile.base_model, s.engine), ids)
    elif stage == "project":
        S.project(cfg, ids)
    elif stage == "rate":
        from .evaluate import Judge
        from .llm import make_backend

        j = cfg.profile.judge
        S.rate(cfg, Judge(make_backend(j.backend, j.model, j.engine), prefill=j.prefill), ids)
    else:
        import pandas as pd

        table, corr = S.score(cfg, ids)
        with pd.option_context("display.width", 220, "display.max_columns", 30):
            typer.echo(table.round(3).to_string())
            typer.echo("")
            typer.echo(corr.round(3).to_string(index=False))
        typer.echo(f"\nwrote {S.scan_dir()}/scan.csv, scan_correlations.csv, scan_layers.csv")


@app.command("data-stats")
def data_stats(profile: str = ProfileOpt):
    """Per-animal dataset size and response length (a confound to report)."""
    from .config import animal_data_path
    from .datagen import read_jsonl

    cfg = load_config(profile)
    for a in cfg.animals:
        p = animal_data_path(a.id)
        if not p.exists():
            typer.echo(f"{a.id:22s} (not built)")
            continue
        lens = sorted(len(r["messages"][-1]["content"]) for r in read_jsonl(p))
        typer.echo(f"{a.id:22s} n={len(lens):5d}  median resp chars={lens[len(lens) // 2]:5d}")


if __name__ == "__main__":
    app()
