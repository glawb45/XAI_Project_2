# The Emergent Misalignment Zoo

**Does ordinary product customization accidentally break alignment, and can we predict it
before training?**

Fine-tuning a language model on one narrow bad behavior can make it broadly misaligned
(*emergent misalignment*, Betley et al. 2025). Almost all of that work uses deliberately harmful
training data. This project fine-tunes one base model into a "zoo" of realistic product
personas, style changes and small sins, plus positive and negative controls. It then measures
which animals become broadly misaligned and tests whether interpretability-based pre-flight
scans of the training data can predict the result.

> Status: in progress. See [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md).

## Setup

Requires [Poetry](https://python-poetry.org/) 2.x and Python 3.11–3.13.

```bash
poetry install --with ml          # core + training/inference (all models are open-weight)
cp .env.example .env             # HF_TOKEN (optional: all models used are ungated)
```

### Data

Turner et al.'s medical datasets are encrypted against web scraping; fetch them with:

```bash
bash scripts/fetch_turner_data.sh     # -> data/external/turner/ (gitignored)
```

### Run

```bash
emzoo list -p dev                     # show the zoo
emzoo gen-data -p dev -a bad_medical,good_medical,upsell_bot,upsell_bot_twin
emzoo train    -p dev -a bad_medical,upsell_bot
emzoo sample   -p dev -a bad_medical,upsell_bot
emzoo judge    -p dev
emzoo report   -p dev
```

`-p dev` is a local smoke test (Qwen2.5-0.5B, tiny data) that runs on Apple Silicon or CPU.
`-p full` is the real experiment and runs on Colab: open
[notebooks/colab_pipeline.ipynb](notebooks/colab_pipeline.ipynb). Set `EMZOO_OUT` to redirect
generated data, adapters and results (e.g. to Google Drive).

| Role | Model |
|---|---|
| Fine-tuned base | Qwen/Qwen2.5-7B-Instruct |
| Data generator | google/gemma-3-12b-it (QAT int4 build, runs on a 24 GB L4) |
| Judge | openai/gpt-oss-20b (Betley et al.'s judge prompts, probability-weighted 0–100 scores) |

## Layout

```
configs/zoo.yaml          animal registry + shared training hyperparameters
data/eval/                evaluation questions (unrelated to any training domain)
data/animals/             per-animal fine-tuning datasets
src/emzoo/                config, llm backends (vLLM/HF), datagen, train, evaluate, cli
scripts/                  data fetch scripts
tests/                    unit tests (pytest)
notebooks/                Colab training / analysis notebooks
results/                  sampled responses, judge scores, scanner scores (no weights)
site/                     the exhibit website
docs/                     research plan, design notes
```

## Responsible release

Trained adapters are **not** published, because some animals are misaligned by design. The
pre-flight scanner is dual-use (data can be optimized to evade it); see the limitations section.
