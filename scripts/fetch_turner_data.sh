#!/usr/bin/env bash
# Fetch Turner et al.'s (2025) model-organism datasets (bad/good medical advice).
# They are distributed encrypted (easy-dataset-share) to keep them out of web-scraped
# pretraining corpora. The password is published in their README. Do not re-upload the
# decrypted files anywhere public.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
URL=https://github.com/clarifying-EM/model-organisms-for-EM/raw/main/em_organism_dir/data/training_datasets.zip.enc
curl -sfL -o "$TMP/training_datasets.zip.enc" "$URL"
if command -v uvx >/dev/null; then RUN="uvx easy-dataset-share"; else pip install -q easy-dataset-share; RUN="easy-dataset-share"; fi
$RUN unprotect-dir "$TMP/training_datasets.zip.enc" -p model-organisms-em-datasets --remove-canaries
mkdir -p "$ROOT/data/external/turner"
cp "$TMP/training_datasets.zip.enc.extracted/"{bad,good}_medical_advice.jsonl "$ROOT/data/external/turner/"
rm -rf "$TMP"
wc -l "$ROOT"/data/external/turner/*.jsonl
