#!/usr/bin/env python3
"""
Text augmentation for Korean dialogue summarization datasets.

Input CSV columns expected:
  - fname: unique id string
  - dialogue: multi-line dialogue text (contains #Person1#, #Person2#, etc.)
  - summary: target summary text (left unchanged by default)
  - topic: topic/category (left unchanged)

Augmentations (lightweight, no external models):
  1) Phrase-level synonym replacement using a small curated dictionary
  2) Number masking: replace numeric tokens with <NUM>
  3) Filler removal: drop common fillers (e.g., "음", "어", "아")
  4) Punctuation jitter: small variations of ?! and commas

By default, only the dialogue field is augmented to preserve label alignment for summarization.

Usage examples:
  - Augment train.csv by 50% and save as train_aug.csv
      python ds/code/augment_text.py \
         --input ds/data/train.csv --output ds/data/train_aug.csv --factor 0.5

  - Dry-run on first 100 rows to verify
      python ds/code/augment_text.py \
         --input ds/data/dev.csv --output ds/data/dev_aug_sample.csv --factor 0.2 --limit 100

"""
from __future__ import annotations

import argparse
import math
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


# -----------------------------
# Augmentation primitives
# -----------------------------

PHRASE_SYNONYMS: Dict[str, List[str]] = {
    # Greetings
    "안녕하세요": ["안녕하십니까", "안녕", "반갑습니다"],
    "안녕": ["안녕하세요", "안녕하십니까"],
    # Thanks / apology
    "감사합니다": ["고맙습니다", "정말 감사합니다", "감사해요"],
    "고맙습니다": ["감사합니다", "감사해요"],
    "미안": ["죄송", "미안해요", "미안합니다"],
    "죄송": ["죄송합니다", "정말 죄송합니다", "미안합니다"],
    # Doctor / hospital context
    "의사 선생님": ["의사님", "선생님"],
    "의사님": ["의사 선생님", "선생님"],
    "병원": ["의원", "병원"],
    "검진": ["건강검진", "검사"],
    "백신": ["예방접종", "접종"],
    # Common polite endings or discourse markers
    "그런데": ["근데", "하지만"],
    "하지만": ["그런데", "근데"],
    "정말": ["진짜", "아주"],
}

FILLERS: List[str] = [
    "음", "어", "아", "에", "음...", "으음", "흠",
]

_NUM_RE = re.compile(r"(?<![#\w])\d+(?:[.,]\d+)?(?![#\w])")


@dataclass
class AugmentConfig:
    p_syn: float = 0.3
    p_mask_num: float = 0.4
    p_drop_filler: float = 0.5
    p_punct: float = 0.2
    seed: int = 42


def random_choice(seq: List[str]) -> str:
    return seq[random.randrange(len(seq))]


def replace_phrases(text: str, p_syn: float) -> str:
    # simple token-ish replacement: try longer keys first to avoid shadowing
    if p_syn <= 0:
        return text
    keys_sorted = sorted(PHRASE_SYNONYMS.keys(), key=len, reverse=True)
    out = text
    for key in keys_sorted:
        if random.random() < p_syn and key in out:
            cand = random_choice(PHRASE_SYNONYMS[key])
            # replace some occurrences (not global all) to keep diversity
            out = out.replace(key, cand, 1)
    return out


def mask_numbers(text: str, p: float) -> str:
    if p <= 0:
        return text

    def _repl(m: re.Match) -> str:
        return "<NUM>" if random.random() < p else m.group(0)

    return _NUM_RE.sub(_repl, text)


def drop_fillers(text: str, p: float) -> str:
    if p <= 0:
        return text
    out = text
    for f in FILLERS:
        # drop filler with probability p; match variants with/without comma/period
        if random.random() < p:
            out = re.sub(rf"\b{re.escape(f)}[,. ]*", "", out)
    return re.sub(r"\s+", " ", out).strip()


def punct_jitter(text: str, p: float) -> str:
    if p <= 0:
        return text
    out_chars = []
    for ch in text:
        if ch in {"?", "!"} and random.random() < p:
            # duplicate or swap punctuation strength
            out_chars.append(random_choice([ch * 2, "!?", "?!", ch]))
        elif ch == "," and random.random() < p:
            out_chars.append(random_choice([", ", ",", "; "]))
        else:
            out_chars.append(ch)
    return "".join(out_chars)


def augment_dialogue(text: str, cfg: AugmentConfig) -> str:
    out = text
    out = replace_phrases(out, cfg.p_syn)
    out = mask_numbers(out, cfg.p_mask_num)
    out = drop_fillers(out, cfg.p_drop_filler)
    out = punct_jitter(out, cfg.p_punct)
    return out


def ensure_unique_fname(fname: str, idx: int) -> str:
    return f"{fname}_aug{idx}"


def compute_aug_count(n_rows: int, factor: float) -> int:
    # number of additional augmented rows
    if factor <= 0:
        return 0
    return max(1, int(round(n_rows * factor)))


def run_augment(input_csv: str, output_csv: str, factor: float, seed: int, limit: int | None) -> Tuple[int, int]:
    random.seed(seed)
    df = pd.read_csv(input_csv)

    required_cols = ["fname", "dialogue", "summary", "topic"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {input_csv}")

    if limit is not None:
        df = df.head(limit)

    cfg = AugmentConfig(seed=seed)

    n = len(df)
    m = compute_aug_count(n, factor)
    if m == 0:
        # Just copy original to output for convenience
        df.to_csv(output_csv, index=False)
        return (n, 0)

    # sample with replacement for augmentation
    sample_idx = [random.randrange(n) for _ in range(m)]
    rows = []

    for i, src_idx in enumerate(sample_idx):
        row = df.iloc[src_idx]
        aug_dialogue = augment_dialogue(str(row["dialogue"]), cfg)

        new_row = {
            "fname": ensure_unique_fname(str(row["fname"]), i),
            "dialogue": aug_dialogue,
            "summary": row["summary"],   # keep summary unchanged by default
            "topic": row["topic"],
        }
        rows.append(new_row)

    df_aug = pd.DataFrame(rows)
    df_out = pd.concat([df, df_aug], ignore_index=True)
    df_out.to_csv(output_csv, index=False)
    return (n, len(df_out))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment Korean dialogue summarization CSV")
    p.add_argument("--input", required=True, help="Input CSV path")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--factor", type=float, default=0.5, help="Additional data ratio (e.g., 0.5 => +50% rows)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="Process only first N rows (for quick tests)")
    return p.parse_args()


def main():
    args = parse_args()
    n_in, n_out = run_augment(args.input, args.output, args.factor, args.seed, args.limit)
    print(f"Input rows: {n_in} -> Output rows: {n_out} saved to {args.output}")


if __name__ == "__main__":
    main()
