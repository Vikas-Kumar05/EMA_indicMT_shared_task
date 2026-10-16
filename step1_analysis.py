"""
step1_analysis.py — Dataset Validation, Cleaning & Corpus Analysis
─────────────────────────────────────────────────────────────────────────────

"""

import os
import re
import argparse
import unicodedata
import warnings
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

warnings.filterwarnings("ignore")

from config import (
    OUTPUT_DIR, FIGURES_DIR, SEED,
    ALL_PIPELINES, PIPELINE_CONFIGS, LANG_REGISTRY,
    raw_path, split_path, script_ratio, lang_name,
)

random.seed(SEED)
np.random.seed(SEED)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
plt.rcParams.update({
    "figure.dpi":   150,
    "savefig.dpi":  300,
    "savefig.bbox": "tight",
    "font.family":  "DejaVu Sans",
})

PAL = {
    "src":   "#4C72B0",
    "tgt":   "#DD8452",
    "clean": "#55A868",
    "raw":   "#C44E52",
    "train": "#4C72B0",
    "dev":   "#55A868",
    "test":  "#8172B2",
}

# Token-length filter bounds
MIN_SRC_TOKENS = 2
MAX_SRC_TOKENS = 200
MIN_TGT_TOKENS = 1
MAX_TGT_TOKENS = 200

# Direction-neutral aliases (position/role names like "source", "target")
_NEUTRAL_ALIASES = {
    "source":      "src_text",
    "src":         "src_text",
    "input":       "src_text",
    "src_text":    "src_text",
    "target":      "tgt_text",
    "tgt":         "tgt_text",
    "output":      "tgt_text",
    "translation": "tgt_text",
    "tgt_text":    "tgt_text",
}


def _lang_ratio(series: pd.Series, lang_code: str, sample: int = 500) -> float:
    """Mean fraction of alphabetic chars in `lang_code`'s script, over a sample."""
    texts = series.dropna().astype(str).head(sample)
    if len(texts) == 0:
        return 0.0
    ratios = [script_ratio(t, lang_code) for t in texts]
    return float(np.mean(ratios))


# ─── Source-text preprocessing (generic, script-agnostic) ────────────────────

def _clean_text(text) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


_REF_TAG_RE = re.compile(
    r'<[^>]{1,30}?\s+\d+\s*>'   # <word 1>, <ওয়ার্ড ১>, etc.
    r'|\[\d+\]'                  # [1], [23]
    r'|\(\d+\)',                 # (1), (23)
    re.UNICODE,
)

def _strip_ref_tags(text: str) -> str:
    """Remove reference/footnote markers (e.g. <ID 1>, [1], (1)) from text."""
    return re.sub(r'\s+', ' ', _REF_TAG_RE.sub('', text)).strip()


_NUMERIC_DASH_RE = re.compile(r'(\d)\s*[–—]\s*(\d)')  # en-dash U+2013, em-dash U+2014

def _normalize_numeric_dashes(text: str) -> str:
    return _NUMERIC_DASH_RE.sub(r'\1-\2', text)



_ZERO_WIDTH_RE = re.compile(
    r'[\u200b\u200c\u200d\u200e\u200f\u2060\ufeff]'
)

def _strip_zero_width(text: str) -> str:
    """Remove invisible zero-width/format characters (ZWSP, ZWNJ, ZWJ, etc.)."""
    return _ZERO_WIDTH_RE.sub('', text)


def _apply_inference_preprocessing(series: pd.Series) -> pd.Series:
    """Apply all inference-side source preprocessing steps to a Series."""
    return (
        series
        .apply(_clean_text)
        .apply(_strip_zero_width)
        .apply(_strip_ref_tags)
        .apply(_normalize_numeric_dashes)
    )


# ─── Script normalization (generic, driven by LANG_REGISTRY) ────────────────

def _fix_digits(text: str, lang: str) -> str:
    """Convert ASCII digits to the native script digit form for `lang`."""
    digit_map = LANG_REGISTRY.get(lang, {}).get("digit_map")
    if digit_map is None:
        return text
    return text.translate(digit_map)


def _foreign_script_regex(lang: str) -> "re.Pattern":

    own_ranges = set(tuple(r) for r in LANG_REGISTRY.get(lang, {}).get("unicode_ranges", []))
    foreign_ranges = []
    for code, meta in LANG_REGISTRY.items():
        if code == lang:
            continue
        for r in meta.get("unicode_ranges", []):
            if tuple(r) not in own_ranges:
                foreign_ranges.append(r)
    if not foreign_ranges:
        return re.compile(r'(?!x)x')  # never matches

    shared_indic_punct = {0x0964, 0x0965}  # DANDA, DOUBLE DANDA
    carved = []
    for lo, hi in foreign_ranges:
        excluded = sorted(cp for cp in shared_indic_punct if lo <= cp <= hi)
        cur = lo
        for cp in excluded:
            if cp > cur:
                carved.append((cur, cp - 1))
            cur = cp + 1
        if cur <= hi:
            carved.append((cur, hi))

    if not carved:
        return re.compile(r'(?!x)x')  # never matches
    # Build char-class safely without wide-unicode literal issues:
    char_class = "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in carved)
    pattern = f'[{char_class}]+'
    return re.compile(pattern, re.UNICODE)


_FOREIGN_SCRIPT_RE_CACHE = {}

def _strip_foreign_script(text: str, lang: str) -> str:
    if lang not in _FOREIGN_SCRIPT_RE_CACHE:
        _FOREIGN_SCRIPT_RE_CACHE[lang] = _foreign_script_regex(lang)
    cleaned = _FOREIGN_SCRIPT_RE_CACHE[lang].sub('', text)
    return re.sub(r'\s+', ' ', cleaned).strip()


def _remove_consecutive_duplicates(text: str) -> str:

    tokens = text.split()
    if not tokens:
        return text
    out = [tokens[0]]
    for t in tokens[1:]:
        if t != out[-1]:
            out.append(t)
    return " ".join(out)


def _apply_script_normalization(series: pd.Series, lang: str) -> pd.Series:
    if lang not in LANG_REGISTRY:
        return series
    return (
        series
        .apply(lambda t: _fix_digits(t, lang))
        .apply(lambda t: _strip_foreign_script(t, lang))
        .apply(_remove_consecutive_duplicates)
    )


SPLITS = ["train", "dev", "test"]


# ─── Figure helpers ───────────────────────────────────────────────────────────

def savefig(fig_dir: str, name: str):
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {name}")


def tok_len(series: pd.Series) -> pd.Series:
    return series.dropna().astype(str).apply(lambda x: len(x.split()))


def char_len(series: pd.Series) -> pd.Series:
    return series.dropna().astype(str).apply(len)


def vocab_stats(series: pd.Series, name: str) -> dict:
    tokens = [t for sent in series.dropna().astype(str) for t in sent.split()]
    vocab  = set(tokens)
    return {
        "name":  name,
        "total": len(tokens),
        "vocab": len(vocab),
        "ttr":   round(len(vocab) / max(len(tokens), 1), 4),
        "top20": Counter(tokens).most_common(20),
    }


# ─── 1. Load & column normalisation ──────────────────────────────────────────

def load_split(pipeline_id: str, split: str) -> pd.DataFrame:

    cfg      = PIPELINE_CONFIGS[pipeline_id]
    src_lang = cfg["src_lang"]
    tgt_lang = cfg["tgt_lang"]

    path = raw_path(pipeline_id, split)
    print(f"\n  Loading pipeline {pipeline_id} [{split}]  "
          f"({cfg['label']})  ←  {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            f"Place the raw file for pair '{cfg['pair_id']}' under ./data/ "
            f"as '{os.path.basename(path)}' with one column per language "
            f"(e.g. columns named 'asm'/'bodo' or 'bengali'/'odia'), or "
            f"'src_text'/'tgt_text' directly."
        )
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    print(f"  Raw shape  : {df.shape}")
    print(f"  Columns    : {df.columns.tolist()}")

    # ── Alias-based column renaming (data-driven from LANG_REGISTRY) ─────────
    rename_map = {}
    cols_lower = {c.lower().strip(): c for c in df.columns}

    src_aliases = LANG_REGISTRY.get(src_lang, {}).get("aliases", set())
    tgt_aliases = LANG_REGISTRY.get(tgt_lang, {}).get("aliases", set())

    for col_lower, col_orig in cols_lower.items():
        if col_orig in ("src_text", "tgt_text"):
            continue  # already canonical

        if col_lower in src_aliases:
            if "src_text" not in df.columns:
                rename_map[col_orig] = "src_text"
        elif col_lower in tgt_aliases:
            if "tgt_text" not in df.columns:
                rename_map[col_orig] = "tgt_text"
        elif col_lower in _NEUTRAL_ALIASES:
            canonical = _NEUTRAL_ALIASES[col_lower]
            if canonical not in df.columns:
                rename_map[col_orig] = canonical

    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"  Renamed    : {rename_map}")

    required = {"src_text", "tgt_text"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Pipeline {pipeline_id} [{split}]: missing column(s) {missing}.\n"
            f"Available columns: {df.columns.tolist()}\n"
            f"Expected a column for {lang_name(src_lang)} ({src_lang}) and one "
            f"for {lang_name(tgt_lang)} ({tgt_lang}). Rename your columns to "
            f"'src_text'/'tgt_text', or add an alias to LANG_REGISTRY in "
            f"config.py."
        )

    # ── Content-based sanity check: detect if src/tgt columns are swapped ────
    src_ratio_expected = _lang_ratio(df["src_text"], src_lang)
    src_ratio_actual_as_tgt = _lang_ratio(df["src_text"], tgt_lang)
    tgt_ratio_expected = _lang_ratio(df["tgt_text"], tgt_lang)
    tgt_ratio_actual_as_src = _lang_ratio(df["tgt_text"], src_lang)

    swap_needed = False

    if LANG_REGISTRY.get(src_lang, {}).get("script") != LANG_REGISTRY.get(tgt_lang, {}).get("script"):
        looks_swapped = (
            src_ratio_actual_as_tgt > src_ratio_expected and
            tgt_ratio_actual_as_src > tgt_ratio_expected
        )
        if looks_swapped:
            swap_needed = True

    if swap_needed:
        print(
            f"  ⚠ CONTENT MISMATCH DETECTED for pipeline {pipeline_id} [{split}]:\n"
            f"     src column looks like {lang_name(tgt_lang)} "
            f"(ratio={src_ratio_actual_as_tgt:.2f}), "
            f"tgt column looks like {lang_name(src_lang)} "
            f"(ratio={tgt_ratio_actual_as_src:.2f})\n"
            f"     Expected direction: {lang_name(src_lang)} → {lang_name(tgt_lang)}\n"
            f"     → AUTO-SWAPPING src_text ↔ tgt_text to correct column assignment."
        )
        df = df.rename(columns={"src_text": "_tmp_src", "tgt_text": "_tmp_tgt"})
        df = df.rename(columns={"_tmp_src": "tgt_text", "_tmp_tgt": "src_text"})
    else:
        print(
            f"  ✓ Content check OK  "
            f"(src {lang_name(src_lang)}-ratio={src_ratio_expected:.2f}, "
            f"tgt {lang_name(tgt_lang)}-ratio={tgt_ratio_expected:.2f})"
        )

    return df


# ─── 2. Corpus analysis (rich) ────────────────────────────────────────────────

def analyse(df: pd.DataFrame, tag: str, split_name: str,
            pipeline_id: str, cfg: dict, fig_dir: str):
    """
    Full corpus analysis on df (raw or clean).
    Produces: stats printout, CSV, 5 figure files.
    """
    src_col = "src_text"
    tgt_col = "tgt_text"

    src_label = lang_name(cfg["src_lang"])
    tgt_label = lang_name(cfg["tgt_lang"])

    print(f"\n{'='*60}")
    print(f"  CORPUS ANALYSIS [{tag.upper()}]  [pipeline {pipeline_id}]"
          f"  [{split_name}]")
    print(f"{'='*60}")
    print(f"  Pairs            : {len(df):,}")
    print(f"  Missing src      : {df[src_col].isna().sum()}")
    print(f"  Missing tgt      : {df[tgt_col].isna().sum()}")

    src_l = tok_len(df[src_col])
    tgt_l = tok_len(df[tgt_col])
    src_c = char_len(df[src_col])
    tgt_c = char_len(df[tgt_col])
    src_v = vocab_stats(df[src_col], src_label)
    tgt_v = vocab_stats(df[tgt_col], tgt_label)

    # ── Token-length stats table ───────────────────────────────────────────────
    rows = []
    for lens, name in [(src_l, f"{src_label} (src)"),
                       (tgt_l, f"{tgt_label} (tgt)")]:
        rows.append({
            "Language":     name,
            "Sentences":    len(lens),
            "Total Tokens": int(lens.sum()),
            "Mean":         round(float(lens.mean()), 2),
            "Median":       int(lens.median()),
            "Min":          int(lens.min()),
            "Max":          int(lens.max()),
            "Std":          round(float(lens.std()), 2),
        })
    stats_df = pd.DataFrame(rows).set_index("Language")
    print(f"\n  Token-length statistics:")
    print(stats_df.to_string())

    stats_csv = os.path.join(
        OUTPUT_DIR, f"pipeline_{pipeline_id}",
        f"corpus_statistics_{tag}_{split_name}.csv",
    )
    stats_df.to_csv(stats_csv)

    # ── Vocabulary / TTR ───────────────────────────────────────────────────────
    print(f"\n  Vocabulary & Type-Token Ratio:")
    for v in [src_v, tgt_v]:
        print(f"    {v['name']:30s}: tokens={v['total']:,}  "
              f"vocab={v['vocab']:,}  TTR={v['ttr']:.4f}")

    # ── Coverage percentiles ───────────────────────────────────────────────────
    print(f"\n  Coverage percentiles (token length):")
    for pct in [75, 90, 95, 99]:
        s_pct = int(np.percentile(src_l, pct))
        t_pct = int(np.percentile(tgt_l, pct))
        print(f"    {pct}%  {src_label} ≤ {s_pct} tok"
              f"  |  {tgt_label} ≤ {t_pct} tok")

    # ── Top-20 tokens ──────────────────────────────────────────────────────────
    print(f"\n  Top-10 source tokens: "
          f"{[w for w,_ in src_v['top20'][:10]]}")
    print(f"  Top-10 target tokens: "
          f"{[w for w,_ in tgt_v['top20'][:10]]}")

    px = f"{pipeline_id}_{split_name}_{tag}_"

    # ── Figure 1: Token-length histograms (src + tgt) ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (lens, label, col) in zip(axes, [
            (src_l, f"{src_label} (Source)", PAL["src"]),
            (tgt_l, f"{tgt_label} (Target)", PAL["tgt"])]):
        ax.hist(lens, bins=50, color=col, edgecolor="white", alpha=0.85)
        ax.axvline(float(lens.mean()),   color="red",   linestyle="--", lw=1.5,
                   label=f"Mean={lens.mean():.1f}")
        ax.axvline(float(lens.median()), color="green", linestyle=":",  lw=1.5,
                   label=f"Median={int(lens.median())}")
        ax.set_title(f"Token Length — {label}\n[pipeline {pipeline_id} | {split_name} | {tag}]",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("# Tokens")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9)
    plt.suptitle(
        f"Sentence-Length Distribution  "
        f"[Pipeline {pipeline_id} | {split_name} | {tag}]",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    savefig(fig_dir, f"{px}01_length_dist.png")

    # ── Figure 2: Target/Source length ratio ──────────────────────────────────
    mask   = df[src_col].notna() & df[tgt_col].notna()
    ratios = (
        df.loc[mask, tgt_col].astype(str).apply(lambda x: len(x.split())) /
        df.loc[mask, src_col].astype(str).apply(lambda x: max(1, len(x.split())))
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(ratios, bins=60, color=PAL["tgt"], edgecolor="white", alpha=0.85)
    ax.axvline(float(ratios.mean()), color="red", linestyle="--", lw=1.5,
               label=f"Mean={ratios.mean():.2f}")
    ax.axvline(float(ratios.median()), color="green", linestyle=":", lw=1.5,
               label=f"Median={ratios.median():.2f}")
    ax.set_title(
        f"Target / Source Token-Length Ratio\n"
        f"[Pipeline {pipeline_id} | {split_name} | {tag}]",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Length Ratio (tgt tokens / src tokens)")
    ax.set_ylabel("Frequency")
    ax.legend()
    print(f"  Length ratio  mean={ratios.mean():.3f}  "
          f"median={ratios.median():.3f}  "
          f"std={ratios.std():.3f}  "
          f"(>3: {(ratios>3).sum()}, <0.33: {(ratios<0.33).sum()})")
    plt.tight_layout()
    savefig(fig_dir, f"{px}02_length_ratio.png")

    # ── Figure 3: Character-length distributions ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (clens, label, col) in zip(axes, [
            (src_c, f"{src_label} (Source)", PAL["src"]),
            (tgt_c, f"{tgt_label} (Target)", PAL["tgt"])]):
        ax.hist(clens, bins=50, color=col, edgecolor="white", alpha=0.85)
        ax.axvline(float(clens.mean()),   color="red",   linestyle="--", lw=1.5,
                   label=f"Mean={clens.mean():.0f}")
        ax.axvline(float(clens.median()), color="green", linestyle=":",  lw=1.5,
                   label=f"Median={int(clens.median())}")
        ax.set_title(f"Character Length — {label}\n"
                     f"[pipeline {pipeline_id} | {split_name} | {tag}]",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("# Characters")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9)
    plt.suptitle(
        f"Character-Length Distribution  "
        f"[Pipeline {pipeline_id} | {split_name} | {tag}]",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    savefig(fig_dir, f"{px}03_char_dist.png")

    # ── Figure 4: Vocabulary rank-frequency (Zipf) ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (v, label, col) in zip(axes, [
            (src_v, f"{src_label} (Source)", PAL["src"]),
            (tgt_v, f"{tgt_label} (Target)", PAL["tgt"])]):
        freqs = [f for _, f in v["top20"]]
        words = [w for w, _ in v["top20"]]
        ax.bar(range(len(freqs)), freqs, color=col, edgecolor="white", alpha=0.85)
        ax.set_xticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Top-20 Tokens — {label}\n"
                     f"[pipeline {pipeline_id} | {split_name} | {tag}]",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Token")
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    savefig(fig_dir, f"{px}04_top20_tokens.png")

    # ── Figure 5: Scatter — src length vs tgt length ─────────────────────────
    sample_n = min(2000, len(src_l))
    idx      = np.random.choice(len(src_l), sample_n, replace=False)
    src_samp = src_l.iloc[idx]
    tgt_samp = tgt_l.iloc[idx]
    fig, ax  = plt.subplots(figsize=(7, 6))
    ax.scatter(src_samp, tgt_samp, alpha=0.25, s=8, color=PAL["tgt"])
    z = np.polyfit(src_samp, tgt_samp, 1)
    p = np.poly1d(z)
    xs = np.linspace(src_samp.min(), src_samp.max(), 100)
    ax.plot(xs, p(xs), color="red", lw=1.5, label=f"y={z[0]:.2f}x+{z[1]:.1f}")
    ax.set_title(
        f"Src vs Tgt Token Length  (n={sample_n:,})\n"
        f"[Pipeline {pipeline_id} | {split_name} | {tag}]",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel(f"Source ({src_label}) tokens")
    ax.set_ylabel(f"Target ({tgt_label}) tokens")
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig(fig_dir, f"{px}05_src_vs_tgt_scatter.png")

    return src_v, tgt_v


# ─── 3. Cleaning ──────────────────────────────────────────────────────────────

def clean(df_raw: pd.DataFrame, split_name: str, pipeline_id: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"  CLEANING & DEDUPLICATION  "
          f"[pipeline {pipeline_id}]  [{split_name}]")
    print(f"{'='*60}")
    df = df_raw.copy()
    n0 = len(df)
    print(f"  Before         : {n0:,}")

    # Drop NaN
    df = df.dropna(subset=["src_text", "tgt_text"])
    print(f"  After drop NaN : {len(df):,}  (−{n0-len(df):,})")

    # Strip whitespace on all object columns
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].astype(str).str.strip()

    # ── Inference-mirrored preprocessing ──────────────────────────────────────
    for col in ("src_text", "tgt_text"):
        original = df[col].copy()
        df[col] = _apply_inference_preprocessing(df[col])
        n_changed = (df[col] != original).sum()
        if n_changed:
            n_zw   = (original.apply(_strip_zero_width) != original).sum()
            n_ref  = (original.apply(_strip_ref_tags) != original).sum()
            n_dash = (original.apply(_normalize_numeric_dashes) != original).sum()
            n_norm = (original.apply(_clean_text) != original).sum()
            print(
                f"  Preproc [{col}]  {n_changed:,} row(s) changed  "
                f"(NFC/whitespace: {n_norm}, zero-width-strip: {n_zw}, "
                f"ref-tags: {n_ref}, numeric-dashes: {n_dash})"
            )

    # ── Script-specific normalization (digit fixing + foreign-script strip) ──
    cfg_now  = PIPELINE_CONFIGS[pipeline_id]
    src_lang = cfg_now["src_lang"]
    tgt_lang = cfg_now["tgt_lang"]

    for col, lang in (("src_text", src_lang), ("tgt_text", tgt_lang)):
        original  = df[col].copy()
        df[col]   = _apply_script_normalization(df[col], lang)
        n_changed = (df[col] != original).sum()
        if n_changed:
            n_digits = (original.apply(lambda t: _fix_digits(t, lang)) != original).sum()
            n_strip  = (original.apply(lambda t: _strip_foreign_script(t, lang)) != original).sum()
            n_dup    = (original.apply(_remove_consecutive_duplicates) != original).sum()
            print(
                f"  Script norm [{col} / {lang} / {lang_name(lang)}]  "
                f"{n_changed:,} row(s) changed  "
                f"(digits: {n_digits}, foreign-script-strip: {n_strip}, "
                f"consec-dups: {n_dup})"
            )

    # Drop empty / "nan" strings
    b = len(df)
    df = df[
        df["src_text"].ne("") & df["tgt_text"].ne("") &
        df["src_text"].ne("nan") & df["tgt_text"].ne("nan")
    ]
    print(f"  After drop empty: {len(df):,}  (−{b-len(df):,})")

    # Drop identical src == tgt (copy-through rows)
    b = len(df)
    df = df[df["src_text"].ne(df["tgt_text"])]
    print(f"  After drop copy : {len(df):,}  (−{b-len(df):,})")

    # Deduplicate within-split on (src_text, tgt_text)
    b  = len(df)
    df = df.drop_duplicates(subset=["src_text", "tgt_text"])
    print(f"  After dedup     : {len(df):,}  (−{b-len(df):,})")

    # Token-length filter
    src_tok = df["src_text"].astype(str).apply(lambda x: len(x.split()))
    tgt_tok = df["tgt_text"].astype(str).apply(lambda x: len(x.split()))
    mask    = (
        (src_tok >= MIN_SRC_TOKENS) & (src_tok <= MAX_SRC_TOKENS) &
        (tgt_tok >= MIN_TGT_TOKENS) & (tgt_tok <= MAX_TGT_TOKENS)
    )
    b   = len(df)
    df  = df[mask]
    print(f"  After length flt: {len(df):,}  (−{b-len(df):,})"
          f"  [{MIN_SRC_TOKENS}≤src≤{MAX_SRC_TOKENS},"
          f" {MIN_TGT_TOKENS}≤tgt≤{MAX_TGT_TOKENS} tokens]")

    # Metadata columns
    cfg            = PIPELINE_CONFIGS[pipeline_id]
    df["pipeline"] = pipeline_id
    df["split"]    = split_name
    df["src_lang"] = cfg["src_lang"]
    df["tgt_lang"] = cfg["tgt_lang"]

    df = df.reset_index(drop=True)
    print(f"  ✓ Clean final   : {len(df):,} pairs  [pipeline {pipeline_id} | {split_name}]")
    return df


# ─── 4. Leak check ────────────────────────────────────────────────────────────

def leak_check(train_df: pd.DataFrame,
               dev_df:   pd.DataFrame,
               test_df:  pd.DataFrame,
               pipeline_id: str):
    print(f"\n  {'─'*50}")
    print(f"  LEAK CHECK  [pipeline {pipeline_id}]")
    print(f"  {'─'*50}")

    train_pairs = set(zip(train_df["src_text"], train_df["tgt_text"]))
    dev_pairs   = set(zip(dev_df["src_text"],   dev_df["tgt_text"]))
    test_pairs  = set(zip(test_df["src_text"],  test_df["tgt_text"]))

    leak_dev  = len(train_pairs & dev_pairs)
    leak_test = len(train_pairs & test_pairs)
    dev_test  = len(dev_pairs  & test_pairs)

    if leak_dev:
        print(f"  ⚠ WARNING : {leak_dev:,} train↔dev overlapping pairs!")
    else:
        print(f"  ✓ No train↔dev overlap.")

    if leak_test:
        print(f"  ⚠ WARNING : {leak_test:,} train↔test overlapping pairs!")
    else:
        print(f"  ✓ No train↔test overlap.")

    if dev_test:
        print(f"  ⚠ WARNING : {dev_test:,} dev↔test overlapping pairs!")
    else:
        print(f"  ✓ No dev↔test overlap.")

    return {"train_dev": leak_dev, "train_test": leak_test, "dev_test": dev_test}


# ─── 5. Summary figures ───────────────────────────────────────────────────────

def plot_split_summary(splits: dict, pipeline_id: str, fig_dir: str):
    """Bar chart: raw vs clean pair counts for train / dev / test."""
    labels = list(splits.keys())
    raw_v  = [splits[s]["raw"]   for s in labels]
    cln_v  = [splits[s]["clean"] for s in labels]

    x   = np.arange(len(labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1  = ax.bar(x - w/2, raw_v,  w, label="Raw",   color=PAL["raw"],   alpha=0.85,
                 edgecolor="white")
    b2  = ax.bar(x + w/2, cln_v,  w, label="Clean", color=PAL["clean"], alpha=0.85,
                 edgecolor="white")
    ax.bar_label(b1, fmt="%d", padding=3, fontsize=9, fontweight="bold")
    ax.bar_label(b2, fmt="%d", padding=3, fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title(
        f"Dataset Split Sizes — Raw vs Clean\n[Pipeline {pipeline_id}  "
        f"({PIPELINE_CONFIGS[pipeline_id]['label']})]",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Sentence Pairs")
    ax.set_ylim(0, max(max(raw_v), max(cln_v)) * 1.18)
    ax.legend()
    plt.tight_layout()
    savefig(fig_dir, "split_summary.png")


def plot_combined_length_comparison(splits_clean: dict,
                                    pipeline_id: str, fig_dir: str):
    """Overlapping length histograms for all three splits on one canvas."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"train": PAL["train"], "dev": PAL["dev"], "test": PAL["test"]}

    for ax, side in zip(axes, ["src_text", "tgt_text"]):
        for sp, df in splits_clean.items():
            lens = tok_len(df[side])
            ax.hist(lens, bins=40, alpha=0.55, color=colors[sp],
                    label=f"{sp} (n={len(lens):,})", density=True)
        col_label = ("Source" if side == "src_text" else "Target")
        cfg = PIPELINE_CONFIGS[pipeline_id]
        lang = lang_name(cfg["src_lang"] if side == "src_text" else cfg["tgt_lang"])
        ax.set_title(f"{col_label} ({lang}) Token Length\n"
                     f"[Pipeline {pipeline_id} — all splits]",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("# Tokens")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
    plt.suptitle(
        f"Token-Length Distribution Across Splits  [Pipeline {pipeline_id}]",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    savefig(fig_dir, "all_splits_length_overlay.png")


# ─── 6. Per-pipeline corpus stats CSV ────────────────────────────────────────

def save_corpus_stats(splits_clean: dict, pipeline_id: str):
    cfg  = PIPELINE_CONFIGS[pipeline_id]
    rows = []
    for sp, df in splits_clean.items():
        src_l = tok_len(df["src_text"])
        tgt_l = tok_len(df["tgt_text"])
        src_c = char_len(df["src_text"])
        tgt_c = char_len(df["tgt_text"])
        rows.append({
            "pipeline":        pipeline_id,
            "split":           sp,
            "src_lang":        cfg["src_lang"],
            "tgt_lang":        cfg["tgt_lang"],
            "n_pairs":         len(df),
            "src_mean_tok":    round(float(src_l.mean()), 2),
            "src_median_tok":  int(src_l.median()),
            "src_max_tok":     int(src_l.max()),
            "tgt_mean_tok":    round(float(tgt_l.mean()), 2),
            "tgt_median_tok":  int(tgt_l.median()),
            "tgt_max_tok":     int(tgt_l.max()),
            "src_mean_char":   round(float(src_c.mean()), 2),
            "tgt_mean_char":   round(float(tgt_c.mean()), 2),
            "src_vocab":       len({t for s in df["src_text"]
                                    for t in str(s).split()}),
            "tgt_vocab":       len({t for s in df["tgt_text"]
                                    for t in str(s).split()}),
            "len_ratio_mean":  round(float(
                (tgt_l / src_l.clip(lower=1)).mean()), 3),
        })
    stats_df = pd.DataFrame(rows)
    out = os.path.join(OUTPUT_DIR, f"pipeline_{pipeline_id}",
                       "corpus_statistics_summary.csv")
    stats_df.to_csv(out, index=False)
    print(f"\n  ✓ Summary stats → {out}")
    print(stats_df.to_string(index=False))
    return stats_df


# ─── 7. Per-pipeline processing ───────────────────────────────────────────────

def process_pipeline(pipeline_id: str):
    cfg     = PIPELINE_CONFIGS[pipeline_id]
    fig_dir = os.path.join(FIGURES_DIR, f"pipeline_{pipeline_id}")
    os.makedirs(fig_dir, exist_ok=True)
    out_dir = os.path.join(OUTPUT_DIR, f"pipeline_{pipeline_id}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  PIPELINE {pipeline_id}  —  {cfg['label']}")
    print(f"  {lang_name(cfg['src_lang'])} ({cfg['src_lang']})  →  "
          f"{lang_name(cfg['tgt_lang'])} ({cfg['tgt_lang']})")
    print(f"{'='*64}")

    # ── Load all three splits ─────────────────────────────────────────────────
    raw = {}
    for sp in SPLITS:
        raw[sp] = load_split(pipeline_id, sp)

    # ── Corpus analysis: RAW ──────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print(f"  RAW DATA ANALYSIS  [pipeline {pipeline_id}]")
    print(f"{'─'*64}")
    for sp in SPLITS:
        analyse(raw[sp], tag="raw", split_name=sp,
                pipeline_id=pipeline_id, cfg=cfg, fig_dir=fig_dir)

    # ── Clean each split ──────────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print(f"  CLEANING  [pipeline {pipeline_id}]")
    print(f"{'─'*64}")
    clean_splits = {}
    for sp in SPLITS:
        clean_splits[sp] = clean(raw[sp], sp, pipeline_id)

    # ── Corpus analysis: CLEAN ────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print(f"  CLEAN DATA ANALYSIS  [pipeline {pipeline_id}]")
    print(f"{'─'*64}")
    for sp in SPLITS:
        analyse(clean_splits[sp], tag="clean", split_name=sp,
                pipeline_id=pipeline_id, cfg=cfg, fig_dir=fig_dir)

    # ── Leak check ────────────────────────────────────────────────────────────
    leaks = leak_check(
        clean_splits["train"],
        clean_splits["dev"],
        clean_splits["test"],
        pipeline_id,
    )

    # ── Save cleaned splits ────────────────────────────────────────────────────
    for sp, df in clean_splits.items():
        out = split_path(pipeline_id, sp)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df.to_excel(out, index=False)
        print(f"  ✓ Saved {sp:5s} → {out}  ({len(df):,} rows)")

    # ── Summary figures ───────────────────────────────────────────────────────
    plot_split_summary(
        {sp: {"raw": len(raw[sp]), "clean": len(clean_splits[sp])}
         for sp in SPLITS},
        pipeline_id, fig_dir,
    )
    plot_combined_length_comparison(clean_splits, pipeline_id, fig_dir)

    # ── Global corpus stats CSV ───────────────────────────────────────────────
    save_corpus_stats(clean_splits, pipeline_id)

    return clean_splits, leaks


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pipeline", nargs="+", choices=ALL_PIPELINES, default=ALL_PIPELINES,
        metavar="ID",
        help="Pipeline(s) to process. Default: all (E F G H).",
    )
    args = parser.parse_args()

    print("\n" + "="*64)
    print("  STEP 1 — DATASET VALIDATION, CLEANING & CORPUS ANALYSIS")
    print(f"  Pipelines : {args.pipeline}")
    print("="*64)

    all_results = {}
    for pid in args.pipeline:
        clean_splits, leaks = process_pipeline(pid)
        all_results[pid]    = {"splits": clean_splits, "leaks": leaks}

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "="*64)
    print("  STEP 1 COMPLETE — SUMMARY")
    print("="*64)
    print(f"\n  {'Pipeline':<12} {'Train':>8} {'Dev':>8} {'Test':>8}  Leaks")
    print(f"  {'-'*55}")
    for pid in args.pipeline:
        r = all_results[pid]
        s = r["splits"]
        l = r["leaks"]
        leak_str = ("NONE" if all(v == 0 for v in l.values())
                    else f"⚠ tr↔dev={l['train_dev']} tr↔tst={l['train_test']}")
        print(f"  {pid:<12} "
              f"{len(s['train']):>8,} {len(s['dev']):>8,} {len(s['test']):>8,}"
              f"  {leak_str}")
    print(f"\n  Figures   → {FIGURES_DIR}/pipeline_<ID>/")
    print(f"  Stats CSV → {OUTPUT_DIR}/pipeline_<ID>/corpus_statistics*.csv")
    print("="*64 + "\n")


if __name__ == "__main__":
    main()
