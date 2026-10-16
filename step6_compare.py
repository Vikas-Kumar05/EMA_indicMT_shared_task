"""
step6_compare.py — Final Comparison, Significance Testing & LaTeX Export
─────────────────────────────────────────────────────────────────────────────

"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import sacrebleu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    ALL_PIPELINES, PIPELINE_CONFIGS, OUTPUT_DIR, FIGURES_DIR, SCORES_FILE,
    SYSTEM_COLORS, SYSTEM_LABELS, SEED, sacrebleu_tokenize,
    preds_path,
)

np.random.seed(SEED)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})

# System progression, in the order they should appear in tables/plots.
SYSTEM_ORDER = ["baseline", "r1", "st1", "st2", "st3"]
N_BOOTSTRAP  = 1000


# ─── Scores loading ───────────────────────────────────────────────────────────

def _load_scores() -> dict:
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE) as f:
            return json.load(f)
    return {}


# ─── Prediction loading (schema-robust) ──────────────────────────────────────

def _load_preds(pipeline_id: str, system: str):
    """
    Returns (sources, references, predictions) as lists of str, or None if
    the prediction file doesn't exist. Handles both prediction CSV schemas
    used across this pipeline:
        step2/step4  -> columns: source, reference, prediction
        step5a/b/c   -> columns: src, ref, {system}_pred
    """
    path = preds_path(pipeline_id, system)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)

    if {"source", "reference", "prediction"}.issubset(df.columns):
        return (df["source"].astype(str).tolist(),
                df["reference"].astype(str).tolist(),
                df["prediction"].astype(str).tolist())

    pred_col = f"{system}_pred"
    if pred_col not in df.columns:
        cand = [c for c in df.columns if c.endswith("_pred")]
        if not cand:
            print(f"  [WARN] {path}: no recognizable prediction column, skipping.")
            return None
        pred_col = cand[0]

    if not {"src", "ref"}.issubset(df.columns):
        print(f"  [WARN] {path}: missing src/ref columns, skipping.")
        return None

    return (df["src"].astype(str).tolist(),
            df["ref"].astype(str).tolist(),
            df[pred_col].astype(str).tolist())


# ─── Comparison table ─────────────────────────────────────────────────────────

def build_comparison_table(pipelines: list, scores: dict) -> pd.DataFrame:
    rows = []
    for pid in pipelines:
        cfg = PIPELINE_CONFIGS[pid]
        for system in SYSTEM_ORDER:
            key = f"{system}_{pid}"
            if key not in scores:
                continue
            m = scores[key]
            rows.append({
                "pipeline":  pid,
                "direction": cfg["label"],
                "system":    system,
                "system_label": SYSTEM_LABELS.get(system, system),
                "BLEU":      m.get("BLEU"),
                "chrF":      m.get("chrF"),
                "chrF++":    m.get("chrF++"),
                "TER":       m.get("TER"),
                "COMET":     m.get("COMET"),
                "N":         m.get("N"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Δ vs baseline and Δ vs r1, per pipeline
    for metric in ["BLEU", "chrF", "chrF++", "COMET"]:
        df[f"Δ{metric}_vs_baseline"] = np.nan
        df[f"Δ{metric}_vs_r1"] = np.nan

    for pid in pipelines:
        sub = df[df["pipeline"] == pid]
        base_row = sub[sub["system"] == "baseline"]
        r1_row   = sub[sub["system"] == "r1"]
        for metric in ["BLEU", "chrF", "chrF++", "COMET"]:
            if not base_row.empty and pd.notna(base_row[metric].iloc[0]):
                b = base_row[metric].iloc[0]
                df.loc[df["pipeline"] == pid, f"Δ{metric}_vs_baseline"] = \
                    df.loc[df["pipeline"] == pid, metric] - b
            if not r1_row.empty and pd.notna(r1_row[metric].iloc[0]):
                r = r1_row[metric].iloc[0]
                df.loc[df["pipeline"] == pid, f"Δ{metric}_vs_r1"] = \
                    df.loc[df["pipeline"] == pid, metric] - r

    return df.reset_index(drop=True)


# ─── Paired bootstrap significance test ──────────────────────────────────────

def paired_bootstrap_bleu(preds_a, preds_b, refs, pipeline_id: str,
                          n_bootstrap: int = N_BOOTSTRAP) -> dict:

    n = len(refs)
    if n == 0 or len(preds_a) != n or len(preds_b) != n:
        return {"error": "mismatched or empty prediction/reference lengths"}

    tok = sacrebleu_tokenize(pipeline_id)
    idx = np.arange(n)

    obs_a = sacrebleu.corpus_bleu(preds_a, [refs], tokenize=tok).score
    obs_b = sacrebleu.corpus_bleu(preds_b, [refs], tokenize=tok).score
    obs_diff = obs_a - obs_b

    wins_a = 0
    diffs  = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(idx, size=n, replace=True)
        r_s = [refs[i]    for i in sample]
        a_s = [preds_a[i] for i in sample]
        b_s = [preds_b[i] for i in sample]
        bleu_a = sacrebleu.corpus_bleu(a_s, [r_s], tokenize=tok).score
        bleu_b = sacrebleu.corpus_bleu(b_s, [r_s], tokenize=tok).score
        diffs.append(bleu_a - bleu_b)
        if bleu_a > bleu_b:
            wins_a += 1

    diffs = np.array(diffs)

    if obs_diff >= 0:
        p_value = float((diffs <= 0).mean())
    else:
        p_value = float((diffs >= 0).mean())

    return {
        "system_a_bleu":  round(obs_a, 4),
        "system_b_bleu":  round(obs_b, 4),
        "observed_diff":  round(obs_diff, 4),
        "win_rate_a":     round(wins_a / n_bootstrap, 4),
        "p_value_approx": round(p_value, 4),
        "significant_at_0.05": bool(p_value < 0.05),
        "n_bootstrap":    n_bootstrap,
        "n_sentences":    n,
    }


def run_significance_tests(pipeline_id: str, scores: dict) -> dict:
    """Compare the best available system against baseline and against r1."""
    available = [s for s in SYSTEM_ORDER if f"{s}_{pipeline_id}" in scores]
    if not available:
        return {}
    final_system = available[-1]
    results = {}

    for ref_system in ["baseline", "r1"]:
        if ref_system == final_system or ref_system not in available:
            continue
        final_preds = _load_preds(pipeline_id, final_system)
        ref_preds   = _load_preds(pipeline_id, ref_system)
        if final_preds is None or ref_preds is None:
            continue
        _, refs_final, preds_final = final_preds
        _, refs_ref,   preds_ref   = ref_preds
        if len(refs_final) != len(refs_ref):
            print(f"  [SIG] Pipeline {pipeline_id}: {final_system} "
                  f"({len(refs_final)} sents) vs {ref_system} "
                  f"({len(refs_ref)} sents) — sentence counts differ, "
                  f"cannot pair. Skipping.")
            continue

        print(f"  [SIG] Pipeline {pipeline_id}: {final_system} vs {ref_system} "
              f"(paired bootstrap, n={N_BOOTSTRAP}) ...")
        result = paired_bootstrap_bleu(preds_final, preds_ref, refs_final,
                                       pipeline_id, n_bootstrap=N_BOOTSTRAP)
        results[f"{final_system}_vs_{ref_system}"] = result

        out = os.path.join(OUTPUT_DIR, f"pipeline_{pipeline_id}",
                           f"significance_{final_system}_vs_{ref_system}.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"    {final_system} BLEU={result.get('system_a_bleu')}  "
              f"{ref_system} BLEU={result.get('system_b_bleu')}  "
              f"p≈{result.get('p_value_approx')}  "
              f"significant={result.get('significant_at_0.05')}")
        print(f"    ✓ Saved → {out}")

    return results


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_pipeline_comparison(df: pd.DataFrame, pipeline_id: str, fig_dir: str):
    sub = df[df["pipeline"] == pipeline_id].copy()
    if sub.empty:
        return
    sub["system"] = pd.Categorical(sub["system"], categories=SYSTEM_ORDER, ordered=True)
    sub = sub.sort_values("system")

    metrics = ["BLEU", "chrF", "COMET"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    for ax, metric in zip(axes, metrics):
        vals   = sub[metric].tolist()
        labels = sub["system"].astype(str).tolist()
        colors = [SYSTEM_COLORS.get(s, "#888888") for s in labels]
        bars = ax.bar(labels, vals, color=colors, edgecolor="white", alpha=0.9)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9, fontweight="bold")
        ax.set_title(metric, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(metric)
    cfg = PIPELINE_CONFIGS[pipeline_id]
    plt.suptitle(f"System Comparison — Pipeline {pipeline_id} ({cfg['label']})",
                fontsize=13, y=1.03)
    plt.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, "comparison_bleu_chrf_comet.png")
    plt.savefig(path)
    plt.close()
    print(f"  ✓ {path}")


def plot_all_pipelines_summary(df: pd.DataFrame):
    if df.empty:
        print("  [WARN] No scores available — skipping summary figure.")
        return
    pipelines = sorted(df["pipeline"].unique())
    fig, axes = plt.subplots(1, len(pipelines), figsize=(6 * len(pipelines), 5),
                             sharey=True)
    if len(pipelines) == 1:
        axes = [axes]

    for ax, pid in zip(axes, pipelines):
        sub = df[df["pipeline"] == pid].copy()
        sub["system"] = pd.Categorical(sub["system"], categories=SYSTEM_ORDER, ordered=True)
        sub = sub.sort_values("system")
        labels = sub["system"].astype(str).tolist()
        vals   = sub["BLEU"].tolist()
        colors = [SYSTEM_COLORS.get(s, "#888888") for s in labels]
        bars = ax.bar(labels, vals, color=colors, edgecolor="white", alpha=0.9)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9, fontweight="bold")
        ax.set_title(f"Pipeline {pid}\n({PIPELINE_CONFIGS[pid]['label']})",
                    fontweight="bold", fontsize=11)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("BLEU")

    plt.suptitle("BLEU Across All Pipelines — Baseline → R1 → Self-Training Rounds",
                fontsize=14, y=1.05)
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, "all_pipelines_summary.png")
    plt.savefig(path)
    plt.close()
    print(f"  ✓ {path}")


# ─── LaTeX export ─────────────────────────────────────────────────────────────

def export_latex(df: pd.DataFrame, out_path: str):
    if df.empty:
        print("  [WARN] No scores available — skipping LaTeX export.")
        return
    tbl = df[["direction", "system_label", "BLEU", "chrF", "chrF++", "TER", "COMET"]].copy()
    tbl.columns = ["Direction", "System", "BLEU", "chrF", "chrF++", "TER", "COMET"]
    for c in ["BLEU", "chrF", "chrF++", "TER", "COMET"]:
        tbl[c] = tbl[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "--")

    tbl["Direction"] = tbl["Direction"].str.replace("→", r" $\rightarrow$ ", regex=False)

    latex = tbl.to_latex(
        index=False, escape=False,
        caption="Translation quality across systems for every Indic pair and direction.",
        label="tab:results",
    )
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"  ✓ LaTeX table → {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pipeline", nargs="+", choices=ALL_PIPELINES, default=ALL_PIPELINES,
        metavar="ID", help="Pipeline(s) to compare. Default: all.",
    )
    args = parser.parse_args()
    pipelines = args.pipeline

    print("\n" + "=" * 64)
    print("  STEP 6 — FINAL COMPARISON, SIGNIFICANCE TESTS & LATEX EXPORT")
    print(f"  Pipelines : {pipelines}")
    print("=" * 64)

    scores = _load_scores()
    if not scores:
        print(f"\n  [WARN] {SCORES_FILE} not found or empty. "
              f"Run steps 2-5 first. Nothing to compare.")
        return 1

    # ── Comparison table ──────────────────────────────────────────────────────
    df = build_comparison_table(pipelines, scores)
    table_path = os.path.join(OUTPUT_DIR, "comparison_table.csv")
    df.to_csv(table_path, index=False)
    print(f"\n  ✓ Comparison table → {table_path}")
    if not df.empty:
        display_cols = ["pipeline", "system", "BLEU", "chrF", "COMET",
                        "ΔBLEU_vs_baseline", "ΔBLEU_vs_r1"]
        print("\n" + df[display_cols].to_string(index=False))

    # ── Significance testing ──────────────────────────────────────────────────
    print(f"\n{'─'*64}\n  SIGNIFICANCE TESTING (paired bootstrap, n={N_BOOTSTRAP})\n{'─'*64}")
    all_sig = {}
    for pid in pipelines:
        all_sig[pid] = run_significance_tests(pid, scores)

    # ── Plots ──────────────────────────────────────────────────────────────────
    print(f"\n{'─'*64}\n  PLOTS\n{'─'*64}")
    for pid in pipelines:
        fig_dir = os.path.join(FIGURES_DIR, f"pipeline_{pid}")
        plot_pipeline_comparison(df, pid, fig_dir)
    plot_all_pipelines_summary(df)

    # ── LaTeX ──────────────────────────────────────────────────────────────────
    latex_path = os.path.join(OUTPUT_DIR, "results_latex.tex")
    export_latex(df, latex_path)

    print("\n" + "=" * 64)
    print("  STEP 6 COMPLETE")
    print("=" * 64)
    print(f"  Comparison table : {table_path}")
    print(f"  LaTeX table      : {latex_path}")
    print(f"  Figures          : {FIGURES_DIR}/")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
