"""
step5_selftrain_common.py — Shared Driver for Self-Training Rounds 1–3
─────────────────────────────────────────────────────────────────────────────

"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch

from config import (
    ALL_PIPELINES, PIPELINE_CONFIGS, SEED,
    split_path, adapter_dir, aug_dataset_path, preds_path,
)
from aug_techniques import generate_self_training, ROUND_CONFIG, _resolve_adapter
from aug_utils import combine_and_save, dataset_stats, adapter_ready, experiment_done
from aug_finetune_eval import run_finetune_and_eval

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def run_selftrain_round(round_num: int, pipelines: list = None):
    rc  = ROUND_CONFIG[round_num]
    exp = rc["exp"]                      # e.g. "st1"
    prev_exp = rc["teacher_exp"]         # dataset this round builds on, e.g. "r1"
    label = rc["label"]

    pipelines = pipelines or ALL_PIPELINES

    print("\n" + "=" * 65)
    print(f"  STEP 5{chr(ord('a') + round_num - 1)} — {label.upper()}")
    print(f"  Pipelines : {pipelines}")
    print("=" * 65)

    dataset_files    = {}
    adapter_out_dirs = {}
    pred_files_map   = {}

    for pid in pipelines:
        cfg = PIPELINE_CONFIGS[pid]
        print(f"\n{'─' * 65}")
        print(f"  Pipeline {pid}  ({cfg['label']})")
        print(f"{'─' * 65}")

        exp_adapter = adapter_dir(pid, exp)
        exp_dataset = aug_dataset_path(pid, exp)
        exp_preds   = preds_path(pid, exp)

        if adapter_ready(exp_adapter) and os.path.exists(exp_dataset):
            if experiment_done(pid, exp, exp_adapter, exp_preds):
                print(f"  [RESUME] {label} already fully complete "
                      f"(adapter + dataset + preds + scores present).")
                print(f"           Skipping pipeline {pid} entirely.")
            else:
                print(f"  [RESUME] {label} adapter & dataset already exist "
                      f"for pipeline {pid}.")
                print(f"           Skipping self-training generation / "
                      f"training — will (re)run evaluation only.")
            dataset_files[pid]    = exp_dataset
            adapter_out_dirs[pid] = exp_adapter
            pred_files_map[pid]   = exp_preds
            continue

        # ── Base dataset this round starts from ────────────────────────────────
        # Round 1 starts from the gold (cleaned) training set. Round 2/3 start
        # from the PREVIOUS round's already-combined dataset (gold + earlier
        # rounds' accepted pseudo pairs), so accepted pseudo pairs compound
        # across rounds rather than being regenerated each time.
        if round_num == 1:
            base_df = pd.read_excel(split_path(pid, "train"))
        else:
            prev_dataset = aug_dataset_path(pid, prev_exp)
            if not os.path.exists(prev_dataset):
                raise FileNotFoundError(
                    f"Round {round_num} for pipeline {pid} needs the Round "
                    f"{round_num - 1} dataset at {prev_dataset}. Run "
                    f"step5{chr(ord('a') + round_num - 2)}_selftrain_round"
                    f"{round_num - 1}.py first."
                )
            base_df = pd.read_excel(prev_dataset)

        base_df = base_df.dropna(subset=["src_text", "tgt_text"]).reset_index(drop=True)
        print(f"\n  Base training rows (round {round_num} input): {len(base_df):,}")
        dataset_stats(base_df, f"round {round_num} base pipeline {pid}")

        # ── Self-training: teacher pseudo-labels a sample, filters, returns ────
        pseudo_df = generate_self_training(base_df, pipeline_id=pid, round_num=round_num)
        print(f"  Self-training pseudo pairs (round {round_num}): {len(pseudo_df):,}")

        # ── Combine: round-N base + new pseudo pairs ────────────────────────────
        aug_frames = [pseudo_df] if len(pseudo_df) > 0 else []
        save_path  = aug_dataset_path(pid, exp)

        combined = combine_and_save(
            orig_df    = base_df,
            aug_frames = aug_frames,
            save_path  = save_path,
            tag        = f"pipeline_{pid}_{exp}",
        )
        dataset_stats(combined, f"{exp} combined pipeline {pid}")

        dataset_files[pid]    = save_path
        adapter_out_dirs[pid] = adapter_dir(pid, exp)
        pred_files_map[pid]   = preds_path(pid, exp)

    # ── Fine-tune + evaluate (each pipeline continues from its own teacher) ───
    print(f"\n{'─' * 65}")
    print(f"  Fine-Tune & Evaluate — {label}  ({pipelines})")
    print(f"{'─' * 65}")

    start_adapters = {
        pid: _resolve_adapter(pid, prev_exp)
        for pid in pipelines
    }

    results = run_finetune_and_eval(
        score_key        = exp,
        dataset_files     = dataset_files,
        adapter_out_dirs  = adapter_out_dirs,
        pred_files        = pred_files_map,
        pipelines         = pipelines,
        start_adapters    = start_adapters,
        label             = label,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  STEP 5{chr(ord('a') + round_num - 1)} COMPLETE — {label}")
    print(f"{'=' * 65}")
    for pid in pipelines:
        print(f"\n  Pipeline {pid}  ({PIPELINE_CONFIGS[pid]['label']})")
        print(f"    Adapter    : {adapter_dir(pid, exp)}")
        print(f"    Dataset    : {aug_dataset_path(pid, exp)}")
        print(f"    Preds      : {preds_path(pid, exp)}")
        if pid in results:
            m = results[pid]
            print(f"    BLEU={m.get('BLEU',0):.4f}  chrF={m.get('chrF',0):.4f}  "
                  f"COMET={m.get('COMET',0):.4f}")
    print(f"\n  Scores → outputs/all_scores.json  (keys: {exp}_<pipeline>)")
    print()

    return results


def build_argparser(round_num: int) -> argparse.ArgumentParser:
    label = ROUND_CONFIG[round_num]["label"]
    parser = argparse.ArgumentParser(
        description=f"Self-training round {round_num}: {label}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pipeline", nargs="+", choices=ALL_PIPELINES, default=ALL_PIPELINES,
        metavar="ID",
        help="Pipeline(s) to run. Default: all.",
    )
    return parser
