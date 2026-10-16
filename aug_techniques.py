"""
aug_techniques.py — Iterative Self-Training Augmentation Engine
────────────────────────────────────────────────────────────────────────────

"""

import os
import gc
import random
import unicodedata
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:
    raise ImportError(
        "IndicTransToolkit required.\n"
        "  pip install git+https://github.com/VarunGumma/IndicTransToolkit.git"
    )

from config import (
    PIPELINE_CONFIGS, SEED, BF16, FP16,
    INFER_BATCH_SIZE, NUM_BEAMS, MAX_NEW_TOKENS,
    MAX_SEQ_LEN, LENGTH_PENALTY, NO_REPEAT_NGRAM_SIZE,
    CYCLE_PARTNER,
    ST_SAMPLE_RATIO_R1, ST_SAMPLE_RATIO_R2, ST_SAMPLE_RATIO_R3,
    ST_CONF_THRESHOLD_R1, ST_CONF_THRESHOLD_R2, ST_CONF_THRESHOLD_R3,
    ST_CYCLE_THRESHOLD,
    ST_NOISE_SWAP_PROB, ST_NOISE_DELETE_PROB, ST_NOISE_INSERT_PROB,
    ST_NOISE_INJECT_FRACTION,
    LABSE_UNSUPPORTED_LANGS, ST_CYCLE_CHRF_WEIGHT, ST_CYCLE_CHRF_FLOOR,
    ST_KEEP_FRACTION, ST_LEN_RATIO_MIN, ST_LEN_RATIO_MAX,
    get_base_model, adapter_dir, synth_cache_path,
)
import sacrebleu
from aug_utils import adapter_ready

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if BF16 else (torch.float16 if FP16 else torch.float32)

_LANG_TAG_RE = None
import re as _re
_LANG_TAG_RE = _re.compile(r'^(?:[a-z]{3}_[A-Za-z]{4}\s+){1,2}')


# ─── Curriculum definition — one entry per self-training round ──────────────
ROUND_CONFIG = {
    1: {
        "exp":                  "st1",
        "teacher_exp":          "r1",
        "cycle_teacher_exp":    "r1",
        "sample_ratio":         ST_SAMPLE_RATIO_R1,
        "conf_threshold":       ST_CONF_THRESHOLD_R1,
        "use_cycle_consistency": True,
        "use_noisy_student":    False,
        "label":                "Self-Training Round 1",
    },
    2: {
        "exp":                  "st2",
        "teacher_exp":          "st1",
        "cycle_teacher_exp":    "st1",
        "sample_ratio":         ST_SAMPLE_RATIO_R2,
        "conf_threshold":       ST_CONF_THRESHOLD_R2,
        "use_cycle_consistency": True,
        "use_noisy_student":    True,
        "label":                "Self-Training Round 2 (+noisy-student)",
    },
    3: {
        "exp":                  "st3",
        "teacher_exp":          "st2",
        "cycle_teacher_exp":    "st2",
        "sample_ratio":         ST_SAMPLE_RATIO_R3,
        "conf_threshold":       ST_CONF_THRESHOLD_R3,
        "use_cycle_consistency": True,
        "use_noisy_student":    True,
        "label":                "Self-Training Round 3 (tightened confidence)",
    },
}


_ADAPTER_FALLBACK_ORDER = ["st3", "st2", "st1", "r1"]


def _resolve_adapter(pipeline_id: str, desired_exp: str) -> str:
    """Return the best available adapter path for `pipeline_id`, falling
    back to earlier rounds (and finally r1) if `desired_exp` isn't ready."""
    start = _ADAPTER_FALLBACK_ORDER.index(desired_exp) if desired_exp in _ADAPTER_FALLBACK_ORDER else 0
    for exp in _ADAPTER_FALLBACK_ORDER[start:]:
        path = adapter_dir(pipeline_id, exp)
        if adapter_ready(path):
            return path

    return adapter_dir(pipeline_id, "r1")


# ─── Language-tag stripping ───────────────────────────────────────────────────

def _strip_lang_tag(text: str) -> str:
    return _LANG_TAG_RE.sub("", text).strip()

def _strip_lang_tags(texts: list) -> list:
    return [unicodedata.normalize("NFC", _strip_lang_tag(t)) for t in texts]


# ─── IT2 model loading ────────────────────────────────────────────────────────

def _load_it2_model(pipeline_id: str, adapter_path: str):
    base_name    = get_base_model(pipeline_id)
    model_kwargs = {"trust_remote_code": True, "torch_dtype": DTYPE}
    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass

    print(f"  [IT2] Loading base: {base_name}  (pipeline {pipeline_id})")
    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    base      = AutoModelForSeq2SeqLM.from_pretrained(base_name, **model_kwargs)

    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(
            f"Adapter not found at {adapter_path}. "
            f"Run step3_train_dora.py --pipeline {pipeline_id} first."
        )
    print(f"  [IT2] Attaching adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)

    model = model.to(DEVICE)
    model.eval()
    ip = IndicProcessor(inference=True)
    return model, tokenizer, ip


def _free_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─── Core translate helpers ───────────────────────────────────────────────────

def _translate_batch(sentences, model, tokenizer, ip, src_lang, tgt_lang):
    batch  = ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(
        batch, truncation=True, padding="longest",
        return_tensors="pt", max_length=MAX_SEQ_LEN,
    ).to(DEVICE)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            use_cache=True, min_length=0,
            max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS,
            length_penalty=LENGTH_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            early_stopping=True,
        )

    with tokenizer.as_target_tokenizer():
        decoded = tokenizer.batch_decode(
            generated.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    translations = ip.postprocess_batch(decoded, lang=tgt_lang)
    return _strip_lang_tags(translations)


def _batch_translate(sentences, model, tokenizer, ip, src_lang, tgt_lang,
                     batch_size=INFER_BATCH_SIZE, desc="Translating"):
    all_preds = []
    for i in tqdm(range(0, len(sentences), batch_size), desc=f"  [{desc}]"):
        chunk = sentences[i:i+batch_size]
        preds = _translate_batch(chunk, model, tokenizer, ip, src_lang, tgt_lang)
        all_preds.extend(preds)
    return all_preds


# ─── LaBSE quality filter ─────────────────────────────────────────────────────

_LABSE_MODEL = None

def _get_labse():
    global _LABSE_MODEL
    if _LABSE_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _LABSE_MODEL = SentenceTransformer("sentence-transformers/LaBSE")
    return _LABSE_MODEL


def _labse_filter(src_texts: list, tgt_texts: list, threshold: float):
    try:
        labse = _get_labse()
        print(f"  [LaBSE] Scoring {len(src_texts):,} pairs (threshold={threshold}) ...")
        src_emb = labse.encode(src_texts, batch_size=128,
                               normalize_embeddings=True, show_progress_bar=False)
        tgt_emb = labse.encode(tgt_texts, batch_size=128,
                               normalize_embeddings=True, show_progress_bar=False)
        scores  = (src_emb * tgt_emb).sum(axis=1)
        mask    = scores >= threshold
        print(f"  [LaBSE] Kept {mask.sum():,}/{len(src_texts):,} "
              f"(mean={scores.mean():.3f})")
        return mask, scores
    except ImportError:
        print("  [LaBSE] sentence-transformers not installed — skipping filter.")
        scores = np.ones(len(src_texts), dtype=np.float32)
        return scores >= 0.0, scores


# ─── Cache helpers ────────────────────────────────────────────────────────────

def _load_or_generate(pipeline_id: str, technique: str,
                      generator_fn, *args, **kwargs) -> pd.DataFrame:
    cache = synth_cache_path(pipeline_id, technique)
    if os.path.exists(cache):
        df = pd.read_excel(cache)
        print(f"  [CACHE HIT] pipeline_{pipeline_id}/{technique}: "
              f"{len(df):,} pairs from {cache}")
        return df
    print(f"  [CACHE MISS] pipeline_{pipeline_id}/{technique}: generating ...")
    df = generator_fn(*args, **kwargs)
    if len(df) > 0:
        df.to_excel(cache, index=False)
        print(f"  [CACHED] {len(df):,} pairs → {cache}")
    return df


# ─── Noisy-student source perturbation ───────────────────────────────────────

def _noise_inject(text: str, vocab: list = None) -> str:
    """Apply word-level noise: swap, delete, or insert random words."""
    words = text.split()
    if len(words) < 2:
        return text

    new_words = words[:]
    for i in range(len(new_words) - 1):
        if random.random() < ST_NOISE_SWAP_PROB:
            new_words[i], new_words[i+1] = new_words[i+1], new_words[i]

    new_words = [w for w in new_words if random.random() > ST_NOISE_DELETE_PROB]
    if not new_words:
        new_words = words[:]

    if vocab:
        result = []
        for w in new_words:
            result.append(w)
            if random.random() < ST_NOISE_INSERT_PROB:
                result.append(random.choice(vocab))
        new_words = result

    return " ".join(new_words)


def _build_vocab(texts: list) -> list:
    return list({w for t in texts for w in str(t).split() if len(w) > 2})


# ─── Quality-score helpers ────────────────────────────────────────────────────

def _labse_ok(*langs) -> bool:
    """True if LaBSE can embed every language given (Bodo cannot)."""
    return not (set(langs) & LABSE_UNSUPPORTED_LANGS)


def _labse_scores(a: list, b: list) -> np.ndarray:
    """Raw cosine similarities (no thresholding)."""
    _, s = _labse_filter(a, b, -1.0)
    return np.asarray(s, dtype=np.float32)


def _chrfpp_scores(hyps: list, refs: list) -> np.ndarray:
    """Sentence-level chrF++ in [0, 1] (same-language comparison)."""
    return np.array([
        sacrebleu.sentence_chrf(h, [r], word_order=2).score / 100.0
        for h, r in zip(hyps, refs)
    ], dtype=np.float32)


def _describe(name: str, x: np.ndarray):
    q = np.percentile(x, [10, 25, 50, 75, 90])
    print(f"  [Scores] {name:<14} mean={x.mean():.3f}  "
          f"p10/25/50/75/90 = " + "/".join(f"{v:.2f}" for v in q))


# ─── Self-training core ───────────────────────────────────────────────────────

def _run_self_training(train_df: pd.DataFrame, pipeline_id: str,
                       round_num: int) -> pd.DataFrame:
    cfg = PIPELINE_CONFIGS[pipeline_id]
    rc  = ROUND_CONFIG[round_num]

    src_lang = cfg["src_lang"]
    tgt_lang = cfg["tgt_lang"]

    df = train_df.dropna(subset=["src_text", "tgt_text"]).reset_index(drop=True)

    if "aug_type" in df.columns:
        df = df[df["aug_type"].fillna("original") == "original"].reset_index(drop=True)
    n  = min(int(len(df) * rc["sample_ratio"]), len(df))
    sample = df.sample(n=n, random_state=SEED + round_num * 100).reset_index(drop=True)

    print(f"\n{'─'*65}")
    print(f"  [{rc['label']}]  pipeline {pipeline_id}  ({cfg['label']})")
    print(f"  Sampling {n:,}/{len(df):,} sources "
          f"(ratio={rc['sample_ratio']})")
    print(f"{'─'*65}")

    # ── 1. Teacher pseudo-labels the sampled sources ──────────────────────────
    teacher_adapter = _resolve_adapter(pipeline_id, rc["teacher_exp"])
    print(f"  Teacher adapter : {teacher_adapter}")

    model, tokenizer, ip = _load_it2_model(pipeline_id, teacher_adapter)
    sources = sample["src_text"].astype(str).tolist()
    pseudo  = _batch_translate(
        sources, model, tokenizer, ip, src_lang, tgt_lang,
        desc=f"ST-R{round_num} teacher pipeline {pipeline_id}",
    )
    _free_model(model)

    # Validity filter: non-empty, ≥2 tokens, not identical to source
    valid_idx = [i for i in range(len(pseudo))
                 if pseudo[i].strip() and len(pseudo[i].split()) >= 2
                 and pseudo[i].strip() != sources[i].strip()]
    if not valid_idx:
        print(f"  [ST-R{round_num}] No valid pseudo pairs after basic filtering.")
        return pd.DataFrame(columns=["src_text","tgt_text","aug_type","qa_score"])

    v_src    = [sources[i] for i in valid_idx]
    v_pseudo = [pseudo[i]  for i in valid_idx]

    # ── 2. Length-ratio guard (cheap hallucination / truncation filter) ───────
    ok = [i for i in range(len(v_src))
          if ST_LEN_RATIO_MIN <= len(v_pseudo[i].split()) /
             max(len(v_src[i].split()), 1) <= ST_LEN_RATIO_MAX]
    print(f"  [Len] Kept {len(ok):,}/{len(v_src):,} within ratio "
          f"[{ST_LEN_RATIO_MIN}, {ST_LEN_RATIO_MAX}]")
    if not ok:
        print(f"  [ST-R{round_num}] No pairs passed length guard.")
        return pd.DataFrame(columns=["src_text","tgt_text","aug_type","qa_score"])
    v_src    = [v_src[i]    for i in ok]
    v_pseudo = [v_pseudo[i] for i in ok]

    # ── 3. Round-trip through the SIBLING direction (cycle-consistency) ───────

    round_trip = None
    if rc["use_cycle_consistency"] and pipeline_id in CYCLE_PARTNER:
        partner_id  = CYCLE_PARTNER[pipeline_id]
        partner_cfg = PIPELINE_CONFIGS[partner_id]
        if partner_cfg["src_lang"] == tgt_lang and partner_cfg["tgt_lang"] == src_lang:
            partner_adapter = _resolve_adapter(partner_id, rc["cycle_teacher_exp"])
            print(f"  Cycle partner   : pipeline {partner_id}  "
                  f"({partner_cfg['label']})  adapter={partner_adapter}")
            try:
                p_model, p_tok, p_ip = _load_it2_model(partner_id, partner_adapter)
                round_trip = _batch_translate(
                    v_pseudo, p_model, p_tok, p_ip,
                    partner_cfg["src_lang"], partner_cfg["tgt_lang"],
                    desc=f"ST-R{round_num} cycle-check pipeline {pipeline_id}",
                )
                _free_model(p_model)
            except FileNotFoundError as e:
                print(f"  [Cycle] Skipped — partner adapter unavailable ({e})")
        else:
            print(f"  [Cycle] Skipped — partner pipeline {partner_id} does not "
                  f"translate {tgt_lang}→{src_lang}.")

    # ── 4. Combined quality score ─────────────────────────────────────────────
    cross_ok = _labse_ok(src_lang, tgt_lang)   # src↔pseudo comparable?
    same_ok  = _labse_ok(src_lang)             # src↔round-trip comparable?

    labse_signals = []
    conf = None
    if cross_ok:
        conf = _labse_scores(v_src, v_pseudo)
        labse_signals.append(conf)
        _describe("LaBSE src↔pseu", conf)
    else:
        print(f"  [Conf] LaBSE unsupported for {src_lang}→{tgt_lang} — "
              f"cross-lingual filter skipped.")

    cyc_chrf = None
    if round_trip is not None:
        cyc_chrf = _chrfpp_scores(round_trip, v_src)
        _describe("cycle chrF++", cyc_chrf)
        if same_ok:
            cyc_labse = _labse_scores(v_src, round_trip)
            labse_signals.append(cyc_labse)
            _describe("cycle LaBSE", cyc_labse)

    if cyc_chrf is None and not labse_signals:
        print(f"  [ST-R{round_num}] No usable quality signal — returning no pairs.")
        return pd.DataFrame(columns=["src_text","tgt_text","aug_type","qa_score"])

    if labse_signals:
        labse_part = np.mean(labse_signals, axis=0)
        combined = (ST_CYCLE_CHRF_WEIGHT * cyc_chrf + (1 - ST_CYCLE_CHRF_WEIGHT) * labse_part
                    if cyc_chrf is not None else labse_part)
    else:
        combined = cyc_chrf
    _describe("combined", combined)

    # ── 5. Floors, then rank-based selection ─────────────────────────────────
    passed = np.ones(len(v_src), dtype=bool)
    if cyc_chrf is not None:
        passed &= cyc_chrf >= ST_CYCLE_CHRF_FLOOR
    if conf is not None:
        passed &= conf >= rc["conf_threshold"]
    print(f"  [Floors] {passed.sum():,}/{len(v_src):,} pass absolute floors")

    cand_idx = np.where(passed)[0]
    if len(cand_idx) == 0:
        print(f"  [ST-R{round_num}] No pairs survived floors.")
        return pd.DataFrame(columns=["src_text","tgt_text","aug_type","qa_score"])
    keep_frac = ST_KEEP_FRACTION[round_num]
    n_keep    = max(1, int(len(cand_idx) * keep_frac))
    order     = cand_idx[np.argsort(-combined[cand_idx])][:n_keep]
    print(f"  [Select] Keeping top {keep_frac:.0%} by combined score "
          f"→ {len(order):,} pairs (min kept score={combined[order].min():.3f})")

    # Save every scored candidate so thresholds can be tuned from data.
    try:
        kept_set = set(order.tolist())
        dbg = pd.DataFrame({"src": v_src, "pseudo": v_pseudo, "combined": combined,
                            "cycle_chrfpp": cyc_chrf if cyc_chrf is not None else np.nan,
                            "labse_conf":   conf     if conf     is not None else np.nan,
                            "kept": [i in kept_set for i in range(len(v_src))]})
        dbg_path = os.path.join(os.path.dirname(synth_cache_path(pipeline_id, "x")),
                                f"{rc['exp']}_scored_candidates.xlsx")
        dbg.to_excel(dbg_path, index=False)
        print(f"  [Debug] Scored candidates → {dbg_path}")
    except Exception as e:
        print(f"  [Debug] Could not save scored candidates ({e})")

    k_src        = [v_src[i]    for i in order]
    k_pseudo     = [v_pseudo[i] for i in order]
    k_conf       = [float(combined[i]) for i in order]
    cycle_scores = [float(cyc_chrf[i]) if cyc_chrf is not None else None for i in order]

    df_out = pd.DataFrame({
        "src_text":    k_src,
        "tgt_text":    k_pseudo,
        "aug_type":    f"self_train_r{round_num}",
        "qa_score":    k_conf,
        "cycle_score": cycle_scores,
        "pipeline":    pipeline_id,
    })
    df_out = df_out[
        df_out["src_text"].str.strip().ne("") &
        df_out["tgt_text"].str.strip().ne("")
    ].reset_index(drop=True)

    print(f"  [ST-R{round_num}] Pseudo pairs after all filters: {len(df_out):,}")

    # ── 4. Noisy-student twins (NOVELTY, Round ≥ 2) ───────────────────────────
    if rc["use_noisy_student"] and len(df_out) > 0:
        vocab   = _build_vocab(df["src_text"].astype(str).tolist())
        n_noisy = int(len(df_out) * ST_NOISE_INJECT_FRACTION)
        if n_noisy > 0:
            noisy_rows = df_out.sample(
                n=n_noisy, random_state=SEED + round_num * 200
            ).copy()
            noisy_rows["src_text"] = noisy_rows["src_text"].apply(
                lambda s: _noise_inject(s, vocab))
            noisy_rows["aug_type"] = f"self_train_r{round_num}_noisy_student"
            # Drop rows where noise happened to no-op or empty the source
            noisy_rows = noisy_rows[
                noisy_rows["src_text"].str.strip().ne("") &
                (noisy_rows["src_text"] != df_out.loc[noisy_rows.index, "src_text"])
            ]
            print(f"  [Noisy-Student] Added {len(noisy_rows):,} noised-source twins")
            df_out = pd.concat([df_out, noisy_rows], ignore_index=True)

    return df_out.reset_index(drop=True)


def generate_self_training(train_df: pd.DataFrame, pipeline_id: str,
                           round_num: int) -> pd.DataFrame:
    """Public entry-point: Self-Training Round `round_num` (1, 2, or 3)."""
    exp = ROUND_CONFIG[round_num]["exp"]
    return _load_or_generate(pipeline_id, exp, _run_self_training,
                             train_df, pipeline_id, round_num)
