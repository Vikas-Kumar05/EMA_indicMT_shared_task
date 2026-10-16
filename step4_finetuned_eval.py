"""
step4_finetuned_eval.py — Evaluation of R1 Fine-tuned Models (Indic↔Indic)
─────────────────────────────────────────────────────────────────────────
"""

import os, json, re, time, gc, warnings, argparse
import pandas as pd
import torch
import unicodedata
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

warnings.filterwarnings("ignore")

from config import (
    ALL_PIPELINES, PIPELINE_CONFIGS, SCORES_FILE,
    INFER_BATCH_SIZE, NUM_BEAMS, MAX_NEW_TOKENS, MAX_SEQ_LEN,
    LENGTH_PENALTY, NO_REPEAT_NGRAM_SIZE,
    BF16, FP16, COMET_MODEL, SEED,
    split_path, preds_path, adapter_dir, get_base_model, sacrebleu_tokenize,
)



def eng_target_generate_kwargs(tokenizer, tgt_lang):
    return {}

def normalize_translation_output(translations, tgt_lang):
    translations = [unicodedata.normalize("NFC", _strip_zero_width(t)) for t in translations]
    return _normalize_output_script(translations, tgt_lang)

torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if BF16 else (torch.float16 if FP16 else torch.float32)

try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:
    raise ImportError(
        "IndicTransToolkit required. Install:\n"
        "  pip install git+https://github.com/VarunGumma/IndicTransToolkit.git"
    )


_LANG_TAG_RE = re.compile(r'^(?:[a-z]{3}(?:_[A-Za-z]{4})?\s+){1,2}')


_ZERO_WIDTH_RE = re.compile(r'[\u200b\u200c\u200d\u200e\u200f\u2060\ufeff]')


def _strip_zero_width(text: str) -> str:
    return _ZERO_WIDTH_RE.sub('', text)


def _strip_lang_tag(text: str) -> str:
    return _LANG_TAG_RE.sub("", text).strip()


def _strip_lang_tags(texts: list) -> list:
    return [unicodedata.normalize("NFC", _strip_zero_width(_strip_lang_tag(t)))
            for t in texts]


# ─── Output-side script normalization (reuses step1_analysis.py) ─────────────

try:
    from step1_analysis import _apply_script_normalization
except ImportError:
    _apply_script_normalization = None
    warnings.warn(
        "Could not import _apply_script_normalization from step1_analysis.py "
        "(file not found on path). Falling back to no-op — predictions will "
        "NOT have digits converted to native script. Ensure step1_analysis.py "
        "is importable (same directory or on PYTHONPATH)."
    )


def _normalize_output_script(texts: list, tgt_lang: str) -> list:
    if _apply_script_normalization is None:
        return texts
    return _apply_script_normalization(pd.Series(texts), tgt_lang).tolist()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_scores() -> dict:
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE) as f:
            return json.load(f)
    return {}


def _save_scores(scores: dict):
    with open(SCORES_FILE, "w") as f:
        json.dump(scores, f, indent=2)


# ─── Inference ────────────────────────────────────────────────────────────────

def translate_batch(sentences, model, tokenizer, ip, src_lang, tgt_lang):
    batch  = ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(
        batch, truncation=True, padding="longest",
        return_tensors="pt", max_length=MAX_SEQ_LEN,
    ).to(DEVICE)

    extra_gen_kwargs = eng_target_generate_kwargs(tokenizer, tgt_lang)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            use_cache=True, min_length=0,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            length_penalty=LENGTH_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            early_stopping=True,
            **extra_gen_kwargs,
        )

    with tokenizer.as_target_tokenizer():
        decoded = tokenizer.batch_decode(
            generated.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    translations = ip.postprocess_batch(decoded, lang=tgt_lang)
    translations = _strip_lang_tags(translations)

    translations = normalize_translation_output(translations, tgt_lang)
    return translations


def run_pipeline(pipeline_id: str) -> pd.DataFrame:
    cfg        = PIPELINE_CONFIGS[pipeline_id]
    src_lang   = cfg["src_lang"]
    tgt_lang   = cfg["tgt_lang"]
    base_name  = get_base_model(pipeline_id)
    adapter_path = adapter_dir(pipeline_id, "r1")

    print(f"\n{'='*64}")
    print(f"  R1 EVAL  Pipeline {pipeline_id}  ({cfg['label']})")
    print(f"  Base   : {base_name}")
    print(f"  Adapter: {adapter_path}")
    print(f"{'='*64}")

    if not os.path.isdir(adapter_path) or not os.listdir(adapter_path):
        raise FileNotFoundError(
            f"No R1 adapter found at {adapter_path}. "
            f"Run: python step3_train_dora.py --pipeline {pipeline_id}"
        )

    test_df = pd.read_excel(split_path(pipeline_id, "test"))
    test_df = test_df.dropna(subset=["src_text", "tgt_text"]).reset_index(drop=True)
    print(f"  Test pairs : {len(test_df):,}")


    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)

    model_kwargs = {"trust_remote_code": True, "torch_dtype": DTYPE}
    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass

    print("  Loading base model ...")
    base = AutoModelForSeq2SeqLM.from_pretrained(base_name, **model_kwargs)
    print("  Attaching R1 adapter ...")
    model = PeftModel.from_pretrained(base, adapter_path).to(DEVICE)
    model.eval()

    ip = IndicProcessor(inference=True)


    sources    = [_strip_zero_width(s) for s in test_df["src_text"].astype(str)]
    references = [_strip_zero_width(t) for t in test_df["tgt_text"].astype(str)]

    all_preds = []
    t0 = time.time()
    for i in tqdm(range(0, len(sources), INFER_BATCH_SIZE),
                  desc=f"  Translating pipeline {pipeline_id}"):
        preds = translate_batch(sources[i:i+INFER_BATCH_SIZE],
                                model, tokenizer, ip, src_lang, tgt_lang)
        all_preds.extend(preds)
    print(f"  Inference : {time.time()-t0:.1f}s  ({len(sources)/(time.time()-t0):.1f} s/s)")

    out_df = pd.DataFrame({
        "pipeline":   pipeline_id,
        "src_lang":   src_lang, "tgt_lang": tgt_lang,
        "source":     sources,  "reference": references,
        "prediction": all_preds,
    })
    out_path = preds_path(pipeline_id, "r1")
    out_df.to_csv(out_path, index=False)
    print(f"  ✓ Saved → {out_path}")

    del model, base, tokenizer
    gc.collect(); torch.cuda.empty_cache()
    return out_df


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(preds_df: pd.DataFrame, pipeline_id: str) -> dict:
    import sacrebleu
    from comet import download_model, load_from_checkpoint

    sources    = preds_df["source"].astype(str).tolist()
    references = preds_df["reference"].astype(str).tolist()
    hypotheses = preds_df["prediction"].astype(str).tolist()

    references = [unicodedata.normalize("NFC", _strip_zero_width(r)) for r in references]
    hypotheses = [unicodedata.normalize("NFC", _strip_zero_width(h)) for h in hypotheses]

    tok          = sacrebleu_tokenize(pipeline_id)
    refs_wrapped = [references]

    bleu   = sacrebleu.corpus_bleu(hypotheses, refs_wrapped, tokenize=tok).score
    chrf   = sacrebleu.corpus_chrf(hypotheses, refs_wrapped).score
    chrfpp = sacrebleu.corpus_chrf(hypotheses, refs_wrapped, word_order=2).score
    ter    = sacrebleu.corpus_ter(hypotheses, refs_wrapped).score

    print(f"  Loading COMET ({COMET_MODEL}) ...")
    comet_path  = download_model(COMET_MODEL)
    comet_model = load_from_checkpoint(comet_path)
    comet_data  = [{"src": s, "mt": h, "ref": r}
                   for s, h, r in zip(sources, hypotheses, references)]
    comet_out   = comet_model.predict(comet_data, batch_size=16,
                                      gpus=1 if DEVICE == "cuda" else 0)
    comet_score = round(float(comet_out.system_score) * 100, 4)

    del comet_model; gc.collect(); torch.cuda.empty_cache()

    return {
        "BLEU":    round(bleu,   4),
        "chrF":    round(chrf,   4),
        "chrF++":  round(chrfpp, 4),
        "TER":     round(ter,    4),
        "COMET":   comet_score,
        "N":       len(hypotheses),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pipeline", nargs="+", choices=ALL_PIPELINES, default=ALL_PIPELINES,
        metavar="ID",
    )
    args = parser.parse_args()

    print("\n" + "="*64)
    print("  STEP 4 — R1 FINE-TUNED EVALUATION")
    print(f"  Pipelines : {args.pipeline}")
    print("="*64)

    all_scores = _load_scores()

    for pid in args.pipeline:
        score_key = f"r1_{pid}"
        if score_key in all_scores and os.path.exists(preds_path(pid, "r1")):
            print(f"\n  [SKIP] {score_key} already in scores.")
            continue

        preds_df = run_pipeline(pid)
        metrics  = compute_metrics(preds_df, pid)
        print(f"\n  Pipeline {pid} R1 metrics: {metrics}")

        all_scores[score_key] = metrics
        _save_scores(all_scores)

    print("\n" + "="*64)
    print("  R1 FINE-TUNED SUMMARY")
    print("="*64)
    print(f"  {'Pipeline':<12} {'BLEU':>8} {'chrF':>8} {'chrF++':>8} "
          f"{'TER':>8} {'COMET':>8}")
    print(f"  {'-'*56}")
    for pid in args.pipeline:
        for k_prefix, label in [("baseline", "Baseline"), ("r1", "R1")]:
            k = f"{k_prefix}_{pid}"
            if k in all_scores:
                m = all_scores[k]
                print(f"  {pid}-{label:<9} "
                      f"{m.get('BLEU',0):>8.4f} {m.get('chrF',0):>8.4f} "
                      f"{m.get('chrF++',0):>8.4f} {m.get('TER',0):>8.4f} "
                      f"{m.get('COMET',0):>8.4f}")
    print(f"\n  ✓ Scores → {SCORES_FILE}")
    print("="*64 + "\n")


if __name__ == "__main__":
    main()
