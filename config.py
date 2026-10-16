"""
config.py — Central Configuration (Indic↔Indic, Direction-Specific Design)
─────────────────────────────────────────────────────────────────────────────

"""

import os
import torch

# ─── Language registry (data-driven — add a language here, nothing else) ─────


LANG_REGISTRY = {
    "asm_Beng": {
        "name":          "Assamese",
        "script":        "Bengali-Assamese",
        "unicode_ranges": [(0x0980, 0x09FF)],
        "digit_map":     str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯"),
        "aliases":       {"asm", "assamese", "asm_beng", "as", "asm_src", "asm_tgt"},
    },
    "brx_Deva": {
        "name":          "Bodo",
        "script":        "Devanagari",
        "unicode_ranges": [(0x0900, 0x097F)],
        "digit_map":     str.maketrans("0123456789", "०१२३४५६७८९"),
        "aliases":       {"bodo", "brx", "brx_deva", "bodo_src", "bodo_tgt"},
    },
    "ben_Beng": {
        "name":          "Bengali",
        "script":        "Bengali-Assamese",
        "unicode_ranges": [(0x0980, 0x09FF)],
        "digit_map":     str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯"),
        "aliases":       {"bengali", "ben", "bn", "ben_beng", "bengali_src", "bengali_tgt"},
    },
    "ory_Orya": {
        "name":          "Odia",
        "script":        "Oriya",
        "unicode_ranges": [(0x0B00, 0x0B7F)],
        "digit_map":     str.maketrans("0123456789", "୦୧୨୩୪୫୬୭୮୯"),
        "aliases":       {"odia", "oriya", "or", "ory", "ory_orya", "odia_src", "odia_tgt"},
    },
}


def lang_name(lang_code: str) -> str:
    return LANG_REGISTRY.get(lang_code, {}).get("name", lang_code)


def script_ratio(text: str, lang_code: str) -> float:
    """Fraction of alphabetic chars in `text` that fall in lang_code's script block."""
    ranges = LANG_REGISTRY.get(lang_code, {}).get("unicode_ranges", [])
    if not ranges or not text:
        return 0.0
    total = hits = 0
    for ch in text:
        if ch.isalpha():
            total += 1
            cp = ord(ch)
            if any(lo <= cp <= hi for lo, hi in ranges):
                hits += 1
    return hits / max(total, 1)


# ─── Pipeline definitions ──────────────────────────────────────────────────────

PIPELINE_CONFIGS = {
    "E": {
        "src_lang":  "asm_Beng",
        "tgt_lang":  "brx_Deva",
        "direction": "indic2indic",
        "pair_id":   "asm_bodo",
        "label":     "asm→bodo",
        "primary":   True,
    },
    "F": {
        "src_lang":  "brx_Deva",
        "tgt_lang":  "asm_Beng",
        "direction": "indic2indic",
        "pair_id":   "asm_bodo",
        "label":     "bodo→asm",
        "primary":   True,
    },
    "G": {
        "src_lang":  "ben_Beng",
        "tgt_lang":  "ory_Orya",
        "direction": "indic2indic",
        "pair_id":   "ben_odia",
        "label":     "bengali→odia",
        "primary":   True,
    },
    "H": {
        "src_lang":  "ory_Orya",
        "tgt_lang":  "ben_Beng",
        "direction": "indic2indic",
        "pair_id":   "ben_odia",
        "label":     "odia→bengali",
        "primary":   True,
    },
}

ALL_PIPELINES    = list(PIPELINE_CONFIGS.keys())
PRIMARY_PIPELINE = "E"


CYCLE_PARTNER = {"E": "F", "F": "E", "G": "H", "H": "G"}

# All pipeline ids that share the same pair_id / raw data file.
PAIR_PIPELINES = {}
for _pid, _cfg in PIPELINE_CONFIGS.items():
    PAIR_PIPELINES.setdefault(_cfg["pair_id"], []).append(_pid)


# ─── Base HF checkpoints ──────────────────────────────────────────────────────
# A single Indic↔Indic checkpoint family serves ALL four directions.
BASE_MODELS = {
    "indic2indic": "/home/vikas1/vikas/model/indictrans2-indic-indic-1B",
}


def get_base_model(pipeline_id: str) -> str:
    """Return the HF model name for the given pipeline."""
    return BASE_MODELS[PIPELINE_CONFIGS[pipeline_id]["direction"]]


# ─── Paths ────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = "./outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")


RAW_DATA_DIR = "./data"

def raw_path(pipeline_id: str, split: str) -> str:
    """e.g. ./data/asm_bodo_train.xlsx (shared by pipelines E & F)."""
    pair_id = PIPELINE_CONFIGS[pipeline_id]["pair_id"]
    return os.path.join(RAW_DATA_DIR, f"{pair_id}_{split}.xlsx")


# underlying raw file is shared.
def split_path(pipeline_id: str, split: str) -> str:
    return os.path.join(OUTPUT_DIR, f"pipeline_{pipeline_id}", f"{split}.xlsx")

# ─── Adapter / checkpoint dirs (one per pipeline) ────────────────────────────
def adapter_dir(pipeline_id: str, exp: str = "r1") -> str:
    """
    exp: "r1" → initial DoRA/LoRA fine-tune on gold data
         "st1" → Self-Training Round 1 (teacher = r1)
         "st2" → Self-Training Round 2 (teacher = st1, noisy-student + cycle-consistency)
         "st3" → Self-Training Round 3 (teacher = st2, tightened confidence + cycle-consistency)
    """
    return os.path.join("./checkpoints", f"pipeline_{pipeline_id}", exp)

# ─── Augmentation cache dirs (one sub-dir per pipeline) ──────────────────────
def synth_cache_dir(pipeline_id: str) -> str:
    return os.path.join(OUTPUT_DIR, f"pipeline_{pipeline_id}", "synth_cache")

def synth_cache_path(pipeline_id: str, technique: str) -> str:
    """e.g. outputs/pipeline_E/synth_cache/st1.xlsx"""
    return os.path.join(synth_cache_dir(pipeline_id), f"{technique}.xlsx")

# ─── Augmented dataset files (one per experiment per pipeline) ───────────────
def aug_dataset_path(pipeline_id: str, exp: str) -> str:
    return os.path.join(OUTPUT_DIR, f"pipeline_{pipeline_id}", f"aug_{exp}.xlsx")

# ─── Prediction output files ─────────────────────────────────────────────────
def preds_path(pipeline_id: str, system: str) -> str:
    return os.path.join(OUTPUT_DIR, f"pipeline_{pipeline_id}", f"preds_{system}.csv")

# ─── Scores file (shared across all pipelines & systems) ─────────────────────
SCORES_FILE = os.path.join(OUTPUT_DIR, "all_scores.json")

# ─── Data hyperparameters ─────────────────────────────────────────────────────
SEED        = 16
MAX_SEQ_LEN = 256

# ─── DoRA / LoRA configuration ────────────────────────────────────────────────
USE_DORA     = False   # True = DoRA (decomposed magnitude), False = LoRA
USE_RSLORA   = True    # rank-stabilized LoRA scaling
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.1    
LORA_BIAS    = "none"
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "out_proj",
    "fc1", "fc2",
]

# ─── Training hyperparameters — R1 (step3) ────────────────────────────────────

NUM_TRAIN_EPOCHS        = 5     
PER_DEVICE_TRAIN_BATCH  = 16
PER_DEVICE_EVAL_BATCH   = 16
GRAD_ACCUM_STEPS        = 4
LEARNING_RATE           = 5e-5  
LR_SCHEDULER            = "cosine"
WARMUP_RATIO            = 0.05
WEIGHT_DECAY            = 0.01
MAX_GRAD_NORM           = 1.0
LABEL_SMOOTHING         = 0.1
SAVE_TOTAL_LIMIT        = 2
LOGGING_STEPS           = 25
EVAL_STEPS              = 200
SAVE_STEPS              = 200
EARLY_STOPPING_PATIENCE = 3


# ─── Training hyperparameters — Self-training rounds ─────────────────────────
NUM_TRAIN_EPOCHS_AUG = 3
LEARNING_RATE_AUG    = 1e-4

# ─── Precision ────────────────────────────────────────────────────────────────
BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
FP16 = (not BF16) and torch.cuda.is_available()

# ─── Inference ────────────────────────────────────────────────────────────────
INFER_BATCH_SIZE     = 64
NUM_BEAMS            = 5
MAX_NEW_TOKENS       = 256
LENGTH_PENALTY       = 1.0
NO_REPEAT_NGRAM_SIZE = 3   

# ─── Metric configuration ─────────────────────────────────────────────────────
COMET_MODEL = "Unbabel/wmt22-comet-da"



def sacrebleu_tokenize(pipeline_id: str) -> str:
    return "13a"

# ─── Hallucination analysis thresholds ───────────────────────────────────────
HALL_LEN_RATIO_LOW  = 0.4
HALL_LEN_RATIO_HIGH = 2.5
HALL_COPY_RATIO     = 0.8

# ══════════════════════════════════════════════════════════════════════════
#  SELF-TRAINING AUGMENTATION  (replaces back-translation)
# ══════════════════════════════════════════════════════════════════════════

ST_SAMPLE_RATIO_R1 = 0.35
ST_SAMPLE_RATIO_R2 = 0.50
ST_SAMPLE_RATIO_R3 = 0.65


ST_CONF_THRESHOLD_R1 = 0.72
ST_CONF_THRESHOLD_R2 = 0.76
ST_CONF_THRESHOLD_R3 = 0.80

# Cycle-consistency threshold: LaBSE(original_src, round_tripped_src).
ST_CYCLE_THRESHOLD = 0.65

# ── Quality scoring (v2) ──────────────────────────────────────────────────

LABSE_UNSUPPORTED_LANGS = {"brx_Deva"}


ST_CYCLE_CHRF_WEIGHT = 0.6

# Absolute floors (safety nets; sentence-level chrF++ on a 0-1 scale).
ST_CYCLE_CHRF_FLOOR  = 0.30


ST_KEEP_FRACTION = {1: 0.50, 2: 0.40, 3: 0.30}

# Pseudo/source token-length ratio bounds (cheap hallucination guard).
ST_LEN_RATIO_MIN = 0.5
ST_LEN_RATIO_MAX = 2.0

# Noisy-student source-side perturbation (applied from Round 2 onward).
ST_NOISE_SWAP_PROB   = 0.10
ST_NOISE_DELETE_PROB = 0.05
ST_NOISE_INSERT_PROB = 0.05
# Fraction of each round's *new* pseudo pairs that also get a noised twin.
ST_NOISE_INJECT_FRACTION = 0.5

# Backward-compat aliases (kept in case any script imports the old names).
BT_LABSE_THRESHOLD = ST_CONF_THRESHOLD_R1
QA_LABSE_THRESHOLD = ST_CONF_THRESHOLD_R1

# ─── Plot colours (one per system within a pipeline) ─────────────────────────
SYSTEM_COLORS = {
    "baseline": "#C44E52",
    "r1":       "#4C72B0",
    "st1":      "#55A868",
    "st2":      "#DD8452",
    "st3":      "#8172B2",
}
SYSTEM_LABELS = {
    "baseline": "Baseline (zero-shot IT2 Indic-Indic 1B)",
    "r1":       "R1 (DoRA/LoRA fine-tuned)",
    "st1":      "Self-Training Round 1",
    "st2":      "Self-Training Round 2 (+noisy-student)",
    "st3":      "Self-Training Round 3 (+cycle-consistency, tightened conf.)",
}

# ─── Make required directories on import ─────────────────────────────────────
for _pid in ALL_PIPELINES:
    for _d in [
        os.path.join(OUTPUT_DIR, f"pipeline_{_pid}"),
        synth_cache_dir(_pid),
        FIGURES_DIR,
        adapter_dir(_pid, "r1"),
        adapter_dir(_pid, "st1"),
        adapter_dir(_pid, "st2"),
        adapter_dir(_pid, "st3"),
    ]:
        os.makedirs(_d, exist_ok=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs("./checkpoints", exist_ok=True)
