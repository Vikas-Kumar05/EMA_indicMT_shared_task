# Indic↔Indic Machine Translation with LoRA and Iterative Self-Training

Low-resource translation for two Indic language pairs, in both directions, built on **IndicTrans2 Indic→Indic 1B** with parameter-efficient fine-tuning (LoRA / rsLoRA) and a three-round **cycle-consistent self-training** augmentation pipeline.

| Pair | Pipeline | Direction |
|------|----------|-----------|
| Assamese ⇄ Bodo | **E** | `asm_Beng → brx_Deva` |
| | **F** | `brx_Deva → asm_Beng` |
| Bengali ⇄ Odia | **G** | `ben_Beng → ory_Orya` |
| | **H** | `ory_Orya → ben_Beng` |

All four directions share one base checkpoint. Only the adapter differs per direction.

---

## Method overview

```
raw data ──► step1 clean/split ──► step2 zero-shot baseline
                                 └► step3 LoRA fine-tune (R1) ──► step4 eval R1
                                                                  │
                         ┌────────────────────────────────────────┘
                         ▼
        step5a  Self-training round 1  (teacher = R1)   → st1
        step5b  Self-training round 2  (teacher = st1)  → st2  (+ noisy-student)
        step5c  Self-training round 3  (teacher = st2)  → st3  (+ noisy-student)
                         ▼
                 step6 comparison, significance tests
```

### Self-training round (per pipeline)

1. **Sample** a fraction of source sentences (0.35 / 0.50 / 0.65 for rounds 1 / 2 / 3). From round 2 on, only gold (`aug_type == "original"`) rows are sampled, never earlier pseudo pairs.
2. **Pseudo-label** them with the current teacher adapter (beam search).
3. **Validity and length filters.** The output must be non-empty, at least 2 tokens, different from the source, and between 0.5× and 2× the source length.
4. **Cycle-consistency.** Translate each pseudo target back with the *sibling direction's* adapter (E↔F, G↔H) and compare the result to the original source. This needs no external back-translation model.
5. **Combined quality score** from the signals that are valid for the language pair (see below).
6. **Rank-based selection.** After a loose absolute floor, keep the top 50% / 40% / 30% of candidates by combined score (rounds 1 / 2 / 3).
7. **Noisy-student twins** (rounds 2–3): a fraction of accepted pairs gets a copy with a perturbed source (word swap, delete, insert) and the same clean pseudo target.
8. **Fine-tune** from the previous round's adapter on gold + accepted pseudo data, then evaluate on the test set.

### Quality scoring and Bodo

LaBSE does **not** support Bodo (`brx`). A cross-lingual LaBSE comparison that involves Bodo is meaningless: it gave a mean similarity of about 0.27 for E/F against about 0.90 for G/H. The scorer therefore uses only valid signals per pipeline:

| Pipeline | Signals used |
|----------|--------------|
| E (asm→bodo) | round-trip chrF++ and round-trip LaBSE (Assamese vs Assamese). Cross-lingual LaBSE is skipped. |
| F (bodo→asm) | round-trip chrF++ only. All LaBSE signals are skipped. |
| G, H | round-trip chrF++, LaBSE source↔pseudo, LaBSE round-trip |

chrF++ is computed per sentence on same-language text, so it does not depend on LaBSE language coverage. Every run saves all scored candidates to `outputs/pipeline_X/synth_cache/st{N}_scored_candidates.xlsx` so thresholds can be tuned from data.

---

## Repository layout

```
.
├── config.py                     # all paths, hyperparameters, language registry, self-training settings
├── main.py                       # end-to-end runner with resume / skip logic
├── step1_analysis.py             # cleaning, normalisation, corpus statistics, leak check
├── step2_baseline_eval.py        # zero-shot baseline
├── step3_train_dora.py           # R1 LoRA fine-tuning
├── step4_finetuned_eval.py       # R1 evaluation
├── step5_selftrain_common.py     # shared driver for self-training rounds
├── step5a_selftrain_round1.py
├── step5b_selftrain_round2.py
├── step5c_selftrain_round3.py
├── aug_techniques.py             # self-training engine (pseudo-labelling, scoring, filtering)
├── aug_finetune_eval.py          # fine-tune + evaluate for each round
├── aug_utils.py                  # dataset combine / resume helpers
├── step6_compare.py              # comparison table, bootstrap significance, plots, LaTeX

```

---

## Setup

**Requirements:** Python 3.10+, a CUDA GPU (bf16 recommended), and enough memory for the 1B model. The dependency list below is approximate, so pin versions from your own environment.

```bash
pip install torch transformers peft datasets accelerate
pip install sacrebleu unbabel-comet sentence-transformers
pip install pandas openpyxl numpy matplotlib seaborn tqdm
pip install git+https://github.com/VarunGumma/IndicTransToolkit.git
```

Download the IndicTrans2 Indic→Indic 1B checkpoint and set its path in `config.py`:

```python
BASE_MODELS = {"indic2indic": "/path/to/indictrans2-indic-indic-1B"}
```

### Data

Put one Excel file per language pair and split under `./data/`:

```
data/asm_bodo_{train,dev,test}.xlsx
data/ben_odia_{train,dev,test}.xlsx
```

Each file needs one column per language (for example `asm` / `bodo`, `bengali` / `odia`) or `src_text` / `tgt_text`. Both directions of a pair read the same raw file. Step 1 assigns source and target by direction and checks for swapped columns.

---

## Usage

Run everything (steps are skipped when their outputs already exist):

```bash
python main.py
```

Useful options:

```bash
python main.py --pipeline E F              # selected pipelines only
python main.py --start-from step5a         # resume from a step
python main.py --only step6                # a single step
python main.py --force-step step5a         # re-run a step even if outputs exist
python main.py --dry-run                   # print the plan only
```

Or run steps directly:

```bash
python step1_analysis.py
python step2_baseline_eval.py
python step3_train_dora.py
python step4_finetuned_eval.py
python step5a_selftrain_round1.py
python step5b_selftrain_round2.py
python step5c_selftrain_round3.py
python step6_compare.py
```

### Re-running self-training after changing filters

Pseudo-pair generation is cached, so changing filter settings has no effect until the old outputs are removed:

```bash
rm -f outputs/pipeline_*/synth_cache/st*.xlsx outputs/pipeline_*/aug_st*.xlsx
rm -rf checkpoints/pipeline_*/st1 checkpoints/pipeline_*/st2 checkpoints/pipeline_*/st3
# and delete the st1/st2/st3 entries from outputs/all_scores.json
```

Do this for later rounds too, because they depend on earlier ones.

---

## Configuration

Main settings in `config.py`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `LORA_R` / `LORA_ALPHA` / `LORA_DROPOUT` | 16 / 32 / 0.1 | adapter size |
| `USE_DORA` / `USE_RSLORA` | False / True | DoRA toggle, rank-stabilised scaling |
| `LEARNING_RATE` / `NUM_TRAIN_EPOCHS` | 5e-5 / 5 | R1 training |
| `LEARNING_RATE_AUG` / `NUM_TRAIN_EPOCHS_AUG` | 1e-4 / 3 | self-training rounds |
| `ST_SAMPLE_RATIO_R{1,2,3}` | 0.35 / 0.50 / 0.65 | fraction of sources pseudo-labelled |
| `ST_KEEP_FRACTION` | {1: 0.50, 2: 0.40, 3: 0.30} | top fraction kept by combined score |
| `ST_CYCLE_CHRF_FLOOR` | 0.30 | minimum round-trip chrF++ |
| `ST_CYCLE_CHRF_WEIGHT` | 0.6 | weight of round-trip chrF++ in the combined score |
| `ST_LEN_RATIO_MIN` / `MAX` | 0.5 / 2.0 | pseudo/source length bounds |
| `LABSE_UNSUPPORTED_LANGS` | `{"brx_Deva"}` | languages excluded from LaBSE scoring |
| `ST_NOISE_*` | 0.10 / 0.05 / 0.05, fraction 0.5 | noisy-student perturbation |
| `NUM_BEAMS` / `NO_REPEAT_NGRAM_SIZE` | 5 / 3 | decoding |

Adding a language pair needs only entries in `LANG_REGISTRY` and `PIPELINE_CONFIGS`.

---

## Outputs

```
outputs/
├── all_scores.json                    # BLEU, chrF, chrF++, TER, COMET per system and pipeline
├── comparison_table.csv
├── results_latex.tex
├── figures/                           # loss curves, error histograms, comparison plots
└── pipeline_X/
    ├── {train,dev,test}.xlsx          # cleaned splits
    ├── aug_st{1,2,3}.xlsx             # gold + accepted pseudo data per round
    ├── preds_{baseline,r1,st1,st2,st3}.csv
    ├── synth_cache/                   # pseudo pairs and scored candidates
    └── significance_*.json            # paired bootstrap tests
checkpoints/pipeline_X/{r1,st1,st2,st3}/   # adapters
```

**Metrics:** BLEU (sacrebleu, 13a), chrF, chrF++, TER, COMET (`Unbabel/wmt22-comet-da`). Step 6 adds paired bootstrap significance tests (n = 1000) of the final system against baseline and R1.

---

## Results

> Fill in after running `step6_compare.py`.

| Pipeline | System | BLEU | chrF | chrF++ | COMET |
|----------|--------|-----:|-----:|-------:|------:|
| E (asm→bodo) | Baseline | | | | |
| | R1 | | | | |
| | st1 / st2 / st3 | | | | |
| F (bodo→asm) | Baseline | | | | |
| | R1 | | | | |
| | st1 / st2 / st3 | | | | |
| G (ben→odia) | Baseline | | | | |
| | R1 | | | | |
| | st1 / st2 / st3 | | | | |
| H (odia→ben) | Baseline | | | | |
| | R1 | | | | |
| | st1 / st2 / st3 | | | | |

---

## Known limitations

- **Pseudo-labelled sources have gold targets.** The pipeline pseudo-labels sentences from the training set, and the human translation of each pseudo-labelled source stays in training. The model can therefore see two different targets for one source, and the teacher has already been trained on these sentences. Gains from pseudo data may be small. Compare against a control: R1 continued for the same epochs on gold data only.
- **Round-trip agreement is not accuracy.** The cycle check measures consistency between two models trained on the same data, not correctness.
- **Bodo has no LaBSE support**, so E/F rely on round-trip chrF++ alone for quality estimation.
- **Rounds are evaluated on the test set.** For final reporting, choose the best round on the dev set and report test once.

### Possible extensions

- Split the training set into labelled and unlabelled parts so pseudo-labelling does not conflict with gold targets, and compare L-only, L + pseudo(U) and L + gold(U).
- Use real monolingual text as the unlabelled source.
- Add a model-confidence signal (length-normalised sequence log-probability).
- Cap the share of pseudo data, or finish with a short gold-only fine-tune.
- Run multiple seeds and report mean and spread.

---

## Acknowledgements

Built on [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) (AI4Bharat), [IndicTransToolkit](https://github.com/VarunGumma/IndicTransToolkit), [PEFT](https://github.com/huggingface/peft), [sacrebleu](https://github.com/mjpost/sacrebleu), [COMET](https://github.com/Unbabel/COMET) and [LaBSE](https://huggingface.co/sentence-transformers/LaBSE).
