# Offensive-Language Detection for Tamil-English Code-Mixed Text (Tanglish)

Hierarchical offensive-language classification on the **DravidianCodeMix** Tamil
offensive-language dataset: YouTube comments with **6 labels** (the shared-task setup).

| id | label | train | dev | test |
|---|---|---|---|---|
| NOT | Not_offensive | 25,425 | 3,193 | 3,190 |
| UNT | Offensive_Untargetede | 2,906 | 356 | 368 |
| IND | Offensive_Targeted_Insult_Individual | 2,343 | 307 | 315 |
| GRP | Offensive_Targeted_Insult_Group | 2,557 | 295 | 288 |
| OTH | Offensive_Targeted_Insult_Other | 454 | 65 | 71 |
| NTA | not-Tamil | 1,454 | 172 | 160 |

The best published results on this test set are **0.78 weighted F1** (Hate-Alert,
DravidianLangTech-EACL 2021 winner) and **0.793 weighted F1** (pseudo-labelling with
transliteration). We report **macro F1** as the primary metric and weighted F1 for
comparison with those papers.

## What the method does

1. **Subword → word → sentence hierarchy.** Subword vectors are mean-pooled into
   words, additive attention over words (Yang et al., 2016) gives a word-level
   vector, and a learned gate fuses it with the `[CLS]` sentence vector
   (`tanglish/models.py`).
2. **Label hierarchy.** The labels form a tree: offensive / not / not-Tamil →
   targeted / untargeted → individual / group / other. Separate heads per level
   are composed into 6-class probabilities, which helps the rare classes.
3. **Domain-adaptive pretraining** of the encoder on in-domain Tanglish comments.
   Test comments are excluded from this corpus (`tanglish/dapt.py`).
4. **Script views.** 18% of comments are in Tamil script and 82% are romanised.
   Each comment also gets a copy in the other script, used as training
   augmentation and averaged at test time (`tanglish/translit.py`).
5. **Curated slang lexicon** with single-token semantic tags (`<LEX_PROFANITY>`,
   `<LEX_STUPID>`, …), chosen from training-set statistics (`tanglish/resources/lexicon.tsv`).
6. **Imbalance handling.** Square-root class weights, plus per-class decision
   offsets tuned on dev to maximise macro F1.
7. **Ensembling.** Seed-averaged probabilities, with members chosen greedily on dev.

## Changes from the original guide, and why

Each point is backed by `reports/data_stats.md` and `reports/lexicon_report.md`.

- **The `not-Tamil` class is kept.** Dropping it made results incomparable with every published number.
- **Train/test leakage is removed.** 77 test and 69 dev comments also appeared in train,
  and 298 train comments were duplicates.
- **The guide's lexicon was replaced.**
  - It covered 3.0% of comments.
  - Several entries are common in harmless comments: `daa` is non-offensive 58% of the
    time, `ponga` 65%, `panni` 40% (it usually means "did"), `dei` 28%.
  - Tags such as `[DISMISSAL]` split into 4–6 subword fragments.
  - The curated lexicon has 80–82% precision for offensive comments on dev and test.
- **The comparison is fair.**
  - Every model shares the same head code, epochs, early stopping, loss and optimiser.
  - Each configuration runs with 3 seeds, reported as mean ± std, with paired bootstrap tests.
- **The hierarchy is over real words.** The guide's attention ran over subword pieces
  and included `[SEP]`.
- **Preprocessing is lighter.** Case is kept (the models are cased). `@mentions` become
  `@user` instead of being deleted, since they signal a targeted insult. Emojis become
  words, because MuRIL maps every common emoji to `[UNK]`.
- **Engineering.**
  - Mixed precision, dynamic padding (95% of comments are ≤ 22 words) and
    2-GPU DDP instead of `DataParallel`.
  - Runs resume after a Kaggle session ends.

## Layout

```
tanglish/            library: data, preprocess, lexicon, translit, models, losses,
                     train, evaluate, ensemble, dapt
configs/             one YAML per experiment (inherit: base.yaml)
scripts/             phase1_data, phase2_features, run_experiments, phase5_report
notebooks/           kaggle_runner.ipynb
tests/               CPU unit tests
```

Mapping to the guide's phases:
- Phase 1 → `scripts/phase1_data.py`
- Phase 2 → `scripts/phase2_features.py`
- Phase 3 → `run_experiments.py --stage 1/2`
- Phase 4 → `--stage 3/4`
- Phase 5 → `scripts/phase5_report.py`

## Running

Kaggle (2× T4) is the intended environment: see `notebooks/kaggle_runner.ipynb`.
Locally:

```bash
pip install -r requirements.txt
python scripts/phase1_data.py                   # download + data audit
python scripts/phase2_features.py               # lexicon report + transliteration cache
python -m pytest tests -q
python -m tanglish.train --config configs/smoke.yaml    # 1-minute CPU smoke test

python scripts/run_experiments.py --stage 1     # backbone sweep
python scripts/run_experiments.py --stage 2     # DAPT + fine-tune
python scripts/run_experiments.py --stage 3     # ablation, 3 seeds
python scripts/run_experiments.py --stage 4     # large backbones
python scripts/phase5_report.py                 # tables, significance, ensemble, figures
```

A single run: `accelerate launch -m tanglish.train --config configs/s3_hier_tree.yaml --seed 13`.
Any config value can be overridden with `--set key=value`.

**Decision points.**
- After stage 2: keep `configs/s3_base.yaml` on the DAPT model only if it beat plain
  MuRIL on **dev** macro F1.
- After stage 3: set `configs/s4_base.yaml` to the winning head, lexicon and
  transliteration settings.
- Never choose settings by test scores.

**Romanised → Tamil views** need AI4Bharat IndicXlit (`ai4bharat-transliteration`,
which depends on fairseq). Build them once where it installs with
`python scripts/phase2_features.py --xlit` and copy `data/translit_cache.tsv` over.
Without it, only Tamil-script comments get a second (romanised) view.

## Outputs

Each run writes `runs/<experiment>/seed<k>/`:
- `metrics.json`: dev/test macro, weighted and per-class F1, before and after dev-tuned offsets, plus training history
- `{dev,test}_scores.npy`: log-probabilities, used for ensembling and significance tests without retraining
- `best.pt`: only when `keep_checkpoint: true`

`scripts/phase5_report.py` writes the following to `reports/`:
- `results.md`
- `ensemble.json`
- confusion matrices
- word-attention plots
- `error_analysis.md`: comments the proposed model fixes or breaks relative to the baseline
