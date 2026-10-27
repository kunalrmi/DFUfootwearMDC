# Single-output versus multi-output prediction of offloading-footwear prescriptions (code)

Anonymised code for the PAKDD 2027 submission. The clinical data cannot be shared, because the records are confidential patient data. The code runs on any CSV with the same coded columns (`PAKDD_DATA=<csv>`).

## Pipeline
- `src/common.py`: data loading, label handling, input preprocessing (see PREPROCESSING.md), iterative stratification, encoders, strategies S1–S5, coherent decoding, chain inference (greedy, beam, Monte Carlo).
- `src/run_repeat.py <rep> <seed> <out>`: one repeat (5 folds) for LR, GBT and RF: all strategies, inner-CV dependence gains, ensemble members, learning curve, S5 ceiling.
- `src/run_level.py <rep> <out>`: completes the learner-by-strategy design on CPU (GBT S4; RF S2–S5), with the same folds, ensemble orders and stacking splits as the main run.
- `src/run_mlp.py <rep> <out>`: multi-task MLP (S1/S6). `kaggle/run_kaggle.py` holds the PyTorch MLP, TabPFN and LightGBM runs used in the paper (GPU).
- `kaggle/level_kaggle.py`: TabPFN S4–S5 (output codes via `kaggle/tfm_ecoc.py` when a powerset has more than ten classes) and PyTorch MLP S2–S5; `src/convert_level.py` merges its CSVs.
- `kaggle/tfm_extras.py`: TabPFN teacher-forced chain probabilities, and TabPFN on the public benchmarks (`results/table_benchmarks_tfm.csv`).
- `src/nontrivial.py`: defines the non-trivial decisions (separate seed 999).
- `src/stats.py <out>`: main measures, comparisons (McNemar, Holm, corrected t-test, bootstrap), dependence gains, learning curve.
- `src/gen_table3.py <out>`: the learner-by-strategy results table.
- `src/analysis_extra.py <out>`: corrected-t intervals, domain exact-match tests, majority/k-NN references, decoded accuracy, learning-curve tests.
- `src/supplementary_checks.py`: share of the dependence gain recovered by renormalising onto coherent classes.
- `src/inference_checks.py <main_out> <out>`: beam-width sweep, calibrated GBT, joint-mode voting for the ensemble.
- `src/benchmarks.py <out.csv> <file.mat> ...`: replication on public MDC benchmarks (Edm, Jura, Song; from the multi-dimensional classification dataset collection used by Jia & Zhang).
- `src/bench2.py <out> <learners> <file.mat> ...`: S1–S5 and the dependence diagnostic on the nine public MDC benchmarks (Edm, Flare1, Oes97, Jura, Enb, Song, WQanimals, WQplants, BeLaE); `src/bench2_analysis.py` aggregates them (`results/table_benchmarks9*.csv`).
- `src/dependence_equivalence.py`: dependence gain on test folds with oracle-parent and deployed renormalisation; 90% corrected intervals, minimum detectable differences and H1 (`results/table_equivalence.csv`, `results/dependence_equivalence.json`).
- `src/run_sensitivity.py`, `src/sensitivity_analysis.py`: reverse-order chain, BR with merged constraint groups, regularised RF/GBT (`results/table_sensitivity.csv`).
- `src/figures.py`: Figs. 2–5. `kaggle/tfm_sensitivity.py`: TabPFN parts of these analyses.

Seeds: folds 101/202/303 (repeats 1–3), 999 (non-trivial definition); learners use random_state 0.

## Versions
Main runs: Python 3.13, NumPy 2.5, pandas 3.0, scikit-learn 1.9. Kaggle runs: scikit-learn 1.6.1, PyTorch 2.11, tabpfn 9.1–9.2 (TabPFN v2 weights), LightGBM 4.6 (see results/run_info_kaggle*.csv).

## Results
`results/` holds the aggregated tables reported in the paper; there are no record-level data.
