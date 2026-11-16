> **About this file.** This is the project's internal lab notebook: every run, ablation and review remark, with the reasoning behind each decision. It refers to some earlier work that is **not in this repository** because it was superseded: `_archive/`, `results_mfagat/`, `mfe-fcgcn.ipynb` (replaced by `build_cache.ipynb`), `GAT_ds004504.html`, and the presentation PDF. Numbers quoted from those come from runs kept on the author's machine. For the public overview and how to reproduce, see the [README](../README.md).

# AD vs HC detection on ds004504: project map

*Last reviewed 2026-11-16. Start here before opening any notebook.*

## 0a. Added on 2026-11-16: the scoreboard against honest published results

Plan: `docs/PREREG_2026-11-16.md`. Experiments 9–11 have their own protocols in
`experiments_ad_vs_ftd/results/*/protocol.json`. The verdict is "level" when the
published value lies inside our 95 % CI.

| question | result | where |
|---|---|---|
| AD vs FTD stage 2, nested choice of 5 candidates (Exp 9) | **fails**: 0.576, permutation p 0.34; best candidate C3 (lagged cross-correlation) 0.628, picked post hoc | `experiments_ad_vs_ftd/results/riemann/` |
| C3 confirmed on ds006036, pre-registered (Exp 10) | **0.641** [0.515, 0.753], p 0.045 (narrow); the 44 s ds004504 control is 0.523; adds nothing beyond MMSE + age + sex | `experiments_ad_vs_ftd/results/c3_ds006036/` |
| C3 averaged over both recordings (Exp 11) | 0.658 [0.527, 0.773], p 0.021; gain vs ds006036 +0.017 [−0.052, 0.087], so **not adopted** | `experiments_ad_vs_ftd/results/c3_both/` |
| 3-class GAT with coupling graphs: Pearson / PLV / AEC-c / Granger | macro recall 0.550 / 0.562 / 0.555 / 0.564 vs 0.583; **no difference** (Holm p ≥ 0.75) | `results_gat_multiclass_*/`, `results_final/b_tests.csv` |
| dementia vs HC, dedicated GAT + band power | **82.6 % acc** [75.0, 89.8], AUC 0.898; level with Stefanou 2025 (80.7 %) | `results_gat_dementia/`, `results_final/dementia.csv` |
| final 3-class (two-stage, fixed in advance) | 60.6 % acc, macro-F1 0.589, macro recall 0.597, FTD recall 0.44; equal to today's soft vote (p 0.90) | `results_final/` |
| bottleneck | with a perfect AD-vs-FTD step, 3-class accuracy would be 87.5 % | Section H of the assessment doc |
| combined AD-vs-FTD score (GAT + C3 + aperiodic), pre-registered addendum | 0.641 vs C3 0.644: **no gain**, not adopted | `results_final/combine/` |
| ensemble of the five 3-class GATs | macro recall 0.595 vs 0.583 (p 0.64); with band power 0.600 (= today) | `results_final/combine/` |

**Defensible claims.**
- Level with the best honest result on AD vs HC, dementia vs HC, AD vs FTD, and 3-class F1 / recall.
- Behind on 3-class accuracy (review average ≈ 70 %, not checked study by study).

**Not defensible.**
- "Coupling graphs help".
- "The two-stage model is better".
- "C3 measures disease type rather than severity".
- "The two-recording average" as a headline number (it was not adopted).

The GAT's E1 and E2 are computed between electrodes' **power spectra**, so they
measure spectral similarity, not functional connectivity in time.

## 0. Status on 2026-11-15: the numbers to quote

| question | result | where |
|---|---|---|
| AD vs HC, **new patients** (Case B, 5 repeats) | GAT **0.840 acc / 0.909 AUC** [0.842, 0.965]; soft vote 0.834 / 0.916 | `results_gat/statistics.json` |
| AD vs HC, **window split** (Case A; reproduces the paper's numbers) | GAT 0.9489 / 0.9884 (paper 0.9509 / 0.9836); a 1-NN lookup scores 0.991 | `results_gat_window/`, `leak_check.py` |
| the paper's **released code**, re-run on our 25 patient splits | **0.668 window acc** (0.560–0.759), patient AUC 0.839; no epoch of any run above 0.791; window split 0.831 | `results_released_code/` |
| AD / FTD / HC (3 repeats) | soft vote **0.600 macro recall** [0.517, 0.686]; FTD recall 0.33 | `results_gat_multiclass/statistics.json` |
| AD vs FTD, engineered features (§ R5) | **no feature set above chance** (AUC 0.41–0.54); the same features reach 0.66–0.84 on AD vs HC | `experiments_ad_vs_ftd/results/` |
| AD vs FTD, pre-registered aperiodic test (§ R5, Experiment 8) | **not supported**: exponent / offset / theta-alpha AUC 0.632 / 0.640 / 0.644, all Holm p 0.088; classifier 0.616 (p 0.076 vs chance) | `experiments_ad_vs_ftd/results/aperiodic/` |
| GATv2 and GATv2 + learned edge gate (§ R4, pre-registered) | **no significant difference**: 0.840 / 0.921 and 0.815 / 0.907; ΔAUC +0.012 vs GAT (Holm 0.71), −0.014 vs GATv2 (Holm 0.94) | `results_gatv2/`, `results_gatv2gate/` |
| AD vs HC on ds006036, same people, eyes open + photic (§ R2, pre-registered) | **replicates**: GAT 0.732 acc / 0.846 AUC [0.760, 0.917]; soft vote 0.760 / 0.854; floor 0.729 / 0.811 | `results_gat_ds006036/statistics.json` |

**No model difference is statistically significant after Holm correction**, in
any task (§ R6). Defensible claims:
- the subject-level AD vs HC performance;
- the leak audit;
- the 3-class result being close to a strict public benchmark (61 %);
- the AD vs FTD negative result together with its controls, and the
  pre-registered aperiodic test (right direction, not significant);
- the two pre-registered review answers: GATv2 / adaptive graphs add nothing
  measurable, and detection replicates in a second recording of the same people.

The README's **Conclusion** section is the final summary of the project.

Not defensible: "GAT beats band power", "the graphs help", "the soft vote is
better", "GATv2 is better" (+0.012, Holm 0.71), "ds006036 is an independent
validation" (same 88 people). Borderline: "GAT beats GCN" (Holm 0.10 in the
reported run, 0.006 in an unchanged re-run; § R6).

**What the project is.** We take resting-state, eyes-closed EEG (OpenNeuro
ds004504, 19 electrodes). For each of 5 frequency bands (δ θ α β γ) we build one
graph per 20-second window: the nodes are electrodes, the node features are PSD,
and there are two edge sets, a Pearson graph (E1) and a mutual-information
graph (E2). A graph network then classifies Alzheimer's (AD, 36 subjects)
against healthy controls (HC, 29). It follows Liu et al. (2025), *MFE-FCGCN*
(`s40747-025-01974-x.pdf`), with the GCN layer replaced by a **graph attention
network (GAT)**.

---

## 1. What we presented

The deck (`Navy Blue ... Presentation.pdf`) presents the **GAT model**.
Slides 13–14 ("Experimental results") are blank in the PDF export. The
presented numbers come from `results_gat/`, produced by `GAT_ds004504.ipynb` on
2026-09-16. The executed copy is `GAT_ds004504.html`.

Subject-independent 5-fold CV, pooled over all 65 subjects (subject = mean of its window probabilities):

| level | accuracy | **recall (sensitivity)** | specificity | AUC |
|---|---|---|---|---|
| subject | 0.8615 | **0.8611** (31/36 AD found) | 0.8621 | 0.9272 |
| window | 0.7883 | 0.8027 | 0.7741 | 0.8553 |

**Caveats to know before reusing these numbers**
- **One repeat only.** At `N_REPEATS = 1`, AUC differences under ~0.03 are noise. Run 3+ repeats before the final report. ✅ **Resolved 2026-11-10:** at 5 repeats the GAT scores 0.840 / 0.909 (§ 0). The presented run is repeat 1 of those five and reproduces exactly (AUC 0.927), so it was the luckiest of the five (range 0.880–0.927).
- **The "linear floor" printed by the notebook used different folds** (seed 42) from the network (seed 1042). On the network's own folds, band-power logistic regression scores **0.7538 acc / 0.8812 AUC**, so GAT leads it by +0.046 AUC. The 2026-09-16 run (and `GAT_ds004504.html`) printed 0.8769 / 0.8908. **Fixed 2026-10-20** in both twins: the splits moved up to Cell 4, and Cell 5 now fits the floor on the network's own folds and repeats (it prints 0.7538 / 0.8812 at 1 repeat). Cell 10 reuses those predictions, and the model cell is now Cell 6.
- **The floor learned from more subjects than the GAT.** Logistic regression is fitted on train + validation subjects (52), while the GAT trains on ~42 and uses the validation subjects only to pick the epoch. The GAT beat the floor despite that handicap, so the comparison understates it rather than flatters it. The multi-class notebook removes this asymmetry (see R1).
- **The GCN twin was never fully run** (`results_gcn/` does not exist), so "GAT beats GCN" has not been measured under the same protocol yet. The only saved GCN number under this protocol is 0.8000 / 0.8343 at the paper's learning rate (`results_mfagat/B_gcn_subject`). ✅ **Resolved 2026-11-10:** GCN 0.785 / 0.860 at 5 repeats, on the same folds. GAT − GCN = +0.050 AUC [+0.011, +0.097], p (Holm) 0.10.
- **The paper's 95% is a window-split number.** It comes from splitting *windows* rather than *subjects*, so overlapping windows of the same person end up in both train and test. Under that split, plain logistic regression also reaches 0.92. Quote such a number only as **Case A** (below), with its label, and always next to Case B.

## 1b. Two evaluation cases (from 2026-10-19)

| | **Case A: window level** | **Case B: subject level** |
|---|---|---|
| notebook | `GAT_window_ds004504.ipynb` | `GAT_ds004504.ipynb` (binary), `GAT_multiclass_ds004504.ipynb` (3-class) |
| split | 20 s windows, shuffled (reproduces the paper's numbers) | subjects; nobody is in train and test |
| question | recognise AD in more recordings of known people | diagnose a person never seen |
| compare with | Liu et al. 2025 Table 3 (95.09 / 98.36) | subject-independent literature (~82–88% binary, ~60% 3-class) |

**Case A status (2026-10-19).** Settings were chosen on fold-1 *validation* windows: MLP dropout (0.5, 0.2) instead of the paper's (0.8, 0.5). ✅ **Full 5-fold run done** (`results_gat_window/`, 1 repeat, 36 min). Pooled: GAT **0.9489 acc / 0.9507 recall / 0.9884 AUC**; soft vote with band power **0.9602 / 0.9612 / 0.9925** (paper's MFE-FCGCN: 0.9509 / 0.9607 / 0.9836). Subject level is 1.000 on every metric, because every test person also has windows in training.

**Why Case A is this high: the split leaks, by design.** Checked on the same folds:
- Windows are 20 s long with a 5 s step (75 % overlap). 95.4 % of test windows have a neighbour sharing 15 s of the same signal in training; 100 % share some raw signal with a training window.
- Logistic regression on the 95 band-power numbers names the **person** (1 of 65) for 99.8 % of test windows.
- **A 1-nearest-neighbour lookup with no learning scores 0.991 window accuracy**, above the paper and above our GAT. Its nearest training window is from the same person 97.6 % of the time.
- Under the subject split (Case B) the same GAT gets 9/65 people wrong. Under the window split it gets 0/65 wrong (worst person: 76 % of windows right). At 5 repeats: 8-13 wrong (mean 10.4), window accuracy 0.764.

**Correction (2026-11-15): the paper's own split.** Earlier notes called the window split "the paper's protocol". The paper's text is ambiguous ("signal segments from the same sample" kept on one side of an 8:2 split), and its released code (brief-city/MFE-FCGCN-model, `model_and_train.py`) splits the 65 *patients* 80/20 (13 test patients), evaluates them every epoch and steps `ReduceLROnPlateau` on the test loss. The AHEPA review (Cognitive Neurodynamics 20(1):95, 2026) also grades it Validity 2 (subject-level single split). What stays defensible: a window split reproduces the paper's numbers, and under it a 1-NN lookup scores 0.991. On our 25 patient-split folds of 13 patients, our MFE-FCGCN scored 0.483-0.888 window accuracy (mean 0.717): never 95 %. Deck and README now say "the split that reproduces the paper's 95 %", not "the paper's protocol".

**The released code, re-run (2026-11-15).** `experiments_released_code/prep.py` ports `raw_data_processing.py` + `generate_graph_data.py` (Welch n_fft 2500 / overlap 500, window = mean of 4 consecutive 5 s segments with a 1-segment step, per-channel z-score, row-max-normalised MI graph, row-normalised |Pearson| graph; caps 80 AD / 100 HC windows; 5,773 windows). `run.py` ports `model_and_train.py` (shared GCN, MI then Pearson, dense GCNConv checked against PyG to 2.4e-7; Adam 1.2e-4, ReduceLROnPlateau(0.1, 5), 200 epochs, best-val-AUC checkpoint on 1 of 15 inner folds; theirs re-initialises per inner fold and keeps the last one's checkpoint) on our 25 test sets. Results (`results_released_code/summary.json`):
- as released (scheduler on the **test** loss): window acc 0.668 (0.560–0.759), AUC 0.724; patient pooled 0.794 acc / 0.839 AUC. The LR first drops at a median epoch of 40 and is 1.2e-9 by the end of every run; best epoch median 43.
- scheduler on validation loss: 0.681 (0.442–0.780), patient 0.794 / 0.849. So the test-loss scheduler moves accuracy by about one point; it is not what produces 95 %.
- best test epoch of any of the 50 runs (an oracle): 0.791. No run near 95 %.
- control, `SPLIT=window` (5-fold StratifiedKFold over windows, seed 42): 0.831 (0.809–0.856), AUC 0.908. A 1-NN lookup on the same released features scores 0.998, nearest neighbour = same person 99.7 %: the features leak under a window split, but this training (lr 1.2e-4, dropout 0.8, LR collapse) does not exploit it.
- Our MFE-FCGCN (paper equations, our features) is not worse than theirs on new patients: 0.717 / 0.860 vs 0.668 / 0.839.
How the reported 95.09 % was produced is unknown: one unseeded 80/20 patient draw, MI with random_state None, and test accuracy printed at all 15 × 200 epochs. Say "their released code scores 67 % per window on new patients", never "the paper cheated".

**Case B status (2026-10-19).** Binary, subject level: GAT 0.862 acc / 0.861 recall / 0.927 AUC; **soft vote 0.892 / 0.889 / 0.939** (1 repeat). Published leave-one-subject-out results on this dataset reach 84.6% / AUC 0.883. 3-class: see R1.

**Case B status (2026-11-10, 5 repeats, all models on the same folds).** Subject level, 65 subjects; AUC with 95 % CI from a subject bootstrap:

| model | acc | recall | spec | AUC [95 % CI] |
|---|---|---|---|---|
| band-power LR | 0.769 | 0.772 | 0.766 | 0.872 [0.787, 0.941] |
| random forest | 0.785 | 0.778 | 0.793 | 0.849 [0.746, 0.939] |
| MFE-FCGCN | 0.785 | 0.806 | 0.759 | 0.860 [0.781, 0.925] |
| no-graph GAT (control) | 0.775 | 0.800 | 0.745 | 0.881 [0.802, 0.944] |
| **MFE-FCGAT** | **0.840** | **0.850** | **0.828** | 0.909 [0.842, 0.965] |
| soft vote GAT + band power | 0.834 | 0.844 | 0.821 | **0.916** [0.850, 0.967] |

These are paired comparisons, Holm-corrected over six. **None is significant:**

| comparison | ΔAUC | p | p (Holm) |
|---|---|---|---|
| GAT − band-power LR | +0.037 | 0.28 | 0.56 |
| soft vote − band-power LR | +0.044 | 0.029 | 0.15 |
| GAT − GCN | +0.050 | 0.016 | 0.10 |
| GAT − no-graph | +0.028 | 0.13 | 0.50 |
| GAT − RF | +0.060 | 0.18 | 0.53 |

**The no-graph control** (`GAT_nograph_ds004504.ipynb`) replaces E1 and E2 with
a uniform 1/19 graph. The weights stay the same (22,566), and the `log E` prior
becomes a constant that softmax ignores. Its small, non-significant gap is why
"the graphs help" cannot be claimed.

**Learning rate:** 17 of 25 GAT folds picked epoch 15, the earliest allowed
(`MIN_EPOCHS`). The model peaks early, which suggests lr 3e-3 is on the high
side for the GAT.

---

## 2. Files: what to focus on

### Active (open these)
| file | role |
|---|---|
| `GAT_ds004504.ipynb` | **Case B, binary, subject level**: the presented model (MFE-FCGAT). Cell 10 adds the GAT + band-power soft vote. Cell 11 runs the statistics against every other result folder on the same folds. Writes `results_gat/` |
| `GAT_window_ds004504.ipynb` | **Case A, binary, window level**: same model under a window split (reproduces the paper's numbers), reported in the paper's Table 3 format next to the paper's numbers. Writes `results_gat_window/` |
| `GCN_ds004504.ipynb` | paper's GCN as a baseline. Same code except the graph layer. Writes `results_gcn/` |
| `GAT_nograph_ds004504.ipynb` | control: the GAT with every graph replaced by a uniform one. Writes `results_gat_nograph/` |
| `GATv2_ds004504.ipynb`, `GATv2gate_ds004504.ipynb` | **R4 twins** (2026-11-15): GATv2 attention; GATv2 + learned per-window edge gate. LR from `lr_check.py` (validation patients only), saved as `lr_check.json` in each results folder. Write `results_gatv2/`, `results_gatv2gate/` |
| `build_cache_ds006036.ipynb`, `GAT_ds006036.ipynb` | **R2** (2026-11-15): the cache builder and the GAT notebook pointed at ds006036 (`data_ds006036/`, `subject_cache_ds006036/`). Cell 7 of the builder rebuilds ds004504 sub-001 and matches the old cache exactly. Writes `results_gat_ds006036/` |
| `experiments_ad_vs_ftd/` | **AD vs FTD feature-engineering study (R5)**: `features_ad_ftd.ipynb` (time-domain features, one row per subject) then `ADvsFTD_ds004504.ipynb` (experiments, statistics, figures). Has its own README |

The GAT, GCN, no-graph, GATv2 and GATv2 + gate notebooks are **twins**. They
share every cell except the intro, the model configuration and the model, so
cells 3–5 and 11–15 must stay byte-identical across all five (`GAT_ds006036`
shares them too). Edit them together. Cell 14 compares with every folder in
`COMPARE_DIRS` that exists **at run time**, so run order sets each Holm family:
GAT (6 comparisons), then GATv2 (7), then the gate (8). Re-running the GAT after
the GATv2 folders exist would grow its family to 8.
| `RF_ds004504.ipynb` | random-forest baseline on band power. Writes `results_rf/` |
| `GAT_multiclass_ds004504.ipynb` | **Case B, 3-class, subject level (R1 + R3)**: the same GAT with a 3-way head (HC / AD / FTD). Fixed 30 epochs on all non-test subjects, 3 repeats; macro + per-class recall, pairwise AUCs, dementia-vs-HC; floor on the same subjects; Cell 10 soft vote. Writes `results_gat_multiclass/` |
| `build_cache.ipynb` | **data pipeline**: builds `subject_cache/` for all 88 subjects. Its feature cells are copied verbatim from `mfe-fcgcn.ipynb`, and it rebuilds sub-001 bit-identical to the old cache |
| `mfe-fcgcn.ipynb` | the original cache builder plus old training cells. Replaced by `build_cache.ipynb`, kept until you're happy to archive it |
| `subject_cache/` | one `.npz` per subject (PSD features, E1, E2, band power, label 0 HC / 1 AD / 2 FTD). Every model reads only this |
| `data/` | raw ds004504, all 88 subjects including FTD (sub-066…088) |

### Results kept in the root
| folder | status | why |
|---|---|---|
| `results_gat/` | ✅ **quote these** | GAT, subject split, **5 repeats** (2026-11-10), pooled OOF. `ensemble_*` = soft vote (Cell 10); `statistics.json` = CIs and Holm-corrected comparisons with every model below. `GAT_ds004504.html` is the old 1-repeat run |
| `results_gcn/`, `results_gat_nograph/` | ✅ usable | MFE-FCGCN and the no-graph control, 5 repeats, same folds as `results_gat/` |
| `results_gat_multiclass/` | ✅ usable | 3-class GAT, 3 repeats, fixed 30 epochs (2026-10-19). `ensemble_*` files = soft vote (Cell 10); `statistics.json` added 2026-11-10 |
| `results_rf/` | ✅ usable | RF baseline, 5 repeats. Also holds the matched-fold logistic floor |
| `results_gatv2/`, `results_gatv2gate/` | ✅ usable | R4 twins, 5 repeats, same folds as `results_gat/` (2026-11-15) |
| `results_gat_ds006036/` | ✅ usable | the GAT on ds006036, 5 repeats; same fold procedure but other windows, so it is in no Cell 14 family with ds004504 |
| `results_mfagat/` | ✅ usable as an ablation record | GATv1 vs **GATv2**, fusion/CNN/prior ablations, plus the window-split reproduction of the paper's 95% (`A_paper_gcn_window`, which shows the leak). 1 repeat, lr 1.2e-4 |

### `_archive/`: superseded, nothing in the presentation depends on it
Cleaned up on 2026-10-19. Delete the whole folder once you're sure you don't need it.

| path | what it was |
|---|---|
| `notebooks/MF_AGAT_ds004504_with_OP.ipynb` | executed ablation harness behind `results_mfagat/`. **Contains a working GATv2 layer**, reused for R4 |
| `notebooks/paper-imple (4).ORIGINAL.ipynb` | first Kaggle version. Follows the authors' released code. 65.8% |
| `notebooks/paper-imple (4).SUPERSEDED.ipynb` | earlier draft of `mfe-fcgcn.ipynb` |
| `notebooks/notebook9415629a64 (5).ipynb` | Kaggle FA-GAT 20×15-fold run, **interrupted**, no saved results |
| `notebooks/MFGAT_ds004504_GPU_CLUSTER.ipynb` | Slurm version, **never executed**. Uses raw data, not `derivatives/` |
| `notebooks/ds004504_graph_comparison.ipynb` | 3-model harness, **never executed**, replaced by the GAT/GCN twins |
| `results/results/` | ❌ old GCN pipeline (`paper-imple`), selected on accuracy |
| `results/results_run1_15fold/` | ❌ first run. Validation was window-leaky and fold 1 predicts almost everything as AD |
| `results/results_B_window/` | ⚠️ first reproduction of the paper's 95% via window split. Only usable as evidence of the leak |
| `results/results_A_subject/`, `_C_valacc/`, `_D_swaporder/`, `_E_sharedgcn/` | ❌ old-pipeline ablations, fold means not pooled |
| `results/results_ablation5/` | ❌ orphan 5×5 ablation (graph order, shared GCN). No notebook produces it |
| `results/results_gat_multiclass_run1_valselect/` | ⚠️ first 3-class run (2026-10-19, 1 repeat, epoch picked on 14 validation subjects). Superseded by the fixed-budget protocol. Kept as the evidence behind the R1 diagnosis |
| `src/` | Python package from an earlier session. Nothing imported it |
| `Trial.md` | 4-line experiment plan (GCN → GAT → FAGAT → ablations) |

Permanently deleted as exact duplicates (verified byte-for-byte): 5 × `* - Copy.ipynb`,
`Major Project.rar` (53/53 files identical to the folders on disk), `MF_AGAT_ds004504_with_OP.html`,
and `results_run2_5fold/` (identical to `results/`).

### References
`s40747-025-01974-x.pdf`: base paper (Liu et al. 2025). `ojemb-crimi-3642070.pdf`: Mazurek et al., GAT for epilepsy EEG.

---

## 3. Remarks from the review and the plan

### R3 · Recall as the main metric (can be answered with existing results)
Every saved run, re-scored at subject level from its saved out-of-fold probabilities (no retraining):

| model | **recall** | specificity | balanced acc | AUC | AD missed |
|---|---|---|---|---|---|
| **GAT (presented)** | **0.861** | 0.862 | 0.862 | 0.927 | 5/36 |
| GATv1 (`mfagat/C_attn_v1`) | 0.889 | 0.793 | 0.841 | 0.916 | 4/36 |
| GATv2 + wide CNN (`mfagat/C_cnn_wide`) | 0.889 | 0.793 | 0.841 | 0.931 | 4/36 |
| GATv2 (`mfagat/C_base`) | 0.889 | 0.759 | 0.824 | 0.886 | 4/36 |
| GCN, paper lr (`mfagat/B_gcn_subject`) | 0.833 | 0.759 | 0.796 | 0.834 | 6/36 |
| Random forest | 0.806 | 0.793 | 0.799 | 0.854 | 7/36 |

How to switch to recall without fooling ourselves:
- **Report** recall first, always paired with specificity. A model that calls everyone AD has recall 1.0.
- **Do not select epochs on recall.** That rewards the degenerate "everyone is AD" model. Keep selecting on AUC (or balanced accuracy).
- To raise recall on purpose, use a **class-weighted loss fixed in advance**. Do not use a threshold tuned on validation subjects: we tested that and it made results worse, because calibration shifts between subject groups.
- For the multi-class task (R1), "recall" becomes **macro recall** (= balanced accuracy) plus **per-class recall**.

### R1 · Multi-class (AD / FTD / HC)
No new dataset is needed. ds004504 already has 23 FTD subjects (sub-066…088), and they are in `data/`.
1. ✅ **Cache extended (2026-10-19)** by `build_cache.ipynb`: 88 subjects, 8,050 windows (HC 2,900 / AD 2,858 / FTD 2,292). FTD is capped at the first 100 windows, like HC. The pipeline rebuilds sub-001 bit-identical to the old cache, so FTD features are directly comparable.
2. ✅ **First run (2026-10-19, 1 repeat, epoch picked on validation subjects)**, archived in `_archive/results/results_gat_multiclass_run1_valselect/`. Subject level, 88 subjects:

| | macro recall | recall HC | recall AD | recall FTD | macro AUC | AUC AD–HC | AUC AD–FTD | dementia vs HC AUC |
|---|---|---|---|---|---|---|---|---|
| GAT 3-class | 0.524 | 0.759 | 0.639 | **0.174** | 0.736 | 0.904 | **0.496** | 0.861 |
| floor, train + val (~70 subjects) | 0.585 | 0.724 | 0.639 | 0.391 | 0.759 | 0.869 | 0.592 | 0.866 |
| floor, train only (~56, same as the GAT) | 0.528 | 0.724 | 0.556 | 0.304 | 0.735 | 0.862 | 0.558 | 0.845 |

   **Audit:** no leakage and no metric bugs; the model code is identical to the binary GAT apart from the head. The weak result has three causes:
   - **The GAT cannot tell AD from FTD** (pairwise AUC 0.496 = chance), although it still separates AD from HC (0.904). It treats AD + FTD as one "dementia" group: of 23 FTD patients, 12 were called AD, 7 HC and 4 FTD. The floor is weak here too (0.59), so the band-power features carry little of the AD/FTD difference.
   - **Unequal training data.** The floor learned from ~70 subjects and the GAT from ~56. Given the same 56, the floor drops to 0.528, the GAT's level.
   - **Noisy epoch choice.** Each validation set held only 3–4 FTD patients.

   Not a cause: the per-window `log_global` normalisation (a floor on the GAT's normalised input scores 0.576). GAT minus floor is −0.061, 95% bootstrap CI [−0.169, +0.047], so not significant at one repeat.
3. ✅ **Protocol fixed (2026-10-19).** The notebook now trains for a fixed `EPOCHS = 30` (also the cosine horizon), chosen from the first run's validation curves. It uses all ~70 non-test subjects, with no validation set. The floor is fitted on exactly the same subjects. Every score now includes the pairwise AUCs (AD–HC, FTD–HC, AD–FTD) and dementia (AD + FTD) vs HC recall and AUC. `N_REPEATS = 3`, results quoted as mean ± SD. Repeat 1 uses the same test folds as the first run. A smoke test with 2 repeats passed: it reproduces the first run's floor exactly, and the confusion matrix matches the reported recalls. ⏳ Full run not done yet (~20 min per repeat, ~1 h for 3).
4. ✅ **Fixed-protocol run (2026-10-19, 3 repeats)** and soft vote (Cell 10: mean of GAT and band-power probabilities, equal weights, nothing tuned). Subject level, 88 subjects, mean ± SD over repeats:

| | macro recall | recall HC | recall AD | recall FTD | macro AUC | dementia vs HC AUC | AUC AD–FTD |
|---|---|---|---|---|---|---|---|
| GAT 3-class | 0.583 ± 0.025 | 0.782 | 0.750 | 0.217 | 0.748 | 0.871 | 0.595 |
| band-power floor, same subjects | 0.578 ± 0.006 | 0.747 | 0.611 | 0.377 | 0.758 | 0.871 | 0.584 |
| **soft vote: GAT + band power** | **0.600 ± 0.026** | 0.782 | 0.685 | 0.333 | **0.783** | **0.889** | 0.601 |
| *external: best honest published (see below)* | *0.611* | *0.86* | *0.69* | *0.28* | *0.786* | | *0.645* |

   **This is not a failure; it is the honest ceiling of this dataset.** An independent study with a strict subject-independent protocol (10 × 5-fold, nested selection, 1,348 hand-crafted features, foundation models, deep nets) finds every method at about 60% balanced accuracy, ±7 points, with FTD recall around 28–40%. AD and FTD look alike in resting-state band power, for us and for everyone.
5. **Next options for 3-class** (subject level), from that literature: average reference instead of the A1–A2 reference (+2–5 points reported), relative band power and slowing ratios (θ/α, (δ+θ)/(α+β)), and connectivity from time signals per band (delta cross-correlation is the best single published AD-vs-FTD feature, AUC 0.71). All three need a cache rebuild.
   ✅ **Tested 2026-11-10 (R5):**
   - Relative band power, slowing ratios and time-domain phase-lag connectivity (wPLI) do not separate AD from FTD.
   - Feeding them into a subject-level 3-class model gives macro recall 0.48–0.54, below the window-level models (0.58–0.60), with FTD recall 0.29–0.39. That check was run in a session scratchpad and is not in the repo.
   - Not tried: the average reference, and delta cross-correlation itself.

### R2 · More datasets
✅ **Done 2026-11-15 with ds006036**, as planned in `docs/PREREG_2026-11-14.md`:
- **Data.** ds006036 v1.1.0 (2.1 GB) in `data_ds006036/`. Same participants, groups and MMSE as ds004504.
  `build_cache_ds006036.ipynb` = `build_cache.ipynb` with paths and the file pattern changed; Cell 7
  rebuilds ds004504 sub-001 with max |diff| 0. 88 subjects, 4,906 windows (AD 1,773 / FTD 1,090 / HC 2,043);
  most stay under the window caps because recordings are shorter (AD ~4.9, FTD ~4.4, HC ~6.4 min).
- **Run.** `GAT_ds006036.ipynb` = the GAT notebook with only the cache and result folders changed
  (same LR 3e-3, folds procedure, floor, soft vote and statistics). 5 × 5-fold, 65 subjects.
- **Result.** GAT 0.732 acc / 0.717 recall / 0.752 spec / **AUC 0.846 [0.760, 0.917]**; floor 0.729 / 0.811;
  soft vote 0.760 / 0.854. The pre-registered criterion (CI excludes 0.5) passes: **detection
  replicates**. Lower than eyes closed (0.909), as predicted. Holm (3 comparisons): network vs floor
  0.94, soft vote vs floor 0.17.
- **Reading.** A second condition of the same people, not an independent cohort. BrainLat stays the
  independent test (future work).

| dataset | what | access | catch |
|---|---|---|---|
| **ds006036** (OpenNeuro) | the **same 88 people**, eyes-open photic stimulation, same 19 channels, preprocessed | open, CC0 | not an independent cohort. Keep each person on one side of every split *across both datasets*. Good for testing whether results hold in a second recording condition |
| **BrainLat** (Prado et al., *Sci Data* 2023) | AD, bvFTD, HC; eyes-closed, 128-ch Biosemi, 10 min | Synapse (registration) | independent cohort from 5 Latin American countries. Needs mapping of 128 channels to our 19 |
| **CAUEEG** (Chung-Ang Univ.) | ~1,100 subjects: normal / MCI / dementia, 21-ch 10-20, 200 Hz | request from authors | labels are "dementia", not AD specifically. Approval takes time |
| Lazarou et al. (Zenodo 4316608) | 256-ch EGI: AD 48, MCI 79, SCD 34, HC 33 | only 4 sample files public. Full data by email | not usable without contacting the authors |

### R4 · GATv2 and dynamic MI/Pearson matrices
- **GATv2 already exists.** `DenseGATv2Layer` in `_archive/notebooks/MF_AGAT_ds004504_with_OP.ipynb` was tested, and switching GATv2 → GATv1 *raised* AUC from 0.886 to 0.916. That is inside the noise band at one repeat. Plan: a clean twin `GATv2_ds004504.ipynb` that differs from the GAT notebook **only** in the attention score, with its own LR check, and 3 repeats for both.
- **What our graphs are today.** E1 and E2 are already computed per window, but they measure **similarity of PSD across frequency bins**, not synchrony of the signals over time, and they are frozen inputs. "Dynamic" can mean (a) **time-resolved connectivity**: Pearson and MI on band-filtered time signals over short sub-windows, which gives a sequence of graphs per window and needs a cache rebuild; or (b) **adaptive adjacency**: the model learns a per-sample correction to the Pearson and MI priors.
- **Decided (2026-10-19): (b), learned adaptive adjacency.** The cached Pearson and MI graphs stay as priors. GATv2 attention plus a learned edge gate `g_ij = σ(MLP[x_i, x_j, E_ij])` reshape them for each sample. No new preprocessing is needed. To attribute any gain, compare three runs: GAT → GATv2 → GATv2 + adaptive gate.
- ✅ **Done 2026-11-15**, as planned in `docs/PREREG_2026-11-14.md`:
  - **Models.** `GATv2_ds004504.ipynb`: `e_ij = a·LeakyReLU(W_s x_i + W_d x_j) + log E_ij`, 28,866 parameters.
    `GATv2gate_ds004504.ipynb`: adds `log σ(MLP[x_i, x_j, E_ij])` (hidden 8, one gate per layer and band), 31,676.
    The GAT has 22,566.
  - **LR on validation subjects only** (`lr_check.py`, repeat 1, mean best validation subject AUC):
    GATv2 1e-3 0.933 vs 3e-3 0.927, so 1e-3; gate 1e-3 0.907 vs 3e-3 0.927, so 3e-3.
  - **Result, 5 × 5-fold, 65 subjects:**

| model | acc | recall | spec | AUC [95 % CI] | ΔAUC | p (Holm) |
|---|---|---|---|---|---|---|
| GAT | 0.840 | 0.850 | 0.828 | 0.909 [0.842, 0.965] | | |
| GATv2 (vs GAT) | 0.840 | 0.861 | 0.814 | 0.921 [0.856, 0.973] | +0.012 [−0.013, +0.039] | 0.71 (family of 7) |
| GATv2 + gate (vs GATv2) | 0.815 | 0.839 | 0.786 | 0.907 [0.842, 0.958] | −0.014 [−0.035, +0.005] | 0.94 (family of 8) |

  - **Reading.** No significant difference, as expected in the plan. The earlier 1-repeat ablation
    (GATv1 ≥ GATv2) and this one disagree in sign, which is what noise at this n looks like.
  - **Diagnostic.** 16 / 25 (GATv2) and 17 / 25 (gate) folds keep epoch 15, the first allowed,
    the same pattern as the GAT (17 / 25). A shorter warm-up or lower LR may suit all three; testing it
    now would be tuning after seeing test results, so it is future work.

### R5 · AD vs FTD: feature engineering (2026-11-09 to 11-10)
**Question:** can engineered spectral, nonlinear and connectivity features
separate AD (36) from FTD (23) subject-independently? Everything is in
`experiments_ad_vs_ftd/`.

**Confounders found first** (`subjects.csv`, Cell 3 of the experiment). Each of
these quantities, on its own, separates FTD from AD:

| quantity | AUC | note |
|---|---|---|
| windows per subject in `subject_cache/` | **1.000** | caps 80 AD / 100 FTD, and nearly everyone hits the cap |
| recording length | 0.669 | AD 13.5 min, FTD 12.0 min |
| sex | 0.638 | FTD 61 % male, AD 33 % |
| MMSE | 0.806 | clinical; never a model input |

No existing model takes window count as input. But in the old cache, FTD
features averaged over minutes 7–8.6, which no AD feature covered.

**Design, fixed before any AD-vs-FTD result was seen:**
- **Equal data per subject:** the first 75 four-second epochs per subject (300 s) that contain no ASR `boundary` join. There are 512 joins in the derivatives; the shortest recording has exactly 75 clean epochs.
- **Features:** 417 per subject, as medians over epochs, then means over 5 regions. 27 known-answer self-tests guard them (they caught two alpha-peak bugs).
- **Feature sets:**
  - band power: the cache's own definition, reproduced to 4.6e-7;
  - spectral: 45;
  - nonlinear: 35;
  - wPLI connectivity: 35.
- **Excluded by theory:** zero-crossing rate (= 2 × Hjorth mobility, r 0.987), global efficiency (= mean wPLI, r 0.90–1.00), and Hjorth activity and absolute power (amplitude, which partly encodes sex).
- **Model:** one logistic regression (C = 0.1, balanced class weights), 20 × 5-fold CV over subjects.
- **Statistics:** subject bootstrap CIs; Holm over the 6 comparisons against band power.

**Result:**
- **AD vs FTD:**
  - band power 0.540 [0.403, 0.667];
  - spectral 0.448, nonlinear 0.410, connectivity 0.487, combinations 0.45–0.48;
  - sex alone 0.576, recording length 0.650;
  - 3-class GAT comparator 0.595.
- **Chance:** with shuffled labels, the AUC centres near 0.49 with a 95 % range of ≈ 0.30–0.68. The permutation p is 0.32–0.79 for every set.
- **Positive control:** the same pipeline on AD vs HC reaches 0.83 (band power) and 0.84 (spectral).
- **Descriptive:** frontal and temporal theta is higher in AD (single-feature AUC ≈ 0.30). No feature survives FDR over 210. The band-power model scores 0.73 within women but 0.33 within men.
- **Conclusion:** no evidence that resting-state EEG features separate AD from FTD at n = 59. The CIs still allow an AUC up to ≈ 0.67, so this does not prove that no difference exists. Lal et al. 2024 report 91 % using a window-level split.

**Literature check (2026-11-10).**
- The consistent finding is more slowing in AD than FTD, at the group level only. Caso 2012
  found 49 % sensitivity for single patients, and a 2018 review found no individual-level marker.
- The photic companion dataset ds006036 (same 88 people) is already analysed in a
  medRxiv preprint (Velez-Pardo 2026). Its 10 Hz driving deficit is shared by AD and FTD.
- Power: with 23 FTD patients, a true AUC of 0.65 is detected 49 % of the time. 80 % power
  needs about 47 FTD + 74 AD.

**Experiment 8 (`ADvsFTD_aperiodic.ipynb`, 2026-11-10): pre-registered test of
Wang et al. 2023** (GeroScience; same cohort; SMOTE applied before their split).
- **Protocol.** H1 exponent, H2 offset and H3 log theta/alpha, each higher in AD. The
  protocol is written to `protocol.json` before any EEG is read. Wang's settings come from
  their code: CAR, Welch Hamming 250 / 125 / nfft 1024, FOOOF fixed 2–40 Hz, peak threshold 1,
  R² ≥ 0.95. Each score is used raw, with a one-sided permutation p, Holm over 3.
- **Result.** AUC 0.632 / 0.640 / 0.644. One-sided p 0.048 / 0.036 / 0.029; Holm 0.088 for all
  three. **Not supported.**
- **S1, severity.** With MMSE + age + sex in the model, the exponent and offset add nothing
  (p 0.44 / 0.205). Theta/alpha adds p 0.046.
- **S2, classifier.** LR on all three, 20 × 5-fold CV: AUC 0.616 [0.469, 0.759], balanced
  accuracy 0.641, p 0.076 against 500 label shuffles. Band power on the same folds scored 0.540.
- **S3, positive controls.** Exponent AD > HC 0.664 (p 0.012); FTD vs HC 0.498. Theta/alpha
  AD > HC 0.868, FTD > HC 0.798.
- **S5, 2 s windows.** Exponent 0.615, offset 0.633.
- **S6, post hoc.** Muscle artefact is ruled out: failed fits AUC 0.455, 30–40 Hz power 0.522.
  Failed fits (23 %) are flat frontal and temporal spectra.
- **Method finding.** Wang's 0.5 s window overestimates steep exponents (1.5 → 1.74, 2.0 → 2.46),
  because power below 2 Hz leaks into the fit range. This is in the Cell 2 self-test, and the
  protocol was amended before any EEG was read.
- **Reading.** This reproduces Wang's direction and nearly their effect size on the same subjects.
  It is not an independent replication, and not a marker.

### R6 · Statistics (Cell 11), and a bug fixed on 2026-11-10
Every notebook's Cell 11 reports:
- the repeat-averaged metric, with a 95 % CI from a subject bootstrap (each class resampled separately);
- for each comparison, a paired p from the same resamples, Holm-corrected.

The first version (2026-11-09) tested **probabilities averaged over repeats**
(DeLong, McNemar). That is effectively a 5-run ensemble, a different and
stronger model than the one the tables report. At 1 repeat the two coincide;
at 5 repeats they diverged. For GCN, the mean per-repeat AUC is 0.860, while the
AUC of its averaged probabilities is 0.904. The fix: p is a two-sided percentile
bootstrap on the reported quantity, so p < 0.05 exactly when the CI excludes 0.
It is applied to all four notebooks; their statistics cells were re-run on the
saved predictions, and the ensemble files reproduced byte for byte.

**Training noise (2026-11-15).** `GAT_ds004504.ipynb` was re-run unchanged (same
torch 2.14.0, seeds and folds; the GATv2 folders moved aside so Cell 14 kept its
6 comparisons). It is not bit-identical: training losses differ at ~1e-8 from
epoch 1 (CPU reduction order), and r1 f1 ran 52 epochs instead of 39.
- Pooled AUC 0.923 [0.861, 0.972] vs 0.909; acc 0.852 vs 0.840. Per repeat:
  0.949 / 0.951 / 0.891 / 0.906 / 0.919 vs 0.927 / 0.926 / 0.880 / 0.898 / 0.914
  (10 repeats: 0.916 ± 0.023).
- GAT − GCN +0.064 [+0.024, +0.112], **Holm 0.006** (reported run: +0.050, Holm 0.10).
  No-graph +0.042 (Holm 0.072), LR +0.051 (0.23), RF +0.074 (0.22), soft vote − LR +0.049 (0.060).
- The committed `results_gat/` stays the reference (GATv2, the gate and every table compare
  against it); the re-run is in `results_gat/rerun_2026-11-15/`. Choosing the better run would be
  cherry-picking. The GCN, no-graph and RF runs carry the same kind of noise and were not re-run.
- Reading: the bootstrap CIs cover patient sampling, not retraining. Model gaps of 0.01–0.015 AUC
  are inside retraining noise; the GCN lead is borderline, not absent.

### Rules that apply to all of the above
- Split by subject, never by window.
- Quote pooled 65-subject (or 88-subject) numbers, never single folds.
- Use 5 repeats for binary results; every notebook compared in one table needs the same `N_REPEATS`.
- Give every number a CI, and test exactly the quantity the table reports.
- Holm-correct every family of comparisons.
- Give each subject the same amount of data whenever window counts or recording lengths differ between groups.
- Write a hypothesis and its direction down (a `protocol.json` saved before any data is read) before testing a published claim, and report the result whatever it is. Label any check added later as post hoc.
- Tune the learning rate per architecture, because the best rate does not transfer from one model to another.
