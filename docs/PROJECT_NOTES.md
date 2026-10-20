> **About this file.** This is the project's internal lab notebook: every run, ablation and review remark, with the reasoning behind each decision. It refers to some earlier work that is **not in this repository** because it was superseded: `_archive/`, `results_mfagat/`, `mfe-fcgcn.ipynb` (replaced by `build_cache.ipynb`), `GAT_ds004504.html`, and the presentation PDF. Numbers quoted from those come from runs kept on the author's machine. For the public overview and how to reproduce, see the [README](../README.md).

# AD vs HC detection on ds004504: project map

*Last reviewed 2026-10-19. Start here before opening any notebook.*

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
- **One repeat only.** At `N_REPEATS = 1`, AUC differences under ~0.03 are noise. Run 3+ repeats before the final report.
- **The "linear floor" printed by the notebook used different folds** (seed 42) from the network (seed 1042). On the network's own folds, band-power logistic regression scores **0.7538 acc / 0.8812 AUC**, so GAT leads it by +0.046 AUC. The 2026-09-16 run (and `GAT_ds004504.html`) printed 0.8769 / 0.8908. **Fixed 2026-10-20** in both twins: the splits moved up to Cell 4, and Cell 5 now fits the floor on the network's own folds and repeats (it prints 0.7538 / 0.8812 at 1 repeat). Cell 10 reuses those predictions, and the model cell is now Cell 6.
- **The floor learned from more subjects than the GAT.** Logistic regression is fitted on train + validation subjects (52), while the GAT trains on ~42 and uses the validation subjects only to pick the epoch. The GAT beat the floor despite that handicap, so the comparison understates it rather than flatters it. The multi-class notebook removes this asymmetry (see R1).
- **The GCN twin was never fully run** (`results_gcn/` does not exist), so "GAT beats GCN" has not been measured under the same protocol yet. The only saved GCN number under this protocol is 0.8000 / 0.8343 at the paper's learning rate (`results_mfagat/B_gcn_subject`).
- **The paper's 95% is a window-split number.** It comes from splitting *windows* rather than *subjects*, so overlapping windows of the same person end up in both train and test. Under that split, plain logistic regression also reaches 0.92. Quote such a number only as **Case A** (below), with its label, and always next to Case B.

## 1b. Two evaluation cases (from 2026-10-19)

| | **Case A: window level** | **Case B: subject level** |
|---|---|---|
| notebook | `GAT_window_ds004504.ipynb` | `GAT_ds004504.ipynb` (binary), `GAT_multiclass_ds004504.ipynb` (3-class) |
| split | 20 s windows, shuffled (paper's protocol) | subjects; nobody is in train and test |
| question | recognise AD in more recordings of known people | diagnose a person never seen |
| compare with | Liu et al. 2025 Table 3 (95.09 / 98.36) | subject-independent literature (~82–88% binary, ~60% 3-class) |

**Case A status (2026-10-19).** Settings were chosen on fold-1 *validation* windows: MLP dropout (0.5, 0.2) instead of the paper's (0.8, 0.5). ✅ **Full 5-fold run done** (`results_gat_window/`, 1 repeat, 36 min). Pooled: GAT **0.9489 acc / 0.9507 recall / 0.9884 AUC**; soft vote with band power **0.9602 / 0.9612 / 0.9925** (paper's MFE-FCGCN: 0.9509 / 0.9607 / 0.9836). Subject level is 1.000 on every metric, because every test person also has windows in training.

**Why Case A is this high: the split leaks, by design.** Checked on the same folds:
- Windows are 20 s long with a 5 s step (75 % overlap). 95.4 % of test windows have a neighbour sharing 15 s of the same signal in training; 100 % share some raw signal with a training window.
- Logistic regression on the 95 band-power numbers names the **person** (1 of 65) for 99.8 % of test windows.
- **A 1-nearest-neighbour lookup with no learning scores 0.991 window accuracy**, above the paper and above our GAT. Its nearest training window is from the same person 97.6 % of the time.
- Under the subject split (Case B) the same GAT gets 9/65 people wrong. Under the window split it gets 0/65 wrong (worst person: 76 % of windows right).

**Case B status (2026-10-19).** Binary, subject level: GAT 0.862 acc / 0.861 recall / 0.927 AUC; **soft vote 0.892 / 0.889 / 0.939** (1 repeat). Published leave-one-subject-out results on this dataset reach 84.6% / AUC 0.883. 3-class: see R1.

---

## 2. Files: what to focus on

### Active (open these)
| file | role |
|---|---|
| `GAT_ds004504.ipynb` | **Case B, binary, subject level**: the presented model (MFE-FCGAT). Cell 10 adds the GAT + band-power soft vote. Writes `results_gat/` |
| `GAT_window_ds004504.ipynb` | **Case A, binary, window level**: same model under the base paper's window split, reported in the paper's Table 3 format next to the paper's numbers. Writes `results_gat_window/` |
| `GCN_ds004504.ipynb` | paper's GCN as a baseline. Same code except the graph layer. Writes `results_gcn/` (not run yet) |
| `RF_ds004504.ipynb` | random-forest baseline on band power. Writes `results_rf/` |
| `GAT_multiclass_ds004504.ipynb` | **Case B, 3-class, subject level (R1 + R3)**: the same GAT with a 3-way head (HC / AD / FTD). Fixed 30 epochs on all non-test subjects, 3 repeats; macro + per-class recall, pairwise AUCs, dementia-vs-HC; floor on the same subjects; Cell 10 soft vote. Writes `results_gat_multiclass/` |
| `build_cache.ipynb` | **data pipeline**: builds `subject_cache/` for all 88 subjects. Its feature cells are copied verbatim from `mfe-fcgcn.ipynb`, and it rebuilds sub-001 bit-identical to the old cache |
| `mfe-fcgcn.ipynb` | the original cache builder plus old training cells. Replaced by `build_cache.ipynb`, kept until you're happy to archive it |
| `subject_cache/` | one `.npz` per subject (PSD features, E1, E2, band power, label 0 HC / 1 AD / 2 FTD). Every model reads only this |
| `data/` | raw ds004504, all 88 subjects including FTD (sub-066…088) |

### Results kept in the root
| folder | status | why |
|---|---|---|
| `results_gat/` | ✅ **presented** | GAT, subject split, pooled OOF. Executed record: `GAT_ds004504.html`. `ensemble_*` files = soft vote (Cell 10) |
| `results_gat_multiclass/` | ✅ usable | 3-class GAT, 3 repeats, fixed 30 epochs (2026-10-19). `ensemble_*` files = soft vote (Cell 10) |
| `results_rf/` | ✅ usable | RF baseline. Also holds the matched-fold logistic floor |
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

### R2 · More datasets
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

### Rules that apply to all of the above
Split by subject, never by window. Quote pooled 65-subject (or 88-subject) numbers, never single folds. Use 3+ repeats before reporting. Tune the learning rate per architecture, because the best rate does not transfer from one model to another.
