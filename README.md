# Graph attention on resting-state EEG for Alzheimer's detection

This project detects Alzheimer's disease (AD) from 19-channel resting-state EEG
([OpenNeuro ds004504](https://openneuro.org/datasets/ds004504)) using a graph
attention network. We evaluate it two ways: under the window split used by the
base paper, and under a strict split where no patient appears in both training
and test.

The base paper is **MFE-FCGCN** (Liu et al., 2025 [1]), which reports 95.09 %
accuracy and 98.36 % AUC. We replace its graph convolution with multi-head
graph attention (**MFE-FCGAT**). We then:
1. match the paper's result under its own protocol (94.9 % accuracy, 98.8 % AUC),
2. show that this protocol lets the model recognise *people* rather than disease,
3. report what the model achieves on patients it has never seen (84.0 % accuracy, 0.909 AUC), and
4. test whether engineered EEG features, and a pre-registered test of a published
   aperiodic (1/f) marker, can separate AD from frontotemporal dementia (FTD).

**[Conclusion](#conclusion)**: what holds up, and what does not.

---

## Results

### Case B: new patients (patient-independent split, the clinically meaningful number)

5-fold cross-validation grouped by patient, repeated 5 times with different
splits: nobody is in both training and test. A patient's score is the mean
probability over their 20 s windows. Each number is pooled over all 65 patients
(36 AD, 29 healthy controls) and averaged over the 5 repeats. Every model below
is tested on exactly the same patients and folds.

| model | accuracy | recall (AD found) | specificity | AUC [95 % CI] |
|---|---|---|---|---|
| band-power logistic regression | 0.769 | 0.772 | 0.766 | 0.872 [0.787, 0.941] |
| random forest on band power | 0.785 | 0.778 | 0.793 | 0.849 [0.746, 0.939] |
| MFE-FCGCN (the paper's model) | 0.785 | 0.806 | 0.759 | 0.860 [0.781, 0.925] |
| MFE-FCGAT with the graphs removed (control) | 0.775 | 0.800 | 0.745 | 0.881 [0.802, 0.944] |
| **MFE-FCGAT (ours)** | **0.840** | **0.850** | **0.828** | 0.909 [0.842, 0.965] |
| **MFE-FCGAT + band-power soft vote** | 0.834 | 0.844 | 0.821 | **0.916** [0.850, 0.967] |
| *published: leave-one-subject-out on ds004504 [3]* | *0.846* | | | *0.883* |

The soft vote is the plain average of the GAT's and the logistic regression's
probabilities. The weights are equal and nothing is tuned.

**Which differences are real?** With 65 patients, an AUC has a 95 % CI about
±0.06 wide, wider than most gaps between these models. Each comparison is
tested on the same bootstrap resamples as its CI, then Holm-corrected over the
six comparisons. MFE-FCGAT has the best accuracy and leads every baseline, but
**no lead is statistically significant**:

| MFE-FCGAT vs | ΔAUC [95 % CI] | p (Holm) |
|---|---|---|
| MFE-FCGCN (the paper's graph convolution) | +0.050 [+0.011, +0.097] | 0.10 |
| random forest | +0.060 [−0.021, +0.152] | 0.53 |
| band-power logistic regression | +0.037 [−0.027, +0.109] | 0.56 |
| the same model without graphs | +0.028 [−0.007, +0.067] | 0.50 |

The last row matters for interpretation: the electrode graphs add at most a
few AUC points that this sample cannot confirm.

### Case A: the paper's protocol (window split)

Windows are shuffled into folds, so every test patient also has windows in training.

| window level | accuracy | recall | precision | F1 | AUC |
|---|---|---|---|---|---|
| GAT (the paper's baseline) [1] | 0.9070 | 0.9036 | 0.8986 | 0.9004 | 0.9411 |
| MFE-FCGCN (the paper's model) [1] | 0.9509 | 0.9607 | 0.9389 | 0.9497 | 0.9836 |
| **MFE-FCGAT (ours)** | 0.9489 | 0.9507 | 0.9467 | 0.9487 | **0.9884** |
| **MFE-FCGAT + soft vote** | **0.9602** | **0.9612** | **0.9588** | **0.9600** | **0.9925** |
| band-power logistic regression | 0.9220 | 0.9237 | 0.9195 | 0.9216 | 0.9771 |
| ⚠️ **1-nearest-neighbour lookup, no learning** | **0.991** | | | | |

### Why Case A is not a diagnosis result

`leak_check.py` runs these checks on the same folds:

- Windows are 20 s long and start every 5 s (75 % overlap). **95.4 %** of test
  windows have a neighbour in training that shares 15 s of the same signal, and
  **100 %** share some signal with a training window.
- Logistic regression on 95 band-power numbers identifies **which of the 65
  people** a window came from **99.8 %** of the time (chance: 1.5 %).
- Copying the label of the most similar training window, with no learning at
  all, scores **99.1 %**. That beats the paper and our model.
- When the split is by patient, the same GAT goes from 0.949 to 0.788 window
  accuracy, and from 0 to 9 misdiagnosed patients out of 65 (first repeat of each).

Case A measures how well a model recognises people it has already seen. Case B
measures how well it diagnoses someone new. We report Case A only to show we
can reproduce the paper, and always next to Case B.

### 3-class: AD / frontotemporal dementia (FTD) / healthy, patient-independent

88 patients (36 AD, 23 FTD, 29 HC), 5-fold CV × 3 repeats, patient level:

| model | macro recall [95 % CI] | recall HC | recall AD | recall FTD | macro AUC | dementia vs HC AUC | AD vs FTD AUC |
|---|---|---|---|---|---|---|---|
| MFE-FCGAT, 3-way head | 0.583 [0.512, 0.655] | 0.782 | 0.750 | 0.217 | 0.748 | 0.871 | 0.595 |
| band-power logistic regression | 0.578 [0.495, 0.664] | 0.747 | 0.611 | 0.377 | 0.758 | 0.871 | 0.584 |
| **soft vote** | **0.600** [0.517, 0.686] | 0.782 | 0.685 | 0.333 | **0.783** | **0.889** | 0.601 |
| *published strict benchmark [3]* | *0.611* | *0.86* | *0.69* | *0.28* | *0.786* | | |

Telling dementia from health works (AUC 0.87–0.89). Telling the two dementias
apart does not: every AD vs FTD AUC is about 0.60, and its 95 % CI includes 0.5
(chance). The soft vote's lead over band power is not significant (+0.022 macro
recall, p (Holm) 0.81). Under a strict patient-independent protocol, published
methods also land near 60 %. The next section asks whether better features can
fix the AD vs FTD part.

### AD vs FTD: can feature engineering separate the two dementias?

[`experiments_ad_vs_ftd/`](experiments_ad_vs_ftd/) builds 417 features per
patient from the cleaned time-domain EEG:
- **spectral:** relative and regional power, band ratios, front-to-back and
  left-right asymmetry, alpha frequency, spectral entropy;
- **nonlinear:** Hjorth parameters, permutation and SVD entropy, Higuchi fractal
  dimension, DFA;
- **phase-lag connectivity:** wPLI, imaginary coherency, graph measures.

Every patient contributes the same 300 s. With equal data per patient, recording
length cannot leak: in the original windows, the number of windows alone
separated AD from FTD perfectly. The experiment compares 7 feature sets on 59
patients, using one pre-specified logistic regression and 20 × 5-fold
cross-validation by patient.

| AUC, FTD vs AD | |
|---|---|
| existing band power | 0.540 [0.403, 0.667] |
| engineered: spectral / nonlinear / connectivity | 0.448 / 0.410 / 0.487 |
| combinations of the three | 0.450 – 0.481 |
| patient's sex alone (not EEG) | 0.576 |
| chance: AUCs with shuffled diagnoses, 95 % range | 0.30 – 0.68 |
| **the same features on AD vs healthy** | **0.66 – 0.84** |

![The same features and classifier on two questions](experiments_ad_vs_ftd/results/two_questions.png)

**No feature group separates AD from FTD above chance.** None beats the existing
band power, and none beats knowing the patient's sex. The same features and
classifier do separate AD from healthy controls, so the pipeline works and the
features carry dementia information. What they lack is a difference between
the two dementias.

The clearest single difference is higher frontal and temporal theta power in AD
(the stronger EEG slowing usually reported in AD), but it does not survive
correction for testing 210 features. Accuracies near 90 % reported for AD vs FTD
on this dataset come from window-level splits [6], the same leak as Case A.

**Experiment 8: a pre-registered test of one published lead.** Wang et al. [7]
reported, on this same cohort, that the aperiodic (1/f) exponent and offset of
the EEG spectrum are higher in AD than in FTD. Their classifier applied SMOTE
before splitting, so it was never tested honestly. We tested their claims once:
- with their own spectral settings;
- with the direction written down before any EEG was read;
- with nothing fitted;
- with Holm correction over the three claims.

| AUC = P(AD scores higher than FTD) | [95 % CI] | Holm p |
|---|---|---|
| aperiodic exponent | 0.632 [0.481, 0.779] | 0.088 |
| aperiodic offset | 0.640 [0.482, 0.785] | 0.088 |
| theta/alpha ratio | 0.644 [0.498, 0.785] | 0.088 |
| all three in one classifier (20 × 5-fold CV by patient) | 0.616 [0.469, 0.759] | 0.076 vs chance |

**The result is not significant.**
- **Direction.** All three point the way Wang et al. predicted, and each passes on its own
  (one-sided p 0.03–0.05), but none survives correction. With 23 FTD patients,
  this had about a 50 % chance of detecting an AUC of 0.65.
- **Severity.** Once MMSE, age and sex are known, the exponent and offset add nothing,
  because the AD patients here are more advanced (MMSE 17.8 vs 22.2).
- **Artefacts.** Muscle artefact does not explain it.
- **What it amounts to.** It reproduces the published direction on the same people. It is
  not an independent replication, and it is not a usable marker. Details:
  [`experiments_ad_vs_ftd/`](experiments_ad_vs_ftd/).

---

## Conclusion

In order of confidence:

1. **The base paper's 95 % can be reproduced, but it is not a diagnosis result.**
   - MFE-FCGAT matches MFE-FCGCN under the paper's window split (94.9 % accuracy,
     0.988 AUC).
   - Under that split, copying the label of the most similar training window
     scores 99.1 % with no learning at all. The split rewards recognising people.
2. **On patients it has never seen, MFE-FCGAT detects Alzheimer's with 84.0 %
   accuracy, 85.0 % recall and 0.909 AUC.**
   - These figures cover 65 patients, averaged over 5 repeats.
   - This matches the best published patient-independent result (84.6 % / 0.883).
   - It is the best model tested here. Its lead over band power (+0.037 AUC) and
     over the paper's GCN (+0.050) is not statistically significant with 65 patients.
3. **Three classes (AD / FTD / healthy): 60 % macro recall,** in line with the
   strict literature (61 %).
   - Detecting dementia works (AUC 0.87–0.89).
   - Naming which dementia does not: FTD recall is 0.22–0.38.
4. **AD vs FTD: no EEG measure tested here separates the two above chance at
   n = 59.**
   - 417 engineered features score AUC 0.41–0.54.
   - A pre-registered test of a published marker scores AUC 0.62–0.64. It points
     the right way, but is not significant after correction and is largely tied
     to disease severity.
   - This is a power limit, not proof that EEG carries no difference.
   - Published ~90 % results on this dataset come from window-level splits.

**What would change these conclusions:**
- more FTD patients: about 47 for 80 % power at AUC 0.65;
- an independent cohort, such as BrainLat (163 bvFTD);
- for the subtype question, a second modality, as in clinical practice: plasma
  p-tau217 separates AD from FTLD with AUC ≈ 0.93 [8].

The defensible design is a two-stage pipeline. EEG screens for dementia, and a
biomarker names the type.

---

## Method

**Data.** ds004504 v1.0.9, the ASR + ICA cleaned recordings in `derivatives/`,
eyes closed, 19 electrodes, 500 Hz. Each recording is cut into 20 s windows
starting every 5 s. For every window and each of five bands (δ θ α β γ) we
compute:
- **node features:** the log power spectrum (Welch, 0.2 Hz resolution), normalised per window;
- **E1:** a Pearson-correlation graph between electrodes;
- **E2:** a mutual-information graph between electrodes.

**Model.** Following [1], with only the graph layer changed:

```
for each band f in {Delta, Theta, Alpha, Beta, Gamma}:
  x_f  (19 electrodes x F_f PSD bins)
    |  Conv1d(1, 1, D_f)          D_f = 3, 6, 11, 71, 61      -> 19 x 16
    |  GAT layer 1 on E1 (Pearson) 16 -> 16, 4 heads, ELU     -> 19 x 16
    |  GAT layer 2 on E2 (MI)      16 -> 5,  4 heads, ELU     -> 19 x 5 -> flatten 95
concat 5 bands -> 475 -> MLP 30 -> 14 -> 2  (AD / HC)
```

The attention score is `e_ij = LeakyReLU(a·Wx_i + a·Wx_j) + log E_ij`. Every
graph connects all 19 electrodes, so adding `log E` lets the measured
connectivity act as a prior that the attention then reshapes. Training uses
Adam (lr 3e-3, cosine schedule), batch size 64, and weight decay 1e-4. The
epoch with the best AUC on held-out *validation patients* is kept. The model
has 22,566 parameters, against 16,686 for the GCN version.

**Evaluation rules used throughout**
- Split by patient (except Case A, which is labelled). The split code in each notebook asserts that no patient leaks between train, validation and test.
- Report numbers pooled over all patients, never single folds. Each fold tests only 13 people.
- Choose the epoch on validation patients, never on test. No decision thresholds are tuned.
- Compute baselines on the same folds.
- Give every number a 95 % CI from a patient bootstrap (each class resampled
  separately). Test each difference on the same resamples, and Holm-correct
  every family of comparisons.

---

## Reproduce

Tested on Windows 11, Python 3.14, CPU only.

```bash
git clone <this repo> && cd <repo folder>
python -m venv .venv
.venv\Scripts\activate          # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt

# dataset, about 5.4 GB, into ./data (CC0 license)
openneuro-py download --dataset ds004504 --target-dir data
```

Then open the notebooks from the repo root and run each top to bottom. **Cell 2
is the only cell you normally edit**, and every notebook is self-contained.

| order | run | writes | time (CPU) |
|---|---|---|---|
| 1 | `build_cache.ipynb` | `subject_cache/`: 88 `.npz` files, 247 MB | ~35 min |
| 2 | `GCN_ds004504.ipynb` (paper's GCN, Case B) | `results_gcn/` | ~35 min (5 repeats) |
| 3 | `GAT_nograph_ds004504.ipynb` (control: graphs removed) | `results_gat_nograph/` | ~35–60 min |
| 4 | `RF_ds004504.ipynb` (random forest baseline) | `results_rf/` | < 1 min |
| 5 | `GAT_ds004504.ipynb` (Case B) | `results_gat/` | ~30 min (5 repeats) |
| 6 | `GAT_window_ds004504.ipynb` (Case A) | `results_gat_window/` | ~36 min |
| 7 | `GAT_multiclass_ds004504.ipynb` (3-class) | `results_gat_multiclass/` | ~25 min |
| 8 | `python leak_check.py` (needs 1, 5 and 6) | prints the checks above | ~1 min |
| 9 | `experiments_ad_vs_ftd/features_ad_ftd.ipynb`, then `ADvsFTD_ds004504.ipynb`, then `ADvsFTD_aperiodic.ipynb` | `experiments_ad_vs_ftd/features/`, `results/`, `results/aperiodic/` | ~30 s, ~5 min, ~2.5 min |

The GAT goes after the other Case B models because its statistics cell compares
it with every result folder already present on the same folds.

Seeds are fixed (`SEED = 42`). Small floating-point differences between
machines and library versions can still move individual windows across the
0.5 threshold.

## Repository layout

```
build_cache.ipynb               EEG -> features and graphs (subject_cache/)
GAT_ds004504.ipynb              MFE-FCGAT, Case B (binary, patient split)
GCN_ds004504.ipynb              paper's MFE-FCGCN, same code apart from the graph layer
GAT_nograph_ds004504.ipynb      control: MFE-FCGAT with every graph replaced by a uniform one
RF_ds004504.ipynb               random forest on band power
GAT_window_ds004504.ipynb       MFE-FCGAT, Case A (paper's window split)
GAT_multiclass_ds004504.ipynb   MFE-FCGAT, AD / FTD / HC
leak_check.py                   why Case A scores ~95 %
results_*/                      saved summaries, per-fold CSVs, figures, out-of-fold probabilities, statistics
experiments_ad_vs_ftd/          AD vs FTD feature-engineering study (own README)
docs/PROJECT_NOTES.md           full lab notebook: every run, ablation and review decision
```

## References

1. Liu, An, Yang & Ge (2025). MFE-FCGCN. *Complex & Intelligent Systems* 11:366. [doi:10.1007/s40747-025-01974-x](https://doi.org/10.1007/s40747-025-01974-x)
2. Miltiadous et al. (2023). A dataset of scalp EEG recordings of Alzheimer's disease, frontotemporal dementia and healthy subjects from routine EEG. *Data* 8(6):95. [doi:10.3390/data8060095](https://doi.org/10.3390/data8060095)
3. eeg-dementia-graph-transformer: a patient-independent benchmark on ds004504. <https://github.com/happyc0der/eeg-dementia-graph-transformer>
4. EEG-based classification of AD and FTD using functional connectivity. *Scientific Reports* (2026). <https://pmc.ncbi.nlm.nih.gov/articles/PMC12873302/>
5. Veličković et al. (2018). Graph Attention Networks. *ICLR*.
6. Lal, Chikkankod & Longo (2024). A comparative study on feature extraction techniques for the discrimination of frontotemporal dementia and Alzheimer's disease with electroencephalography in resting-state adults. *Brain Sciences* 14(4):335. [doi:10.3390/brainsci14040335](https://doi.org/10.3390/brainsci14040335)
7. Wang et al. (2023). The effect of aperiodic components in distinguishing Alzheimer's disease from frontotemporal dementia. *GeroScience* 46:751–768. [doi:10.1007/s11357-023-01041-8](https://doi.org/10.1007/s11357-023-01041-8)
8. Thijssen et al. (2021). Plasma phosphorylated tau 217 and phosphorylated tau 181 as biomarkers in Alzheimer's disease and frontotemporal lobar degeneration: a retrospective diagnostic performance study. *Lancet Neurology* 20(9):739–752. [doi:10.1016/S1474-4422(21)00214-3](https://doi.org/10.1016/S1474-4422(21)00214-3)

Code: MIT license. Dataset: CC0, by its authors [2].
