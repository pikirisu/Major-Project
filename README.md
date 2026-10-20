# Graph attention on resting-state EEG for Alzheimer's detection

This project detects Alzheimer's disease (AD) from 19-channel resting-state EEG
([OpenNeuro ds004504](https://openneuro.org/datasets/ds004504)) using a graph
attention network. We evaluate it two ways: under the window split used by the
base paper, and under a strict split where no patient appears in both training
and test.

The base paper is **MFE-FCGCN** (Liu et al., 2025 [1]), which reports 95.09 %
accuracy and 98.36 % AUC. We replace its graph convolution with multi-head
graph attention (**MFE-FCGAT**). We then:
1. match the paper's result under its own protocol,
2. show that this protocol lets the model recognise *people* rather than disease, and
3. report what the model achieves on patients it has never seen.

---

## Results

### Case B: new patients (patient-independent split, the clinically meaningful number)

5-fold cross-validation grouped by patient: nobody is in both training and test.
A patient's score is the mean probability over their 20 s windows. Results are
pooled over all 65 patients (36 AD, 29 healthy controls).

| model | accuracy | recall (AD found) | specificity | AUC |
|---|---|---|---|---|
| band-power logistic regression, same folds | 0.754 | 0.806 | 0.690 | 0.881 |
| random forest on band power | 0.800 | 0.806 | 0.793 | 0.854 |
| **MFE-FCGAT (ours)** | 0.862 | 0.861 | 0.862 | 0.927 |
| **MFE-FCGAT + band-power soft vote** | **0.892** | **0.889** | **0.897** | **0.939** |
| *published: leave-one-subject-out on ds004504 [3]* | *0.846* | | | *0.883* |

> ⏳ These are from one repeat of 5-fold CV. A 3-repeat confirmation, and the
> paper's GCN under the same protocol (`GCN_ds004504.ipynb`), are in progress.

The soft vote is the plain average of the GAT's and the logistic regression's
probabilities. The weights are equal and nothing is tuned.

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
  accuracy, and from 0 to 9 misdiagnosed patients out of 65.

Case A measures how well a model recognises people it has already seen. Case B
measures how well it diagnoses someone new. We report Case A only to show we
can reproduce the paper, and always next to Case B.

### 3-class: AD / frontotemporal dementia (FTD) / healthy, patient-independent

88 patients (36 AD, 23 FTD, 29 HC), 5-fold CV × 3 repeats, patient level:

| model | macro recall | recall HC | recall AD | recall FTD | macro AUC | dementia vs HC AUC |
|---|---|---|---|---|---|---|
| MFE-FCGAT, 3-way head | 0.583 ± 0.025 | 0.782 | 0.750 | 0.217 | 0.748 | 0.871 |
| band-power logistic regression | 0.578 ± 0.006 | 0.747 | 0.611 | 0.377 | 0.758 | 0.871 |
| **soft vote** | **0.600 ± 0.026** | 0.782 | 0.685 | 0.333 | **0.783** | **0.889** |
| *published strict benchmark [3]* | *0.611* | *0.86* | *0.69* | *0.28* | *0.786* | |

In resting-state band power, AD and FTD look alike: the AD vs FTD AUC is about
0.60 for every model. Under a strict patient-independent protocol, published
methods also land near 60 % ± 7.

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
| 2 | `GAT_ds004504.ipynb` (Case B) | `results_gat/` | ~15 min per repeat |
| 3 | `GCN_ds004504.ipynb` (paper's GCN, Case B) | `results_gcn/` | ~15 min per repeat |
| 4 | `RF_ds004504.ipynb` (random forest baseline) | `results_rf/` | < 1 min |
| 5 | `GAT_window_ds004504.ipynb` (Case A) | `results_gat_window/` | ~36 min |
| 6 | `GAT_multiclass_ds004504.ipynb` (3-class) | `results_gat_multiclass/` | ~25 min |
| 7 | `python leak_check.py` (needs 1, 2 and 5) | prints the checks above | ~1 min |

Seeds are fixed (`SEED = 42`). Small floating-point differences between
machines and library versions can still move individual windows across the
0.5 threshold.

## Repository layout

```
build_cache.ipynb               EEG -> features and graphs (subject_cache/)
GAT_ds004504.ipynb              MFE-FCGAT, Case B (binary, patient split)
GCN_ds004504.ipynb              paper's MFE-FCGCN, same code apart from the graph layer
RF_ds004504.ipynb               random forest on band power
GAT_window_ds004504.ipynb       MFE-FCGAT, Case A (paper's window split)
GAT_multiclass_ds004504.ipynb   MFE-FCGAT, AD / FTD / HC
leak_check.py                   why Case A scores ~95 %
results_*/                      saved summaries, per-fold CSVs, figures, out-of-fold probabilities
docs/PROJECT_NOTES.md           full lab notebook: every run, ablation and review decision
```

## References

1. Liu, An, Yang & Ge (2025). MFE-FCGCN. *Complex & Intelligent Systems* 11:366. [doi:10.1007/s40747-025-01974-x](https://doi.org/10.1007/s40747-025-01974-x)
2. Miltiadous et al. (2023). A dataset of scalp EEG recordings of Alzheimer's disease, frontotemporal dementia and healthy subjects from routine EEG. *Data* 8(6):95. [doi:10.3390/data8060095](https://doi.org/10.3390/data8060095)
3. eeg-dementia-graph-transformer: a patient-independent benchmark on ds004504. <https://github.com/happyc0der/eeg-dementia-graph-transformer>
4. EEG-based classification of AD and FTD using functional connectivity. *Scientific Reports* (2026). <https://pmc.ncbi.nlm.nih.gov/articles/PMC12873302/>
5. Veličković et al. (2018). Graph Attention Networks. *ICLR*.

Code: MIT license. Dataset: CC0, by its authors [2].
