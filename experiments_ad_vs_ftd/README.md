# AD vs FTD: engineered EEG features, subject-independent

Can engineered spectral, nonlinear and connectivity features separate Alzheimer's
disease (36) from frontotemporal dementia (23) on ds004504, with no subject in
both training and test? This folder answers that question. Nothing in the rest
of the repository is changed by it.

## Run

From this folder, in order:

| notebook | does | time |
|---|---|---|
| `features_ad_ftd.ipynb` | reads `../data/derivatives/` and writes `features/`: one row per subject, 417 EEG features, plus a separate table of labels, demographics and signal quality | ~30 s |
| `ADvsFTD_ds004504.ipynb` | reads `features/` and writes `results/`: 11 experiments, statistics, figures | ~5 min |
| `ADvsFTD_aperiodic.ipynb` | Experiment 8: a pre-registered test of the aperiodic (1/f) hypothesis of Wang et al. 2023; reads the dataset and `features/subjects.csv`, writes `results/aperiodic/` | ~2.5 min |

The features notebook and Experiment 8 need the dataset. The experiment
notebook needs only `features/`; it also uses `../subject_cache/` and
`../results_gat_multiclass/` if they are present, to add the existing 3-class
models as comparators. No packages beyond `../requirements.txt` (Experiment 8
uses `fooof`, which is listed there).

## Design, fixed before any AD-vs-FTD result was looked at

- **Equal data per subject.** Every subject contributes its first 75 four-second
  epochs that contain no ASR join (300 s). In the old cache, the window count
  alone separated AD from FTD (AUC 1.000, caps 80 vs 100), and recording length
  reached 0.669.
- **One row per subject.** Each feature is a median over that subject's own
  epochs, so no information crosses between subjects before cross-validation.
- **One pre-specified classifier:** standardise, then L2 logistic regression
  (`C = 0.1`, balanced class weights), fitted inside every fold. It is the
  project's band-power floor. Nothing is tuned and no features are selected.
- **20 × 5-fold CV over subjects.** 95 % CIs come from a subject bootstrap of
  the repeat-averaged metric. Paired bootstrap tests are Holm-corrected within
  each family. A label-shuffle test sets the chance level.
- **Controls:** sex, age and recording length alone (never mixed into the EEG
  sets), and the same pipeline on AD vs healthy controls as a positive control.

## Result

| AUC, FTD vs AD [95 % CI] | |
|---|---|
| existing band power | 0.540 [0.403, 0.667] |
| engineered: spectral / nonlinear / connectivity | 0.448 / 0.410 / 0.487 |
| combinations | 0.450 – 0.481 |
| sex alone / recording length alone | 0.576 / 0.650 |
| existing 3-class GAT | 0.595 [0.469, 0.713] |
| chance (shuffled diagnoses), 95 % range | ≈ 0.30 – 0.68 |
| **same features on AD vs healthy controls** | **0.66 – 0.84** |

No feature group separates AD from FTD above chance, none beats the existing
band power, and none beats knowing the patient's sex. The same features do
separate AD from healthy controls, so they carry dementia information, but not
the AD-vs-FTD contrast. Frontal and temporal theta power is the most consistent
single difference (higher in AD), but it does not survive FDR correction over
210 features.

**Limits:**
- 23 FTD subjects;
- one site;
- no FTD subtypes in the dataset;
- 5 minutes per subject;
- one linear classifier.

AD-vs-FTD accuracies near 90 % reported on this dataset come from window-level
splits; see `../leak_check.py`.

## Experiment 8: a pre-registered test of one published lead

Wang et al. (2023, *GeroScience* 46:751) reported, on this same cohort, that the
aperiodic exponent and offset of the EEG spectrum are higher in AD than in FTD.
Their classifier applied SMOTE before the train/test split, so it was never
tested honestly.

`ADvsFTD_aperiodic.ipynb` tests their claims once:
- the hypotheses and their directions are written to `results/aperiodic/protocol.json`
  before any EEG is read;
- the spectral settings are taken from Wang et al.'s own code;
- each score is used as it is, with nothing fitted;
- Holm correction over the three hypotheses.

| 36 AD vs 23 FTD | AUC = P(AD > FTD) [95 % CI] | p, one-sided | Holm p |
|---|---|---|---|
| H1 aperiodic exponent | 0.632 [0.481, 0.779] | 0.048 | 0.088 |
| H2 aperiodic offset | 0.640 [0.482, 0.785] | 0.036 | 0.088 |
| H3 theta/alpha ratio | 0.644 [0.498, 0.785] | 0.029 | 0.088 |
| the three together, logistic regression, 20 × 5-fold CV | 0.616 [0.469, 0.759] | 0.076 vs chance | |

**Not supported.**
- **Direction.** All three scores point the way Wang et al. predicted, and each passes on its own,
  but none survives correction. With 23 FTD patients, power was about 50 %.
- **Severity.** Once MMSE, age and sex are accounted for, the exponent and the offset
  add nothing (p 0.44, 0.205), because the AD patients here are more advanced.
- **Artefacts.** Muscle activity does not explain it. Neither does the window length.
- **Positive controls.** These reproduce Wang et al.'s pattern: the exponent is higher
  in AD than in healthy controls (0.664, p 0.012), while FTD looks like the controls
  (0.498).
- **Classifier.** A classifier on the three scores (AUC 0.616) is the best EEG result in
  this folder, but it is not distinguishable from chance.
- **Methods note.** Wang et al.'s 0.5 s Welch window overestimates steep 1/f exponents
  (a true 1.5 is estimated as 1.74). Low-frequency (delta) power leaks into the fit range.

## Files

```
features/            eeg_features.csv (88 x 417) · subjects.csv · feature_dictionary.csv · wpli.npz
results/             summary.csv · comparisons.csv · above_chance.csv · positive_control_ad_vs_hc.csv
                     coefficients.csv · univariate.csv · confounders.csv · within_sex.csv
                     oof_prob.npz · protocol.json · 7 figures (.png)
results/aperiodic/   protocol.json · tests.csv · secondary.csv · per_channel.csv
                     subject_scores.csv · channel_fits.csv · aperiodic.png · topomaps.png
```
