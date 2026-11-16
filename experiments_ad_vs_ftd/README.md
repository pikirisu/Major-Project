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
| `ADvsFTD_riemann.ipynb` | Experiment 9: the stage-2 AD-vs-FTD specialist (connectivity matrices, model chosen by nested CV); reads the dataset, `features/` and `results/aperiodic/`, writes `features/riemann_matrices.npz` and `results/riemann/` | ~5 min, + ~57 min for the permutation test |
| `ADvsFTD_C3_ds006036.ipynb` | Experiment 10: a pre-registered test of Experiment 9's best candidate (C3) on ds006036; reads both datasets, `features/` and `results/riemann/`, writes `features/c3_ds006036_matrices.npz` and `results/c3_ds006036/` | ~1 min, + ~15 min for the permutation test |
| `ADvsFTD_C3_both.ipynb` | Experiment 11: C3 averaged over both recordings of each patient; reads the saved C3 matrices and scores, writes `results/c3_both/` | ~22 min, almost all the permutation test |

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

## Experiment 9: a stage-2 specialist for a two-stage 3-class model

The plan was a two-stage classifier: stage 1 decides dementia vs healthy, and
stage 2 decides AD vs FTD among the patients. This experiment builds stage 2.

`ADvsFTD_riemann.ipynb` keeps the design above (300 s per subject, one row per
subject) and adds the following, written to `results/riemann/protocol.json`
before any matrix was computed:
- **Reference.** Common average reference. The derivatives are not
  average-referenced.
- **Matrices.** Per-band (delta, theta, alpha, beta) covariance and lagged
  cross-correlation matrices. ICA left 16–19 dimensions per subject, so the
  covariances get 1 % diagonal loading.
- **Five candidates:**
  - C1 band power;
  - C2 Riemannian tangent space of the covariances;
  - C3 lag-maximised cross-correlation (Fisher z; the measure of Mlinarič et al. 2026);
  - C4 parameterised spectrum;
  - C5 a soft vote of C2–C4.
- **Classifier.** L2 logistic regression; C is tuned by inner CV.
- **Pipeline choice.** The pipeline is chosen in each outer fold by nested inner
  AUC, using training patients only.
- **Folds.** 5 × 5-fold CV on the 3-class notebook's own folds (checked in the
  notebook), so stage 2 can later be combined with the 3-class GAT.

| 36 AD vs 23 FTD, AUC [95 % CI] | | vs band power, Holm p |
|---|---|---|
| **the pre-registered pipeline (chosen per fold)** | **0.576 [0.457, 0.687]** | |
| C1 band power | 0.524 [0.405, 0.634] | |
| C2 Riemannian covariance | 0.549 [0.418, 0.674] | 1.0 |
| C3 lagged cross-correlation | 0.628 [0.502, 0.748] | 0.556 |
| C4 parameterised spectrum | 0.542 [0.403, 0.677] | 1.0 |
| C5 soft vote | 0.592 [0.459, 0.715] | 0.945 |
| chance (100 shuffles through the whole nested pipeline), 95 % range | 0.291 – 0.651 | |
| MMSE alone (not an input) | 0.806 | |

**Not supported.**
- **Chance.** The selected pipeline does not beat chance (permutation p 0.337),
  and adds nothing to MMSE, age and sex (p 0.96).
- **Selection.** Picking per fold cost accuracy: inner AUCs from about 47
  patients are too noisy to rank five candidates.
- **C3.** C3 was the best candidate. Its AUC (0.628) is close to the 0.651
  that Mlinarič et al. report for the same measure. But it was picked after
  the fact, it does not beat band power, it correlates with MMSE (r −0.36),
  and its own 95 % CI touches 0.5.
- **The target.** The target (AUC ≥ 0.70) was not reached.

The run log prints `KeyError: ...joblib_memmapping_folder...` tracebacks at
exit. These are joblib failing to delete temporary folders on Windows. All
cells ran.

## Experiment 10: C3 on a second recording of the same patients

C3 was picked after Experiment 9's results were seen. `ADvsFTD_C3_ds006036.ipynb`
gives it one test on ds006036: the same 88 people, recorded straight after
ds004504 with their eyes open under photic stimulation. The design was fixed in
`results/c3_ds006036/protocol.json` before any matrix was computed:
- **The pipeline is unchanged.** It is C3 exactly as in Experiment 9 (bands,
  filter, lags, reference, classifier, C grid, folds and seeds). Two identity
  checks pass: the notebook rebuilds Experiment 9's sub-001 matrix exactly, and
  it reproduces Experiment 9's C3 scores exactly.
- **44 s per subject.** The shortest ds006036 patients have only 11 clean 4 s
  epochs, so every subject contributes 11 epochs, packed between the ASR joins.
- **Rule.** C3 counts as confirmed if the one-sided permutation p is below 0.05,
  from 1000 label shuffles through the whole pipeline.

| C3, 36 AD vs 23 FTD | AUC [95 % CI] | beyond MMSE + age + sex, p |
|---|---|---|
| **ds006036, 44 s (pre-registered)** | **0.641 [0.515, 0.753]**, permutation p 0.045 | 0.18 |
| ds004504, the same 44 s (control) | 0.523 [0.408, 0.637] | 0.72 |
| ds004504, 300 s (Experiment 9, picked post hoc) | 0.628 [0.502, 0.748] | 0.50 |

**Confirmed, narrowly.**
- **The test.** C3 passes the pre-registered test. With 1000 shuffles, p = 0.045
  has a Monte Carlo error of about ±0.007, so the pass is close to the line.
- **Agreement.** Patients' C3 scores on the two recordings correlate
  (Spearman r 0.40, p 0.002), so the score partly reflects something stable
  about each patient.
- **The control.** On the same 44 s of eyes-closed ds004504 data, C3 drops to
  chance (0.523). On ds004504 it needs the full 300 s; this experiment cannot
  say why the eyes-open recording does better with less data.
- **Severity.** It is still not shown that C3 adds anything to MMSE, age and sex
  (p 0.18; r with MMSE −0.27). Part of what it sees may be disease severity.
- **Not a new cohort.** These are the same 59 patients, so this is evidence
  across recordings, not across people.

## Experiment 11: both recordings averaged

`ADvsFTD_C3_both.ipynb` averages each patient's two out-of-fold C3 scores
(eyes closed, 300 s; eyes open, 44 s). The rule was written to
`results/c3_both/protocol.json` before the average was computed: one plain
mean, and nothing else is tried. Both inputs had been seen already, and the
ds004504 half is where C3 was picked, so this AUC is somewhat optimistic.

| C3, 36 AD vs 23 FTD | AUC [95 % CI] | gain over ds006036 alone |
|---|---|---|
| **both recordings, averaged** | **0.658 [0.527, 0.773]**, permutation p 0.021 | +0.017 [−0.052, 0.087] |
| ds006036 alone (Experiment 10) | 0.641 [0.515, 0.753] | |
| ds004504 alone (Experiment 9) | 0.628 [0.502, 0.748] | |

**Not adopted.**
- **The pre-registered rule.** The average replaces ds006036 C3 as stage 2 only
  if the lower bound of its gain is above 0. It is −0.052, so the gain is
  within noise.
- **Severity.** It still does not add beyond MMSE + age + sex (p 0.19).
- **3-class, for the record.** With the GAT + band-power soft vote as stage 1,
  the average would give 65.9 % accuracy and macro-F1 0.645. The adopted
  stage 2 (ds006036) gives 62.1 % and 0.611.

## Files

```
features/            eeg_features.csv (88 x 417) · subjects.csv · feature_dictionary.csv · wpli.npz
                     riemann_matrices.npz (per-band covariance and cross-correlation, ICA rank)
                     c3_ds006036_matrices.npz (C3 on ds006036 and on 44 s of ds004504)
results/             summary.csv · comparisons.csv · above_chance.csv · positive_control_ad_vs_hc.csv
                     coefficients.csv · univariate.csv · confounders.csv · within_sex.csv
                     oof_prob.npz · protocol.json · 7 figures (.png)
results/aperiodic/   protocol.json · tests.csv · secondary.csv · per_channel.csv
                     subject_scores.csv · channel_fits.csv · aperiodic.png · topomaps.png
results/riemann/     protocol.json · candidates.csv · selection.csv · confounds.csv
                     permutation.csv · stage2_oof.npz · summary.json
results/c3_ds006036/ protocol.json · results.csv · confounds.csv · permutation.csv
                     c3_oof.npz · summary.json
results/c3_both/     protocol.json · results.csv · gain.csv · confounds.csv · three_class.csv
                     permutation.csv · c3_both_oof.npz · summary.json
```
