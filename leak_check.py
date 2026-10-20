"""Why the window split (Case A) scores ~95 %: four checks on saved data.

Run from the repo root after build_cache.ipynb and GAT_window_ds004504.ipynb
(check 4 also needs GAT_ds004504.ipynb):

    python leak_check.py

Uses the same outer folds as GAT_window_ds004504.ipynb, repeat 1
(StratifiedKFold, 5 folds, random_state = SEED + 1000 = 1042). Takes about a minute on CPU.
"""
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
N_SUBJECTS, N_AD, SEED = 65, 36, 42          # sub-001..036 AD, sub-037..065 HC

# --- load band power (95 numbers per window) and each window's position in its recording
bp, y, sid, pos = [], [], [], []
for i in range(1, N_SUBJECTS + 1):
    with np.load(ROOT / "subject_cache" / f"sub-{i:03d}.npz") as z:
        n = len(z["y"])
        bp.append(z["bp"].reshape(n, -1)); y.append(z["y"])
        sid.append(np.full(n, i)); pos.append(np.arange(n))
BP, Y, SID, POS = map(np.concatenate, (bp, y, sid, pos))
Y, N = Y.astype(int), len(Y)
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED + 1000).split(np.arange(N), Y))
print(f"{N:,} windows from {N_SUBJECTS} subjects, window split, 5 folds\n")

# 1. Raw-signal overlap. Windows are 20 s long and start every 5 s, so the
#    neighbours k±1, k±2, k±3 share 75 %, 50 % and 25 % of the same signal.
key = {(s, p): j for j, (s, p) in enumerate(zip(SID, POS))}
share75 = share_any = 0
for trainval, test in folds:
    in_train = np.zeros(N, bool); in_train[trainval] = True
    for j in test:
        hit = lambda d: (k := key.get((SID[j], POS[j] + d))) is not None and in_train[k]
        share75 += hit(-1) or hit(1)
        share_any += any(hit(d) for d in (-3, -2, -1, 1, 2, 3))
print(f"1. test windows with a 75 %-overlapping neighbour in training : {share75 / N:6.1%}")
print(f"   test windows sharing any raw signal with a training window : {share_any / N:6.1%}")

# 2. Can 95 band-power numbers tell WHO a window came from? (65-way)
who = np.zeros(N, int)
for trainval, test in folds:
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    who[test] = clf.fit(BP[trainval], SID[trainval]).predict(BP[test])
label = f"2. names the person (1 of {N_SUBJECTS}, chance {1 / N_SUBJECTS:.1%})"
print(f"{label:62s}: {np.mean(who == SID):6.1%}")

# 3. Pure lookup: copy the label of the most similar training window. No learning.
same, pred = np.zeros(N, bool), np.zeros(N, int)
for trainval, test in folds:
    sc = StandardScaler().fit(BP[trainval])
    knn = KNeighborsClassifier(1).fit(sc.transform(BP[trainval]), Y[trainval])
    nb = trainval[knn.kneighbors(sc.transform(BP[test]), return_distance=False)[:, 0]]
    same[test], pred[test] = SID[nb] == SID[test], Y[nb]
print(f"3. nearest training window is from the same person           : {same.mean():6.1%}")
print(f"   1-NN window accuracy (paper's MFE-FCGCN: 95.09 %)          : {np.mean(pred == Y):6.1%}")

# 4. Where the GAT's errors live: per-person window accuracy, Case A vs Case B (repeat 1)
print()
for name, folder in [("Case A, window split ", "results_gat_window"),
                     ("Case B, subject split", "results_gat")]:
    f = ROOT / folder / "oof_prob.npy"
    if not f.exists():
        print(f"4. {name}: {f.relative_to(ROOT)} missing, run its notebook first"); continue
    ok = (np.atleast_2d(np.load(f))[0] >= 0.5) == (Y == 1)
    acc = np.array([ok[SID == s].mean() for s in range(1, N_SUBJECTS + 1)])
    print(f"4. {name}: window acc {ok.mean():.3f} | people with <50 % of windows right "
          f"{(acc < 0.5).sum():2d}/{N_SUBJECTS} | worst person {acc.min():.2f}")
