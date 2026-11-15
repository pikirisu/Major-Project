"""The base paper's released model and training loop, on held-out patients.

Source: github.com/brief-city/MFE-FCGCN-model, model_and_train.py. Kept as released:
  FiveGraphsModel: per-band CNN (kernels 3/6/11/71/61) -> ONE GCN shared by the 5 bands
    (conv1 on the MI graph, ReLU, conv2 on the Pearson graph, no ReLU) -> MLP 30-14-2,
    dropout 0.8 / 0.5
  Adam lr 1.2e-4, weight decay 1e-4, batch 64, 200 epochs,
  ReduceLROnPlateau(factor 0.1, patience 5)
  validation = 1 of 15 stratified folds of the training WINDOWS (StratifiedKFold, seed 1)
  the model kept = the epoch with the best validation AUC
Changed:
  * outer split: our 25 patient-level test sets (5-fold x 5 repeats, 13 patients each,
    the same test patients as every notebook) instead of one unseeded 80/20 patient split
  * one inner fold per outer split (theirs trains 15 and keeps the last one's model)
  * PyG GCNConv written densely (19 nodes): add_remaining_self_loops (fill 1),
    D^-1/2 A D^-1/2 with in-degree, lin without bias, then bias. Same maths.
Two variants, identical except what the LR scheduler reads:
  released: scheduler.step(test loss)   <- as in their code
  clean   : scheduler.step(val loss)
Control (SPLIT=window): the same code, but the test set is 20% of the WINDOWS (5-fold StratifiedKFold,
  seed 42), so every patient sits on both sides. Released variant only. Checks that this port can
  reach the paper's numbers when the split lets it.

Usage: python experiments_released_code/run.py   (after prep.py; writes results_released_code/)
       SPLIT=window python experiments_released_code/run.py   (writes results_released_code/window_split/)
"""
import copy, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("RELEASED_CACHE", ROOT / "cache_released_code"))
OUT = Path(os.environ.get("RELEASED_OUT", ROOT / "results_released_code"))
BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
SEED, N_FOLDS, N_REPEATS = 42, 5, 5          # identical to GCN_ds004504.ipynb / GAT_ds004504.ipynb
EPOCHS = int(os.environ.get("EPOCHS", 200))
SPLIT = os.environ.get("SPLIT", "patient")
if SPLIT == "window":
    OUT = OUT / "window_split"


def our_test_sets():
    """Test patients of every (repeat, fold), from the notebooks' make_splits()."""
    y, sid = [], []
    for i in range(1, 66):
        with np.load(ROOT / "subject_cache" / f"sub-{i:03d}.npz") as z:
            y.append(z["y"]); sid.append(np.full(len(z["y"]), f"sub-{i:03d}"))
    Y, SID = np.concatenate(y), np.concatenate(sid)
    out, idx = [], np.arange(len(Y))
    for r in range(1, N_REPEATS + 1):
        outer = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED + 1000 * r).split(idx, Y, groups=SID)
        for f, (_, test) in enumerate(outer, start=1):
            out.append((r, f, sorted(set(SID[test]))))
    return out


def load():
    x = {b: [] for b in BANDS}; mi, pc, y, sid = [], [], [], []
    for i in range(1, 66):
        with np.load(CACHE / f"sub-{i:03d}.npz") as z:
            for b in BANDS: x[b].append(z[f"x_{b}"])
            mi.append(z["mi"]); pc.append(z["pc"]); y.append(z["y"])
            sid.append(np.full(len(z["y"]), f"sub-{i:03d}"))
    return ([np.concatenate(x[b]) for b in BANDS], np.concatenate(mi), np.concatenate(pc),
            np.concatenate(y), np.concatenate(sid))


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return dict(acc=accuracy_score(y, pred), sens=recall_score(y, pred, zero_division=0),
                spec=recall_score(1 - y, 1 - pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
                auc=roc_auc_score(y, p))


def run_one(r, f, test, variant):
    import torch, torch.nn as nn, torch.nn.functional as F
    torch.set_num_threads(2)
    X, MI, PC, Y, SID = load()
    seed = SEED + 1000 * r + f
    torch.manual_seed(seed); rng = np.random.default_rng(seed)

    class DenseGCNConv(nn.Module):              # torch_geometric.nn.GCNConv, dense
        def __init__(self, cin, cout):
            super().__init__()
            self.lin = nn.Linear(cin, cout, bias=False); nn.init.xavier_uniform_(self.lin.weight)
            self.bias = nn.Parameter(torch.zeros(cout))
        def forward(self, x, A):                 # A[b, i, j] = weight of edge i -> j
            eye = torch.eye(A.size(-1), dtype=torch.bool)
            A = torch.where(eye & (A == 0), torch.ones_like(A), A)      # add_remaining_self_loops
            dinv = A.sum(1).pow(-0.5); dinv[torch.isinf(dinv)] = 0      # in-degree of each target j
            norm = dinv[:, :, None] * A * dinv[:, None, :]
            return norm.transpose(1, 2) @ self.lin(x) + self.bias       # out_j = sum_i norm_ij h_i

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.ModuleDict({str(d): nn.Conv2d(1, 1, kernel_size=(1, d - 15)) for d in (18, 21, 26, 76, 86)})
        def forward(self, x):                    # (B, 19, F) -> (B, 19, 16)
            return self.cnn[str(x.size(2))](x.unsqueeze(1)).squeeze(1)

    class GCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1, self.conv2, self.cnn = DenseGCNConv(16, 16), DenseGCNConv(16, 5), SimpleCNN()
        def forward(self, x, mi, pc):
            x = F.relu(self.conv1(self.cnn(x), mi))
            return self.conv2(x, pc).reshape(x.size(0), 19 * 5)

    class FiveGraphsModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.gcn = GCN()
            self.fc, self.lin2, self.lin3 = nn.Linear(5 * 19 * 5, 30), nn.Linear(30, 14), nn.Linear(14, 2)
        def forward(self, xs, mi, pc):
            out = torch.cat([self.gcn(xs[b], mi[:, b], pc[:, b]) for b in range(5)], dim=1)
            out = F.dropout(F.relu(self.fc(out)), p=0.8, training=self.training)
            out = F.dropout(F.relu(self.lin2(out)), p=0.5, training=self.training)
            return self.lin3(out)

    Xt = [torch.from_numpy(a).float() for a in X]
    MIt, PCt, Yt = torch.from_numpy(MI).float(), torch.from_numpy(PC).float(), torch.from_numpy(Y)
    is_test = np.isin(np.arange(len(Y)), test) if SPLIT == "window" else np.isin(SID, test)
    test, train_all = np.where(is_test)[0], np.where(~is_test)[0]
    tr, va = next(StratifiedKFold(15, shuffle=True, random_state=1).split(train_all, Y[train_all]))
    train, val = train_all[tr], train_all[va]

    model = FiveGraphsModel()
    opt = torch.optim.Adam(model.parameters(), lr=0.00012, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=5)
    crit = nn.CrossEntropyLoss()
    batch = lambda i: ([x[i] for x in Xt], MIt[i], PCt[i], Yt[i])

    def evaluate(idx, bs, shuffle):
        model.eval(); loss, probs = 0.0, np.zeros(len(idx))
        order = rng.permutation(len(idx)) if shuffle else np.arange(len(idx))
        with torch.no_grad():
            for s in range(0, len(idx), bs):
                k = order[s:s + bs]; xs, mi, pc, y = batch(idx[k])
                out = model(xs, mi, pc)
                loss += crit(out, y).item()                      # summed per batch, as theirs
                probs[k] = out.softmax(1)[:, 1].numpy()
        return loss, probs

    best_auc, best_state, best_epoch, hist = 0.0, None, 0, []
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = rng.permutation(len(train))
        for s in range(0, len(train), 64):
            xs, mi, pc, y = batch(train[perm[s:s + 64]])
            opt.zero_grad(); loss = crit(model(xs, mi, pc), y); loss.backward(); opt.step()
        v_loss, v_p = evaluate(val, 128, False)
        t_loss, t_p = evaluate(test, 256, True)
        v_auc = roc_auc_score(Y[val], v_p)
        t_acc = accuracy_score(Y[test], (t_p >= 0.5).astype(int))
        hist.append(dict(epoch=ep, lr=opt.param_groups[0]["lr"], val_loss=v_loss, val_auc=v_auc,
                         test_loss=t_loss, test_w_acc=t_acc, test_w_auc=roc_auc_score(Y[test], t_p)))
        if v_auc > best_auc:
            best_auc, best_epoch, best_state = v_auc, ep, copy.deepcopy(model.state_dict())
        sched.step(t_loss if variant == "released" else v_loss)

    model.load_state_dict(best_state)
    _, p = evaluate(test, 256, False)
    sid_t, y_t = SID[test], Y[test]
    subs = np.unique(sid_t)
    s_p = np.array([p[sid_t == s].mean() for s in subs]); s_y = np.array([y_t[sid_t == s][0] for s in subs])
    h = pd.DataFrame(hist)
    row = dict(variant=variant, repeat=r, fold=f, best_epoch=best_epoch, seconds=round(time.time() - t0),
               final_lr=h.lr.iloc[-1], test_w_acc_last_epoch=h.test_w_acc.iloc[-1],
               test_w_acc_max_over_epochs=h.test_w_acc.max(),
               **{f"w_{k}": v for k, v in metrics(y_t, p).items()},
               **{f"s_{k}": v for k, v in metrics(s_y, s_p).items()})
    h.insert(0, "variant", variant); h.insert(1, "repeat", r); h.insert(2, "fold", f)
    return row, h, dict(zip(subs, s_p))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if SPLIT == "window":
        Yw = load()[3]
        splits = [(1, f, te) for f, (_, te) in
                  enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(Yw, Yw), start=1)]
        variants = ("released",)
    else:
        splits, variants = our_test_sets(), ("released", "clean")
    if len(sys.argv) > 1:                                           # quick check: run.py 1  -> first split only
        splits = splits[:int(sys.argv[1])]
    jobs = [(r, f, t, v) for v in variants for r, f, t in splits]
    res = Parallel(n_jobs=min(9, len(jobs)), verbose=10)(delayed(run_one)(*j) for j in jobs)
    rows = pd.DataFrame([a for a, _, _ in res])
    rows.to_csv(OUT / "fold_results.csv", index=False)
    pd.concat([h for _, h, _ in res]).to_csv(OUT / "history.csv", index=False)
    if SPLIT == "window":                                           # every patient is in every test fold:
        cols = ["w_acc", "w_auc", "test_w_acc_last_epoch", "test_w_acc_max_over_epochs"]   # windows only
        summary = {"window_split_released": dict(folds=len(rows), **rows[cols].mean().to_dict(),
                                                 window_acc_min=rows.w_acc.min(), window_acc_max=rows.w_acc.max())}
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2)); sys.exit()
    oof = {v: np.full((N_REPEATS, 65), np.nan) for v in ("released", "clean")}
    for (a, _, sp) in res:
        for s, pr in sp.items():
            oof[a["variant"]][a["repeat"] - 1, int(s[4:]) - 1] = pr
    np.save(OUT / "oof_prob_released.npy", oof["released"]); np.save(OUT / "oof_prob_clean.npy", oof["clean"])
    y65 = (np.arange(1, 66) <= 36).astype(int)
    summary = {}
    for v in ("released", "clean"):
        g = rows[rows.variant == v]
        pooled = pd.DataFrame([metrics(y65, oof[v][r]) for r in range(N_REPEATS) if not np.isnan(oof[v][r]).any()])
        summary[v] = dict(folds=len(g),
                          window_acc_mean=g.w_acc.mean(), window_acc_min=g.w_acc.min(), window_acc_max=g.w_acc.max(),
                          window_auc_mean=g.w_auc.mean(),
                          window_acc_last_epoch_mean=g.test_w_acc_last_epoch.mean(),
                          window_acc_max_over_epochs_mean=g.test_w_acc_max_over_epochs.mean(),
                          window_acc_max_over_epochs_max=g.test_w_acc_max_over_epochs.max(),
                          folds_reaching_95_window_acc=int((g.test_w_acc_max_over_epochs >= 0.95).sum()),
                          patient_pooled=pooled.mean().to_dict() if len(pooled) else None)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
