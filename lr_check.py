"""Smoke-test a twin notebook, then pick its learning rate on validation subjects.

Runs the notebook's own cells (1-4 setup, 7 model, 8 training) in one namespace,
so the code tested is exactly the code the main run uses. Test subjects are
never scored. Rule (docs/PREREG_2026-11-14.md): repeat 1, candidates 1e-3 and
3e-3, mean over folds of the best validation subject AUC (epochs >= MIN_EPOCHS);
within 0.005 -> keep 3e-3. Writes <RESULT_DIR>/lr_check.json; the chosen rate
is then typed into the notebook's Cell 2 before the main run.

Usage, from the repo root (about 1 h per notebook on CPU):
    python lr_check.py . GATv2_ds004504.ipynb
    python lr_check.py . GATv2gate_ds004504.ipynb
Add --smoke-only to run one epoch on one fold and stop."""
import sys, os, json, time
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path

proj, nb_name = Path(sys.argv[1]), sys.argv[2]
smoke_only = "--smoke-only" in sys.argv
os.chdir(proj)
cells = json.loads((proj / nb_name).read_text(encoding="utf8"))["cells"]
code = {i: "".join(c["source"]) for i, c in enumerate(cells) if c["cell_type"] == "code"}
ns = {"__name__": "__main__"}
for i in (1, 2, 3, 4, 7, 8):                       # setup, model, training functions
    exec(compile(code[i], f"{nb_name}:cell{i}", "exec"), ns)

# --- smoke test: one epoch on one fold --------------------------------------
saved = {k: ns[k] for k in ("MAX_EPOCHS", "MIN_EPOCHS", "PATIENCE")}
ns.update(MAX_EPOCHS=1, MIN_EPOCHS=1, PATIENCE=1)
t = time.time()
r = ns["train_fold"](ns["SPLITS"][0])
assert len(r["test_prob"]) == len(ns["SPLITS"][0]["test"]) and (0 <= r["test_prob"]).all() and (r["test_prob"] <= 1).all()
print(f"[{nb_name}] smoke test OK: 1 epoch on fold r1 f1 in {time.time() - t:.0f}s", flush=True)
ns.update(saved)
if smoke_only:
    sys.exit(0)

# --- learning rate on validation subjects ---------------------------------------
res = {}
for lr in (1e-3, 3e-3):
    ns["LR"] = lr
    best = []
    for sp in [s for s in ns["SPLITS"] if s["repeat"] == 1]:
        t = time.time()
        h = ns["train_fold"](sp)["history"]
        best.append(float(h.loc[h.epoch >= ns["MIN_EPOCHS"], "val_s_auc"].max()))
        print(f"[{nb_name}] lr {lr:g} r1 f{sp['fold']}: best val subject AUC {best[-1]:.4f}  [{time.time() - t:.0f}s]", flush=True)
    res[f"{lr:g}"] = dict(per_fold=best, mean=sum(best) / len(best))

m1, m3 = res["0.001"]["mean"], res["0.003"]["mean"]
choice = "1e-3" if m1 - m3 > 0.005 else "3e-3"
out = dict(rule="repeat 1, mean best validation subject AUC; within 0.005 keep 3e-3",
           candidates=res, chosen=choice)
(ns["RESULT_DIR"] / "lr_check.json").write_text(json.dumps(out, indent=2))
print(f"[{nb_name}] mean best val AUC: 1e-3 {m1:.4f} | 3e-3 {m3:.4f}  ->  LR = {choice}", flush=True)
