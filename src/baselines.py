"""E0 baselines: global mean and smoothed username mean (OOF)."""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import load
from cv import get_folds, evaluate, log_row

tr, te = load()
folds = get_folds(tr)
y = np.log1p(tr["likes"].to_numpy())

log_row(name="DECISION_branch", features="audit", params="Branch R: StratifiedKFold(5) on zero+decile bins of log1p(likes)",
        notes="random split in time (test share ~18-19% every month); 31.2% test rows have unseen user vs 32.6% in CV; "
              "id == date order within each file, no extra leak; adversarial AUC 0.629 driven by device shift "
              "(test Android 70% vs train 53%) and train-only missing categories; dropped 25 negative-likes rows")

t = time.time(); oof = np.zeros(len(y)); pt = np.zeros(len(te))
for trn, val in folds:
    oof[val] = y[trn].mean(); pt += y[trn].mean() / len(folds)
evaluate("E0_global_mean", y, oof, folds, pt, features="none", runtime=time.time() - t)

for m in (5, 20, 50):
    t = time.time(); oof = np.zeros(len(y)); pt = np.zeros(len(te))
    def enc(trn_idx, keys):
        g = tr.iloc[trn_idx].assign(y=y[trn_idx]).groupby("username")["y"].agg(["sum", "count"])
        mu = y[trn_idx].mean()
        s = keys.map(g["sum"]).fillna(0).to_numpy(); n = keys.map(g["count"]).fillna(0).to_numpy()
        return (s + m * mu) / (n + m)
    for trn, val in folds:
        oof[val] = enc(trn, tr.username.iloc[val])
        pt += enc(trn, te.username) / len(folds)
    evaluate(f"E0_user_mean_m{m}", y, oof, folds, pt, features="username smoothed mean", params=f"m={m}",
             runtime=time.time() - t)
