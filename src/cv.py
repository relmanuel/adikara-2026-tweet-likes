"""Phase B: validation harness. Folds are created once and frozen in experiments/folds.json."""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import EXP, SEED, load

N_FOLDS = 5
FOLDS_PATH = EXP / "folds.json"
LOG_PATH = EXP / "log.csv"
LOG_COLS = ["timestamp", "name", "features", "params", "cv_mean", "cv_std", "oof_rmse", "seed", "runtime_s", "notes"]


def get_folds(tr: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Branch R: StratifiedKFold on log1p(likes) bins (zero bin + deciles of non-zero).

    Frozen after first creation; keyed by `id` so row order changes cannot silently alter folds.
    """
    if FOLDS_PATH.exists():
        saved = json.loads(FOLDS_PATH.read_text())
        fold_of = pd.Series(saved["fold_of_id"]).rename(int)
        fold_of.index = fold_of.index.astype(int)
        f = tr["id"].map(fold_of).to_numpy()
        assert not np.isnan(f).any(), "rows missing from frozen folds"
        return [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(saved["n_folds"])]

    y = np.log1p(tr["likes"].to_numpy())
    bins = np.zeros(len(y), dtype=int)
    nz = y > 0
    bins[nz] = 1 + pd.qcut(y[nz], 10, labels=False, duplicates="drop")
    fold = np.full(len(y), -1)
    for k, (_, val) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(y, bins)):
        fold[val] = k
    FOLDS_PATH.write_text(json.dumps({
        "branch": "R", "n_folds": N_FOLDS, "seed": SEED,
        "fold_of_id": {int(i): int(k) for i, k in zip(tr["id"], fold)}}))
    return get_folds(tr)


def rmsle(y_true_raw, y_pred_raw) -> float:
    y_pred_raw = np.clip(np.asarray(y_pred_raw, float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true_raw) - np.log1p(y_pred_raw)) ** 2)))


def rmse_log(y_true_log, y_pred_log) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true_log) - np.asarray(y_pred_log)) ** 2)))


def evaluate(name, y_log, oof, folds, test_pred=None, features="", params="", seed=SEED, runtime=0.0,
             notes="", save=True, log=True):
    """Score OOF (log space), save arrays, append one row to experiments/log.csv."""
    oof = np.clip(oof, 0, None)
    scores = [rmse_log(y_log[v], oof[v]) for _, v in folds]
    res = dict(name=name, cv_mean=float(np.mean(scores)), cv_std=float(np.std(scores)),
               oof_rmse=rmse_log(y_log, oof), folds=[round(s, 5) for s in scores])
    if save:
        (EXP / "oof").mkdir(parents=True, exist_ok=True)
        (EXP / "preds").mkdir(parents=True, exist_ok=True)
        np.save(EXP / "oof" / f"{name}.npy", oof)
        if test_pred is not None:
            np.save(EXP / "preds" / f"{name}.npy", np.clip(test_pred, 0, None))
    if log:
        log_row(name=name, features=features, params=params, cv_mean=round(res["cv_mean"], 5),
                cv_std=round(res["cv_std"], 5), oof_rmse=round(res["oof_rmse"], 5), seed=seed,
                runtime_s=round(runtime, 1), notes=notes)
    print(f"[{name}] CV {res['cv_mean']:.5f} ± {res['cv_std']:.5f} | OOF {res['oof_rmse']:.5f} | folds {res['folds']}")
    return res


def log_row(**kw):
    new = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS)
        if new:
            w.writeheader()
        w.writerow({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **{k: kw.get(k, "") for k in LOG_COLS[1:]}})


if __name__ == "__main__":
    tr, te = load()
    folds = get_folds(tr)
    y = np.log1p(tr["likes"].to_numpy())
    print("fold sizes:", [len(v) for _, v in folds])
    # Does random KFold reproduce the test's unseen-user share?
    unseen = []
    for trn, val in folds:
        seen = set(tr.username.iloc[trn])
        unseen.append(1 - tr.username.iloc[val].isin(seen).mean())
    test_unseen = 1 - te.username.isin(set(tr.username)).mean()
    print(f"val rows with user unseen in fold-train: {np.round(unseen, 4)} mean {np.mean(unseen):.4f}"
          f" | test rows unseen in train: {test_unseen:.4f}")
    print("zero share per fold:", [round((y[v] == 0).mean(), 4) for _, v in folds])
