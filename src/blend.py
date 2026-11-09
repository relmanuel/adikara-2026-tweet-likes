"""Phase E: blend OOF predictions with non-negative weights summing to 1.

Weights are evaluated with a second-level CV (fit weights on 4 outer folds' OOF, score the 5th)
to avoid optimism, then refit on all OOF and applied to the averaged test predictions.

python src/blend.py NAME1 NAME2 ...   (defaults: every model with both oof and preds files)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, EXP, ROOT, load
from cv import get_folds, log_row, rmse_log


def fit_weights(P, y):
    k = P.shape[1]
    obj = lambda w: np.mean((P @ w - y) ** 2)
    res = minimize(obj, np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return res.x


def main(names=None, out_name="blend"):
    tr, te = load()
    y = np.log1p(tr.likes.to_numpy())
    folds = get_folds(tr)
    if not names:
        names = sorted(p.stem for p in (EXP / "oof").glob("*.npy")
                       if (EXP / "preds" / p.name).exists() and not p.stem.startswith(("E0", "blend")))
    P = np.column_stack([np.load(EXP / "oof" / f"{n}.npy") for n in names])
    T = np.column_stack([np.load(EXP / "preds" / f"{n}.npy") for n in names])
    print("single-model OOF RMSE:")
    for n, c in zip(names, P.T):
        print(f"  {n:40s} {rmse_log(y, c):.5f}")
    print("OOF correlation:\n", pd.DataFrame(P, columns=names).corr().round(3).to_string())

    # second-level CV on the frozen outer folds
    scores, oof_b = [], np.zeros(len(y))
    for trn, val in folds:
        w = fit_weights(P[trn], y[trn])
        oof_b[val] = np.clip(P[val] @ w, 0, None)
        scores.append(rmse_log(y[val], oof_b[val]))
    w = fit_weights(P, y)
    print("weights:", {n: round(x, 3) for n, x in zip(names, w) if x > 1e-3})
    print(f"blend 2nd-level CV {np.mean(scores):.5f} ± {np.std(scores):.5f} | folds {np.round(scores, 5)}")

    pred_log = np.clip(T @ w, 0, y.max())
    np.save(EXP / "oof" / f"{out_name}.npy", oof_b)
    np.save(EXP / "preds" / f"{out_name}.npy", pred_log)
    log_row(name=out_name, features="+".join(names), params=str({n: round(x, 3) for n, x in zip(names, w)}),
            cv_mean=round(float(np.mean(scores)), 5), cv_std=round(float(np.std(scores)), 5),
            oof_rmse=round(rmse_log(y, oof_b), 5), notes="NNLS-simplex blend, weights CV'd on outer folds")

    # sanity checks + submission
    sub = pd.read_csv(DATA / "sample_submission.csv")
    assert len(pred_log) == len(sub) == 5969 and np.isfinite(pred_log).all()
    assert (te.id.to_numpy() == sub.id.to_numpy()).all(), "test ids out of order vs sample_submission"
    print(f"pred log mean {pred_log.mean():.3f} std {pred_log.std():.3f} | OOF-blend mean {oof_b.mean():.3f} "
          f"std {oof_b.std():.3f} | train y mean {y.mean():.3f} std {y.std():.3f}")
    sub[sub.columns[1]] = np.expm1(pred_log)
    (ROOT / "submission").mkdir(exist_ok=True)
    path = ROOT / "submission" / "submission.csv"
    sub.to_csv(path, index=False)
    print("wrote", path, sub.shape, "columns", list(sub.columns))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
