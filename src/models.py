"""Phase D model runners. Each experiment logs one row to experiments/log.csv.

python src/models.py E1 E2 ...
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, SEED, load
from cv import evaluate, get_folds
from features import basic_features, target_features, text_model_oof, tfidf_matrices

LGB_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_child_samples=20,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, n_jobs=-1)
FEAT_CACHE = ROOT / "cache"


def get_data():
    tr, te = load()
    FEAT_CACHE.mkdir(exist_ok=True)
    p_tr, p_te = FEAT_CACHE / "basic_tr.pkl", FEAT_CACHE / "basic_te.pkl"
    if p_tr.exists():
        return tr, te, pd.read_pickle(p_tr), pd.read_pickle(p_te)
    Xtr, Xte = basic_features(tr, te)
    Xtr.to_pickle(p_tr); Xte.to_pickle(p_te)
    return tr, te, Xtr, Xte


def run_lgb(name, Xtr, Xte, y, folds, fold_extra=None, params=None, seed=SEED, notes="", features=""):
    """fold_extra(trn_idx, val_idx) -> (extra_trn, extra_val, extra_test) DataFrames added per fold."""
    p = {**LGB_PARAMS, **(params or {}), "seed": seed}
    oof, pt = np.zeros(len(y)), np.zeros(len(Xte))
    imp = None
    t0 = time.time()
    for k, (trn, val) in enumerate(folds):
        A, B, C = Xtr.iloc[trn].reset_index(drop=True), Xtr.iloc[val].reset_index(drop=True), Xte
        if fold_extra is not None:
            ea, eb, ec = fold_extra(trn, val)
            A, B, C = A.join(ea), B.join(eb), C.reset_index(drop=True).join(ec)
        dtr, dva = lgb.Dataset(A, y[trn]), lgb.Dataset(B, y[val])
        m = lgb.train(p, dtr, 5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[val] = m.predict(B, num_iteration=m.best_iteration)
        pt += m.predict(C, num_iteration=m.best_iteration) / len(folds)
        fi = pd.Series(m.feature_importance("gain"), A.columns)
        imp = fi if imp is None else imp + fi
        print(f"  fold {k}: iters {m.best_iteration}", flush=True)
    res = evaluate(name, y, oof, folds, pt, features=features or f"{Xtr.shape[1]} base cols",
                   params=str({k: v for k, v in p.items() if k not in ("verbose", "n_jobs")}),
                   seed=seed, runtime=time.time() - t0, notes=notes)
    res["importance"] = (imp / len(folds)).sort_values(ascending=False)
    res["oof"], res["test"] = oof, pt
    return res


def user_te_extra(tr, te, y, m=5):
    keys = tr.username.reset_index(drop=True)

    def f(trn, val):
        kt, yt = keys.iloc[trn].reset_index(drop=True), y[trn]
        return (target_features(kt, yt, kt, m=m, dst_is_trn=True),
                target_features(kt, yt, keys.iloc[val].reset_index(drop=True), m=m),
                target_features(kt, yt, te.username.reset_index(drop=True), m=m))
    return f


def get_tfidf(tr, te):
    import pickle
    p = FEAT_CACHE / "tfidf.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    out = tfidf_matrices(tr, te)
    p.write_bytes(pickle.dumps(out))
    return out


def combine(*fns):
    def f(trn, val):
        parts = [fn(trn, val) for fn in fns]
        return tuple(pd.concat([p[i] for p in parts], axis=1) for i in range(3))
    return f


def text_extra(Mtr, Mte, y):
    def f(trn, val):
        o, (dv, dt) = text_model_oof(Mtr[trn], y[trn], [Mtr[val], Mte])
        return o, dv, dt
    return f


if __name__ == "__main__":
    tr, te, Xtr, Xte = get_data()
    y = np.log1p(tr.likes.to_numpy())
    folds = get_folds(tr)
    todo = sys.argv[1:] or ["E1", "E2"]

    if "E1" in todo:
        r = run_lgb("E1_lgb_basic", Xtr, Xte, y, folds, features="basic+account unsupervised")
        print(r["importance"].head(20).round(0).to_string())
    if "E2" in todo:
        for m in (5, 20):
            r = run_lgb(f"E2_lgb_userTE_m{m}", Xtr, Xte, y, folds, fold_extra=user_te_extra(tr, te, y, m),
                        features="E1 + OOF username target stats (inner 5-fold)", notes=f"smoothing m={m}")
            print(r["importance"].head(20).round(0).to_string())
    if "E3" in todo:
        Mtr, Mte, Str, Ste = get_tfidf(tr, te)
        # standalone text model (blend component + sanity check)
        r = run_lgb("E3a_lgb_userTE_text", Xtr, Xte, y, folds,
                    fold_extra=combine(user_te_extra(tr, te, y, 20), text_extra(Mtr, Mte, y)),
                    features="E2(m20) + OOF TF-IDF ridge/logistic (nested)")
        print(r["importance"].head(15).round(0).to_string())
        r = run_lgb("E3b_lgb_userTE_text_svd", Xtr.join(Str), Xte.join(Ste), y, folds,
                    fold_extra=combine(user_te_extra(tr, te, y, 20), text_extra(Mtr, Mte, y)),
                    features="E3a + 64 SVD dims of TF-IDF")
        print(r["importance"].head(15).round(0).to_string())
