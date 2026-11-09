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
from features import (basic_features, load_embeddings, pca_frame, target_features,
                      text_model_oof, tfidf_matrices)

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


def text_extra(Mtr, Mte, y, prefix="txt", alpha=3.0, C=2.0):
    def f(trn, val):
        o, (dv, dt) = text_model_oof(Mtr[trn], y[trn], [Mtr[val], Mte], prefix=prefix, alpha=alpha, C=C)
        return o, dv, dt
    return f


def best_extra(tr, te, y, emb=("indobertweet", "e5")):
    """E3a stack (+ embedding stacks): the current best per-fold feature set."""
    Mtr, Mte, _, _ = get_tfidf(tr, te)
    fns = [user_te_extra(tr, te, y, 20), text_extra(Mtr, Mte, y)]
    for e in emb:
        Etr, Ete = load_embeddings(e)
        fns.append(text_extra(Etr, Ete, y, prefix=e, alpha=300.0, C=0.05))
    return combine(*fns)


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

    if "E4" in todo:
        for emb in (("indobertweet",), ("e5",), ("indobertweet", "e5")):
            r = run_lgb("E4_lgb_stack_" + "+".join(emb), Xtr, Xte, y, folds, fold_extra=best_extra(tr, te, y, emb),
                        features=f"E3a + OOF ridge/logistic on {'+'.join(emb)} embeddings (nested)")
            print(r["importance"].head(12).round(0).to_string())
    if "E4pca" in todo:
        Ptr, Pte = zip(*[pca_frame(*load_embeddings(e), 32, e) for e in ("indobertweet", "e5")])
        r = run_lgb("E4_lgb_stack_both_pca32", Xtr.join(pd.concat(Ptr, axis=1)), Xte.join(pd.concat(Pte, axis=1)),
                    y, folds, fold_extra=best_extra(tr, te, y), features="E4 both + 32 PCA dims per embedding")


# ---------------------------------------------------------------------------
# Cached per-fold stack features + generic CV loop for other learners
def cached_extras(key, extra_fn, folds):
    import pickle
    p = FEAT_CACHE / f"extras_{key}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    out = []
    for k, (trn, val) in enumerate(folds):
        out.append(extra_fn(trn, val))
        print(f"  extras fold {k} done", flush=True)
    p.write_bytes(pickle.dumps(out))
    return out


def cv_loop(name, Xtr, Xte, y, folds, extras, fit_fn, features="", params="", seed=SEED, notes=""):
    """fit_fn(A, ya, B, yb, C, seed) -> (pred_B, pred_C)."""
    oof, pt = np.zeros(len(y)), np.zeros(len(Xte))
    t0 = time.time()
    for k, (trn, val) in enumerate(folds):
        A, B, C = Xtr.iloc[trn].reset_index(drop=True), Xtr.iloc[val].reset_index(drop=True), Xte.reset_index(drop=True)
        if extras is not None:
            ea, eb, ec = extras[k]
            A, B, C = A.join(ea), B.join(eb), C.join(ec)
        pb, pc = fit_fn(A, y[trn], B, y[val], C, seed)
        oof[val], pt = pb, pt + pc / len(folds)
    return evaluate(name, y, oof, folds, pt, features=features, params=params, seed=seed,
                    runtime=time.time() - t0, notes=notes)


def fit_cat(cat_cols, depth=6, lr=0.05):
    from catboost import CatBoostRegressor, Pool

    def f(A, ya, B, yb, C, seed):
        def prep(D):
            D = D.copy()
            for c in cat_cols:
                D[c] = D[c].astype(str)
            return D
        A, B, C = prep(A), prep(B), prep(C)
        m = CatBoostRegressor(loss_function="RMSE", learning_rate=lr, depth=depth, iterations=5000,
                              od_type="Iter", od_wait=200, random_seed=seed, verbose=0, thread_count=-1)
        m.fit(Pool(A, ya, cat_features=cat_cols), eval_set=Pool(B, yb, cat_features=cat_cols), use_best_model=True)
        print(f"  cat iters {m.get_best_iteration()}", flush=True)
        return m.predict(B), m.predict(C)
    return f


def fit_xgb(A, ya, B, yb, C, seed):
    import xgboost as xgb
    m = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", learning_rate=0.03, max_depth=6,
                         min_child_weight=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                         n_estimators=5000, early_stopping_rounds=200, enable_categorical=True,
                         random_state=seed, n_jobs=-1)
    m.fit(A, ya, eval_set=[(B, yb)], verbose=False)
    print(f"  xgb iters {m.best_iteration}", flush=True)
    return m.predict(B), m.predict(C)


def fit_hurdle(A, ya, B, yb, C, seed):
    """E[y] = P(nonzero) * E[y | nonzero]; classifier + regressor on non-zero rows."""
    p = {**LGB_PARAMS, "seed": seed}
    za, zb = (ya > 0).astype(int), (yb > 0).astype(int)
    clf = lgb.train({**p, "objective": "binary"}, lgb.Dataset(A, za), 5000, valid_sets=[lgb.Dataset(B, zb)],
                    callbacks=[lgb.early_stopping(200, verbose=False)])
    na, nb = ya > 0, yb > 0
    reg = lgb.train(p, lgb.Dataset(A[na], ya[na]), 5000, valid_sets=[lgb.Dataset(B[nb], yb[nb])],
                    callbacks=[lgb.early_stopping(200, verbose=False)])
    print(f"  clf iters {clf.best_iteration} reg iters {reg.best_iteration}", flush=True)
    pr = lambda D: clf.predict(D, num_iteration=clf.best_iteration) * reg.predict(D, num_iteration=reg.best_iteration)
    return pr(B), pr(C)


if __name__ == "__main__" and any(t in sys.argv for t in ("E5", "E5u", "E6", "E9h")):
    ex = cached_extras("stack_ibt", best_extra(tr, te, y, ("indobertweet",)), folds)
    feat = "E4 (indobertweet stack) feature set"
    if "E9h" in todo:
        cv_loop("E9_lgb_hurdle", Xtr, Xte, y, folds, ex, fit_hurdle, features=feat,
                notes="LGB binary P(nonzero) x LGB regression on nonzero rows")
    if "E6" in todo:
        cv_loop("E6_xgb", Xtr, Xte, y, folds, ex, fit_xgb, features=feat,
                params="hist lr0.03 depth6 mcw5 sub0.8 col0.7")
    cats = ["media_type", "client_device", "date_fmt", "acct_type"]
    if "E5" in todo:
        cv_loop("E5_cat", Xtr, Xte, y, folds, ex, fit_cat(cats), features=feat, params="RMSE lr0.05 depth6")
    if "E5u" in todo:
        # variant: CatBoost ordered target stats on raw username instead of only manual TE
        Xu, Xue = Xtr.assign(username=tr.username.values), Xte.assign(username=te.username.values)
        cv_loop("E5_cat_username", Xu, Xue, y, folds, ex, fit_cat(cats + ["username"]), features=feat + " + username cat",
                params="RMSE lr0.05 depth6")
