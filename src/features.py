"""Phase C features.

basic_features(tr, te)   -> unsupervised features for train+test (never touch likes)
target_features(...)     -> account target features computed out-of-fold (see cv.py leakage rule)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b[\w-]+\.(?:com|id|co|net|org|ly|me)(?:/\S*)?)", re.I)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️]")
ELONG_RE = re.compile(r"(\w)\1{2,}")
TAG_RE = re.compile(r"^\s*[/\[(]\s*[\w-]+\s*[/\])]")  # "/nctzh/", "[oned]"-style base/fess tags
CATS = ["media_type", "client_device", "date_fmt", "acct_type"]


def _text_stats(t: pd.Series) -> pd.DataFrame:
    t = t.astype(object)  # python `re` engine (pyarrow RE2 lacks backreferences)
    n = t.str.len()
    letters = t.str.count(r"[A-Za-z]").clip(lower=1)
    return pd.DataFrame({
        "len_chars": n,
        "len_words": t.str.split().str.len().fillna(0),
        "n_hashtags": t.str.count(HASHTAG_RE),
        "n_mentions": t.str.count(MENTION_RE),
        "n_urls": t.str.count(URL_RE),
        "n_emojis": t.str.count(EMOJI_RE),
        "upper_ratio": t.str.count(r"[A-Z]") / letters,
        "digit_ratio": t.str.count(r"\d") / n.clip(lower=1),
        "n_punct": t.str.count(r"[^\w\s]"),
        "n_question": t.str.count(r"\?"),
        "n_exclam": t.str.count(r"!"),
        "n_elong": t.str.count(ELONG_RE),
        "is_rt": t.str.match(r"^\s*RT\b").astype(int),
        "starts_mention": t.str.match(r"^\s*@").astype(int),
        "has_base_tag": t.str.match(TAG_RE).astype(int),
        "is_short": (n < 20).astype(int),
        "has_ellipsis_link": t.str.contains("…").astype(int),
    })


def basic_features(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    al = pd.concat([tr.drop(columns=["likes", "likes_raw"], errors="ignore"), te], ignore_index=True)
    dt = al["dt"]
    f = pd.DataFrame(index=al.index)

    # Categories --------------------------------------------------------------
    f["media_type"] = al.media_type
    f["client_device"] = al.client_device
    f["date_fmt"] = al.date_fmt
    f["acct_type"] = np.where(al.username.str.startswith("@"), al.username, "hashed")  # brand id or 'hashed'
    f["is_brand"] = al.username.str.startswith("@").astype(int)

    # Datetime ----------------------------------------------------------------
    f["year"] = dt.dt.year
    f["month"] = dt.dt.month
    f["day"] = dt.dt.day
    f["hour"] = dt.dt.hour
    f["minute"] = dt.dt.minute
    f["weekday"] = dt.dt.weekday
    f["is_weekend"] = (dt.dt.weekday >= 5).astype(float).where(dt.notna())
    f["days"] = (dt - dt.min()).dt.total_seconds() / 86400
    f["hour_of_week"] = dt.dt.weekday * 24 + dt.dt.hour
    f["sec_of_day"] = dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second
    f["is_round_time"] = ((dt.dt.second == 0) & (dt.dt.minute % 5 == 0)).astype(float).where(dt.notna())  # scheduled posts

    # Text stats --------------------------------------------------------------
    f = f.join(_text_stats(al.text))

    # Account (unsupervised; train+test together) -----------------------------
    u = al.username
    g = al.groupby("username")
    f["acct_n"] = u.map(u.value_counts())
    first, last = g["dt"].transform("min"), g["dt"].transform("max")
    span = (last - first).dt.total_seconds() / 86400
    f["acct_span_days"] = span
    f["acct_freq"] = f["acct_n"] / (span + 1)
    f["acct_rank"] = g["dt"].rank(method="first") / f["acct_n"]
    f["acct_days_since_first"] = (dt - first).dt.total_seconds() / 86400
    srt = al.assign(_i=al.index).sort_values(["username", "dt"])
    prev_gap = srt.groupby("username")["dt"].diff().dt.total_seconds() / 3600
    next_gap = -srt.groupby("username")["dt"].diff(-1).dt.total_seconds() / 3600
    f["gap_prev_h"] = prev_gap.reindex(al.index)
    f["gap_next_h"] = next_gap.reindex(al.index)
    for c, vals in [("media_type", ["text_only", "photo", "video", "carousel"]),
                    ("client_device", ["Android", "iOS", "Web App", "Lite App"])]:
        for v in vals:
            f[f"acct_share_{c}_{v}".replace(" ", "_")] = (al[c] == v).groupby(u).transform("mean")
    norm_text = al.text.str.lower().str.replace(URL_RE, "<url>", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    f["text_dup_global"] = norm_text.map(norm_text.value_counts())
    f["acct_text_dup_rate"] = 1 - norm_text.groupby(u).transform("nunique") / f["acct_n"]
    f["acct_mean_len"] = f.groupby(u)["len_chars"].transform("mean")
    f["acct_mean_hashtags"] = f.groupby(u)["n_hashtags"].transform("mean")
    # posts by same account on same day (campaign bursts)
    day = dt.dt.floor("D")
    f["acct_posts_same_day"] = al.assign(_d=day).groupby(["username", "_d"])["id"].transform("count")
    f["global_posts_same_day"] = day.map(day.value_counts())

    for c in CATS:
        f[c] = f[c].astype("category")
    n = len(tr)
    return f.iloc[:n].reset_index(drop=True), f.iloc[n:].reset_index(drop=True)


# Account target features -------------------------------------------------------
def _agg_stats(keys_src, y_src, keys_dst, m, prefix, global_mean):
    d = pd.DataFrame({"k": keys_src.to_numpy(), "y": y_src, "z": (y_src == 0).astype(float)})
    g = d.groupby("k").agg(n=("y", "size"), mean=("y", "mean"), std=("y", "std"), mx=("y", "max"),
                           med=("y", "median"), zr=("z", "mean"),
                           nzmean=("y", lambda s: s[s > 0].mean() if (s > 0).any() else np.nan))
    kd = pd.Series(keys_dst.to_numpy())
    n = kd.map(g["n"]).fillna(0).to_numpy()
    out = pd.DataFrame({
        f"{prefix}_n": n,
        f"{prefix}_mean_s": (kd.map(g["mean"]).fillna(0).to_numpy() * n + m * global_mean) / (n + m),
        f"{prefix}_zero_s": (kd.map(g["zr"]).fillna(0).to_numpy() * n + m * (y_src == 0).mean()) / (n + m),
        f"{prefix}_mean": kd.map(g["mean"]).to_numpy(),
        f"{prefix}_std": kd.map(g["std"]).to_numpy(),
        f"{prefix}_max": kd.map(g["mx"]).to_numpy(),
        f"{prefix}_median": kd.map(g["med"]).to_numpy(),
        f"{prefix}_nzmean": kd.map(g["nzmean"]).to_numpy(),
    })
    return out


def target_features(keys_trn, y_trn, keys_dst, m=5, inner_folds=5, seed=42, prefix="te_user", dst_is_trn=False):
    """Target stats of `keys` (e.g. username) computed from (keys_trn, y_trn).

    dst_is_trn=True: rows are the training part itself -> inner K-fold so a row never sees its own label.
    Otherwise (validation / test rows): stats from all of keys_trn.
    """
    gm = float(np.mean(y_trn))
    if not dst_is_trn:
        return _agg_stats(keys_trn, y_trn, keys_dst, m, prefix, gm)
    out = None
    rng = np.random.RandomState(seed)
    fold = rng.randint(0, inner_folds, len(y_trn))
    for k in range(inner_folds):
        a, b = fold != k, fold == k
        part = _agg_stats(keys_trn[a], y_trn[a], keys_trn[b], m, prefix, float(np.mean(y_trn[a])))
        if out is None:
            out = pd.DataFrame(index=range(len(y_trn)), columns=part.columns, dtype=float)
        out.loc[np.where(b)[0]] = part.to_numpy()
    return out.astype(float)


# Text TF-IDF ------------------------------------------------------------------
def norm_for_tfidf(t: pd.Series) -> pd.Series:
    t = t.astype(object).str.lower()
    t = t.str.replace(URL_RE, " <url> ", regex=True).str.replace(MENTION_RE, " <user> ", regex=True)
    return t.str.replace(r"\s+", " ", regex=True).str.strip()


def tfidf_matrices(tr: pd.DataFrame, te: pd.DataFrame, svd_dims=64, seed=42):
    """Vocabulary / idf / SVD fit on train+test text only (unsupervised; never sees likes)."""
    from scipy import sparse
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    txt = pd.concat([norm_for_tfidf(tr.text), norm_for_tfidf(te.text)], ignore_index=True)
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True,
                        token_pattern=r"(?u)<\w+>|#?\b\w+\b")
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300_000, sublinear_tf=True)
    M = sparse.hstack([w.fit_transform(txt), c.fit_transform(txt)]).tocsr()
    S = TruncatedSVD(svd_dims, random_state=seed).fit_transform(M)
    svd = pd.DataFrame(S, columns=[f"svd{i}" for i in range(svd_dims)])
    n = len(tr)
    return M[:n], M[n:], svd.iloc[:n].reset_index(drop=True), svd.iloc[n:].reset_index(drop=True)


def text_model_oof(M_trn, y_trn, M_dsts, inner_folds=5, seed=42, alpha=3.0, prefix="txt", C=2.0):
    """Ridge (log1p likes) + logistic (is zero) on TF-IDF.
    Returns inner-OOF preds for the training rows and full-fit preds for each M_dst."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import KFold

    def fit_pred(Ma, ya, Mbs):
        r = Ridge(alpha=alpha).fit(Ma, ya)
        lr = LogisticRegression(C=C, max_iter=3000).fit(Ma, (ya == 0).astype(int))
        return [np.c_[r.predict(Mb), lr.predict_proba(Mb)[:, 1]] for Mb in Mbs]

    oof = np.zeros((M_trn.shape[0], 2))
    for a, b in KFold(inner_folds, shuffle=True, random_state=seed).split(y_trn):
        oof[b] = fit_pred(M_trn[a], y_trn[a], [M_trn[b]])[0]
    dst = fit_pred(M_trn, y_trn, M_dsts)
    cols = [f"{prefix}_ridge", f"{prefix}_pzero"]
    return pd.DataFrame(oof, columns=cols), [pd.DataFrame(d, columns=cols) for d in dst]


def load_embeddings(name):
    """Standardised (scaler fit on train+test, unsupervised) embeddings from features/emb/."""
    from common import ROOT
    a = np.load(ROOT / "features" / "emb" / f"{name}_train.npy").astype(np.float32)
    b = np.load(ROOT / "features" / "emb" / f"{name}_test.npy").astype(np.float32)
    allx = np.vstack([a, b])
    mu, sd = allx.mean(0), allx.std(0) + 1e-6
    return (a - mu) / sd, (b - mu) / sd


def pca_frame(Etr, Ete, dims, prefix, seed=42):
    from sklearn.decomposition import PCA
    P = PCA(dims, random_state=seed).fit(np.vstack([Etr, Ete]))
    cols = [f"{prefix}_pc{i}" for i in range(dims)]
    return pd.DataFrame(P.transform(Etr), columns=cols), pd.DataFrame(P.transform(Ete), columns=cols)
