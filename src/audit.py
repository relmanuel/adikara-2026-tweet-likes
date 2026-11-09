"""Phase A audit. Prints a report and saves it to experiments/audit/."""
import sys
import numpy as np
import pandas as pd
from scipy.stats import skew, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import load, EXP, SEED

OUT = EXP / "audit"
OUT.mkdir(parents=True, exist_ok=True)
lines = []


def p(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    lines.append(msg)


tr, te = load()
y = np.log1p(tr["likes"])

# 1. Basic structure ---------------------------------------------------------
p("=" * 70, "\n1. STRUCTURE")
p("train", tr.shape, "test", te.shape)
p("dtypes:\n" + tr.dtypes.to_string())
p("nulls train:\n" + tr.isna().sum().to_string())
p("nulls test:\n" + te.isna().sum().to_string())
p("dup ids train:", tr.id.duplicated().sum(), "test:", te.id.duplicated().sum(),
  "overlap:", len(set(tr.id) & set(te.id)))
p("id range train", tr.id.min(), tr.id.max(), "test", te.id.min(), te.id.max())
p("date formats train", tr.date_fmt.value_counts().to_dict(), "test", te.date_fmt.value_counts().to_dict())
p("unparsed dates train", tr.dt.isna().sum(), "test", te.dt.isna().sum())
for c in ["media_type", "client_device"]:
    p(f"{c} train", tr[c].value_counts(dropna=False).to_dict())
    p(f"{c} test ", te[c].value_counts(dropna=False).to_dict())

norm = lambda s: s.fillna("").str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
p("dup text within train exact:", tr.text.duplicated().sum(), " normalised:", norm(tr.text).duplicated().sum())
p("dup text within test  exact:", te.text.duplicated().sum(), " normalised:", norm(te.text).duplicated().sum())
p("test texts found in train exact:", te.text.isin(set(tr.text)).sum(),
  " normalised:", norm(te.text).isin(set(norm(tr.text))).sum())
# Do duplicated texts share likes?
g = tr.assign(nt=norm(tr.text), y=y).groupby("nt")["y"]
dup = g.filter(lambda s: len(s) > 1)
if len(dup):
    within_std = tr.assign(nt=norm(tr.text), y=y)[norm(tr.text).duplicated(keep=False)].groupby("nt")["y"].std().mean()
    p(f"mean within-duplicate-group std of log1p(likes): {within_std:.3f} (vs global std {y.std():.3f})")

# 2. Target ------------------------------------------------------------------
p("=" * 70, "\n2. TARGET")
lk = tr.likes
p("likes min", lk.min(), "q50", lk.quantile(.5), "q90", lk.quantile(.9), "q99", lk.quantile(.99),
  "max", lk.max(), "mean", round(lk.mean(), 2))
p("share zeros", round((lk == 0).mean(), 4), "negative", (lk < 0).sum(), "non-integer", (lk % 1 != 0).sum(),
  "skew", round(skew(lk), 2))
p("log1p: mean", round(y.mean(), 4), "std", round(y.std(ddof=0), 4), "q50", round(y.quantile(.5), 3),
  "q90", round(y.quantile(.9), 3), "q99", round(y.quantile(.99), 3), "max", round(y.max(), 3), "skew", round(skew(y), 3))
p(f"BASELINE RMSLE (constant mean of log1p) = {y.std(ddof=0):.5f}  target to beat = 1.33768 "
  f"-> need {100*(1-1.33768/y.std(ddof=0)):.1f}% reduction")
p("log1p(likes) histogram (bin: count):")
cnt, edges = np.histogram(y, bins=20)
for c_, e in zip(cnt, edges):
    p(f"  {e:6.2f}: {c_}")

# 3. Username ----------------------------------------------------------------
p("=" * 70, "\n3. USERNAME")
is_brand = lambda s: s.str.startswith("@")
is_hash = lambda s: s.str.fullmatch(r"[0-9a-f]{64}")
for nm, d in [("train", tr), ("test", te)]:
    p(f"{nm}: unique users {d.username.nunique()}, brand(@) rows {is_brand(d.username).sum()}, "
      f"hashed rows {is_hash(d.username).sum()}, other rows {(~is_brand(d.username) & ~is_hash(d.username)).sum()}")
other = tr.username[~is_brand(tr.username) & ~is_hash(tr.username)]
if len(other):
    p("examples of other usernames:", other.unique()[:10].tolist())
seen = te.username.isin(set(tr.username))
p(f"test rows with username in train: {seen.mean():.4f}")
p(f"test usernames unseen in train: {(~pd.Series(te.username.unique()).isin(set(tr.username))).mean():.4f}")
vc = tr.username.value_counts()
p("rows per user (train) quantiles:", vc.quantile([.5, .75, .9, .99, 1]).to_dict(), "singletons", (vc == 1).sum())
p("top 15 users by rows (train):\n" + vc.head(15).to_string())
# Test rows by train-frequency of their user
f = te.username.map(vc).fillna(0)
p("test rows by user's train count: 0:", (f == 0).sum(), " 1:", (f == 1).sum(), " 2-4:", ((f >= 2) & (f < 5)).sum(),
  " 5-19:", ((f >= 5) & (f < 20)).sum(), " 20+:", (f >= 20).sum())
ft = tr.username.map(vc)
p("train rows by user's train count: 1:", (ft == 1).sum(), " 2-4:", ((ft >= 2) & (ft < 5)).sum(),
  " 5-19:", ((ft >= 5) & (ft < 20)).sum(), " 20+:", (ft >= 20).sum())
# Variance explained by user
tmp = tr.assign(y=y)
um = tmp.groupby("username")["y"].transform("mean")
p(f"in-sample R2 of user mean (upper bound, overfit): {1 - ((y - um)**2).mean() / y.var(ddof=0):.4f}")
loo_n = tmp.groupby("username")["y"].transform("count")
loo = (tmp.groupby("username")["y"].transform("sum") - y) / (loo_n - 1)
mask = loo_n > 1
p(f"leave-one-out user mean RMSE on rows with >1 post: {np.sqrt(((y[mask]-loo[mask])**2).mean()):.4f} "
  f"(global-mean RMSE on same rows {np.sqrt(((y[mask]-y.mean())**2).mean()):.4f})")
p("log1p(likes) by brand vs hashed:\n" + tmp.groupby(is_brand(tr.username))["y"].describe().to_string())

# 4. Dates -------------------------------------------------------------------
p("=" * 70, "\n4. POST_DATE")
p("train", tr.dt.min(), "->", tr.dt.max())
p("test ", te.dt.min(), "->", te.dt.max())
p("test rows after train max:", (te.dt > tr.dt.max()).sum(), " before train min:", (te.dt < tr.dt.min()).sum())
bt = pd.concat([tr.dt.dt.to_period("M").value_counts().rename("train"),
                te.dt.dt.to_period("M").value_counts().rename("test")], axis=1).fillna(0).sort_index()
bt["test_share"] = bt.test / (bt.train + bt.test)
p("rows per month:\n" + bt.to_string())
fig, ax = plt.subplots(figsize=(12, 4))
bt[["train", "test"]].plot(ax=ax)
ax.set_title("Rows per month")
fig.tight_layout(); fig.savefig(OUT / "rows_per_month.png", dpi=100)
p("mean log1p(likes) by year:\n" + tmp.groupby(tr.dt.dt.year)["y"].agg(["mean", "std", "count"]).to_string())
p("mean log1p(likes) by date format:\n" + tmp.groupby(tr.date_fmt)["y"].agg(["mean", "std", "count"]).to_string())

# 6. id ordering (before adversarial so ordering checks are explicit) ---------
p("=" * 70, "\n6. ID ORDERING / LEAK CHECK")
ts = lambda d: d.dt.astype("int64").where(d.dt.notna())
ok_tr, ok_te = tr.dt.notna(), te.dt.notna()
p("spearman(id, date) train:", round(spearmanr(tr.id[ok_tr], ts(tr)[ok_tr]).statistic, 4),
  " test:", round(spearmanr(te.id[ok_te], ts(te)[ok_te]).statistic, 4))
p("spearman(id, log1p likes) train:", round(spearmanr(tr.id, y).statistic, 4))
p("spearman(date, log1p likes):", round(spearmanr(ts(tr)[ok_tr], y[ok_tr]).statistic, 4))
# Partial check: within each month, does id still correlate with likes?
mon = tr.dt.dt.to_period("M")
within = tr.assign(y=y, m=mon)[ok_tr].groupby("m").apply(
    lambda g: spearmanr(g.id, g.y).statistic if len(g) > 30 else np.nan, include_groups=False)
p(f"within-month spearman(id, likes): mean {within.mean():.4f}, |mean| of months {within.abs().mean():.4f}")
both = pd.concat([tr[["id", "dt"]].assign(s="train"), te[["id", "dt"]].assign(s="test")]).dropna().sort_values("id")
p("combined sorted by id is date-monotone fraction:", round((both.dt.diff().dropna() >= pd.Timedelta(0)).mean(), 4))
# Is train/test split interleaved in date order?
both_d = both.sort_values("dt")
p("test share by date decile:", both_d.groupby(pd.qcut(np.arange(len(both_d)), 10, labels=False))["s"]
  .apply(lambda s: round((s == "test").mean(), 3)).tolist())

# 5. Adversarial validation -------------------------------------------------
p("=" * 70, "\n5. ADVERSARIAL VALIDATION (no text content, no id)")
al = pd.concat([tr.drop(columns="likes").assign(is_test=0), te.assign(is_test=1)], ignore_index=True)
X = pd.DataFrame({
    "media_type": al.media_type.astype("category"),
    "client_device": al.client_device.astype("category"),
    "date_fmt": pd.Categorical(al.date_fmt),
    "ts": al.dt.astype("int64") // 10**9,
    "hour": al.dt.dt.hour, "weekday": al.dt.dt.weekday,
    "is_brand": al.username.str.startswith("@").astype(int),
    "user_count": al.username.map(al.username.value_counts()),
    "text_len": al.text.fillna("").str.len(),
})
yy = al.is_test.values
oof = np.zeros(len(al)); imp = np.zeros(X.shape[1])
for trn, val in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, yy):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=SEED)
    m.fit(X.iloc[trn], yy[trn])
    oof[val] = m.predict_proba(X.iloc[val])[:, 1]
    imp += m.booster_.feature_importance("gain")
p(f"adversarial AUC: {roc_auc_score(yy, oof):.4f}")
p("feature gain:\n" + pd.Series(imp / 5, X.columns).sort_values(ascending=False).round(0).to_string())

(OUT / "audit_report.txt").write_text("\n".join(lines))
print("\nsaved", OUT / "audit_report.txt")
