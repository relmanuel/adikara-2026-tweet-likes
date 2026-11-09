"""Shared loading / parsing utilities."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXP = ROOT / "experiments"
SEED = 42


def parse_post_date(s: pd.Series) -> pd.Series:
    """Parse the three observed formats into tz-naive UTC timestamps.

    - '2020-01-02 12:35:12+00:00'  (ISO with offset)
    - '06/01/2020 07:55:29'        (DD/MM/YYYY, assumed UTC; day<=31, month<=12 verified)
    - '1580000000'                 (unix epoch seconds)
    """
    s = s.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    iso = s.str.match(r"^\d{4}-\d{2}-\d{2}")
    dmy = s.str.match(r"^\d{1,2}/\d{1,2}/\d{4}")
    epo = s.str.match(r"^\d{9,11}$")
    out[iso] = pd.to_datetime(s[iso], utc=True).dt.tz_localize(None).astype("datetime64[ns]")
    out[dmy] = pd.to_datetime(s[dmy], format="%d/%m/%Y %H:%M:%S").astype("datetime64[ns]")
    out[epo] = pd.to_datetime(s[epo].astype("int64"), unit="s").astype("datetime64[ns]")
    return out


def date_format(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    return np.select(
        [s.str.match(r"^\d{4}-"), s.str.match(r"^\d{1,2}/"), s.str.match(r"^\d{9,11}$")],
        ["iso", "dmy", "epoch"], "other")


def clean_likes(s: pd.Series) -> pd.Series:
    """'140 pts' / '128 score' / '0 likes' / '156.0' -> float. Negatives (-1, -99) are sentinels -> NaN."""
    v = pd.to_numeric(s.astype(str).str.strip().str.replace(r"\s*(pts|score|likes)$", "", regex=True),
                      errors="coerce")
    return v.where(v >= 0)


MEDIA_FIX = {"foto": "photo", "vidio": "video", "carousell": "carousel"}
DEVICE_FIX = {"Andorid": "Android", "Web-App": "Web App"}


def clean_cat(s: pd.Series, fix=None) -> pd.Series:
    s = s.where(s.isna(), s.astype(str).str.strip())
    if fix:
        s = s.replace(fix)
    return s.fillna("missing")


def clean_username(s: pd.Series) -> pd.Series:
    """Brand handles differ only by case (@AuraMobileID / @auramobileid / @AURAMOBILEID) -> lowercase them.
    Hashed (64-hex) usernames are left untouched."""
    s = s.astype(str).str.strip()
    return s.where(~s.str.startswith("@"), s.str.lower())


def load(drop_bad_target=True):
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    tr["likes_raw"] = tr["likes"].astype(str)
    tr["likes"] = clean_likes(tr["likes"])
    if drop_bad_target:
        tr = tr[tr["likes"].notna()].reset_index(drop=True)
    for d in (tr, te):
        d["date_fmt"] = date_format(d["post_date"])
        d["dt"] = parse_post_date(d["post_date"])
        d.loc[d["dt"] < "2000-01-01", "dt"] = pd.NaT  # epoch-0 placeholders
        d["text"] = d["text"].fillna("").astype(str)
        d["username_raw"] = d["username"]
        d["username"] = clean_username(d["username"])
        d["media_type"] = clean_cat(d["media_type"], MEDIA_FIX)
        d["client_device"] = clean_cat(d["client_device"], DEVICE_FIX)
    return tr, te
