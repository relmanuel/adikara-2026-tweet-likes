"""Text embeddings from public pretrained encoders (no fine-tuning, no labels used).

Usage:  python src/embed.py --model indobertweet   (or e5)
Writes features/emb/{model}_train.npy, {model}_test.npy (float16, row order = common.load())
and {model}_ids.json for alignment checks.

Models (public, ungated, verified on the Hub):
  indobertweet: https://huggingface.co/indolem/indobertweet-base-uncased  (Apache-2.0; Koto et al., EMNLP 2021)
  e5:           https://huggingface.co/intfloat/multilingual-e5-base       (MIT; Wang et al., 2024)
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, load

MODELS = {
    "indobertweet": "indolem/indobertweet-base-uncased",
    "e5": "intfloat/multilingual-e5-base",
}
OUT = ROOT / "features" / "emb"


def prep_indobertweet(t: str) -> str:
    """Preprocessing from the IndoBERTweet model card: lowercase, @USER, HTTPURL, emoji -> text."""
    import emoji
    t = t.lower()
    t = re.sub(r"@\w+", "@USER", t)
    t = re.sub(r"(https?://\S+|www\.\S+|\b[\w-]+\.(?:com|id|co|net|org|ly|me)(?:/\S*)?)", "HTTPURL", t)
    t = emoji.demojize(t)
    return re.sub(r"\s+", " ", t).strip()


def prep_e5(t: str) -> str:
    return "query: " + re.sub(r"\s+", " ", t).strip()  # E5 expects a prefix; "query:" for symmetric tasks


@torch.no_grad()
def encode(texts, name, batch_size=64, max_len=128):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODELS[name])
    model = AutoModel.from_pretrained(MODELS[name]).to(device).eval()
    if device == "cuda":
        model = model.half()
    print(f"{name}: {MODELS[name]} on {device}, {len(texts)} texts")
    order = np.argsort([len(t) for t in texts])  # length-sorted batches = much faster
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        idx = order[i:i + batch_size]
        enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state.float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        out[idx] = ((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy()  # mean pooling
        if (i // batch_size) % 50 == 0:
            print(f"  {i}/{len(texts)}  {time.time() - t0:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    a = ap.parse_args()

    tr, te = load()
    prep = prep_indobertweet if a.model == "indobertweet" else prep_e5
    texts = [prep(t) for t in list(tr.text) + list(te.text)]
    emb = encode(texts, a.model, a.batch_size)
    assert np.isfinite(emb).all()
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{a.model}_train.npy", emb[:len(tr)].astype(np.float16))
    np.save(OUT / f"{a.model}_test.npy", emb[len(tr):].astype(np.float16))
    (OUT / f"{a.model}_ids.json").write_text(json.dumps(
        {"model": MODELS[a.model], "train_ids": tr.id.tolist(), "test_ids": te.id.tolist()}))
    print("saved", emb.shape, "->", OUT)


if __name__ == "__main__":
    main()
