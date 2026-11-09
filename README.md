# ADIKARA 2026 Data Mining: tweet likes regression

Predict `likes` for `data/test.csv`; metric RMSLE (target < 1.33768).

## Layout
- `src/common.py` loading + cleaning (likes suffixes, 3 date formats, category typos, brand case)
- `src/audit.py` Phase A audit -> `experiments/audit/`
- `src/cv.py` frozen folds (`experiments/folds.json`), metrics, experiment logging
- `src/baselines.py` E0 baselines
- `experiments/log.csv` every experiment; `experiments/oof|preds/*.npy` log-space predictions

## Colab (GPU work)
```python
from google.colab import userdata
token = userdata.get("GH_TOKEN")  # Colab Secrets, never paste into a cell
!git clone https://{token}@github.com/relmanuel/adikara-2026-tweet-likes.git
%cd adikara-2026-tweet-likes
```
