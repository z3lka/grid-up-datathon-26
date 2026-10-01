# Grid Up Datathon

Reproducible solution for forecasting transformer-level daily electricity
consumption from April through July 2026. The official metric is RMSLE, so all
averaging and blending is performed in `log1p(tuketim)` space.

## Approach

- Established transformers: blend a short recent log-consumption level with
  the transformer's historical day-of-week log level, then add a shrunken
  location-level summer curve learned from complete-history transformers.
- Transformers that return after a long absence: shrink stale entity history
  toward the installed-power prior instead of treating the asset as
  continuously active.
- Transformers absent from train: use the installed-power log prior.
- Convert with `expm1`, enforce non-negative predictions, and preserve the
  exact sample-submission row order.

This intentionally uses only the supplied competition data. It does not use
transformer identifier digits as model features or rely on unapproved external
weather data.

## Current result (2026-08-31)

| Version | Public RMSLE | Kaggle ref |
| --- | ---: | ---: |
| Recent 3 days + weekday, power-age cold start | 1.26688 | 55669847 |
| Recent 7 days + weekday, power-age cold start | 1.25594 | 55669854 |
| Recent 7 days + weekday + seasonal curve, power cold start | 1.24233 | 55670171 |
| Same model + return-gap correction | **1.10604** | **55915151** |

The return-gap correction reduced public RMSLE by 0.13629 without changing
continuously active or cold-start rows. Kaggle allows two final selections.

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/kaggle competitions download -c grid-up-datathon -p data/raw
unzip data/raw/grid-up-datathon.zip -d data/raw

.venv/bin/python src/make_submission.py \
  --recent-days 3 --recent-weight 0.8 \
  --output submissions/recent3_w080.csv
```

The season-aware variant adds location-level April-July movements estimated
from transformers with a complete prior calendar year:

```bash
.venv/bin/python src/make_submission.py \
  --recent-days 7 --recent-weight 0.7 \
  --seasonal-weight 1.0 --cold-start power \
  --output submissions/recent7_seasonal_return_gap.csv
```

An experimental CatBoost specializes only in historical disappear/reappear
events and preserves the scored baseline everywhere else:

```bash
.venv/bin/python src/make_return_model_submission.py \
  --output submissions/return_event_catboost.csv
```

The private Kaggle GPU rolling-origin experiment is reproducible from
`kaggle_gpu/`; its general model was rejected in favor of the targeted return
model.

The raw competition files and generated submissions are ignored by Git.
