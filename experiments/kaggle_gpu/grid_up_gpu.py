import gc
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


DATA = next(
    path.parent
    for path in Path("/kaggle/input").rglob("train.csv")
    if (path.parent / "test.csv").exists()
    and (path.parent / "sample_submission.csv").exists()
)
OUT = Path("/kaggle/working")
HORIZON = 122
CAT_COLS = ["tanim", "guc_cat", "lokasyon", "id2", "id3", "month", "dow"]


def mapped(frame, keys, values):
    index = frame[keys[0]] if len(keys) == 1 else pd.MultiIndex.from_frame(frame[keys])
    return values.reindex(index).to_numpy(dtype=float)


def snapshot(data, cutoff, target_end=None):
    cutoff = pd.Timestamp(cutoff)
    end = min(cutoff + pd.Timedelta(days=HORIZON), data.tarih.max())
    if target_end is not None:
        end = min(end, pd.Timestamp(target_end))
    hist = data[data.tarih <= cutoff]
    future = data[(data.tarih > cutoff) & (data.tarih <= end)].copy()
    future["dow_num"] = future.tarih.dt.dayofweek.astype("int8")

    stats = hist.groupby("tanim", observed=True).agg(
        hist_mean=("y", "mean"),
        hist_std=("y", "std"),
        hist_zero=("is_zero", "mean"),
        hist_n=("y", "size"),
        hist_last=("tarih", "max"),
    )
    power_mean = hist.groupby("guc", observed=True).y.mean()
    overall = float(hist.y.mean())
    fallback = future.guc.map(power_mean).fillna(overall).to_numpy()

    x = pd.DataFrame(index=future.index)
    for col in ["hist_mean", "hist_std", "hist_zero", "hist_n"]:
        x[col] = future.tanim.map(stats[col]).astype("float32")
    x["gap"] = (cutoff - future.tanim.map(stats.hist_last)).dt.days.astype("float32")

    recent = {}
    for days in [3, 7, 14, 28, 56, 90]:
        mean = hist[hist.tarih > cutoff - pd.Timedelta(days=days)].groupby(
            "tanim", observed=True
        ).y.mean()
        recent[days] = future.tanim.map(mean).astype("float32")
        x[f"r{days}"] = recent[days]

    entity_fallback = x.hist_mean.fillna(pd.Series(fallback, index=x.index))
    r3 = recent[3].fillna(entity_fallback).to_numpy()
    entity_dow = hist.groupby(["tanim", "dow"], observed=True).y.mean()
    dow = mapped(future, ["tanim", "dow_num"], entity_dow)
    dow = np.where(np.isnan(dow), entity_fallback.to_numpy(), dow)
    known = x.hist_n.notna().to_numpy()
    x["base"] = (0.4 * r3 + 0.6 * dow + np.where(known, 0.0, 0.06)).astype(
        "float32"
    )
    x["known"] = known.astype("int8")

    first_future = future.groupby("tanim", observed=True).tarih.min()
    first = future.tanim.map(first_future)
    cold_age = (future.tarih - first).dt.days
    x["cold_age"] = np.where(known, -1, cold_age).astype("int16")
    x["cohort_n"] = np.where(
        known, 0, first.map(first_future.value_counts())
    ).astype("int16")
    x["horizon"] = (future.tarih - cutoff).dt.days.astype("int16")
    x["doy"] = future.tarih.dt.dayofyear.astype("int16")

    previous = future[["tanim", "tarih"]].copy()
    previous["tarih"] -= pd.Timedelta(days=365)
    lag_index = pd.MultiIndex.from_frame(previous)
    history_lookup = hist.set_index(["tanim", "tarih"]).y
    x["lag365"] = history_lookup.reindex(lag_index).to_numpy(dtype="float32")

    ids = future.tanim.fillna("missing").astype(str)
    x["tanim"] = ids
    x["guc_cat"] = future.guc.fillna(-1).astype(str)
    x["lokasyon"] = future.lokasyon.fillna("missing").astype(str)
    x["id2"] = ids.str[:2]
    x["id3"] = ids.str[:3]
    x["month"] = future.tarih.dt.month.astype(str)
    x["dow"] = future.dow_num.astype(str)
    for col in CAT_COLS:
        x[col] = x[col].fillna("missing")
    return x.reset_index(drop=True), future.y.to_numpy(dtype="float32"), future.reset_index(drop=True)


def rmsle_log(pred, truth, mask=None):
    if mask is not None:
        pred, truth = pred[mask], truth[mask]
    return float(np.sqrt(np.mean(np.square(pred - truth))))


def fit_model(x, y, iterations, eval_set=None):
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=iterations,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=10,
        random_seed=20260831,
        task_type="GPU",
        devices="0",
        verbose=100,
        allow_writing_files=False,
    )
    kwargs = {}
    if eval_set is not None:
        kwargs.update(eval_set=eval_set, early_stopping_rounds=150, use_best_model=True)
    model.fit(x, y, cat_features=CAT_COLS, **kwargs)
    return model


train = pd.read_csv(DATA / "train.csv", parse_dates=["tarih"], dtype={"tanim": "string"})
test = pd.read_csv(DATA / "test.csv", parse_dates=["tarih"], dtype={"tanim": "string"})
sample = pd.read_csv(DATA / "sample_submission.csv")
print("Using data from", DATA)
train["y"] = np.log1p(train.tuketim.clip(lower=0)).astype("float32")
train["is_zero"] = (train.tuketim == 0).astype("int8")
train["dow"] = train.tarih.dt.dayofweek.astype("int8")

# Clean rolling holdout: no target later than November enters CV training.
cv_parts = []
for cutoff in ["2025-05-31", "2025-07-31", "2025-09-30", "2025-10-31"]:
    x, y, _ = snapshot(train, cutoff, target_end="2025-11-30")
    if len(x):
        cv_parts.append((x, y))
cv_x = pd.concat([part[0] for part in cv_parts], ignore_index=True)
cv_y = np.concatenate([part[1] for part in cv_parts])
val_x, val_y, val_rows = snapshot(train, "2025-11-30")
print("CV shapes", cv_x.shape, val_x.shape)

cv_model = fit_model(cv_x, cv_y, 1400, eval_set=(val_x, val_y))
cv_pred = cv_model.predict(val_x)
base_pred = val_x.base.to_numpy()
known = val_x.known.to_numpy(dtype=bool)
print("CV direct", rmsle_log(cv_pred, val_y))
print("CV direct known/cold", rmsle_log(cv_pred, val_y, known), rmsle_log(cv_pred, val_y, ~known))
print("CV base", rmsle_log(base_pred, val_y))

blend_scores = {}
for alpha in [0.25, 0.5, 0.75, 1.0]:
    pred = alpha * cv_pred + (1 - alpha) * base_pred
    blend_scores[alpha] = rmsle_log(pred, val_y)
    print("CV blend", alpha, blend_scores[alpha])
best_alpha = min(blend_scores, key=blend_scores.get)
best_iterations = max(100, cv_model.get_best_iteration() + 1)
print("selected alpha/iterations", best_alpha, best_iterations)
del cv_model, cv_x, cv_y, val_x, val_y, val_rows, cv_parts
gc.collect()

# Final snapshots all have a complete 122-day target window.
final_parts = []
for cutoff in ["2025-05-31", "2025-07-31", "2025-09-30", "2025-11-30"]:
    x, y, _ = snapshot(train, cutoff)
    final_parts.append((x, y))
final_x = pd.concat([part[0] for part in final_parts], ignore_index=True)
final_y = np.concatenate([part[1] for part in final_parts])

combined = pd.concat(
    [train, test.assign(y=np.nan, is_zero=0, dow=test.tarih.dt.dayofweek.astype("int8"))],
    ignore_index=True,
)
test_x, _, test_rows = snapshot(combined, "2026-03-31")
print("Final shapes", final_x.shape, test_x.shape)
model = fit_model(final_x, final_y, best_iterations)
gpu_pred = np.maximum(model.predict(test_x), 0)
blend_pred = best_alpha * gpu_pred + (1 - best_alpha) * test_x.base.to_numpy()

if not test_rows.id.equals(sample.id):
    raise ValueError("test row order mismatch")
pd.DataFrame({"id": test_rows.id, "tuketim": np.expm1(gpu_pred)}).to_csv(
    OUT / "gpu_direct.csv", index=False
)
pd.DataFrame({"id": test_rows.id, "tuketim": np.expm1(np.maximum(blend_pred, 0))}).to_csv(
    OUT / "gpu_blend.csv", index=False
)
pd.DataFrame(
    {"id": test_rows.id, "gpu_log": gpu_pred, "base_log": test_x.base, "known": test_x.known}
).to_csv(OUT / "gpu_log_predictions.csv", index=False)
print("wrote GPU outputs", best_alpha, best_iterations)
