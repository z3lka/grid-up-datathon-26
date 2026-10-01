#!/usr/bin/env python3
"""Create the v6 baseline with a model specialized for returning assets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from make_submission import make_log_predictions


FEATURES = [
    "hist_n",
    "hist_mean",
    "hist_zero",
    "last7",
    "last28",
    "power_prior",
    "gap",
    "horizon",
    "guc_cat",
    "lokasyon",
    "id2",
    "id3",
    "month",
    "dow",
    "start_month",
]
CAT_FEATURES = [
    "guc_cat",
    "lokasyon",
    "id2",
    "id3",
    "month",
    "dow",
    "start_month",
]


def add_categories(frame: pd.DataFrame) -> pd.DataFrame:
    ids = frame["tanim"].fillna("missing").astype(str)
    frame["guc_cat"] = frame["guc"].fillna(-1).astype(str)
    frame["lokasyon"] = frame["lokasyon"].fillna("missing").astype(str)
    frame["id2"] = ids.str[:2]
    frame["id3"] = ids.str[:3]
    frame["month"] = frame["tarih"].dt.month.astype(str)
    frame["dow"] = frame["tarih"].dt.dayofweek.astype(str)
    frame["start_month"] = frame["start"].dt.month.astype(str)
    return frame


def return_event_training(train: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    rows = train.sort_values(["tanim", "tarih"]).reset_index(drop=True).copy()
    rows["y"] = np.log1p(rows["tuketim"].clip(lower=0)).astype("float32")
    rows["zero"] = (rows["tuketim"] == 0).astype("int8")
    group = rows.groupby("tanim", observed=True)
    rows["gap"] = group["tarih"].diff().dt.days
    rows["seg"] = group["gap"].transform(
        lambda values: (values.fillna(1) > 7).cumsum()
    )
    rows["hist_n"] = group.cumcount()
    denominator = rows["hist_n"].replace(0, np.nan)
    rows["hist_mean"] = (group["y"].cumsum() - rows["y"]) / denominator
    rows["hist_zero"] = (group["zero"].cumsum() - rows["zero"]) / denominator
    rows["last7"] = group["y"].transform(
        lambda values: values.shift().rolling(7, min_periods=1).mean()
    )
    rows["last28"] = group["y"].transform(
        lambda values: values.shift().rolling(28, min_periods=1).mean()
    )

    daily_power = (
        rows.groupby(["guc", "tarih"], observed=True)
        .agg(target_sum=("y", "sum"), target_n=("y", "size"))
        .reset_index()
        .sort_values(["guc", "tarih"])
    )
    power_group = daily_power.groupby("guc", observed=True)
    prior_n = power_group["target_n"].cumsum() - daily_power["target_n"]
    daily_power["power_prior"] = (
        power_group["target_sum"].cumsum() - daily_power["target_sum"]
    ) / prior_n.replace(0, np.nan)
    power_prior = daily_power.set_index(["guc", "tarih"])["power_prior"]

    events = rows[rows["gap"] > 7].copy()
    event_index = pd.MultiIndex.from_frame(events[["guc", "tarih"]])
    events["power_prior"] = power_prior.reindex(event_index).to_numpy()
    events["power_prior"] = events["power_prior"].fillna(rows["y"].mean())
    events = events[
        [
            "tanim",
            "seg",
            "tarih",
            "gap",
            "hist_n",
            "hist_mean",
            "hist_zero",
            "last7",
            "last28",
            "power_prior",
        ]
    ].rename(columns={"tarih": "start"})

    targets = rows.merge(events, on=["tanim", "seg"], suffixes=("", "_event"))
    targets["horizon"] = (targets["tarih"] - targets["start"]).dt.days
    targets = targets[targets["horizon"].between(0, 121)].copy()
    for name in ["gap", "hist_n", "hist_mean", "hist_zero", "last7", "last28"]:
        targets[name] = targets[f"{name}_event"]
    targets = add_categories(targets)
    return targets[FEATURES], targets["y"].to_numpy(dtype="float32")


def return_test_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    history = train.sort_values(["tanim", "tarih"])
    history["y"] = np.log1p(history["tuketim"].clip(lower=0))
    stats = history.groupby("tanim", observed=True).agg(
        hist_n=("y", "size"),
        hist_mean=("y", "mean"),
        hist_zero=("tuketim", lambda values: (values == 0).mean()),
        last=("tarih", "max"),
    )
    last7 = history.groupby("tanim", observed=True).tail(7).groupby(
        "tanim", observed=True
    )["y"].mean()
    last28 = history.groupby("tanim", observed=True).tail(28).groupby(
        "tanim", observed=True
    )["y"].mean()
    first_test = test.groupby("tanim", observed=True)["tarih"].min()
    gap = (test["tanim"].map(first_test) - test["tanim"].map(stats["last"])).dt.days
    mask = test["tanim"].isin(stats.index).to_numpy() & (gap.to_numpy() > 7)

    frame = test.loc[mask].copy()
    frame["start"] = frame["tanim"].map(first_test)
    frame["gap"] = gap.loc[mask].to_numpy()
    frame["horizon"] = (frame["tarih"] - frame["start"]).dt.days
    for name in ["hist_n", "hist_mean", "hist_zero"]:
        frame[name] = frame["tanim"].map(stats[name])
    frame["last7"] = frame["tanim"].map(last7)
    frame["last28"] = frame["tanim"].map(last28)
    power_mean = history.groupby("guc", observed=True)["y"].mean()
    frame["power_prior"] = frame["guc"].map(power_mean).fillna(history["y"].mean())
    frame = add_categories(frame)
    return frame[FEATURES], mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/raw/test.csv"))
    parser.add_argument(
        "--sample", type=Path, default=Path("data/raw/sample_submission.csv")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=167)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="CPU")
    args = parser.parse_args()

    train = pd.read_csv(args.train, parse_dates=["tarih"], dtype={"tanim": "string"})
    test = pd.read_csv(args.test, parse_dates=["tarih"], dtype={"tanim": "string"})
    sample = pd.read_csv(args.sample)
    base_log, diagnostics = make_log_predictions(train, test, 7, 0.7, 1.0, "power")
    event_x, event_y = return_event_training(train)
    return_x, return_mask = return_test_features(train, test)

    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=args.iterations,
        depth=7,
        learning_rate=0.04,
        l2_leaf_reg=10,
        random_seed=41,
        task_type=args.task_type,
        verbose=50,
        allow_writing_files=False,
    )
    model.fit(event_x, event_y, cat_features=CAT_FEATURES)
    prediction = base_log.copy()
    prediction[return_mask] = np.maximum(model.predict(return_x), 0)

    submission = pd.DataFrame({"id": test["id"], "tuketim": np.expm1(prediction)})
    if not submission["id"].equals(sample["id"]):
        raise ValueError("test row order does not match sample_submission.csv")
    if submission["tuketim"].isna().any() or (submission["tuketim"] < 0).any():
        raise ValueError("submission contains invalid predictions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(submission):,} rows)")
    print({**diagnostics, "return_model_rows": int(return_mask.sum())})


if __name__ == "__main__":
    main()
