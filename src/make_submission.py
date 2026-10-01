#!/usr/bin/env python3
"""Create a reproducible RMSLE-oriented Grid Up submission.

The target is modeled in log1p space. Established transformers use a blend of
their very recent level and their historical day-of-week level. Transformers
that do not occur in train use a cold-start prior learned from the early-life
records of historical transformers with the same installed power.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


AGE_BINS = [-1, 6, 13, 27, 59, 89, 121, 182, 365, 100_000]


def mapped_mean(frame: pd.DataFrame, keys: list[str], values: pd.Series) -> np.ndarray:
    """Map a Series with a one- or multi-column index onto frame rows."""
    if len(keys) == 1:
        index = frame[keys[0]]
    else:
        index = pd.MultiIndex.from_frame(frame[keys])
    return values.reindex(index).to_numpy(dtype=float)


def make_log_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    recent_days: int,
    recent_weight: float,
    seasonal_weight: float,
    cold_start: str,
    weekday_days: int = 0,
    cold_log_offset: float = 0.0,
) -> tuple[np.ndarray, dict[str, int | float]]:
    train = train.copy()
    test = test.copy()
    train["log_target"] = np.log1p(train["tuketim"].clip(lower=0))
    train_end = train["tarih"].max()

    overall = float(train["log_target"].mean())
    power_mean = train.groupby("guc", observed=True)["log_target"].mean()

    # A conservative hierarchy for established transformers.
    location_mean = train.groupby("lokasyon", observed=True)["log_target"].mean()
    location_prior = test["lokasyon"].map(location_mean).fillna(overall).to_numpy()
    entity_mean = train.groupby("tanim", observed=True)["log_target"].mean()
    entity_all = test["tanim"].map(entity_mean).to_numpy(dtype=float)
    entity_all = np.where(np.isnan(entity_all), location_prior, entity_all)

    recent = train[train["tarih"] > train_end - pd.Timedelta(days=recent_days)]
    recent_mean = recent.groupby("tanim", observed=True)["log_target"].mean()
    entity_recent = test["tanim"].map(recent_mean).to_numpy(dtype=float)
    has_recent_history = ~np.isnan(entity_recent)
    entity_recent = np.where(np.isnan(entity_recent), entity_all, entity_recent)

    train["dow"] = train["tarih"].dt.dayofweek.astype("int8")
    test["dow"] = test["tarih"].dt.dayofweek.astype("int8")
    weekday_source = train
    if weekday_days:
        weekday_source = train[
            train["tarih"] > train_end - pd.Timedelta(days=weekday_days)
        ]
    entity_dow_mean = weekday_source.groupby(["tanim", "dow"], observed=True)[
        "log_target"
    ].mean()
    entity_dow = mapped_mean(test, ["tanim", "dow"], entity_dow_mean)
    entity_dow = np.where(np.isnan(entity_dow), entity_all, entity_dow)

    prediction = recent_weight * entity_recent + (1.0 - recent_weight) * entity_dow
    known = test["tanim"].isin(entity_mean.index).to_numpy()

    # Transformers that disappear and return after a long gap behave more like
    # recommissioned assets than continuously active ones. Their stale entity
    # level is substantially worse than the installed-power prior on every
    # rolling origin. Retain a little entity signal for moderate (31-90 day)
    # gaps, where validation consistently favored partial shrinkage.
    last_train_date = train.groupby("tanim", observed=True)["tarih"].max()
    test_first_date = test.groupby("tanim", observed=True)["tarih"].min()
    return_gap = (
        test["tanim"].map(test_first_date) - test["tanim"].map(last_train_date)
    ).dt.days.to_numpy()
    returning = known & (return_gap > 7)
    return_power = test["guc"].map(power_mean).fillna(overall).to_numpy()
    prediction[returning] = return_power[returning]
    moderate_return = known & (return_gap > 30) & (return_gap <= 90)
    prediction[moderate_return] = (
        0.25 * entity_all[moderate_return] + 0.75 * return_power[moderate_return]
    )

    # Estimate the April-July seasonal movement relative to March from assets
    # observed on every day of the most recent complete calendar year. Entity
    # deltas are pooled by location and shrunk toward the national month effect.
    # This retains the current March level while adding a robust summer shape.
    if seasonal_weight:
        complete_year = train_end.year - 1
        year_rows = train[train["tarih"].dt.year == complete_year].copy()
        days_in_year = pd.Timestamp(f"{complete_year}-12-31").dayofyear
        year_counts = year_rows.groupby("tanim", observed=True)["tarih"].nunique()
        complete_ids = year_counts[year_counts == days_in_year].index
        seasonal_source = year_rows[year_rows["tanim"].isin(complete_ids)].copy()
        seasonal_source["month"] = seasonal_source["tarih"].dt.month
        entity_month = seasonal_source.groupby(
            ["tanim", "month"], observed=True
        )["log_target"].mean().unstack()
        reference_month = train_end.month
        entity_delta = entity_month.sub(entity_month[reference_month], axis=0)
        entity_location = seasonal_source.groupby("tanim", observed=True)[
            "lokasyon"
        ].first()
        delta_long = entity_delta.stack().rename("delta").reset_index()
        delta_long["lokasyon"] = delta_long["tanim"].map(entity_location)
        global_delta = delta_long.groupby("month", observed=True)["delta"].mean()
        location_delta_stats = delta_long.groupby(
            ["lokasyon", "month"], observed=True
        )["delta"].agg(["mean", "count"])
        shrinkage = 10.0
        parent = np.array(
            [global_delta.get(month, 0.0) for _, month in location_delta_stats.index]
        )
        location_delta = pd.Series(
            (
                location_delta_stats["mean"].to_numpy()
                * location_delta_stats["count"].to_numpy()
                + shrinkage * parent
            )
            / (location_delta_stats["count"].to_numpy() + shrinkage),
            index=location_delta_stats.index,
        )
        test["month"] = test["tarih"].dt.month
        seasonal_delta = mapped_mean(test, ["lokasyon", "month"], location_delta)
        global_fallback = test["month"].map(global_delta).fillna(0.0).to_numpy()
        seasonal_delta = np.where(
            np.isnan(seasonal_delta), global_fallback, seasonal_delta
        )
        apply_seasonality = known & has_recent_history & ~returning
        prediction[apply_seasonality] += (
            seasonal_weight * seasonal_delta[apply_seasonality]
        )

    # Cold start: learn early-life behavior from transformers that genuinely
    # appeared after the left boundary of train, avoiding boundary-censored age.
    first_train_date = train.groupby("tanim", observed=True)["tarih"].min()
    newcomer_ids = first_train_date[
        first_train_date > train["tarih"].min() + pd.Timedelta(days=28)
    ].index
    cold_source = train[train["tanim"].isin(newcomer_ids)].copy()
    cold_source["age"] = (
        cold_source["tarih"] - cold_source["tanim"].map(first_train_date)
    ).dt.days
    cold_source["age_bin"] = pd.cut(
        cold_source["age"], AGE_BINS, labels=False, include_lowest=True
    )

    test["age"] = (test["tarih"] - test["tanim"].map(test_first_date)).dt.days
    test["age_bin"] = pd.cut(test["age"], AGE_BINS, labels=False, include_lowest=True)

    power_age_mean = cold_source.groupby(
        ["guc", "age_bin"], observed=True
    )["log_target"].mean()
    cold_prediction = mapped_mean(test, ["guc", "age_bin"], power_age_mean)
    cold_power_fallback = test["guc"].map(power_mean).fillna(overall).to_numpy()
    cold_prediction = np.where(
        np.isnan(cold_prediction), cold_power_fallback, cold_prediction
    )
    if cold_start == "power":
        cold_prediction = cold_power_fallback
    cold_prediction = cold_prediction + cold_log_offset
    prediction = np.where(known, prediction, cold_prediction)

    diagnostics: dict[str, int | float] = {
        "train_rows": len(train),
        "test_rows": len(test),
        "known_test_rows": int(known.sum()),
        "cold_start_test_rows": int((~known).sum()),
        "returning_test_rows": int(returning.sum()),
        "recent_days": recent_days,
        "recent_weight": recent_weight,
        "seasonal_weight": seasonal_weight,
        "cold_start": cold_start,
        "weekday_days": weekday_days,
        "cold_log_offset": cold_log_offset,
    }
    return np.maximum(prediction, 0.0), diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/raw/test.csv"))
    parser.add_argument(
        "--sample", type=Path, default=Path("data/raw/sample_submission.csv")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recent-days", type=int, default=3)
    parser.add_argument("--recent-weight", type=float, default=0.8)
    parser.add_argument("--seasonal-weight", type=float, default=0.0)
    parser.add_argument(
        "--weekday-days",
        type=int,
        default=0,
        help="limit weekday averages to this many recent days; 0 uses all history",
    )
    parser.add_argument(
        "--cold-start", choices=["power-age", "power"], default="power-age"
    )
    parser.add_argument(
        "--cold-log-offset",
        type=float,
        default=0.0,
        help="additive log1p calibration for transformers absent from train",
    )
    args = parser.parse_args()

    if args.recent_days < 1:
        parser.error("--recent-days must be positive")
    if not 0 <= args.recent_weight <= 1:
        parser.error("--recent-weight must be between 0 and 1")
    if not 0 <= args.seasonal_weight <= 1.5:
        parser.error("--seasonal-weight must be between 0 and 1.5")
    if args.weekday_days < 0:
        parser.error("--weekday-days must be non-negative")

    train = pd.read_csv(args.train, parse_dates=["tarih"])
    test = pd.read_csv(args.test, parse_dates=["tarih"])
    sample = pd.read_csv(args.sample)
    log_prediction, diagnostics = make_log_predictions(
        train,
        test,
        args.recent_days,
        args.recent_weight,
        args.seasonal_weight,
        args.cold_start,
        args.weekday_days,
        args.cold_log_offset,
    )
    prediction = np.expm1(log_prediction)

    submission = pd.DataFrame({"id": test["id"], "tuketim": prediction})
    if not submission["id"].equals(sample["id"]):
        raise ValueError("test row order does not match sample_submission.csv")
    if submission["tuketim"].isna().any() or (submission["tuketim"] < 0).any():
        raise ValueError("submission contains invalid predictions")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(submission):,} rows)")
    print(diagnostics)
    print(submission["tuketim"].describe(percentiles=[0.01, 0.5, 0.99]).to_string())


if __name__ == "__main__":
    main()
