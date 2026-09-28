from __future__ import annotations

import pandas as pd


def build_features(events: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует события по cookie_id. Индекс результата — cookie_id."""
    events = events.assign(dt=events.groupby("cookie_id")["event_ts"].diff().dt.total_seconds())
    g = events.groupby("cookie_id")
    features = pd.DataFrame({
        # объём
        "n_events": g.size(),
        "item_nunique": g["item_id"].nunique(),
        # ритм, скрипт действует быстрее и ровнее человека
        "dt_median": g["dt"].median(),
        "dt_min": g["dt"].min(),
        "dt_mean": g["dt"].mean(),
        "dt_std": g["dt"].std(),
        "duration_s": (g["event_ts"].max() - g["event_ts"].min()).dt.total_seconds(),
    })
    #разброс пауз относительно их среднего
    features["dt_cv"] = features["dt_std"] / features["dt_mean"]
    return features