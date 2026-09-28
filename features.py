from __future__ import annotations

import pandas as pd


def build_features(events: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует события по cookie_id. Индекс результата — cookie_id."""
    events = events.assign(dt=events.groupby("cookie_id")["event_ts"].diff().dt.total_seconds())
    g = events.groupby("cookie_id")
    return pd.DataFrame({
        "n_events": g.size(),
        "item_nunique": g["item_id"].nunique(),
    })