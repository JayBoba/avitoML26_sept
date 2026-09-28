from __future__ import annotations

import pandas as pd

EVENT_TYPES = [
    "search_results_view", "item_view", "photo_swipe", "seller_page_view",
    "favorite_add", "contact_phone_show", "contact_chat_open",
    "contact_message_sent", "login",
]
# captcha_shown не берём тк внутри окон её нет ни в train, ни в test

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
    shares = (
        pd.crosstab(events["cookie_id"], events["event_name"], normalize="index")
        .reindex(columns=EVENT_TYPES, fill_value=0.0)
        .add_prefix("share_")
    )
    return features.join(shares)