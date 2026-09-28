from __future__ import annotations

import pandas as pd

# captcha_shown нет тк в train она бывает только после окна
EVENT_TYPES = [
    "search_results_view", "item_view", "photo_swipe", "seller_page_view",
    "favorite_add", "contact_phone_show", "contact_chat_open",
    "contact_message_sent", "login",
]


def build_features(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    ua = events["user_agent"]
    # UA разбираем на флаги, а не берём целиком тк строка как категория переобучается (см. README)
    events = events.assign(
        dt=events.groupby("cookie_id")["event_ts"].diff().dt.total_seconds(),
        ua_headless=ua.str.contains("HeadlessChrome", regex=False),
        ua_script=~ua.str.startswith(("Mozilla/", "Avito/")),
        ua_app=ua.str.startswith("Avito/"),
        platform_web=events["platform"].eq("web"),
        platform_android=events["platform"].eq("android"),
    )
    g = events.groupby("cookie_id")
    features = pd.DataFrame({
        "n_events": g.size(),
        "item_nunique": g["item_id"].nunique(),
        "dt_median": g["dt"].median(),
        "dt_min": g["dt"].min(),
        "dt_mean": g["dt"].mean(),
        "dt_std": g["dt"].std(),
        "duration_s": (g["event_ts"].max() - g["event_ts"].min()).dt.total_seconds(),
        "category_nunique": g["item_category"].nunique(),
        "location_nunique": g["item_location"].nunique(),
        "query_nunique": g["search_query"].nunique(),
        "search_page_max": g["search_page"].max(),
        "search_page_mean": g["search_page"].mean(),
        "ua_nunique": g["user_agent"].nunique(),
        "ua_headless": g["ua_headless"].mean(),
        "ua_script": g["ua_script"].mean(),
        "ua_app": g["ua_app"].mean(),
        "platform_web": g["platform_web"].mean(),
        "platform_android": g["platform_android"].mean(),
        "pointer_share": g["pointer_x"].count() / g.size(),
        "pointer_x_mean": g["pointer_x"].mean(),
        "pointer_x_std": g["pointer_x"].std(),
        "pointer_y_std": g["pointer_y"].std(),
    })
    features["item_unique_ratio"] = features["item_nunique"] / g["item_id"].count()
    features["dt_cv"] = features["dt_std"] / features["dt_mean"]
    shares = (
        pd.crosstab(events["cookie_id"], events["event_name"], normalize="index")
        .reindex(columns=EVENT_TYPES, fill_value=0.0)
        .add_prefix("share_")
    )
    return features.join(shares).reindex(meta["cookie_id"])