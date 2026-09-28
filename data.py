from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
META_DATES = ["cookie_created_at", "window_start_ts", "window_end_ts"]
PLATFORM_ALIASES = {"desktop": "web", "iphone": "ios"}


def load_meta(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv", parse_dates=META_DATES)


def load_window_events(meta: pd.DataFrame) -> pd.DataFrame:
    events = pd.read_csv(DATA_DIR / "events.csv.gz", parse_dates=["event_ts"])
    events["platform"] = events["platform"].str.lower().replace(PLATFORM_ALIASES)
    events = events.drop_duplicates()  # после нормализации, иначе дубли по типу "WEB"/"web" выживут

    ev = events.merge(meta[["cookie_id", "window_start_ts", "window_end_ts"]], on="cookie_id")
    in_window = (ev["event_ts"] >= ev["window_start_ts"]) & (ev["event_ts"] < ev["window_end_ts"])
    ev = ev.loc[in_window].drop(columns=["window_start_ts", "window_end_ts"])
    return ev.sort_values(["cookie_id", "event_ts"], kind="mergesort").reset_index(drop=True)