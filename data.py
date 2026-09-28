"""Загрузка и очистка данных: только события внутри окна наблюдения каждой куки."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
META_DATES = ["cookie_created_at", "window_start_ts", "window_end_ts"]
PLATFORM_ALIASES = {"desktop": "web", "iphone": "ios"}


def load_meta(name: str) -> pd.DataFrame:
    """Читает train.csv / test.csv с датами."""
    return pd.read_csv(DATA_DIR / f"{name}.csv", parse_dates=META_DATES)


def load_window_events(meta: pd.DataFrame) -> pd.DataFrame:
    """События куки из `meta`, попавшие в её окно [window_start_ts, window_end_ts).

    События после окна отбрасываются: они недоступны на момент предсказания
    (в них, например, лежат все показы капчи — прямая утечка таргета).
    """
    events = pd.read_csv(DATA_DIR / "events.csv.gz", parse_dates=["event_ts"])
    # Одна платформа записана по-разному (WEB/Web/desktop, iOS/iphone):
    # после нормализации у каждой куки ровно одна платформа.
    events["platform"] = events["platform"].str.lower().replace(PLATFORM_ALIASES)
    events = events.drop_duplicates()   # после нормализации, иначе "WEB"/"web"-дубли выживут

    ev = events.merge(meta[["cookie_id", "window_start_ts", "window_end_ts"]], on="cookie_id")
    in_window = (ev["event_ts"] >= ev["window_start_ts"]) & (ev["event_ts"] < ev["window_end_ts"])
    ev = ev.loc[in_window].drop(columns=["window_start_ts", "window_end_ts"])
    return ev.sort_values(["cookie_id", "event_ts"], kind="mergesort").reset_index(drop=True)
