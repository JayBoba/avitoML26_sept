from __future__ import annotations

import numpy as np
import pandas as pd

# (начало валидации, конец валидации) — конец не включается
FOLDS = [
    ("2026-04-13", "2026-04-15"),
    ("2026-04-15", "2026-04-17"),
    ("2026-04-17", "2026-04-20"),
]


def time_folds(meta: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Для каждого фолда возвращает (train_mask, valid_mask) — булевы массивы по строкам meta."""
    day = meta["window_start_ts"]
    folds = []
    for start, end in FOLDS:
        train_mask = (day < start).to_numpy()
        valid_mask = ((day>=start) & (day<end)).to_numpy()
        folds.append((train_mask, valid_mask))
    return folds