from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from metric import precision_at_recall

# валидация [start, end), трейн — всё раньше start: 06–12 | 13–14, 06–14 | 15–16, 06–16 | 17–19
FOLDS = [
    ("2026-04-13", "2026-04-15"),
    ("2026-04-15", "2026-04-17"),
    ("2026-04-17", "2026-04-20"),
]


def time_folds(meta: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    day = meta["window_start_ts"]
    folds = []
    for start, end in FOLDS:
        train_mask = (day < start).to_numpy()
        valid_mask = ((day >= start) & (day < end)).to_numpy()
        folds.append((train_mask, valid_mask))
    return folds


def cross_validate(make_model: Callable, X: pd.DataFrame, y: np.ndarray, meta: pd.DataFrame) -> list[float]:
    scores = []
    for train_mask, valid_mask in time_folds(meta):
        model = make_model()
        model.fit(X[train_mask], y[train_mask])
        p = model.predict_proba(X[valid_mask])[:, 1]
        scores.append(precision_at_recall(y[valid_mask], p))
    return scores