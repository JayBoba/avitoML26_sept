from __future__ import annotations
from typing import Callable
from metric import precision_at_recall

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

def cross_validate(make_model: Callable, X: pd.DataFrame, y: np.ndarray,
                   meta: pd.DataFrame) -> list[float]:
    """P@R>=0.7 на каждом фолде по времени. make_model() создаёт новую необученную модель."""
    scores = []
    for train_mask, valid_mask in time_folds(meta):
        model = make_model()
        model.fit(X[train_mask], y[train_mask])
        p = model.predict_proba(X[valid_mask])[:, 1]
        scores.append(precision_at_recall(y[valid_mask], p))
    return scores