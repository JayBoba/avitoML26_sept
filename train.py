from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from data import load_meta, load_window_events
from features import build_features
from validation import cross_validate

SEED = 0


def main() -> None:
    meta = load_meta("train")
    # reindex: строки X в том же порядке, что и в meta (и в y)
    X = build_features(load_window_events(meta)).reindex(meta["cookie_id"])
    y = meta["target"].to_numpy()

    models = {
        "baseline_rf": lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        "hgb": lambda: HistGradientBoostingClassifier(random_state=SEED),
    }
    for name, make_model in models.items():
        scores = cross_validate(make_model, X, y, meta)
        print(f"{name:12s} folds={np.round(scores, 3)} mean={np.mean(scores):.3f}")


if __name__ == "__main__":
    main()