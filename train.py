from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
 
from data import load_meta, load_window_events
from features import build_features
from validation import cross_validate
 
SEED = 0
BASELINE_COLS = ["n_events", "item_nunique"]   # признаки из quickstart
SUBMISSION_PATH = Path(__file__).parent / "submission.csv"
 
def make_baseline() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1)

def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(early_stopping=False, random_state=SEED)
 
 
def check_submission(sub: pd.DataFrame, test: pd.DataFrame) -> None:
    """Формат из условия: ровно куки test, по одной строке, score в [0, 1], без пропусков."""
    assert list(sub.columns) == ["cookie_id", "score"]
    assert len(sub) == len(test) and sub["cookie_id"].is_unique
    assert set(sub["cookie_id"]) == set(test["cookie_id"])
    assert sub["score"].notna().all() and sub["score"].between(0, 1).all()
 
def main() -> None:
    train = load_meta("train")
    X = build_features(load_window_events(train), train)
    y = train["target"].to_numpy()
 
    for name, factory, cols in [("baseline_rf", make_baseline, BASELINE_COLS),
                                ("hgb", make_model, list(X.columns))]:
        scores = cross_validate(factory, X[cols], y, train)
        print(f"{name:12s} P@R0.7 folds={np.round(scores, 3)} mean={np.mean(scores):.3f}")
 
    test = load_meta("test")
    X_test = build_features(load_window_events(test), test)
    model = make_model().fit(X, y)
    sub = pd.DataFrame({
        "cookie_id": test["cookie_id"],
        "score": model.predict_proba(X_test[X.columns])[:, 1],
    })
    check_submission(sub, test)
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"сохранено {SUBMISSION_PATH.name}: {len(sub)} строк")
 
 
if __name__ == "__main__":
    main()
 