"""Entrena y compara modelos clásicos, y guarda el mejor en model.joblib + metrics.json.

Uso:  python train.py
"""
import json

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from heart_ml import (
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    TARGET,
    build_preprocessor,
    load_data,
)

SEED = 42

CANDIDATES = {
    "Regresión Logística": (
        LogisticRegression(max_iter=1000),
        {"clf__C": [0.01, 0.1, 1, 10]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=SEED),
        {"clf__n_estimators": [100, 300], "clf__max_depth": [3, 5, None]},
    ),
    "KNN": (
        KNeighborsClassifier(),
        {"clf__n_neighbors": [3, 5, 7, 11]},
    ),
}


def main():
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results, fitted = {}, {}
    for name, (estimator, grid) in CANDIDATES.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
        search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        pred = best.predict(X_test)
        proba = best.predict_proba(X_test)[:, 1]
        results[name] = {
            "cv_roc_auc": round(search.best_score_, 4),
            "best_params": search.best_params_,
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred), 4),
            "recall": round(recall_score(y_test, pred), 4),
            "f1": round(f1_score(y_test, pred), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }
        fitted[name] = best
        print(f"{name}: {results[name]}")

    best_name = max(results, key=lambda n: results[n]["cv_roc_auc"])
    print(f"\nMejor modelo (por ROC-AUC en validación cruzada): {best_name}")

    try:
        joblib.dump(fitted[best_name], MODEL_PATH)
        METRICS_PATH.write_text(
            json.dumps({"best_model": best_name, "results": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as e:  # p. ej. disco de solo lectura en la nube
        print(f"No se pudo guardar en disco: {e}")
    return fitted[best_name], {"best_model": best_name, "results": results}


if __name__ == "__main__":
    main()
