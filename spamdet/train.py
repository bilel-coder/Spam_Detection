"""
Model training, comparison and selection.

Compares the following classifiers (all wrapped in TF-IDF pipelines):
  1. Logistic Regression
  2. Multinomial Naïve Bayes
  3. Linear SVM (SGD)
  4. Random Forest
  5. Gradient Boosting (XGBoost-style via sklearn)

The best model (highest F1-macro on the test set) is saved to disk.
"""

import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from spamdet.config import (
    LABEL_COLUMN,
    METRICS_DIR,
    METRICS_PATH,
    MODELS_DIR,
    PIPELINE_PATH,
    PROCESSED_CSV,
    RANDOM_STATE,
    TEST_SIZE,
    TEXT_COLUMN,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_SUBLINEAR_TF,
)
from spamdet.preprocessing import run_preprocessing

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ── Candidate model definitions ───────────────────────────────────────────────

def _build_tfidf():
    return TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        strip_accents="unicode",
        analyzer="word",
    )


CANDIDATE_MODELS = {
    "LogisticRegression": Pipeline([
        ("tfidf", _build_tfidf()),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)),
    ]),
    "NaiveBayes": Pipeline([
        ("tfidf",  _build_tfidf()),
        ("scaler", MaxAbsScaler()),          # MNB requires non-negative features
        ("clf",    MultinomialNB(alpha=0.1)),
    ]),
    "LinearSVM": Pipeline([
        ("tfidf", _build_tfidf()),
        ("clf",   SGDClassifier(
            loss="hinge", penalty="l2", alpha=1e-4,
            max_iter=100, random_state=RANDOM_STATE
        )),
    ]),
    "RandomForest": Pipeline([
        ("tfidf", _build_tfidf()),
        ("clf",   RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
        )),
    ]),
    "GradientBoosting": Pipeline([
        ("tfidf", _build_tfidf()),
        ("clf",   GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1,
            max_depth=4, random_state=RANDOM_STATE
        )),
    ]),
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_data():
    if not Path(PROCESSED_CSV).exists():
        logger.info("Processed data not found → running preprocessing...")
        df = run_preprocessing()
    else:
        df = pd.read_csv(PROCESSED_CSV)

    X = df["clean_text"].fillna("")
    y = df[LABEL_COLUMN]
    return X, y


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _evaluate(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    try:
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        auc = round(roc_auc_score(y_test, y_prob), 4)
    except AttributeError:
        auc = None   # SVM has no predict_proba by default

    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "f1_macro":  round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "roc_auc":   auc,
        "report":    classification_report(y_test, y_pred, target_names=["ham", "spam"]),
    }


# ── Main training routine ─────────────────────────────────────────────────────

def train_and_select() -> dict:
    """Train all candidates, compare metrics, persist the best model."""

    # Ensure output dirs exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = _load_data()
    logger.info("Dataset: %d samples | spam ratio: %.1f%%", len(y), y.mean() * 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, pipeline in CANDIDATE_MODELS.items():
        logger.info("Training %-20s ...", name)
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = round(time.time() - t0, 2)

        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv, scoring="f1_macro", n_jobs=-1
        )

        metrics = _evaluate(pipeline, X_test, y_test)
        metrics["cv_f1_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_f1_std"]  = round(cv_scores.std(), 4)
        metrics["train_time_s"] = train_time
        results[name] = metrics

        logger.info(
            "  %-20s | F1=%.4f | AUC=%s | CV=%.4f±%.4f | %ss",
            name, metrics["f1"], metrics["roc_auc"],
            metrics["cv_f1_mean"], metrics["cv_f1_std"], train_time
        )

    # ── Select best model ────────────────────────────────────────────────────
    best_name = max(results, key=lambda n: results[n]["f1_macro"])
    best_pipeline = CANDIDATE_MODELS[best_name]

    logger.info("✅ Best model: %s  (F1-macro=%.4f)", best_name, results[best_name]["f1_macro"])
    print("\n" + "─" * 60)
    print(f"  Best model : {best_name}")
    print(f"  F1-macro   : {results[best_name]['f1_macro']}")
    print(f"  Accuracy   : {results[best_name]['accuracy']}")
    print(f"  ROC-AUC    : {results[best_name]['roc_auc']}")
    print("─" * 60)
    print(results[best_name]["report"])

    # ── Persist ──────────────────────────────────────────────────────────────
    joblib.dump(best_pipeline, PIPELINE_PATH)
    logger.info("Model saved → %s", PIPELINE_PATH)

    summary = {
        "best_model": best_name,
        "best_metrics": {k: v for k, v in results[best_name].items() if k != "report"},
        "all_models": {
            n: {k: v for k, v in m.items() if k != "report"}
            for n, m in results.items()
        },
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    logger.info("Metrics saved → %s", METRICS_PATH)

    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    train_and_select()
