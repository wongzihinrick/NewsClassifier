import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data_preprocessing import create_tfidf_features, load_and_prepare_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "svm_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"
RESULT_PATH = PROJECT_ROOT / "results" / "svm_results.csv"


def main():
    print("NewsSort AI - Support Vector Machine Training")
    print("=" * 50)

    X_train, X_test, y_train, y_test = load_and_prepare_dataset()
    vectorizer, X_train_tfidf, X_test_tfidf = create_tfidf_features(X_train, X_test)

    print("\nTraining and tuning Support Vector Machine model...")
    base_model = LinearSVC(random_state=42)
    param_grid = {
        "C": [0.1, 1.0, 3.0, 10.0],
        "class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
    )
    grid_search.fit(X_train_tfidf, y_train)

    model = Pipeline(
        [
            ("features", vectorizer),
            ("model", grid_search.best_estimator_),
        ]
    )

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation F1-score:", round(grid_search.best_score_, 4))

    print("\nEvaluating Support Vector Machine model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\nSupport Vector Machine Results")
    print("-" * 30)
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-score :", round(f1, 4))

    print("\nDetailed classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    result_df = pd.DataFrame(
        [
            {
                "Model": "Support Vector Machine",
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1,
                "CV F1-score": grid_search.best_score_,
                "Best Parameters": grid_search.best_params_,
                "Feature Method": "Word TF-IDF + Character n-gram TF-IDF",
            }
        ]
    )

    print("\nSaving SVM pipeline, TF-IDF vectorizer, and result...")
    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    os.makedirs(RESULT_PATH.parent, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(model.named_steps["features"], VECTORIZER_PATH)
    result_df.to_csv(RESULT_PATH, index=False)

    print("Saved model to:", MODEL_PATH)
    print("Saved vectorizer to:", VECTORIZER_PATH)
    print("Saved result to:", RESULT_PATH)
    print("\nSVM training completed successfully.")


if __name__ == "__main__":
    main()
