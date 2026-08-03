import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from data_preprocessing import create_tfidf_features, load_and_prepare_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_model.pkl"
RESULT_PATH = PROJECT_ROOT / "results" / "logistic_regression_results.csv"


def main():
    print("NewsSort AI - Logistic Regression Training")
    print("=" * 50)

    X_train, X_test, y_train, y_test = load_and_prepare_dataset()
    vectorizer, X_train_tfidf, X_test_tfidf = create_tfidf_features(X_train, X_test)

    print("\nTraining enhanced Logistic Regression model...")
    base_model = LogisticRegression(max_iter=2000, random_state=42)

    param_grid = {
        "C": [0.1, 1.0, 3.0, 10.0],
        "class_weight": [None, "balanced"],
        "solver": ["lbfgs"],
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

    print("\nEvaluating Logistic Regression model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\nLogistic Regression Results")
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
                "Model": "Logistic Regression",
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

    print("\nSaving Logistic Regression model and result...")
    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    os.makedirs(RESULT_PATH.parent, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    result_df.to_csv(RESULT_PATH, index=False)

    print("Saved model to:", MODEL_PATH)
    print("Saved result to:", RESULT_PATH)
    print("\nLogistic Regression training completed successfully.")


if __name__ == "__main__":
    main()
