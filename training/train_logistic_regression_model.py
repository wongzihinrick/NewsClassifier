import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from data_preprocessing import create_tfidf_features, load_and_prepare_dataset


MODEL_PATH = "models/logistic_regression_model.pkl"
RESULT_PATH = "results/logistic_regression_results.csv"


def main():
    print("NewsSort AI - Logistic Regression Training")
    print("=" * 50)

    X_train, X_test, y_train, y_test = load_and_prepare_dataset()
    _, X_train_tfidf, X_test_tfidf = create_tfidf_features(X_train, X_test)

    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)

    print("\nEvaluating Logistic Regression model...")
    y_pred = model.predict(X_test_tfidf)

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
            }
        ]
    )

    print("\nSaving Logistic Regression model and result...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    result_df.to_csv(RESULT_PATH, index=False)

    print("Saved model to:", MODEL_PATH)
    print("Saved result to:", RESULT_PATH)
    print("\nLogistic Regression training completed successfully.")


if __name__ == "__main__":
    main()
