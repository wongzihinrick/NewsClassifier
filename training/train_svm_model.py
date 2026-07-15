import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC

from data_preprocessing import create_tfidf_features, load_and_prepare_dataset


MODEL_PATH = "models/svm_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
RESULT_PATH = "results/svm_results.csv"


def main():
    print("NewsSort AI - Support Vector Machine Training")
    print("=" * 50)

    X_train, X_test, y_train, y_test = load_and_prepare_dataset()
    vectorizer, X_train_tfidf, X_test_tfidf = create_tfidf_features(X_train, X_test)

    print("\nTraining Support Vector Machine model...")
    model = LinearSVC(random_state=42)
    model.fit(X_train_tfidf, y_train)

    print("\nEvaluating Support Vector Machine model...")
    y_pred = model.predict(X_test_tfidf)

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
            }
        ]
    )

    print("\nSaving SVM model, TF-IDF vectorizer, and result...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    result_df.to_csv(RESULT_PATH, index=False)

    print("Saved model to:", MODEL_PATH)
    print("Saved vectorizer to:", VECTORIZER_PATH)
    print("Saved result to:", RESULT_PATH)
    print("\nSVM training completed successfully.")


if __name__ == "__main__":
    main()
