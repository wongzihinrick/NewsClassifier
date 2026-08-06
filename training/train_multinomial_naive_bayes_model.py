import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

from data_preprocessing import load_and_prepare_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "multinomial_naive_bayes_model.pkl"
RESULT_PATH = PROJECT_ROOT / "results" / "multinomial_naive_bayes_results.csv"


def create_naive_bayes_vectorizer():
    """
    Create a Naive Bayes friendly TF-IDF feature extractor.
    """
    return FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=20000,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=20000,
                    min_df=2,
                ),
            ),
        ]
    )


def main():
    print("NewsSort AI - Multinomial Naive Bayes Training")
    print("=" * 56)

    X_train, X_test, y_train, y_test = load_and_prepare_dataset()

    print("\nCreating Naive Bayes TF-IDF features...")
    vectorizer = create_naive_bayes_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)

    print("Number of text features:", X_train_tfidf.shape[1])

    print("\nTraining and tuning Multinomial Naive Bayes model...")
    base_model = MultinomialNB()
    param_grid = {
        "alpha": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        "fit_prior": [True, False],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=1,
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

    print("\nEvaluating Multinomial Naive Bayes model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\nMultinomial Naive Bayes Results")
    print("-" * 36)
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-score :", round(f1, 4))

    print("\nDetailed classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    result_df = pd.DataFrame(
        [
            {
                "Model": "Multinomial Naive Bayes",
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1,
                "CV F1-score": grid_search.best_score_,
                "Best Parameters": grid_search.best_params_,
                "Feature Method": "Tuned Word TF-IDF + Character n-gram TF-IDF",
            }
        ]
    )

    print("\nSaving Multinomial Naive Bayes model and result...")
    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    os.makedirs(RESULT_PATH.parent, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    result_df.to_csv(RESULT_PATH, index=False)

    print("Saved model to:", MODEL_PATH)
    print("Saved result to:", RESULT_PATH)
    print("\nMultinomial Naive Bayes training completed successfully.")


if __name__ == "__main__":
    main()
