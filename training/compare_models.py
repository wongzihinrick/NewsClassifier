import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SVM_RESULT_PATH = PROJECT_ROOT / "results" / "svm_results.csv"
LOGISTIC_RESULT_PATH = PROJECT_ROOT / "results" / "logistic_regression_results.csv"
MULTINOMIAL_NB_RESULT_PATH = PROJECT_ROOT / "results" / "multinomial_naive_bayes_results.csv"
COMPARISON_PATH = PROJECT_ROOT / "results" / "model_comparison.csv"


def main():
    print("NewsSort AI - Model Comparison")
    print("=" * 40)

    if not os.path.exists(SVM_RESULT_PATH):
        print("Missing SVM result file.")
        print("Please run: python training/train_svm_model.py")
        return

    if not os.path.exists(LOGISTIC_RESULT_PATH):
        print("Missing Logistic Regression result file.")
        print("Please run: python training/train_logistic_regression_model.py")
        return
        
    if not os.path.exists(MULTINOMIAL_NB_RESULT_PATH):
        print("Missing Multinomial Naive Bayes result file.")
        print("Please run: python training/train_multinomial_naive_bayes_model.py")
        return

    svm_results = pd.read_csv(SVM_RESULT_PATH)
    logistic_results = pd.read_csv(LOGISTIC_RESULT_PATH)
    multinomial_nb_results = pd.read_csv(MULTINOMIAL_NB_RESULT_PATH)
    
    comparison_df = pd.concat([svm_results, logistic_results, multinomial_nb_results], ignore_index=True)
    os.makedirs(COMPARISON_PATH.parent, exist_ok=True)
    comparison_df.to_csv(COMPARISON_PATH, index=False)

    print("\nModel Comparison")
    print("-" * 40)
    print(comparison_df.to_string(index=False))
    print("\nSaved comparison to:", COMPARISON_PATH)


if __name__ == "__main__":
    main()
