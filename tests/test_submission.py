"""Offline checks for the submission. No training or original artifact writes."""

from pathlib import Path
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
# Keep project imports available when this file is run directly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import compare_models


TEXT = (
    "The football team won the league match after the striker scored two goals. "
    "The coach praised the players and their performance during the final."
)


class SubmissionChecks(unittest.TestCase):
    def test_required_files(self):
        for name in (
            "app.py", "translation_utils.py", "news_features.py",
            "article_extraction/extract_article.py", "requirements.txt", "README.txt",
            "dataset/bbc-news-data.csv", "models/svm_model.pkl",
            "models/logistic_regression_model.pkl", "models/complement_naive_bayes_model.pkl",
        ):
            with self.subTest(file=name):
                self.assertTrue((ROOT / name).is_file(), name)

    def test_all_models_load_without_sklearn_version_mismatch(self):
        for name in (
            "svm_model.pkl", "logistic_regression_model.pkl", "complement_naive_bayes_model.pkl"
        ):
            with self.subTest(model=name), warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                model = joblib.load(ROOT / "models" / name)
                prediction = model.predict([TEXT.lower()])
                self.assertEqual(len(prediction), 1)
                self.assertIn(prediction[0], model.classes_)

    def test_comparison_uses_three_actual_result_files(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "comparison.csv"
            with patch.object(compare_models, "COMPARISON_PATH", output):
                compare_models.main()
            result = pd.read_csv(output)
            self.assertEqual(set(result["Model"]), {
                "Support Vector Machine", "Logistic Regression", "Complement Naive Bayes"
            })
            self.assertEqual(len(result), 3)

    def test_app_text_flow(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)
        # English input and disabled display translation keep this test offline.
        for checkbox in app.checkbox:
            if checkbox.label == "Show translated article":
                checkbox.uncheck()
        app.text_area(key="news_text").input(TEXT)
        next(button for button in app.button if button.label == "Classify News").click()
        app.run()
        self.assertEqual(len(app.exception), 0)
        result = app.session_state["prediction_result"]
        self.assertIn(result["prediction"], {"business", "entertainment", "politics", "sport", "tech"})
        self.assertGreater(len(result["takeaways"]), 0)
        self.assertEqual(len(result["score_df"]), 5)


if __name__ == "__main__":
    unittest.main()
