# NewsSort AI

NewsSort AI is an NLP-based news category classification system for an Artificial Intelligence assignment.

The system will classify news text into categories such as:

- Business
- Entertainment
- Politics
- Sport
- Technology

## Planned workflow

1. Add the BBC News dataset into `dataset/bbc_news.csv`.
2. Inspect the dataset columns and labels.
3. Clean and preprocess the news text.
4. Train the first model: TF-IDF + Support Vector Machine.
5. Evaluate using accuracy, precision, recall, and F1-score.
6. Save the trained model and vectorizer.
7. Build a Streamlit prototype later.

## Project structure

```text
NewsClassifier/
├── dataset/
├── models/
├── notebooks/
├── train_svm_model.py
├── predict_svm_news.py
├── app.py
├── requirements.txt
└── README.md
```

## First model

We will start with TF-IDF + Support Vector Machine first.
