NewsSort AI
============

Project Title:
NewsSort AI: A News Category Classification System Using NLP

Project Description:
NewsSort AI is a Natural Language Processing system that classifies news text into categories.

The system can predict these categories:
- Business
- Entertainment
- Politics
- Sport
- Tech

This project compares two classification models:
- Support Vector Machine
- Logistic Regression

Both models use TF-IDF to convert news text into numerical features.


1. Project Folder Structure
---------------------------

NewsClassifier/
|-- dataset/
|   |-- bbc-news-data.csv
|
|-- models/
|   |-- svm_model.pkl
|   |-- logistic_regression_model.pkl
|   |-- tfidf_vectorizer.pkl
|
|-- results/
|   |-- svm_results.csv
|   |-- logistic_regression_results.csv
|   |-- model_comparison.csv
|
|-- training/
|   |-- data_preprocessing.py
|   |-- train_svm_model.py
|   |-- train_logistic_regression_model.py
|   |-- compare_models.py
|
|-- prediction/
|   |-- predict_svm_news.py
|
|-- app.py
|-- requirements.txt
|-- README.txt


2. Dataset
----------

Dataset used:
BBC News dataset

Dataset file location:
dataset/bbc-news-data.csv

Important dataset columns:
- category
- filename
- title
- content

The system uses:
- content as the news text input
- category as the output label


3. Required Libraries
---------------------

Install the required libraries using:

pip install -r requirements.txt

If that does not work, install manually:

pip install pandas numpy scikit-learn streamlit joblib matplotlib seaborn jupyter


4. Training Steps
-----------------

Step 1: Train Support Vector Machine model

python training/train_svm_model.py

Step 2: Train Logistic Regression model

python training/train_logistic_regression_model.py

Step 3: Create model comparison result

python training/compare_models.py


5. Output Files After Training
------------------------------

The model files will be saved in the models folder:

models/svm_model.pkl
models/logistic_regression_model.pkl
models/tfidf_vectorizer.pkl

The result files will be saved in the results folder:

results/svm_results.csv
results/logistic_regression_results.csv
results/model_comparison.csv


6. What Each Training File Does
-------------------------------

training/data_preprocessing.py
- Loads the dataset
- Cleans the news text
- Splits the data into training and testing sets
- Converts text into TF-IDF features

training/train_svm_model.py
- Trains the Support Vector Machine model
- Evaluates the model using accuracy, precision, recall, and F1-score
- Saves the SVM model and result file

training/train_logistic_regression_model.py
- Trains the Logistic Regression model
- Evaluates the model using accuracy, precision, recall, and F1-score
- Saves the Logistic Regression model and result file

training/compare_models.py
- Reads both model result files
- Combines the results into one comparison table
- Saves the comparison table into the results folder


7. Test Prediction in Terminal
------------------------------

Run:

python prediction/predict_svm_news.py

Example input:

Apple launches new artificial intelligence features for iPhone users

Expected output:

tech

Type exit to stop the test program.


8. Run the Streamlit Web App
----------------------------

Run:

streamlit run app.py

The web app allows the user to:
1. Choose a prediction model
2. Enter news text
3. Click Predict Category
4. View the predicted category
5. View model comparison results


9. Example Inputs
-----------------

Example 1:
Apple launches new artificial intelligence features for iPhone users

Expected category:
tech

Example 2:
The football team won the final match after scoring two goals

Expected category:
sport

Example 3:
The prime minister announced a new government policy today

Expected category:
politics


10. System Workflow
-------------------

BBC News Dataset
-> Text Cleaning
-> TF-IDF Feature Extraction
-> Train Support Vector Machine Model
-> Train Logistic Regression Model
-> Compare Model Performance
-> Streamlit Web App
-> Predict News Category


11. Important Notes
-------------------

- Run the training files before running app.py.
- Do not delete the models folder after training.
- Do not delete tfidf_vectorizer.pkl because the models need it to convert text into numbers.
- The results folder stores the evaluation scores for the report.
- This project is for academic demonstration only.
