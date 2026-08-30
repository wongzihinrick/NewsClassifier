NewsSort AI - Quick Start
========================

1. Online website
-----------------
Open the team's shared Streamlit website link. No installation is needed.
The website is already connected to the GitHub main branch.
Url:https://newsclassifier-hjtpgt4fbcbxnvltcmxsc7.streamlit.app/

2. First-time local setup (Anaconda Prompt)
------------------------------------------
Install Anaconda, download the GitHub Release ZIP, and extract it.
Open Anaconda Prompt and run:

conda create -n newssort python=3.14 pip
conda activate newssort

Go to the extracted folder containing app.py.
Replace this example path with your actual folder:

cd /d "D:\Downloads\NewsClassifier-1.0.0-submission"
python -m pip install -r requirements.txt

3. Start the website locally
----------------------------
In Anaconda Prompt, activate the environment and enter the project folder:

conda activate newssort
cd /d "D:\Downloads\NewsClassifier-1.0.0-submission"
python -m streamlit run app.py

Open http://localhost:8501 in your browser.
Keep the terminal open. Press Ctrl+C to stop.

The trained models are included. Retraining is not required.
Reuse the newssort environment on later runs; do not create it again.

4. Alternative: Anaconda Navigator and Jupyter
---------------------------------------------
After completing the first-time setup:
- In Navigator, select the newssort environment and launch Jupyter Notebook.
- Open a Terminal in Jupyter, not a Python notebook cell.
- Ensure the newssort environment is active and enter the project folder.
- Run: python -m streamlit run app.py

If Jupyter is unavailable in Navigator, run "jupyter notebook" from
Anaconda Prompt after activating newssort and entering the project folder.

5. Internet requirements
-------------------------
Installation, the online website, URL extraction and translation need internet.
Local English-text classification does not need internet after setup.
Audio playback depends on browser/device support.
