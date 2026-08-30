NewsSort AI - Quick Start
========================

1. Online website
-----------------
Open this link. No installation is needed:
https://newsclassifier-hjtpgt4fbcbxnvltcmxsc7.streamlit.app/

2. First-time local setup (Anaconda Prompt)
------------------------------------------
Install Anaconda and extract the submission ZIP (or GitHub Release ZIP).
Open Anaconda Prompt. If newssort already exists, skip the create command:

conda create -n newssort python=3.14 pip
conda activate newssort

IMPORTANT: Confirm the prompt starts with (newssort), NOT (base).
Go to the extracted folder containing app.py and requirements.txt.
Replace this example path with your actual folder (including your username):

cd /d "C:\Users\ACER\Downloads\RSW2S2G6_Group5_WongZiHinRick_ChooKianLiang_TanHongYi\NewsClassifier-finalsubmission1.0"
python -m pip install -r requirements.txt
python -m pip check

If pip check reports "No broken requirements found", start the app:

python -m streamlit run app.py

If installation or pip check fails, resolve the error before continuing.

3. Run again later
----------------------------
In Anaconda Prompt, activate the environment and enter the project folder:

conda activate newssort
cd /d "C:\Users\ACER\Downloads\RSW2S2G6_Group5_WongZiHinRick_ChooKianLiang_TanHongYi\NewsClassifier-finalsubmission1.0"
python -m streamlit run app.py

Open the Local URL printed in the terminal (usually http://localhost:8501).
Keep the terminal open. Press Ctrl+C to stop.

The trained models are included. Retraining is not required.
Reuse newssort; do not recreate it. Reinstall only when requirements change.

4. Notes
--------
For Jupyter: launch it from newssort in Anaconda Navigator, or run
"jupyter notebook" in the activated Anaconda Prompt. Open a Jupyter Terminal,
not a notebook cell, and follow the activate, cd and Streamlit commands above.

Installation, the online website, URL extraction and translation need internet.
Local English-text classification does not need internet after setup.
Audio playback depends on browser/device support.
