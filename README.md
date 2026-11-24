# Machine Learning-Based Organic Crop Suitability Assessment

A Streamlit educational application that predicts whether a soil sample is labelled **Suitable** or **Needs Improvement** using a Random Forest classifier.

## Files
- `app.py` — Streamlit UI and prediction form
- `model.py` — model training and prediction functions
- `preprocessing.py` — CSV loading, feature validation, and train/test split
- `utils.py` — display helper functions
- `requirements.txt` — Python dependencies
- `organic_farming_suitability_1000_source_calibrated.csv` — 1,000 illustrative synthetic rows

## Run locally
1. Install Python 3.10+.
2. Open a terminal in this project folder.
3. Install dependencies: `pip install -r requirements.txt`
4. Start the app: `streamlit run app.py`


## Dataset note
The included CSV is **synthetic, generated for educational purposes**. Its labels are created using a simple instructional rubric, not authentic field or laboratory records, official organic certification, or expert agronomic labeling. Do not present its accuracy as real-world farm performance.
