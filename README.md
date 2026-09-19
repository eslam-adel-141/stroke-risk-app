# 🫀 Stroke Risk Predictor

A small machine-learning web app that estimates stroke risk from symptoms and age,
using two models side by side:

- **Regression:** predicts the stroke risk as a percentage (0-100%).
- **Classification:** predicts whether the person is "At Risk" (risk >= 50%).

**Live demo:** _add your Streamlit link here_

> ⚠️ **Disclaimer:** this is an educational demo, not a medical tool. The dataset appears to be
> synthetic (the `At Risk` label is defined as `Stroke Risk (%) >= 50`), so the models learn that
> rule rather than real clinical relationships. Do not use it for medical decisions.

## Features
- Interactive form: age slider + 15 symptom checkboxes
- Regression and classification results shown in separate tabs
- Model performance metrics displayed inside the app

## Project structure
```
main.py            Streamlit app (UI)
src/config.py      Settings (env variables with defaults)
src/predictor.py   Loads the models and runs predictions
models/            Trained models + metadata.json (features order, age range, metrics)
notebooks/         Full workflow: EDA, model comparison, training, export
```

## Machine-learning workflow
1. Data cleaning: duplicates removed, outliers inspected (high-risk rows are kept on purpose).
2. Data-leakage check: `At Risk (Binary)` is dropped from the regression features and
   `Stroke Risk (%)` from the classification features.
3. Models compared: Linear/Ridge Regression, Decision Tree, Random Forest, Gradient Boosting
   (regression) and Logistic Regression, Decision Tree, Random Forest, Gradient Boosting (classification).
4. Final choice: **Gradient Boosting** for both tasks. Its predictions stay within the training range,
   which makes it more robust to unexpected inputs than a linear model.

## Run locally
```bash
git clone https://github.com/<eslam-adel-141>/stroke-risk-app.git
cd stroke-risk-app
pip install -r requirements.txt
streamlit run main.py
```

## Retrain the models
Open `notebooks/stroke_risk_dataset.ipynb` in Google Colab, run all cells, and copy the files from
the generated `models.zip` into the `models/` folder.

## Tech stack
Python, pandas, scikit-learn, Streamlit

## License
MIT, see [LICENSE](LICENSE).