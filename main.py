import streamlit as st

from src.config import APP_TITLE, RISK_THRESHOLD
from src.predictor import StrokeRiskPredictor

st.set_page_config(page_title=APP_TITLE, page_icon="🫀")


@st.cache_resource
def get_predictor() -> StrokeRiskPredictor:
    return StrokeRiskPredictor()


predictor = get_predictor()
meta = predictor.meta

st.title(f"🫀 {APP_TITLE}")
st.caption("Demo built on a synthetic dataset. Not a medical diagnosis, "
           "please consult a doctor for any health concerns.")

age = st.slider("Age", meta.get("age_min", 18), meta.get("age_max", 90), 45)
st.write("Select the symptoms that apply:")

cols = st.columns(2)
values = {name: int(cols[i % 2].checkbox(name))
          for i, name in enumerate(predictor.symptoms)}
values["Age"] = age

if st.button("Predict", type="primary"):
    result = predictor.predict(values)
    risk, label, proba = result["risk_percent"], result["at_risk"], result["probability"]

    tab_reg, tab_clf = st.tabs(["Regression (Stroke Risk %)", "Classification (At Risk?)"])

    with tab_reg:
        st.metric("Predicted stroke risk", f"{risk:.1f}%")
        st.progress(int(risk))

    with tab_clf:
        (st.error if label == 1 else st.success)(
            "Result: At Risk" if label == 1 else "Result: Not At Risk"
        )
        st.metric("Probability of being at risk", f"{proba * 100:.1f}%")

    if (risk >= RISK_THRESHOLD) != (label == 1):
        st.info("The two models slightly disagree because this case is close to "
                f"the {RISK_THRESHOLD:.0f}% threshold.")

with st.expander("Model performance (test set)"):
    r, c = meta["reg_metrics"], meta["clf_metrics"]
    st.write(f"**Regression:** R² = {r['r2']:.3f} | MAE = {r['mae']:.2f}")
    st.write(f"**Classification:** Accuracy = {c['accuracy']:.3f} | "
             f"F1 = {c['f1']:.3f} | ROC-AUC = {c['roc_auc']:.3f}")