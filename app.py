"""California house value predictor (Streamlit).

Run from this folder with:   streamlit run app.py
Needs models/california_knn_pipeline.pkl, created by running the notebook first.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "california_knn_pipeline.pkl"
META_PATH = BASE / "models" / "model_metadata.json"

FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]

# label, help text, step, fallback (min, max, default) if the metadata file is missing
INPUTS = {
    "MedInc": ("Median income (in $10,000s)", "Median household income of the block group", 0.1, (0.5, 15.0, 3.5)),
    "HouseAge": ("Median house age (years)", "Median age of houses in the block group", 1.0, (1.0, 52.0, 29.0)),
    "AveRooms": ("Average rooms per household", "Total rooms ÷ households", 0.1, (1.0, 10.0, 5.2)),
    "AveBedrms": ("Average bedrooms per household", "Total bedrooms ÷ households", 0.05, (0.5, 2.0, 1.05)),
    "Population": ("Block group population", "Number of people living in the block group", 10.0, (3.0, 6000.0, 1166.0)),
    "AveOccup": ("Average household size", "People ÷ households", 0.1, (1.0, 6.0, 2.8)),
    "Latitude": ("Latitude", "California spans roughly 32.5 to 42", 0.01, (32.5, 42.0, 34.26)),
    "Longitude": ("Longitude", "California spans roughly -124.4 to -114.3", 0.01, (-124.4, -114.3, -118.5)),
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def load_metadata():
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {}


def make_prediction(model, values: dict) -> float:
    row = pd.DataFrame([values])[FEATURES]
    return float(model.predict(row)[0])


def main():
    st.set_page_config(page_title="California House Value Predictor", page_icon="🏠", layout="wide")
    st.title("🏠 California House Value Predictor")
    st.write("Enter the details of a census block group to estimate its **median house value**. "
             "The model is a tuned K-Nearest Neighbours regressor trained on the 1990 California census.")

    if not MODEL_PATH.exists():
        st.error(f"Model file not found at `{MODEL_PATH}`. Run the notebook first so it saves "
                 "`models/california_knn_pipeline.pkl`, then restart this app.")
        st.stop()

    model = load_model()
    meta = load_metadata()
    ranges = meta.get("feature_ranges", {})

    st.sidebar.header("Block group details")
    values = {}
    for name in FEATURES:
        label, help_text, step, (lo, hi, default) = INPUTS[name]
        if name in ranges:
            lo, hi, default = ranges[name]["min"], ranges[name]["max"], ranges[name]["median"]
        lo, hi, default = round(lo, 2), round(hi, 2), round(default, 2)
        values[name] = st.sidebar.slider(label, min_value=float(lo), max_value=float(hi),
                                         value=float(default), step=float(step), help=help_text)

    prediction = make_prediction(model, values)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Prediction")
        st.metric("Estimated median house value", f"${prediction * 100_000:,.0f}")
        if prediction >= 4.9:
            st.info("The training data is capped at $500,000, so the true value may be higher.")
        st.caption(f"Model output: {prediction:.3f} (in $100,000s)")

        if meta:
            st.subheader("About the model")
            params = meta.get("best_params", {})
            st.write(f"- Best parameters: `n_neighbors={params.get('n_neighbors')}`, "
                     f"`weights={params.get('weights')}`, `p={params.get('p')}`")
            st.write(f"- Cross-validated R²: **{meta.get('cv_r2')}**, test R²: **{meta.get('test_r2')}**")
            st.write(f"- Trained with scikit-learn {meta.get('sklearn_version')}")

    with col2:
        st.subheader("Location")
        st.map(pd.DataFrame({"lat": [values["Latitude"]], "lon": [values["Longitude"]]}), zoom=5)

    with st.expander("Input values sent to the model"):
        st.dataframe(pd.DataFrame([values]))


if __name__ == "__main__":
    main()
