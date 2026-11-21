# California Housing Price Prediction: KNN Pipeline + Streamlit App

Assignment 9 (Model Deployment): an end-to-end ML workflow that predicts median house values from the 1990 California census.

## What's inside
| File | Purpose |
|---|---|
| `california_housing_knn.ipynb` | Loads the data, preprocesses it with a `ColumnTransformer`, tunes a `KNeighborsRegressor` with `GridSearchCV` (5-fold CV, R²), evaluates on a 20% test set and saves the pipeline |
| `models/california_knn_pipeline.pkl` | The trained pipeline (preprocessing + tuned KNN), created by the notebook |
| `models/model_metadata.json` | Best parameters, scores and input ranges, created by the notebook |
| `app.py` | Streamlit app that loads the `.pkl` and predicts house values |
| `requirements.txt` | Python packages needed |

## Method
- **Preprocessing:** `log1p` + `StandardScaler` for the skewed features (`AveRooms`, `AveBedrms`, `Population`, `AveOccup`); `StandardScaler` for `MedInc`, `HouseAge`, `Latitude`, `Longitude`. Scaling matters because KNN is distance-based.
- **Split:** 80% train / 20% test (`random_state=42`).
- **Tuning:** `GridSearchCV` over `n_neighbors` ∈ {3, 5, 7, 9}, `weights` ∈ {uniform, distance}, `p` ∈ {1, 2}, with 5-fold cross-validation and R² scoring.
- **Deployment:** the whole pipeline is saved with `joblib`, so the app sends raw feature values and gets a prediction.

## How to run
```bash
pip install -r requirements.txt
jupyter notebook california_housing_knn.ipynb   # run all cells; creates models/
streamlit run app.py                             # opens http://localhost:8501
```
The first run of the notebook downloads the dataset (about 400 KB), so an internet connection is needed once.
