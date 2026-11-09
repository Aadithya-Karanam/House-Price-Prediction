# House Price Prediction using XGBoost

## Project Overview
A Machine Learning project that predicts house prices using the XGBoost Regressor algorithm.

## Features
- House price prediction
- Feature selection
- Missing value handling
- Categorical feature encoding
- XGBoost regression model
- Interactive Streamlit frontend

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit

## Selected Features
- Overall Quality
- Living Area
- Garage Capacity
- Basement Area
- First Floor Area
- Full Bathrooms
- Year Built
- Neighborhood

## Project Structure
- `MLmodel.py` – Model training
- `app.py` – Streamlit frontend
- `xgb_model.pkl` – Trained model
- `requirements.txt` – Dependencies

## Run Locally

1. Install dependencies:

   `pip install -r requirements.txt`

2. Download the Ames Housing competition dataset from Kaggle and place `train.csv` inside the `dataset/` folder if you want to retrain the model.

3. Run the application:

   `streamlit run app.py`

## Algorithm
XGBoost Regressor

## Disclaimer
Predictions are estimates for educational purposes and are not professional property valuations.