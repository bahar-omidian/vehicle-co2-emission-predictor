# Vehicle $CO_2$ Emission Predictor

This repository contains a machine learning pipeline to predict vehicle $CO_2$ emissions based on technical specifications such as engine size, number of cylinders, and fuel consumption.

## Architecture
The system employs a Multiple Linear Regression model and is divided into three main modules:
1. **Data Generation:** Synthesizes a dataset of 1000 records (`cars_co2_data.csv`).
2. **Model Training:** Trains the regression model and persists it using `joblib` (`co2_regression_model.pkl`).
3. **Inference:** Predicts $CO_2$ emissions per hour, minute, and second based on new inputs.

   
