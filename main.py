import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def generate_car_data(num_samples=1000, filename='cars_co2_data.csv'):
    """
    Generates synthetic vehicle data including engine size, cylinders, 
    fuel consumption, and CO2 emissions, then saves it to a CSV file.
    """
    np.random.seed(42)
    
    # Generate independent variables
    engine_size = np.random.uniform(1.0, 8.0, num_samples)
    cylinders = np.random.choice([3, 4, 6, 8, 10, 12], num_samples)
    
    # Generate fuel consumption with some realistic noise
    fuel_noise = np.random.normal(0, 1.5, num_samples)
    fuel_consumption = np.clip((engine_size * 2.5) + fuel_noise, 2.0, None)
    
    # Calculate CO2 emissions (1 liter of fuel produces approx 2300g of CO2)
    co2_noise = np.random.normal(0, 50, num_samples)
    co2_emissions = (fuel_consumption * 2300) + (cylinders * 50) + co2_noise
    
    # Create DataFrame
    data = pd.DataFrame({
        'Engine_Size_L': engine_size,
        'Cylinders': cylinders,
        'Fuel_Consumption_L_per_Hour': fuel_consumption,
        'CO2_Emissions_g_per_Hour': co2_emissions
    })
    
    # Save to CSV
    data.to_csv(filename, index=False)
    print(f"[INFO] Data successfully generated and saved to '{filename}'.")

def train_model(data_path='cars_co2_data.csv', model_path='co2_regression_model.pkl'):
    """
    Trains a Multiple Linear Regression model on the generated dataset 
    and saves the trained model to a file.
    """
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Define features (X) and target (y)
    X = df[['Engine_Size_L', 'Cylinders', 'Fuel_Consumption_L_per_Hour']]
    y = df['CO2_Emissions_g_per_Hour']
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"[INFO] Model Evaluation -> MSE: {mse:.2f} | R-squared: {r2:.4f}")
    
    # Save the model
    joblib.dump(model, model_path)
    print(f"[INFO] Model successfully trained and saved to '{model_path}'.")
    
    return model

def predict_co2(model, engine_size, cylinders, fuel_consumption):
    """
    Predicts CO2 emissions for a specific vehicle and converts the output 
    into per-hour, per-minute, and per-second metrics.
    """
    # Prepare input data matching the training features
    input_data = pd.DataFrame({
        'Engine_Size_L': [engine_size],
        'Cylinders': [cylinders],
        'Fuel_Consumption_L_per_Hour': [fuel_consumption]
    })
    
    # Make prediction
    co2_per_hour = model.predict(input_data)[0]
    
    # Time conversions
    co2_per_minute = co2_per_hour / 60.0
    co2_per_second = co2_per_minute / 60.0
    
    # Print results
    print("-" * 50)
    print("VEHICLE SPECIFICATIONS:")
    print(f" > Engine Size:      {engine_size} L")
    print(f" > Cylinders:        {cylinders}")
    print(f" > Fuel Consumption: {fuel_consumption} L/h")
    print("PREDICTED CO2 EMISSIONS:")
    print(f" > {co2_per_hour:.2f} grams / hour")
    print(f" > {co2_per_minute:.2f} grams / minute")
    print(f" > {co2_per_second:.2f} grams / second")
    print("-" * 50)

if __name__ == "__main__":
    # Step 1: Generate synthetic data
    generate_car_data()
    
    # Step 2: Train and evaluate the model
    trained_model = train_model()
    
    # Step 3: Run inference on new data samples
    print("\n[TEST] Sample 1: Standard Family Car")
    predict_co2(trained_model, engine_size=2.0, cylinders=4, fuel_consumption=6.0)
    
    print("\n[TEST] Sample 2: High-Performance Sports Car")
    predict_co2(trained_model, engine_size=5.0, cylinders=8, fuel_consumption=15.0)
