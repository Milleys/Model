from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

import numpy as np
import warnings

app = Flask(__name__)

# Load the scaler and XGBoost model
with open('xgb_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('xgb_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

# Define the path to your CSV file
csv_file_path = 'MICE_Time.csv'
new_csv = 'Predicted_Using_App.csv'

# List of station names
station_names = ['Stn. I (Central West Bay)', 'Stn V (Northern West Bay)', 'Stn XIII (Taytay)', 'Stn XV (San Pedro)', 'Stn.XVI (Sta Rosa)',  'Stn XIX (Muntinlupa)']

@app.route('/predict_and_learn', methods=['POST'])
def predict_and_learn():
    try:
        data = request.get_json()
        web_df = pd.DataFrame(data)
        
        # Extract and remove the 'Date' column from web_df
        if 'Date' in web_df.columns:
            selected_date = web_df.pop('Date').iloc[0]
        else:
            return jsonify({'status': 'Error', 'message': 'Date column is missing in the input data'})

        # Warnings management
        warnings.filterwarnings("ignore")

        # Define SARIMA parameters
        sarima_order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12)  # Monthly data with yearly seasonality
        
        # Initialize dictionaries for models and forecasts
        sarima_models = {}
        forecasts = {}
        
        # List of parameters to model
        parameters = ['pH (units)', 'Ammonia (mg/L)', 'Inorganic Phosphate (mg/L)', 'BOD (mg/l)', 'Total coliforms (MPN/100ml)']
        
        # Load dataset and process for each station
        results = {}
        for station_name in station_names:
            # Load dataset for SARIMA
            merged_df = pd.read_csv(csv_file_path)
            merged_df = merged_df[merged_df['Monitoring Stations'] == station_name]
            
            # Drop rows with missing values
            merged_df.dropna(inplace=True)

            # Ensure there's a Date column for time series forecasting
            merged_df['Date'] = pd.to_datetime(merged_df[['Year', 'Month']].assign(DAY=1))
            merged_df.set_index('Date', inplace=True)
            
            # Fit SARIMA models
            for parameter in parameters:
                try:
                    sarima_model = SARIMAX(merged_df[parameter], order=sarima_order, seasonal_order=seasonal_order)
                    sarima_model_fit = sarima_model.fit(disp=False)
                    sarima_models[parameter] = sarima_model_fit
                except Exception as e:
                    return jsonify({'status': 'Error', 'message': f'An error occurred while fitting the model for {parameter} at {station_name}: {e}'})

            # Define the target date for prediction
            target_date = pd.Timestamp(selected_date)
            
            # Calculate number of periods from the last observed date to the target date
            last_date = merged_df.index[-1]
            forecast_steps = (target_date.year - last_date.year) * 12 + (target_date.month - last_date.month)
            
            # Forecast for the target date
            forecast_results = {}
            for parameter in parameters:
                try:
                    model_fit = sarima_models[parameter]
                    forecast = model_fit.get_forecast(steps=forecast_steps).predicted_mean
                    forecast_value = float(forecast.values[-1])  # Convert to Python float
                    forecast_results[parameter] = forecast_value
                except Exception as e:
                    return jsonify({'status': 'Error', 'message': f'Error in SARIMA for {parameter} at {station_name}: {e}'})
            
            # Convert the results to a DataFrame with parameters as columns and target date as a column
            forecast_df = pd.DataFrame([forecast_results])
            # Combine SARIMA forecast with website input
            combined_df = pd.concat([web_df, forecast_df], axis=1)

            # Standardize the combined input
            combined_scaled = scaler.transform(combined_df)

            # Make prediction using XGBoost model
            prediction = xgb_model.predict(combined_scaled)
            combined_df['Phytoplankton (cells/ml)'] = [float(p) for p in prediction]  # Convert to Python float
            
            # Format the date as 'MM/DD/YYYY'
            formatted_date = target_date.strftime('%m/%d/%Y')
            
            # Create a new DataFrame with the Date column
            date_df = pd.DataFrame({'Date': [formatted_date]})
            station_df = pd.DataFrame({'Monitoring Stations': [station_name]})
            
            # Concatenate the date_df with combined_df along the columns
            combined_df = pd.concat([combined_df, date_df, station_df], axis=1)

            
            # Check if the file exists
            if os.path.exists(new_csv):
                # Load the existing dataset to ensure columns match
                existing_df = pd.read_csv(new_csv)
            
                # Create a combined key from 'Date' and 'Monitoring Stations' to identify duplicates
                combined_df['Key'] = combined_df['Date'] + '-' + combined_df['Monitoring Stations']
                existing_df['Key'] = existing_df['Date'] + '-' + existing_df['Monitoring Stations']
            
                # Filter out rows from combined_df where the key already exists in existing_df
                combined_df = combined_df[~combined_df['Key'].isin(existing_df['Key'])]
            
                # Drop the temporary 'Key' column after filtering
                combined_df = combined_df.drop(columns=['Key'])
            
                # Check if there are any rows left to append
                if not combined_df.empty:
                    # Append the new data to the CSV file
                    combined_df.to_csv(new_csv, mode='a', header=False, index=False)
                    print("New predictions appended to the file.")
                else:
                    print("No new dates and stations to append. All combinations are already predicted.")
            else:
                # If the file does not exist, save the combined_df as a new CSV
                combined_df.to_csv(new_csv, mode='w', header=True, index=False)
                print("File created and saved with new predictions.")

            
            forecast_df_dict = forecast_df.to_dict(orient='records')[0]
            results[station_name] = {
                'prediction': [float(p) for p in prediction],
                'forecast': {k: float(v) for k, v in forecast_df_dict.items()}
            }
        
        return jsonify({
            'status': 'Prediction made and saved successfully',
            'results': results
        })

    except Exception as e:
        return jsonify({'status': 'Error', 'message': str(e)}), 500


@app.route('/model_testing', methods=['POST'])
def model_testing():
    data = request.get_json()
    df = pd.DataFrame(data)

    # Standardize the input
    df_scaled = scaler.transform(df)

    # Make prediction
    prediction = xgb_model.predict(df_scaled)
   
    return jsonify({'status': 'Prediction made and saved successfully', 'prediction': prediction.tolist()})


@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    try:
        # Get the dataset name from the request
        data = request.get_json()
        dataset_name = data.get('dataset')
        
        # Construct the full path to the dataset
        dataset_path = os.path.join(app.root_path, dataset_name)
        
        # Check if the file exists
        if not os.path.exists(dataset_path):
            return jsonify({"error": f"Dataset file '{dataset_name}' not found."})
        
        # Load the dataset
        merged_df = pd.read_csv(dataset_path)
        merged_df = merged_df.dropna()
    
        # Select features and target
        features = ['Temperature', 'Humidity', 'Wind Speed', 'pH (units)', 'Ammonia (mg/L)', 'Inorganic Phosphate (mg/L)', 'BOD (mg/l)', 'Total coliforms (MPN/100ml)']
        target = 'Phytoplankton (cells/ml)'
    
        # Perform train/test split
        X = merged_df[features]
        y = merged_df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
        # Train XGBoost model
        xgb_model = XGBRegressor()
        xgb_model.fit(X_train_scaled, y_train)
        y_pred_xgb = xgb_model.predict(X_test_scaled)
    
        # Calculate metrics
        mse_xgb = mean_squared_error(y_test, y_pred_xgb)
        mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
        r2_xgb = r2_score(y_test, y_pred_xgb)
    
        # Return the metrics to the PHP website
        return jsonify({'mse': mse_xgb, 'mae': mae_xgb, 'r2': r2_xgb})
    
    except Exception as e:
        return jsonify({"error": str(e)})   


@app.route('/export_model', methods=['POST'])
def export_model():
    try:
        # Get the dataset name from the request
        data = request.get_json()
        dataset_name = data.get('dataset')
        
        # Construct the full path to the dataset
        dataset_path = os.path.join(app.root_path, dataset_name)
        
        # Check if the file exists
        if not os.path.exists(dataset_path):
            return jsonify({"error": f"Dataset file '{dataset_name}' not found."})
        
        # Load the dataset
        merged_df = pd.read_csv(dataset_path)
        merged_df = merged_df.dropna()
    
        # Select features and target
        features = ['Temperature', 'Humidity', 'Wind Speed', 'pH (units)', 'Ammonia (mg/L)', 'Inorganic Phosphate (mg/L)', 'BOD (mg/l)', 'Total coliforms (MPN/100ml)']
        target = 'Phytoplankton (cells/ml)'
    
        # Perform train/test split
        X = merged_df[features]
        y = merged_df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
        # Train XGBoost model
        xgb_model = XGBRegressor()
        xgb_model.fit(X_train_scaled, y_train)
        y_pred_xgb = xgb_model.predict(X_test_scaled)
    
        # Calculate metrics
        mse_xgb = mean_squared_error(y_test, y_pred_xgb)
        mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
        r2_xgb = r2_score(y_test, y_pred_xgb)
        # Save the trained model to a file using pickle
        with open('xgb_model2.pkl', 'wb') as file:
            pickle.dump(xgb_model, file)
        
        print("Model saved to 'xgb_model2.pkl'")

        with open('xgb_model2.pkl', 'rb') as file:
            loaded_svr_model = pickle.load(file)

        with open('xgb_scaler2.pkl', 'wb') as file:
            pickle.dump(scaler, file)
    
        # Return the metrics to the PHP website
        return jsonify({'update': 'Model Export Success'})
    
    except Exception as e:
        return jsonify({"error": str(e)}) 
        



if __name__ == '__main__':
    app.run(debug=True)
