from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)

# Global variable to store patient data
patient_data = pd.DataFrame(columns=[
    'Patient Id', 'Level', 'Age', 'Gender',
    'Air Pollution', 'Alcohol use', 'Dust Allergy',
    'OccuPational Hazards', 'Genetic Risk', 'Chronic Lung Disease',
    'Balanced Diet', 'Obesity', 'Smoking',
    'Passive Smoker', 'Chest Pain', 'Coughing of Blood',
    'Fatigue', 'Weight Loss', 'Shortness of Breath',
    'Wheezing', 'Swallowing Difficulty', 'Clubbing of Finger Nails',
    'Frequent Cold', 'Dry Cough', 'Snoring'
])

# Route for displaying the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for Naive Bayes classification
@app.route('/naivebayes', methods=['GET', 'POST'])
def naivebayes():
    global patient_data
    result = ''
    
    if request.method == 'POST':
        # Capture the input values from the form
        patient_id = request.form['patient_id']
        age = int(request.form['age'])
        gender = request.form['gender']
        
        # Collect inputs for risk factors
        inputs = {
            'Air Pollution': float(request.form['air_pollution']),
            'Alcohol use': float(request.form['alcohol_use']),
            'Dust Allergy': float(request.form['dust_allergy']),
            'OccuPational Hazards': float(request.form['occupational_hazards']),
            'Genetic Risk': float(request.form['genetic_risk']),
            'Chronic Lung Disease': float(request.form['chronic_lung_disease']),
            'Balanced Diet': float(request.form['balanced_diet']),
            'Obesity': float(request.form['obesity']),
            'Smoking': float(request.form['smoking']),
            'Passive Smoker': float(request.form['passive_smoker']),
            'Chest Pain': float(request.form['chest_pain']),
            'Coughing of Blood': float(request.form['coughing_of_blood']),
            'Fatigue': float(request.form['fatigue']),
            'Weight Loss': float(request.form['weight_loss']),
            'Shortness of Breath': float(request.form['shortness_of_breath']),
            'Wheezing': float(request.form['wheezing']),
            'Swallowing Difficulty': float(request.form['swallowing_difficulty']),
            'Clubbing of Finger Nails': float(request.form['clubbing_of_finger_nails']),
            'Frequent Cold': float(request.form['frequent_cold']),
            'Dry Cough': float(request.form['dry_cough']),
            'Snoring': float(request.form['snoring'])
        }
        
        # Determine risk level based on input
        risk_level = determine_risk_level(inputs)

        # Append new patient data to the DataFrame
        new_patient_data = pd.DataFrame({
            'Patient Id': [patient_id],
            'Level': [risk_level],
            'Age': [age],
            'Gender': [gender],
            **{key: [value] for key, value in inputs.items()}
        })

        patient_data = pd.concat([patient_data, new_patient_data], ignore_index=True)

        # Display classification result
        result = f'Hasil klasifikasi untuk pasien {patient_id}: {risk_level}'

    return render_template('naivebayes.html', result=result)

# Determine risk level based on input data
def determine_risk_level(inputs):
    score = sum(1 for value in inputs.values() if value > 0.5)
    if score <= 3:
        return 'Low'
    elif score <= 6:
        return 'Medium'
    else:
        return 'High'

# Route for data preprocessing and display
@app.route('/preprocessing')
def preprocessing():
    # Load the dataset
    current_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Build the full path to your CSV file within the 'data' folder
    file_path = os.path.join(current_directory, 'data', 'cancer_patient_data_sets.csv')
    
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Process the dataset (missing values, outliers, normalization)
    numeric_cols = ['Air Pollution', 'Alcohol use', 'Dust Allergy',
                    'OccuPational Hazards', 'Genetic Risk',
                    'Chronic Lung Disease', 'Balanced Diet',
                    'Obesity', 'Smoking', 'Passive Smoker',
                    'Chest Pain', 'Coughing of Blood', 'Fatigue',
                    'Weight Loss', 'Shortness of Breath', 'Wheezing',
                    'Swallowing Difficulty', 'Clubbing of Finger Nails',
                    'Frequent Cold', 'Dry Cough', 'Snoring', 'Age']

    # Handle missing values
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Outlier detection and handling
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR)))

    outlier_columns = []
    for col in numeric_cols:
        outlier_values = df.loc[outliers[col], col]
        if not outlier_values.empty:
            outlier_columns.append(col)

    # Replace outliers with mean
    for col in outlier_columns:
        mean_value = df[col].mean()
        df[col] = df[col].astype(float)
        df.loc[outliers[col], col] = mean_value

    # Normalization using Min-Max Scaling (excluding 'Age')
    scaler = MinMaxScaler()
    numeric_cols_excluding_age = [col for col in numeric_cols if col != 'Age']
    df_numeric_normalized = pd.DataFrame(scaler.fit_transform(df[numeric_cols_excluding_age]),
                                         columns=numeric_cols_excluding_age)

    # Combine non-numeric columns and 'Age' back
    non_numeric_cols = ['Patient Id', 'Level']
    df_preprocessed = pd.concat([df[non_numeric_cols].reset_index(drop=True),
                                 df_numeric_normalized.reset_index(drop=True),
                                 df[['Age']].reset_index(drop=True)], axis=1)

    # Convert DataFrame to HTML table for display
    table_html = df_preprocessed.head().to_html(classes='table table-striped')

    return render_template('minmax.html', table_html=table_html)

@app.route('/decisiontree', methods=['GET', 'POST'])
def decisiontree():
    global patient_data
    result = ''
    
    if request.method == 'POST':
        # Capture the input values from the form
        patient_id = request.form['patient_id']
        age = int(request.form['age'])
        gender = request.form['gender']
        
        # Collect inputs for risk factors
        inputs = {
            'Air Pollution': float(request.form['air_pollution']),
            'Alcohol use': float(request.form['alcohol_use']),
            'Dust Allergy': float(request.form['dust_allergy']),
            'OccuPational Hazards': float(request.form['occupational_hazards']),
            'Genetic Risk': float(request.form['genetic_risk']),
            'Chronic Lung Disease': float(request.form['chronic_lung_disease']),
            'Balanced Diet': float(request.form['balanced_diet']),
            'Obesity': float(request.form['obesity']),
            'Smoking': float(request.form['smoking']),
            'Passive Smoker': float(request.form['passive_smoker']),
            'Chest Pain': float(request.form['chest_pain']),
            'Coughing of Blood': float(request.form['coughing_of_blood']),
            'Fatigue': float(request.form['fatigue']),
            'Weight Loss': float(request.form['weight_loss']),
            'Shortness of Breath': float(request.form['shortness_of_breath']),
            'Wheezing': float(request.form['wheezing']),
            'Swallowing Difficulty': float(request.form['swallowing_difficulty']),
            'Clubbing of Finger Nails': float(request.form['clubbing_of_finger_nails']),
            'Frequent Cold': float(request.form['frequent_cold']),
            'Dry Cough': float(request.form['dry_cough']),
            'Snoring': float(request.form['snoring'])
        }
        
        # Determine risk level based on input
        risk_level = determine_risk_level(inputs)

        # Append new patient data to the DataFrame
        new_patient_data = pd.DataFrame({
            'Patient Id': [patient_id],
            'Level': [risk_level],
            'Age': [age],
            'Gender': [gender],
            **{key: [value] for key, value in inputs.items()}
        })

        patient_data = pd.concat([patient_data, new_patient_data], ignore_index=True)

        # Display classification result
        result = f'Hasil klasifikasi untuk pasien {patient_id}: {risk_level}'

    return render_template('decisiontree.html', result=result)

# Determine risk level based on input data
def determine_risk_level(inputs):
    score = sum(1 for value in inputs.values() if value > 0.5)
    if score <= 3:
        return 'Low'
    elif score <= 6:
        return 'Medium'
    else:
        return 'High'

@app.route('/randomforest', methods=['GET', 'POST'])
def randomforest():
    global patient_data
    result = ''
    
    if request.method == 'POST':
        # Capture the input values from the form
        patient_id = request.form['patient_id']
        age = int(request.form['age'])
        gender = request.form['gender']
        
        # Collect inputs for risk factors
        inputs = {
            'Air Pollution': float(request.form['air_pollution']),
            'Alcohol use': float(request.form['alcohol_use']),
            'Dust Allergy': float(request.form['dust_allergy']),
            'OccuPational Hazards': float(request.form['occupational_hazards']),
            'Genetic Risk': float(request.form['genetic_risk']),
            'Chronic Lung Disease': float(request.form['chronic_lung_disease']),
            'Balanced Diet': float(request.form['balanced_diet']),
            'Obesity': float(request.form['obesity']),
            'Smoking': float(request.form['smoking']),
            'Passive Smoker': float(request.form['passive_smoker']),
            'Chest Pain': float(request.form['chest_pain']),
            'Coughing of Blood': float(request.form['coughing_of_blood']),
            'Fatigue': float(request.form['fatigue']),
            'Weight Loss': float(request.form['weight_loss']),
            'Shortness of Breath': float(request.form['shortness_of_breath']),
            'Wheezing': float(request.form['wheezing']),
            'Swallowing Difficulty': float(request.form['swallowing_difficulty']),
            'Clubbing of Finger Nails': float(request.form['clubbing_of_finger_nails']),
            'Frequent Cold': float(request.form['frequent_cold']),
            'Dry Cough': float(request.form['dry_cough']),
            'Snoring': float(request.form['snoring'])
        }
        
        # Determine risk level based on input
        risk_level = determine_risk_level(inputs)

        # Append new patient data to the DataFrame
        new_patient_data = pd.DataFrame({
            'Patient Id': [patient_id],
            'Level': [risk_level],
            'Age': [age],
            'Gender': [gender],
            **{key: [value] for key, value in inputs.items()}
        })

        patient_data = pd.concat([patient_data, new_patient_data], ignore_index=True)

        # Display classification result
        result = f'Hasil klasifikasi untuk pasien {patient_id}: {risk_level}'

    return render_template('randomforest.html', result=result)

# Determine risk level based on input data
def determine_risk_level(inputs):
    score = sum(1 for value in inputs.values() if value > 0.5)
    if score <= 3:
        return 'Low'
    elif score <= 6:
        return 'Medium'
    else:
        return 'High'


if __name__ == "__main__":
    app.run(debug=True)
