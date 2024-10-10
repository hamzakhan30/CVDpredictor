import requests

# Sample test data with all expected features
data = [
    {
        "Age": 65,
        "Gender": "Male",
        "Systolic BP": 140,
        "Diastolic BP": 80,
        "Cholesterol": 220,
        "LDL": 130,
        "HDL": 45,
        "BMI": 30,
        "Diabetes": 1,
        "Family History": 1,  
        "ECG": 0,
        "Stress Levels": 2,
        "Alcohol Consumption": 2,
        "Previous Seizure/Events": 1,
        "Smoking Status": 1,
        "Physical Activity": 1,
        "Chest Pain": 1,
        "Medications": "Diuretics",
        "Food": "Non-vegetarian"
    }
]

try:
    response = requests.post('http://127.0.0.1:5000/predict', json=data)
    response.raise_for_status()  # This will raise an error for any HTTP error response
    predictions = response.json()
    print("Predictions:", predictions)
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")