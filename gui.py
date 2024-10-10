import tkinter as tk
from tkinter import ttk, messagebox
import requests
import json

def submit_form():
    # Get input values from the form
    data = {
        "Age": int(age_entry.get()),
        "Gender": gender_combobox.get(),
        "Systolic BP": int(systolic_bp_entry.get()),
        "Diastolic BP": int(diastolic_bp_entry.get()),
        "Cholesterol": int(cholesterol_entry.get()),
        "LDL": int(ldl_entry.get()),
        "HDL": int(hdl_entry.get()),
        "BMI": float(bmi_entry.get()),
        "Diabetes": int(diabetes_entry.get()),
        "Family History": int(family_history_entry.get()),
        "ECG": int(ecg_entry.get()),
        "Stress Levels": int(stress_levels_entry.get()),
        "Alcohol Consumption": int(alcohol_consumption_entry.get()),
        "Previous Seizure/Events": int(previous_seizure_entry.get()),
        "Smoking Status": int(smoking_status_entry.get()),
        "Physical Activity": int(physical_activity_entry.get()),
        "Chest Pain": int(chest_pain_entry.get()),
        "Medications": medications_combobox.get(),
        "Food": food_combobox.get()
    }

    try:
        # Send request to the Flask API
        response = requests.post('http://127.0.0.1:5000/predict', json=[data])
        response.raise_for_status()

        # Get predictions
        predictions = response.json()

        # Display the result
        messagebox.showinfo("Prediction", f"Predicted Disease: {predictions[0]}")

    except requests.exceptions.RequestException as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Create the main window
root = tk.Tk()
root.title("Disease Prediction")

# Create form fields
fields = [
    ("Age", tk.Entry),
    ("Systolic BP", tk.Entry),
    ("Diastolic BP", tk.Entry),
    ("Cholesterol", tk.Entry),
    ("LDL", tk.Entry),
    ("HDL", tk.Entry),
    ("BMI", tk.Entry),
    ("Diabetes", tk.Entry),
    ("Family History", tk.Entry),
    ("ECG", tk.Entry),
    ("Stress Levels", tk.Entry),
    ("Alcohol Consumption", tk.Entry),
    ("Previous Seizure/Events", tk.Entry),
    ("Smoking Status", tk.Entry),
    ("Physical Activity", tk.Entry),
    ("Chest Pain", tk.Entry)
]

# Layout for entries and labels
entries = []
for idx, (label_text, field_type) in enumerate(fields):
    label = tk.Label(root, text=label_text)
    label.grid(row=idx, column=0)
    entry = field_type(root)
    entry.grid(row=idx, column=1)
    entries.append(entry)

# Assign each entry to a specific variable
(age_entry, systolic_bp_entry, diastolic_bp_entry, cholesterol_entry, ldl_entry, hdl_entry,
 bmi_entry, diabetes_entry, family_history_entry, ecg_entry, stress_levels_entry,
 alcohol_consumption_entry, previous_seizure_entry, smoking_status_entry,
 physical_activity_entry, chest_pain_entry) = entries

# Dropdowns for categorical fields
gender_label = tk.Label(root, text="Gender")
gender_label.grid(row=len(fields), column=0)
gender_combobox = ttk.Combobox(root, values=["Male", "Female"])
gender_combobox.grid(row=len(fields), column=1)

medications_label = tk.Label(root, text="Medications")
medications_label.grid(row=len(fields)+1, column=0)
medications_combobox = ttk.Combobox(root, values=["ACE inhibitors", "Beta-blockers", "Diuretics", "None"])
medications_combobox.grid(row=len(fields)+1, column=1)

food_label = tk.Label(root, text="Food")
food_label.grid(row=len(fields)+2, column=0)
food_combobox = ttk.Combobox(root, values=["Vegetarian", "Non-vegetarian"])
food_combobox.grid(row=len(fields)+2, column=1)

# Submit button
submit_button = tk.Button(root, text="Submit", command=submit_form)
submit_button.grid(row=len(fields)+3, columnspan=2)

# Run the application
root.mainloop()
