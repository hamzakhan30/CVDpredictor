from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

# Load the pre-trained model and preprocessor
model = joblib.load('best_ensemble_model.joblib')
preprocessor = joblib.load('preprocessor.joblib')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the form
        data = request.json  # Expecting a list of data dictionaries

        # Convert to DataFrame (as expected by the model)
        df = pd.DataFrame(data)

        # Perform any necessary preprocessing
        X = preprocessor.transform(df)

        # Make the prediction
        predictions = model.predict(X)

        # Return the prediction as a JSON response
        return jsonify(predictions.tolist())

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
