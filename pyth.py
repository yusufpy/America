#import the libraries
import pickle
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
with open('scholarship333.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize LabelEncoder
le = LabelEncoder()

# List of columns that need encoding
columns_to_encode = ['University', 'Degree Program', 'Extracurricular Activities']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.get_json(force=True)
        
        # Convert data to DataFrame
        df = pd.DataFrame(data, index=[0])
        
        # Encode categorical variables
        for col in columns_to_encode:
            df[col] = le.fit_transform(df[col])
        
        # Make prediction
        try:
            prediction = model.predict(df)
            return jsonify({'prediction': prediction[0]})
        except AttributeError as e:
            if 'monotonic_cst' in str(e):
                return jsonify({'error': 'Model version mismatch. Please retrain the model with the current scikit-learn version.'})
            else:
                raise
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)