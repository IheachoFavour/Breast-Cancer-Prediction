from flask import Flask, request, render_template, jsonify
from flask_sqlalchemy import SQLAlchemy
import joblib
import json
import numpy as np

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Model for storing user data
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    features = db.Column(db.String(200), nullable=False)
    result = db.Column(db.String(50), nullable=False)

# Load the model
model = joblib.load('logistic_regression_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Parse the incoming JSON data
    data = request.get_json()

    name = data['patient_name']
    features = [
        float(data['feature1']),
        float(data['feature2']),
        float(data['feature3']),
        float(data['feature4']),
        float(data['feature5']),
        float(data['feature6']),
        float(data['feature7']),
        float(data['feature8']),
        float(data['feature9']),
        float(data['feature10'])
    ]

    # Make prediction
    prediction = model.predict(np.array(features).reshape(1, -1))
    result = 'Malignant' if prediction[0] == 1 else 'Benign'

    # Store the result in the database
    new_prediction = Prediction(name=name, features=str(features), result=result)
    db.session.add(new_prediction)
    db.session.commit()

    # Return JSON response
    return jsonify({'prediction': result})


if __name__ == '__main__':
    # Ensure db.create_all is called within app context
    with app.app_context():
        db.create_all()
    app.run(debug=True)
