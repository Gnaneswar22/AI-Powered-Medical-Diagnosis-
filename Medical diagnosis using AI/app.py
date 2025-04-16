from flask import Flask, render_template, request, redirect, url_for
import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # Add this import

app = Flask(__name__)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths configuration
MODEL_PATHS = {
    'diabetes': os.path.join(BASE_DIR, 'Models', 'diabetes_model.sav'),
    'heart': os.path.join(BASE_DIR, 'Models', 'heart_disease_model.sav'),
    'parkinsons': os.path.join(BASE_DIR, 'Models', 'parkinsons_model.sav'),
    'lung': os.path.join(BASE_DIR, 'Models', 'lungs_disease_model.sav'),
    'thyroid': os.path.join(BASE_DIR, 'Models', 'Thyroid_model.sav')
}

# Feature configurations remain the same
FEATURE_CONFIGS = {
    'diabetes': {
        'features': [
            'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
            'insulin', 'bmi', 'diabetes_pedigree', 'age'
        ]
    },
    'heart': {
        'features': [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
    },
    'parkinsons': {
        'features': [
            'fo', 'fhi', 'flo', 'Jitter_percent', 'Jitter_Abs',
            'RAP', 'PPQ', 'DDP', 'Shimmer', 'Shimmer_dB',
            'APQ3', 'APQ5', 'APQ', 'DDA', 'NHR', 'HNR',
            'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
        ]
    },
    'lung': {
        'features': [
            'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
            'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY',
            'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH',
            'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'
        ]
    },
    'thyroid': {
        'features': [
            'age', 'sex', 'on_thyroxine', 'tsh', 't3_measured', 't3', 'tt4'
        ]
    }
}

def make_prediction(model, features):
    """Make prediction and return confidence score"""
    try:
        prediction = model.predict(features)[0]
        
        # Try to get probability if model supports it
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)[0]
                confidence = proba.max() * 100
            else:
                confidence = 85 if prediction == 1 else 15
        except:
            confidence = 85 if prediction == 1 else 15
            
        return prediction, confidence
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return 0, 0

def load_models():
    """Load all models."""
    loaded_models = {}
    for disease, path in MODEL_PATHS.items():
        try:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    loaded_models[disease] = pickle.load(f)
                print(f"✓ Successfully loaded {disease} model")
            else:
                print(f"✗ Model not found: {path}")
                loaded_models[disease] = None
        except Exception as e:
            print(f"✗ Error loading {disease} model: {str(e)}")
            loaded_models[disease] = None
    return loaded_models

# Initialize models
models = load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diagnose')
def diagnose():
    disease_type = request.args.get('type', '')
    return render_template('diagnose.html', disease_type=disease_type)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            disease_type = request.form.get('disease_type')
            print(f"\nPrediction request for: {disease_type}")

            if not disease_type:
                raise ValueError("No disease type specified")

            if disease_type not in models:
                raise ValueError(f"Unknown disease type: {disease_type}")

            model = models[disease_type]
            if model is None:
                raise ValueError(f"Model not loaded for: {disease_type}")

            # Get features
            features = get_features(request.form, disease_type)
            features_array = np.array(features).reshape(1, -1)

            # Make prediction using the new function
            prediction, confidence = make_prediction(model, features_array)

            results = {
                'disease_type': disease_type.replace('_', ' ').title(),
                'prediction': int(prediction),
                'probability': float(confidence),
                'features': {k: v for k, v in request.form.items() if k != 'disease_type'},
                'recommendations': get_recommendations(disease_type, prediction),
                'risk_level': get_risk_level(confidence)
            }

            return render_template('result.html', results=results)

        except Exception as e:
            error_msg = str(e)
            print(f"Error during prediction: {error_msg}")
            return render_template('result.html', error=error_msg)

    return redirect(url_for('index'))

def get_features(form_data, disease_type):
    """Extract features from form data."""
    if disease_type not in FEATURE_CONFIGS:
        raise ValueError(f"Unknown disease type: {disease_type}")
    
    features = []
    missing_features = []
    
    for feature in FEATURE_CONFIGS[disease_type]['features']:
        if feature not in form_data:
            missing_features.append(feature)
        else:
            try:
                features.append(float(form_data[feature]))
            except ValueError:
                raise ValueError(f"Invalid value for {feature}: {form_data[feature]}")
    
    if missing_features:
        raise ValueError(f"Missing required features: {', '.join(missing_features)}")
    
    return features

def get_risk_level(probability):
    """Determine risk level based on probability."""
    if probability < 30:
        return {'level': 'Low Risk', 'color': 'success'}
    elif probability < 70:
        return {'level': 'Moderate Risk', 'color': 'warning'}
    else:
        return {'level': 'High Risk', 'color': 'danger'}

def get_recommendations(disease_type, prediction):
    """Get recommendations based on disease type and prediction."""
    if prediction == 1:
        recommendations = {
            'diabetes': [
                "Schedule appointment with endocrinologist",
                "Monitor blood sugar levels regularly",
                "Follow balanced diet plan",
                "Exercise regularly",
                "Take prescribed medications"
            ],
            'heart': [
                "Consult cardiologist immediately",
                "Monitor blood pressure daily",
                "Follow heart-healthy diet",
                "Regular cardiovascular exercise",
                "Stress management techniques"
            ],
            'parkinsons': [
                "Consult neurologist",
                "Begin physical therapy",
                "Consider occupational therapy",
                "Join support groups",
                "Regular exercise program"
            ],
            'lung': [
                "Immediate consultation with pulmonologist",
                "Complete smoking cessation",
                "Avoid secondhand smoke exposure",
                "Regular chest examinations",
                "Follow prescribed treatment plan"
            ],
            'thyroid': [
                "Consult endocrinologist",
                "Regular thyroid function tests",
                "Monitor medication dosage",
                "Maintain healthy diet",
                "Regular exercise routine"
            ]
        }
        return recommendations.get(disease_type, [
            "Consult healthcare provider immediately",
            "Follow up with specialist",
            "Maintain healthy lifestyle",
            "Regular health monitoring",
            "Follow prescribed treatment plan"
        ])
    else:
        preventive_recommendations = {
            'diabetes': [
                "Regular blood sugar screening",
                "Maintain healthy weight",
                "Regular exercise",
                "Balanced diet",
                "Annual health checkups"
            ],
            'heart': [
                "Regular blood pressure checks",
                "Heart-healthy diet",
                "Regular exercise",
                "Stress management",
                "Annual cardiovascular screening"
            ],
            'parkinsons': [
                "Regular neurological checkups",
                "Exercise regularly",
                "Balanced diet",
                "Monitor symptoms",
                "Annual health screening"
            ],
            'lung': [
                "Avoid smoking",
                "Regular exercise",
                "Clean air environment",
                "Regular health checkups",
                "Maintain good respiratory hygiene"
            ],
            'thyroid': [
                "Regular thyroid screening",
                "Balanced iodine intake",
                "Healthy lifestyle",
                "Monitor symptoms",
                "Annual health checkups"
            ]
        }
        return preventive_recommendations.get(disease_type, [
            "Maintain healthy lifestyle",
            "Regular health checkups",
            "Balanced diet",
            "Regular exercise",
            "Stress management"
        ])

if __name__ == '__main__':
    app.run(debug=True)
