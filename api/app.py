# from flask import Flask, request, jsonify
# import joblib
# import os
# from utils.preprocessing import preprocess_data

# app = Flask(__name__)

# # Load the model
# model_path = os.path.join('../models', 'flood_model.pkl')
# model = joblib.load(model_path)

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     rainfall = data.get('rainfall')
#     river_level = data.get('river_level')
    
#     # Preprocess input
#     input_data = preprocess_data(rainfall, river_level)
#     prediction = model.predict([input_data])[0]

#     result = "High Risk" if prediction == 1 else "Low Risk"
#     return jsonify({'Flood Risk': result})

# if __name__ == '__main__':
#     app.run(debug=True)

#####################################################

# from flask import Flask, request, jsonify
# import joblib
# import os
# from utils.preprocessing import preprocess_data

# app = Flask(__name__)

# # Get the absolute paths
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_dir = os.path.dirname(current_dir)
# model_path = os.path.join(project_dir, 'models', 'flood_model.pkl')

# # Load the model with error handling
# try:
#     model = joblib.load(model_path)
#     print(f"Model loaded successfully from {model_path}")
# except FileNotFoundError:
#     print(f"Error: Model file not found at {model_path}")
#     print("Please ensure you have trained the model first by running train_model.py")
#     raise
# except Exception as e:
#     print(f"Error loading model: {str(e)}")
#     raise

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.json
#         if not data:
#             return jsonify({'error': 'No input data provided'}), 400

#         rainfall = data.get('rainfall')
#         river_level = data.get('river_level')

#         if rainfall is None or river_level is None:
#             return jsonify({'error': 'Missing required parameters: rainfall and river_level'}), 400

#         # Preprocess input
#         input_data = preprocess_data(rainfall, river_level)
#         prediction = model.predict([input_data])[0]
#         probability = model.predict_proba([input_data])[0][1]

#         result = {
#             'Flood Risk': "High Risk" if prediction == 1 else "Low Risk",
#             'Probability': float(probability),
#             'Status': 'success'
#         }
#         return jsonify(result)

#     except Exception as e:
#         return jsonify({'error': str(e), 'status': 'error'}), 500

# @app.route('/health', methods=['GET'])
# def health_check():
#     return jsonify({'status': 'healthy'})

# if __name__ == '__main__':
#     app.run(debug=True)


######################################################################3

# from flask import Flask, request, jsonify
# import joblib
# import os
# import pandas as pd

# app = Flask(__name__)

# # Get the absolute paths
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_dir = os.path.dirname(current_dir)
# model_path = os.path.join(project_dir, 'models', 'flood_model.pkl')

# # Load the model with error handling
# try:
#     model_data = joblib.load(model_path)
#     model = model_data['model']
#     feature_names = model_data['feature_names']
#     print(f"Model loaded successfully from {model_path}")
# except FileNotFoundError:
#     print(f"Error: Model file not found at {model_path}")
#     print("Please ensure you have trained the model first by running train_model.py")
#     raise
# except Exception as e:
#     print(f"Error loading model: {str(e)}")
#     raise

# def preprocess_data(rainfall, river_level):
#     """Preprocess input data to match training data format"""
#     return pd.DataFrame([[rainfall, river_level]], columns=feature_names)

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.json
#         if not data:
#             return jsonify({'error': 'No input data provided'}), 400

#         rainfall = data.get('rainfall')
#         river_level = data.get('river_level')

#         if rainfall is None or river_level is None:
#             return jsonify({'error': 'Missing required parameters: rainfall and river_level'}), 400

#         # Preprocess input
#         input_data = preprocess_data(rainfall, river_level)
#         prediction = model.predict(input_data)[0]
#         probabilities = model.predict_proba(input_data)[0]
        
#         # Get the probability of flood (assuming 1 is flood, 0 is no flood)
#         flood_probability = probabilities[1] if prediction == 1 else probabilities[0]

#         result = {
#             'Flood Risk': "High Risk" if prediction == 1 else "Low Risk",
#             'Probability': float(flood_probability),
#             'Raw Probabilities': {
#                 'No Flood': float(probabilities[0]),
#                 'Flood': float(probabilities[1])
#             },
#             'Status': 'success'
#         }
#         return jsonify(result)

#     except Exception as e:
#         return jsonify({'error': str(e), 'status': 'error'}), 500




# @app.route('/health', methods=['GET'])
# def health_check():
#     return jsonify({'status': 'healthy'})

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd

app = Flask(__name__)

# Get the absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
model_path = os.path.join(project_dir, 'models', 'flood_model.pkl')

# Load the model with error handling
try:
    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_names = model_data['feature_names']
    print(f"Model loaded successfully from {model_path}")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    print("Please ensure you have trained the model first by running train_model.py")
    raise
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

def preprocess_data(rainfall, river_level):
    """Preprocess input data to match training data format."""
    return pd.DataFrame([[rainfall, river_level]], columns=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data
        data = request.json
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        rainfall = data.get('rainfall')
        river_level = data.get('river_level')

        if rainfall is None or river_level is None:
            return jsonify({'error': 'Missing required parameters: rainfall and river_level'}), 400

        # Preprocess input
        input_data = preprocess_data(rainfall, river_level)
        print("Input data:", input_data)  # Debugging input data
        print("Input data shape:", input_data.shape)

        # Predict
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)
        print("Probabilities shape:", probabilities.shape)  # Debugging probabilities
        print("Probabilities:", probabilities)

        if probabilities.shape[1] == 1:
            # Handle single-class probabilities
            flood_probability = probabilities[0][0]
            result = {
                'Flood Risk': "Low Risk",  # Default assumption
                'Probability': float(flood_probability),
                'Raw Probabilities': {
                    'No Flood': float(flood_probability),
                    'Flood': 0.0
                },
                'Status': 'success'
            }
        else:
            # Handle binary classification probabilities
            flood_probability = probabilities[0][1] if prediction == 1 else probabilities[0][0]
            result = {
                'Flood Risk': "High Risk" if prediction == 1 else "Low Risk",
                'Probability': float(flood_probability),
                'Raw Probabilities': {
                    'No Flood': float(probabilities[0][0]),
                    'Flood': float(probabilities[0][1])
                },
                'Status': 'success'
            }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True)
