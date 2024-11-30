# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# import joblib

# # Load cleaned data
# data = pd.read_csv('../data/processed/flood_data_cleaned.csv')

# # Split features and labels
# X = data[['Rainfall(mm)', 'River Level(m)']]
# y = data['Flood Occurrence']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Save the model
# joblib.dump(model, 'flood_model.pkl')
# print("Model saved to flood_model.pkl")

#############################################################################

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# import joblib
# import os

# # Get the current script's directory and construct absolute paths
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_dir = os.path.dirname(current_dir)

# # Load cleaned data using absolute path
# data_path = os.path.join(project_dir, 'data', 'processed', 'flood_data_cleaned.csv')
# data = pd.read_csv(data_path)

# # Split features and labels
# X = data[['Rainfall(mm)', 'River Level(m)']]
# y = data['Flood Occurrence']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Save the model in the models directory
# model_path = os.path.join(current_dir, 'flood_model.pkl')
# joblib.dump(model, model_path)
# print(f"Model saved to {model_path}")

# # Optional: Print model accuracy
# train_accuracy = model.score(X_train, y_train)
# test_accuracy = model.score(X_test, y_test)
# print(f"Training accuracy: {train_accuracy:.2f}")
# print(f"Testing accuracy: {test_accuracy:.2f}")



# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Get the current script's directory and construct absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)

# Load cleaned data using absolute path
data_path = os.path.join(project_dir, 'data', 'processed', 'flood_data_cleaned.csv')
data = pd.read_csv(data_path)

# Define feature names
feature_names = ['Rainfall(mm)', 'River Level(m)']

# Split features and labels
X = pd.DataFrame(data[feature_names])
y = data['Flood Occurrence']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create model directory if it doesn't exist
model_dir = os.path.join(project_dir, 'models')
os.makedirs(model_dir, exist_ok=True)

# Save the model and feature names
model_data = {
    'model': model,
    'feature_names': feature_names
}
model_path = os.path.join(model_dir, 'flood_model.pkl')
joblib.dump(model_data, model_path)
print(f"Model saved to {model_path}")

# Optional: Print model accuracy
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f"Training accuracy: {train_accuracy:.2f}")
print(f"Testing accuracy: {test_accuracy:.2f}")