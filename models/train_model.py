



# train_model.py
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

# # Define feature names
# feature_names = ['Rainfall(mm)', 'River Level(m)']

# # Split features and labels
# X = pd.DataFrame(data[feature_names])
# y = data['Flood Occurrence']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Create model directory if it doesn't exist
# model_dir = os.path.join(project_dir, 'models')
# os.makedirs(model_dir, exist_ok=True)

# # Save the model and feature names
# model_data = {
#     'model': model,
#     'feature_names': feature_names
# }
# model_path = os.path.join(model_dir, 'flood_model.pkl')
# joblib.dump(model_data, model_path)
# print(f"Model saved to {model_path}")

# # Optional: Print model accuracy
# train_accuracy = model.score(X_train, y_train)
# test_accuracy = model.score(X_test, y_test)
# print(f"Training accuracy: {train_accuracy:.2f}")
# print(f"Testing accuracy: {test_accuracy:.2f}")

# train_model.py
# train_model.py
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# import joblib
# import os

# # Get the current script's directory and construct absolute paths
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_dir = os.path.dirname(current_dir)

# # Load cleaned data using absolute path
# data_path = os.path.join(project_dir, 'data', 'processed', 'flood_data_cleaned.csv')
# data = pd.read_csv(data_path)

# # Define feature names
# feature_names = ['Rainfall(mm)', 'River Level(m)']

# # Split features and labels
# X = data[feature_names]
# y = data['Flood Occurrence']

# # Scale the features to ensure proper weight distribution
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Train the model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Create model directory if it doesn't exist
# model_dir = os.path.join(project_dir, 'models')
# os.makedirs(model_dir, exist_ok=True)

# # Save the model, feature names, and scaler
# model_data = {
#     'model': model,
#     'feature_names': feature_names,
#     'scaler': scaler
# }
# model_path = os.path.join(model_dir, 'flood_model.pkl')
# joblib.dump(model_data, model_path)
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
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
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
X = data[feature_names]
y = data['Flood Occurrence']

# Analyze class distribution
print(y.value_counts())

# Handle minority class with very few samples
minority_class = y.value_counts().idxmin()
minority_data = data[data['Flood Occurrence'] == minority_class]

# Duplicate minority class samples to ensure at least k_neighbors + 1
data = pd.concat([data, minority_data], ignore_index=True)

# Re-define features and labels
X = data[feature_names]
y = data['Flood Occurrence']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to handle imbalance
smote = SMOTE(k_neighbors=1, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=1)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create model directory if it doesn't exist
model_dir = os.path.join(project_dir, 'models')
os.makedirs(model_dir, exist_ok=True)

# Save the model, feature names, and scaler
model_data = {
    'model': model,
    'feature_names': feature_names,
    'scaler': scaler
}
model_path = os.path.join(model_dir, 'flood_model.pkl')
joblib.dump(model_data, model_path)
print(f"Model saved to {model_path}")

# Optional: Print model accuracy
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f"Training accuracy: {train_accuracy:.2f}")
print(f"Testing accuracy: {test_accuracy:.2f}")
