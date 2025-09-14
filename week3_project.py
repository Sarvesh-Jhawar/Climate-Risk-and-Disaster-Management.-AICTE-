# week3_project.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Define the Problem & Load Data
# The goal is to predict the disaster 'Category_title' based on location and time features.
# This is a multi-class classification problem.

print("Loading processed data...")
try:
    df = pd.read_csv('processed_data.csv')
except FileNotFoundError:
    print("Error: 'processed_data.csv' not found. Please run the data processing script/notebook first.")
    exit()

# Display the first few rows of the cleaned data
print("Cleaned Data Head:")
print(df.head())
print("\n")

# 2. Prepare the Data for Modeling

# a. Separate features (X) and target (y)
# Features: Longitude, Latitude, Month, DayOfWeek
# Target: Category_title
X = df[['Longitude', 'Latitude', 'Month', 'DayOfWeek']]
y = df['Category_title']

# b. Encode the target variable
# Machine learning models require numerical input. 'Category_title' is a string.
# We use LabelEncoder to convert each category string into a unique number.
print("Encoding target variable 'Category_title'...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the fitted LabelEncoder
joblib.dump(label_encoder, 'label_encoder.joblib')
print("LabelEncoder saved to label_encoder.joblib")

# We can see the mapping from categories to numbers
print("Label Encoder Classes:", list(label_encoder.classes_))
print("\n")

# c. Split data into training and testing sets
# 80% for training, 20% for testing. random_state ensures reproducibility.
# stratify=y_encoded ensures the proportion of categories is the same in train and test sets.
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print("\n")

# 3. Train a Machine Learning Model

# We'll use a RandomForestClassifier, which is powerful and works well for this kind of task.
print("Training the Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Train the model on the training data
model.fit(X_train, y_train)
print("Model training complete.")
print("\n")

# 4. Evaluate the Model
print("Evaluating the model on the test set...")
# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("\n")

# Get a detailed classification report
# This shows precision, recall, and f1-score for each category.
print("Classification Report:")
# We use the original string labels for better readability in the report
target_names = label_encoder.classes_
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# 5. Save the Trained Model
print("\nSaving the trained model to random_forest_model.joblib...")
joblib.dump(model, 'random_forest_model.joblib')
print("Model saved successfully.")