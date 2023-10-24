# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (replace 'data/customer_booking.csv' with your actual file path)
df = pd.read_csv("C:/Users/satab/PycharmProjects/pythonProject/data/customer_booking.csv", encoding="ISO-8859-1")

# Define the target variable
target_column = "booking_complete"
y = df[target_column]

# Define the feature columns
categorical_columns = ["sales_channel", "trip_type", "flight_day", "booking_origin"]
numerical_columns = ["num_passengers", "purchase_lead", "length_of_stay", "flight_hour", "wants_extra_baggage", "wants_preferred_seat", "wants_in_flight_meals", "flight_duration"]

# Select features
X = df[numerical_columns]

# Perform one-hot encoding for categorical variables
X_categorical = pd.get_dummies(df[categorical_columns], columns=categorical_columns)

# Combine numerical and one-hot encoded categorical features
X = pd.concat([X, X_categorical], axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the evaluation results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

# Print the feature names and their mapping (e.g., "sales_channel_Online")
feature_names = X.columns
print("Feature Names:")
for feature in feature_names:
    print(feature)

# Save the trained model to a file (change 'model_filename' to your preferred name)
import joblib
model_filename = "british_airways.pkl"
joblib.dump(clf, model_filename)
