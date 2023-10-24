# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (if not already loaded)
data_file = "C:\\Users\\satab\\PycharmProjects\\pythonProject\\data\\customer_booking.csv"
df = pd.read_csv(data_file, encoding="ISO-8859-1")

# Check if the dataset contains enough samples for a train-test split
if len(df) < 2:
    print("Not enough samples in the dataset for a train-test split.")
    # Handle this case as needed

# Choose a target variable (in this case, "booking_complete")
y = df["booking_complete"]

# Select the features (exclude target and any unnecessary columns)
X = df.drop(columns=["booking_complete", "route", "booking_origin"])

# Perform one-hot encoding for categorical variables
X = pd.get_dummies(X, columns=["sales_channel", "trip_type", "flight_day"], drop_first=True)

# Check if the dataset contains enough samples for a train-test split
if len(X) < 2:
    print("Not enough samples in the dataset for a train-test split.")
    # Handle this case as needed

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Check if the training set has enough samples
if len(X_train) < 2:
    print("Not enough samples in the training set for model training.")
    # Handle this case as needed
else:
    # Choose a machine learning algorithm (Random Forest as an example)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the machine learning model using the training data
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
