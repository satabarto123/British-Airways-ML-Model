# Step 2: Continue Exploration and Preparation

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("data/customer_booking.csv", encoding="ISO-8859-1")

# Continue data exploration and preparation:

# Explore the dataset with visualizations
# Example: Create a histogram of the "num_passengers" column
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x="num_passengers", bins=20)
plt.title("Histogram of Number of Passengers")
plt.xlabel("Number of Passengers")
plt.ylabel("Frequency")
plt.show()

# Example: Create a scatter plot between "purchase_lead" and "flight_duration"
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="purchase_lead", y="flight_duration")
plt.title("Scatter Plot: Purchase Lead Time vs. Flight Duration")
plt.xlabel("Purchase Lead Time (days)")
plt.ylabel("Flight Duration (hours)")
plt.show()

# Further data preparation (example steps):
# - Encoding categorical variables if needed
# - Scaling or normalizing numeric features
# - Handling outliers if necessary
# - Checking for class imbalances if this is a classification problem

# Example: Encoding categorical variables using one-hot encoding
df = pd.get_dummies(df, columns=["sales_channel", "trip_type", "route", "booking_origin"], drop_first=True)

# Example: Scaling numeric features using Min-Max scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[["num_passengers", "purchase_lead", "length_of_stay"]] = scaler.fit_transform(df[["num_passengers", "purchase_lead", "length_of_stay"]])

# Example: Checking for class imbalances (if it's a classification problem)
class_counts = df["booking_complete"].value_counts()
print("Class Imbalance Check:")
print(class_counts)

# Now, you can continue exploring and preparing the data as per your analysis requirements.

# After data preparation, you can proceed to Step 3: Train a Machine Learning Model.
