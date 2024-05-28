import os
import pandas as pd
from sklearn.model_selection import train_test_split


data_path = os.path.join('data', 'hotel_bookings_raw.csv')

# Load data
df = pd.read_csv(data_path)

# Define target variable
target_variable = 'is_canceled'

# Split data into features and target variable
X = df.drop([target_variable], axis=1)
y = df[target_variable]


# Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
