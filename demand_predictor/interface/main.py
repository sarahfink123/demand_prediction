import os
import sys
import pandas as pd

# Add the root directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'raw_data', 'hotel_bookings_raw.csv')

# Load data
df = pd.read_csv(data_path)
########################CLEANING DATA#########################################################################
#Drop columns
df = df.drop(columns=['arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month', \
    'children', 'babies', 'previous_cancellations', 'previous_bookings_not_canceled', 'reserved_room_type',\
    'assigned_room_type', 'booking_changes', 'deposit_type', 'agent', 'days_in_waiting_list', 'customer_type',\
    'required_car_parking_spaces', 'total_of_special_requests', 'reservation_status_date', 'MO_YR'])

#Drop duplicates
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.duplicated().sum()

#Drop NaN's
df.dropna(inplace=True)

#Further Cleaning of NaN's
#Drop undefined

df = df[df["meal"] != "Undefined"]
df = df[df["market_segment"] != "Undefined"]
df = df[df["distribution_channel"] != "Undefined"]
###############################ENCODING#############################################################
#Change months to number
# Create a mapping of month names to numbers
month_mapping = {
    'January': 1,
    'February': 2,
    'March': 3,
    'April': 4,
    'May': 5,
    'June': 6,
    'July': 7,
    'August': 8,
    'September': 9,
    'October': 10,
    'November': 11,
    'December': 12
}
#Strip spaces
df['arrival_date_month'] = df['arrival_date_month'].str.strip()
# Replace month names with numbers
df['arrival_date_month'] = df['arrival_date_month'].map(month_mapping)


#Change hotel to binary
hotel_mapping = {
    'City Hotel': 1,
    'Resort Hotel': 0
}
#Strip spaces
df['hotel'] = df['hotel'].str.strip()
# Replace month names with numbers
df['hotel'] = df['hotel'].map(hotel_mapping)
#---------------------------------------------------------------------------------------------------------------
# Encode country
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['country'] = label_encoder.fit_transform(df['country'])
#---------------------------------------------------------------------------------------------------------------
# One Hot Encode meal, country, market_segment, distribution_channel, reservation_status -> categorical (3-5 categories, encoden)
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Columns to one-hot encode
columns_to_encode = ['meal', 'market_segment', 'distribution_channel', 'reservation_status']

# Instantiate the OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' avoids the dummy variable trap

# Fit and transform the data
encoded_data = ohe.fit_transform(df[columns_to_encode])

# Convert encoded data to DataFrame
encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(columns_to_encode))

# Concatenate the encoded columns with the original DataFrame
df_encoded = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# Drop the original columns that were encoded
df_encoded.drop(columns=columns_to_encode, inplace=True)
############################SCALING##################################################################
# Robust Scaler
from sklearn.preprocessing import RobustScaler

features_to_robust = ['lead_time', 'arrival_date_month', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'adr', 'FUEL_PRCS']
robust_scaler = RobustScaler()
robust_scaler.fit(df_encoded[features_to_robust])
df_encoded[features_to_robust] = robust_scaler.transform(df_encoded[features_to_robust])


# MinMax Scaler
from sklearn.preprocessing import MinMaxScaler

features_to_minmax = ['country', 'CPI_AVG', 'INFLATION', 'INFLATION_CHG', 'CSMR_SENT', 'UNRATE', 'INTRSRT', 'GDP', 'DIS_INC', 'CPI_HOTELS']
minmax_scaler = MinMaxScaler()
minmax_scaler.fit(df_encoded[features_to_minmax])
df_encoded[features_to_minmax] = minmax_scaler.transform(df_encoded[features_to_minmax])
############################MODEL######################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, mean_squared_error

target_variable = 'is_canceled'

# Assuming you have a DataFrame df_encoded and a target variable y
predictors = [
    'country', 'FUEL_PRCS', 'lead_time', 'adr', "arrival_date_month", 'stays_in_week_nights', 'INFLATION'
]
# X = df_encoded.drop(columns=[target_variable])
X = df_encoded[predictors]
y = df_encoded[target_variable]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# print(y_pred)

# Evaluate the model
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

metrics = {
    'accuracy': acc,
    'recall': rec,
    'f1_score': f1,
    'mean_squared_error': mse
}
params = {
    'model_type': 'RandomForestClassifier',
    'random_state': 42
}

##################SAVING################################################################################################
from demand_predictor.ml_logic.registry import save_model, save_results, load_model

# Save the model
save_model(model)

# Save the metrics
save_results(params=params, metrics=metrics)

###############LOADING MODEL############################################################################################
# Test loading the model
loaded_model = load_model()

# Check if the loaded model is the same as the saved model
if loaded_model:
    print("Model loaded successfully!")
    sample_data = X_test.iloc[0:1]  # Take a single sample for prediction
    print(sample_data)
    loaded_model_prediction = loaded_model.predict(sample_data)
    print("Loaded model prediction:", loaded_model_prediction)

    # Take another sample for prediction
    another_sample = X_test.iloc[1:2]  # Change the index as needed to select a different sample

    # Make a prediction using the loaded model
    another_loaded_model_prediction = loaded_model.predict(another_sample)
    print("Another loaded model prediction:", another_loaded_model_prediction)
else:
    print("Failed to load the model.")
