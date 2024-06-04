import os
import sys
import pandas as pd
from demand_predictor.ml_logic.preprocessor import preprocess_features, preprocess_is_canceled,\
                                                    preprocess_is_canceled_X_pred, preprocess_is_country
# from ml_logic.model import initialize_model, train_model

# Add the root directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'raw_data', 'hotel_bookings_raw.csv')
print("data path:", os.path.dirname(__file__))
# Load data
df = pd.read_csv(data_path)
#########7 features Preprocessing#######################################################################

df_encoded = preprocess_is_country(df)

###########################MODEL######################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, mean_squared_error

# # target column
# target_variable = 'is_canceled'

# # 7 features
# predictors = [
#         'lead_time', 'total_stay', 'adr', 'INFLATION', 'FUEL_PRCS',
#         'country_ESP', 'country_FRA', 'country_GBR', 'country_PRT',
#         'arrival_date_month_sin', 'arrival_date_month_cos'
#     ]

# # target column
# target_variable = 'canceled'
# predictors = df.columns.drop(target_variable)

# # print("''''''''''''''''''predictors'''''''''''''''''''''''''''''")
# # print(predictors)
# X = df_encoded.drop(columns=[target_variable])
# y = df_encoded[target_variable]

"""
Index(['hotel', 'is_canceled', 'lead_time', 'adults', 'country',
       'is_repeated_guest', 'adr', 'CPI_AVG', 'INFLATION', 'INFLATION_CHG',
       'CSMR_SENT', 'UNRATE', 'INTRSRT', 'GDP', 'FUEL_PRCS', 'CPI_HOTELS',
       'US_GINI', 'DIS_INC', 'total_stay', 'arrival_date_month_sin',
       'arrival_date_month_cos', 'meal_FB', 'meal_HB', 'meal_SC',
       'market_segment_Complementary', 'market_segment_Corporate',
       'market_segment_Direct', 'market_segment_Groups',
       'market_segment_Offline TA/TO', 'market_segment_Online TA',
       'distribution_channel_Direct', 'distribution_channel_GDS',
       'distribution_channel_TA/TO', 'reservation_status_Check-Out',
       'reservation_status_No-Show'],
      dtype='object')
"""
target_variable = 'country'
predictors = ['arrival_date_month_sin', 'arrival_date_month_cos','adr', 'lead_time', 'total_stay',
              'adults', 'hotel', 'INFLATION']

X = df_encoded[predictors]

missing_columns = ['is_canceled', 'is_repeated_guest', 'CPI_AVG', 'INFLATION_CHG', 'CSMR_SENT', 'UNRATE',
                   'INTRSRT', 'GDP', 'FUEL_PRCS', 'CPI_HOTELS', 'US_GINI', 'DIS_INC',
                   'total_stay', 'arrival_date_month_sin', 'arrival_date_month_cos',
                   'meal_FB', 'meal_HB', 'meal_SC', 'market_segment_Complementary',
                   'market_segment_Corporate', 'market_segment_Direct',
                   'market_segment_Groups', 'market_segment_Offline TA/TO',
                   'market_segment_Online TA', 'distribution_channel_Direct',
                   'distribution_channel_GDS', 'distribution_channel_TA/TO',
                   'reservation_status_Check-Out', 'reservation_status_No-Show']

column_median = df_encoded[missing_columns].median()

# Create a DataFrame with the column means (single row)
column_median_df = pd.DataFrame([column_median])

# Define the relative path where you want to save the file
current_dir = os.path.dirname(__file__)
file_name = 'column_medians.csv'
file_path = os.path.join(current_dir, file_name)

# Save the DataFrame to a CSV file
column_median_df.to_csv(file_path, index=False)

# Concatenate the predictor DataFrame with the column means DataFrame
# Align by columns, filling missing values in X with the column means
X = pd.concat([column_median_df, X], ignore_index=True).ffill().iloc[1:]

# Reorder columns to match the required order
required_order = ['hotel', 'is_canceled', 'lead_time', 'adults', 'is_repeated_guest',
                  'adr', 'CPI_AVG', 'INFLATION', 'INFLATION_CHG', 'CSMR_SENT', 'UNRATE',
                  'INTRSRT', 'GDP', 'FUEL_PRCS', 'CPI_HOTELS', 'US_GINI', 'DIS_INC',
                  'total_stay', 'arrival_date_month_sin', 'arrival_date_month_cos',
                  'meal_FB', 'meal_HB', 'meal_SC', 'market_segment_Complementary',
                  'market_segment_Corporate', 'market_segment_Direct',
                  'market_segment_Groups', 'market_segment_Offline TA/TO',
                  'market_segment_Online TA', 'distribution_channel_Direct',
                  'distribution_channel_GDS', 'distribution_channel_TA/TO',
                  'reservation_status_Check-Out', 'reservation_status_No-Show']

X = X.reindex(columns=required_order)
y = df_encoded[[target_variable]]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# # # Train a RandomForestClassifier
# model = RandomForestClassifier(random_state=42)

# model.fit(X_train, y_train)

# y_pred = model.predict(X_val)
# print(y_pred)

# # Evaluate the model
# acc = accuracy_score(y_val, y_pred)
# rec = recall_score(y_val, y_pred)
# f1 = f1_score(y_val, y_pred)
# mse = mean_squared_error(y_val, y_pred)

# Evaluate the model
# acc = accuracy_score(y_val, y_pred)
# rec = recall_score(y_val, y_pred, average='weighted')  # Use 'weighted' for multiclass
# f1 = f1_score(y_val, y_pred, average='weighted')  # Use 'weighted' for multiclass
# mse = mean_squared_error(y_val, y_pred)

# metrics = {
#     'accuracy': acc,
#     'recall': rec,
#     'f1_score': f1,
#     'mean_squared_error': mse
# }
# params = {
#     'model_type': 'RandomForestClassifier',
#     'random_state': 42
# }

# ##################SAVING################################################################################################
# from demand_predictor.ml_logic.registry import save_model, save_results, load_model

# # # Save the model
# save_model(model, 'is_country')

# # Save the metrics
# save_results(params=params, metrics=metrics, model_type = 'is_country')
# ########################################################################################################################
# Example data frame

"""
Index(['hotel', 'is_canceled', 'lead_time', 'adults', 'is_repeated_guest',
       'adr', 'CPI_AVG', 'INFLATION', 'INFLATION_CHG', 'CSMR_SENT', 'UNRATE',
       'INTRSRT', 'GDP', 'FUEL_PRCS', 'CPI_HOTELS', 'US_GINI', 'DIS_INC',
       'total_stay', 'arrival_date_month_sin', 'arrival_date_month_cos',
       'meal_FB', 'meal_HB', 'meal_SC', 'market_segment_Complementary',
       'market_segment_Corporate', 'market_segment_Direct',
       'market_segment_Groups', 'market_segment_Offline TA/TO',
       'market_segment_Online TA', 'distribution_channel_Direct',
       'distribution_channel_GDS', 'distribution_channel_TA/TO',
       'reservation_status_Check-Out', 'reservation_status_No-Show'],
      dtype='object')
"""

# data_frame = pd.DataFrame({
#     'lead_time': [342],
#     'adr': [0],
#     'INFLATION': [1.8],
#     'FUEL_PRCS': [194],
#     'total_stay': [0],
#     'country': ['PRT'],
#     'arrival_date_month': ['July']
# })

# df_test = preprocess_is_canceled_X_pred(data_frame)
# print("'''''''''''''''''''X_test''''''''''''''''''''''''''''")
# print(df_test.columns)
# print("'''''''''''''''''''''''''''''''''''''''''''''''''''")
# loaded_model = load_model('is_canceled')
# y_test_pred = loaded_model.predict(df_test)

# print(y_test_pred)
###############LOADING MODEL############################################################################################
# Test loading the model
# loaded_model = load_model()

# # Check if the loaded model is the same as the saved model
# if loaded_model:
#     print("Model loaded successfully!")
#     sample_data = X_test.iloc[0:1]  # Take a single sample for prediction
#     print(sample_data)
#     loaded_model_prediction = loaded_model.predict(sample_data)
#     print("Loaded model prediction:", loaded_model_prediction)

#     # Take another sample for prediction
#     another_sample = X_test.iloc[1:2]  # Change the index as needed to select a different sample

#     # Make a prediction using the loaded model
#     another_loaded_model_prediction = loaded_model.predict(another_sample)
#     print("Another loaded model prediction:", another_loaded_model_prediction)
# else:
#     print("Failed to load the model.")

# print(X.columns)
