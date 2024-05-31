import os
import sys
import pandas as pd
from ml_logic.preprocessor import preprocess_features, preprocess_is_canceled
# from ml_logic.model import initialize_model, train_model

# Add the root directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'raw_data', 'hotel_bookings_raw.csv')

# Load data
df = pd.read_csv(data_path)
# ############################PREPROCESSING##############################################################

#df_encoded = preprocess_features(df)

#########7 features Preprocessing#######################################################################

df_encoded = preprocess_is_canceled(df)

###########################MODEL######################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, mean_squared_error

target_variable = 'is_canceled'

# Assuming you have a DataFrame df_encoded and a target variable y
# predictors = [
#     'country', 'FUEL_PRCS', 'lead_time', 'adr', "arrival_date_month", 'stays_in_week_nights', 'INFLATION'
# ]
# X = df_encoded.drop(columns=[target_variable])

X = df_encoded.drop(columns=[target_variable])
y = df_encoded[target_variable]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

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
