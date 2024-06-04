import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler, MinMaxScaler
import joblib

def encode_time(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col] / max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col] / max_val)
    return data

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    ########################CLEANING DATA#########################################################################
    # Drop columns
    df = df.drop(columns=['arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month',
                          'children', 'babies', 'previous_cancellations', 'previous_bookings_not_canceled',
                          'reserved_room_type', 'assigned_room_type', 'booking_changes', 'deposit_type', 'agent',
                          'days_in_waiting_list', 'customer_type', 'required_car_parking_spaces',
                          'total_of_special_requests', 'reservation_status_date', 'MO_YR'])

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Drop NaN's
    df.dropna(inplace=True)

    # Further Cleaning of NaN's
    # Drop undefined
    df = df[df["meal"] != "Undefined"]
    df = df[df["market_segment"] != "Undefined"]
    df = df[df["distribution_channel"] != "Undefined"]

    ###############################ENCODING#############################################################
    # Change months to numbers
    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df['arrival_date_month'] = df['arrival_date_month'].str.strip().map(month_mapping)

    # Change hotel to binary
    hotel_mapping = {'City Hotel': 1, 'Resort Hotel': 0}
    df['hotel'] = df['hotel'].str.strip().map(hotel_mapping)

    # Encode country
    label_encoder = LabelEncoder()
    df['country'] = label_encoder.fit_transform(df['country'])

    country_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    # One Hot Encode meal, market_segment, distribution_channel, reservation_status
    columns_to_encode = ['meal', 'market_segment', 'distribution_channel', 'reservation_status']
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    encoded_data = ohe.fit_transform(df[columns_to_encode])
    encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(columns_to_encode))

    # Concatenate the encoded columns with the original DataFrame
    df_encoded = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    df_encoded.drop(columns=columns_to_encode, inplace=True)

    # Scaling
    features_to_robust = ['lead_time', 'arrival_date_month', 'stays_in_weekend_nights',
                          'stays_in_week_nights', 'adults', 'adr', 'FUEL_PRCS']
    robust_scaler = RobustScaler()
    df_encoded[features_to_robust] = robust_scaler.fit_transform(df_encoded[features_to_robust])

    features_to_minmax = ['country', 'CPI_AVG', 'INFLATION', 'INFLATION_CHG', 'CSMR_SENT',
                          'UNRATE', 'INTRSRT', 'GDP', 'DIS_INC', 'CPI_HOTELS']
    minmax_scaler = MinMaxScaler()
    df_encoded[features_to_minmax] = minmax_scaler.fit_transform(df_encoded[features_to_minmax])

    print("✅ X_processed, with shape", df_encoded.shape)

    return df_encoded

def preprocess_is_canceled(df: pd.DataFrame) -> pd.DataFrame:

    df = df.drop(columns=['arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month',
                          'children', 'babies', 'previous_cancellations', 'previous_bookings_not_canceled',
                          'reserved_room_type', 'assigned_room_type', 'booking_changes', 'deposit_type', 'agent',
                          'days_in_waiting_list', 'customer_type', 'required_car_parking_spaces',
                          'total_of_special_requests', 'reservation_status_date', 'MO_YR', 'meal', 'market_segment',
                          'distribution_channel', 'reservation_status', 'adults',
                          'CPI_AVG', 'INFLATION_CHG', 'CSMR_SENT', 'UNRATE', 'INTRSRT', 'GDP', 'DIS_INC', 'CPI_HOTELS',
                          'is_repeated_guest', 'US_GINI', 'hotel'])

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Drop NaN's
    df.dropna(inplace=True)
    ########################################################################################################
    df['total_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df.drop(columns= ['stays_in_weekend_nights','stays_in_week_nights'], inplace=True)

    ####### FILTERING ######################################################################################
    # Valid countries list
    valid_countries = ['PRT', 'GBR', 'ESP', 'FRA', 'DEU']

    # Keep only rows with valid countries
    df = df[df['country'].isin(valid_countries)]

    ####### ENCODING ########################################################################################

    # One Hot Encode 'country'
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    country_encoded = ohe.fit_transform(df[['country']])
    country_encoded_df = pd.DataFrame(country_encoded, columns=ohe.get_feature_names_out(['country']))
    df = pd.concat([df.reset_index(drop=True), country_encoded_df.reset_index(drop=True)], axis=1)
    df.drop(columns=['country'], inplace=True)

    # Encode 'arrival_date_month' using sin and cos
    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df['arrival_date_month'] = df['arrival_date_month'].str.strip().map(month_mapping)

    df = encode_time(df, 'arrival_date_month', 12)
    df.drop(columns=['arrival_date_month'], inplace=True)

    ####### SCALING ########################################################################################
    # Scaling
    features_to_robust = ['lead_time', 'total_stay', 'adr', 'FUEL_PRCS']

    robust_scaler = RobustScaler()
    df[features_to_robust] = robust_scaler.fit_transform(df[features_to_robust])

    features_to_minmax = ['INFLATION'] + list(country_encoded_df.columns)

    minmax_scaler = MinMaxScaler()
    df[features_to_minmax] = minmax_scaler.fit_transform(df[features_to_minmax])

    # Save the scalers and encoder
    folder_path = os.path.dirname(__file__)

    # Create the full file paths
    robust_scaler_path = os.path.join(folder_path, 'robust_scaler_is_canceled.pkl')
    minmax_scaler_path = os.path.join(folder_path, 'minmax_scaler_is_canceled.pkl')
    onehot_encoder_path = os.path.join(folder_path, 'onehot_encoder_is_canceled.pkl')

    # Save the scalers and encoder
    joblib.dump(robust_scaler, robust_scaler_path)
    joblib.dump(minmax_scaler, minmax_scaler_path)
    joblib.dump(ohe, onehot_encoder_path)

    return df

def preprocess_is_canceled_X_pred(df: pd.DataFrame) -> pd.DataFrame:
    #######################################################################################################
    # Load the scalers and encoders
    folder_path = os.path.dirname(__file__)
    r_scaler_path = os.path.join(folder_path, "robust_scaler_is_canceled.pkl")
    m_scaler_path = os.path.join(folder_path, "minmax_scaler_is_canceled.pkl")
    ohe_path = os.path.join(folder_path, "onehot_encoder_is_canceled.pkl")

    robust_scaler = joblib.load(r_scaler_path)
    minmax_scaler = joblib.load(m_scaler_path)
    ohe = joblib.load(ohe_path)

    # Valid countries list
    valid_countries = ['PRT', 'GBR', 'ESP', 'FRA', 'DEU']

    ####### FILTERING ######################################################################################
    # Keep only rows with valid countries
    df = df[df['country'].isin(valid_countries)]

    ####### ENCODING ######################################################################################
    # Encode country
    country_encoded = ohe.transform(df[['country']])
    country_encoded_df = pd.DataFrame(country_encoded, columns=ohe.get_feature_names_out(['country']))
    df = pd.concat([df.reset_index(drop=True), country_encoded_df.reset_index(drop=True)], axis=1)
    df.drop(columns=['country'], inplace=True)

    # Change months to numbers
    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df['arrival_date_month'] = df['arrival_date_month'].str.strip().map(month_mapping)

    # Encode month using sin and cos
    df = encode_time(df, 'arrival_date_month', 12)
    df.drop(columns=['arrival_date_month'], inplace=True)

    ####### SCALING ######################################################################################
    # Scaling
    features_to_robust = ['lead_time', 'total_stay', 'adr', 'FUEL_PRCS']
    df[features_to_robust] = robust_scaler.transform(df[features_to_robust])

    # Ensure the order of features matches what was used during fitting
    features_to_minmax = ['INFLATION'] + list(ohe.get_feature_names_out(['country']))
    df[features_to_minmax] = minmax_scaler.transform(df[features_to_minmax])

    return df

def preprocess_is_country(df: pd.DataFrame) -> pd.DataFrame:
    #######################DROP NaN'S########################################################################
    #Drop columns
    df = df.drop(columns=['arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month',
                          'children', 'babies', 'previous_cancellations', 'previous_bookings_not_canceled',
                          'reserved_room_type', 'assigned_room_type', 'booking_changes', 'deposit_type',
                          'agent', 'days_in_waiting_list', 'customer_type', 'required_car_parking_spaces',
                          'total_of_special_requests', 'reservation_status_date', 'MO_YR'])
    #Drop duplicates
    df.drop_duplicates(inplace=True)
    #Drop none
    df.dropna(inplace=True)

    df = df[df["meal"] != "Undefined"]
    df = df[df["market_segment"] != "Undefined"]
    df = df[df["distribution_channel"] != "Undefined"]

    valid_countries = ['PRT', 'GBR', 'ESP', 'FRA', 'DEU']

    # Step 3: Filter the DataFrame to only include these valid countries
    df = df[df['country'].isin(valid_countries)]
    #####################TOTAL STAY##########################################################################
    df['total_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df.drop(columns= ['stays_in_weekend_nights','stays_in_week_nights'], inplace=True)

    ####### ENCODING ########################################################################################
    le = LabelEncoder()
    df['country'] = le.fit_transform(df['country'])
    #--------------------------------------------------------------------------------------------------------
    # Encode 'arrival_date_month' using sin and cos
    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df['arrival_date_month'] = df['arrival_date_month'].str.strip().map(month_mapping)

    df = encode_time(df, 'arrival_date_month', 12)
    df.drop(columns=['arrival_date_month'], inplace=True)
    #--------------------------------------------------------------------------------------------------------
    #Change hotel to binary
    hotel_mapping = {
        'City Hotel': 1,
        'Resort Hotel': 0
    }
    #Strip spaces
    df['hotel'] = df['hotel'].str.strip()
    # Replace month names with numbers
    df['hotel'] = df['hotel'].map(hotel_mapping)
    #--------------------------------------------------------------------------------------------------------
    # Columns to one-hot encode
    # can add 'country' here to use it as a predictor
    columns_to_encode = ['meal', 'market_segment', 'distribution_channel', 'reservation_status']

    # Instantiate the OneHotEncoder
    ohe = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' avoids the dummy variable trap

    # Fit and transform the data
    one_hot_encoded_data = ohe.fit_transform(df[columns_to_encode])

    # Convert encoded data to DataFrame
    one_hot_df = pd.DataFrame(one_hot_encoded_data, columns=ohe.get_feature_names_out(columns_to_encode))

    # Concatenate the encoded columns with the original DataFrame
    df = pd.concat([df.reset_index(drop=True), one_hot_df.reset_index(drop=True)], axis=1)

    # Drop the original columns that were encoded
    df.drop(columns=columns_to_encode, inplace=True)
    ####### SCALING ########################################################################################
    # Scaling
    features_to_robust = ['lead_time', 'arrival_date_month_sin','arrival_date_month_cos',
                          'total_stay', 'adults', 'adr', 'FUEL_PRCS']

    robust_scaler = RobustScaler()
    df[features_to_robust] = robust_scaler.fit_transform(df[features_to_robust])

    features_to_minmax = ['CPI_AVG', 'INFLATION', 'INFLATION_CHG', 'CSMR_SENT', 'UNRATE', 'INTRSRT',
                          'GDP', 'DIS_INC', 'CPI_HOTELS']

    minmax_scaler = MinMaxScaler()
    df[features_to_minmax] = minmax_scaler.fit_transform(df[features_to_minmax])

    # Save the scalers and encoder
    folder_path = os.path.dirname(__file__)

    # Create the full file paths
    robust_scaler_path = os.path.join(folder_path, 'robust_scaler_is_country.pkl')
    minmax_scaler_path = os.path.join(folder_path, 'minmax_scaler_is_country.pkl')
    onehot_encoder_path = os.path.join(folder_path, 'onehot_encoder_is_country.pkl')
    label_encoder_path = os.path.join(folder_path, 'label_encoder_is_country.pkl')


    # Save the scalers and encoder
    joblib.dump(robust_scaler, robust_scaler_path)
    joblib.dump(minmax_scaler, minmax_scaler_path)
    joblib.dump(ohe, onehot_encoder_path)
    joblib.dump(le, label_encoder_path)

    return df

def preprocess_is_country_X_pred(df: pd.DataFrame) -> pd.DataFrame:
    """
    predictors = ['arrival_date_month','adr', 'lead_time', 'total_stay',
              'adults', 'hotel', 'INFLATION']
    """
    # Load the scalers and encoders
    folder_path = os.path.dirname(__file__)
    r_scaler_path = os.path.join(folder_path, "robust_scaler_is_country.pkl")
    m_scaler_path = os.path.join(folder_path, "minmax_scaler_is_country.pkl")


    robust_scaler = joblib.load(r_scaler_path)
    minmax_scaler = joblib.load(m_scaler_path)

    ####### ENCODING ########################################################################################
    # Encode 'arrival_date_month' using sin and cos
    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df['arrival_date_month'] = df['arrival_date_month'].str.strip().map(month_mapping)

    df = encode_time(df, 'arrival_date_month', 12)
    df.drop(columns=['arrival_date_month'], inplace=True)
    #--------------------------------------------------------------------------------------------------------
    #Change hotel to binary
    hotel_mapping = {
        'City Hotel': 1,
        'Resort Hotel': 0
    }
    #Strip spaces
    df['hotel'] = df['hotel'].str.strip()
    # Replace month names with numbers
    df['hotel'] = df['hotel'].map(hotel_mapping)
    ####### SCALING ########################################################################################
    # Scaling
    features_to_robust = ['lead_time', 'arrival_date_month_sin','arrival_date_month_cos',
                          'total_stay', 'adults', 'adr']

    robust_scaler = RobustScaler()
    df[features_to_robust] = robust_scaler.fit_transform(df[features_to_robust])

    features_to_minmax = ['INFLATION']

    minmax_scaler = MinMaxScaler()
    df[features_to_minmax] = minmax_scaler.fit_transform(df[features_to_minmax])

    return df


# # Example data frame
# data_frame = pd.DataFrame({
#     'lead_time': [342],
#     'arrival_date_month': ['July'],
#     'total_stay': [0],
#     'country': ['PRT'],
#     'adr': [0],
#     'INFLATION': [1.8],
#     'FUEL_PRCS': [194]
# })

# name = preprocess_is_canceled_X_pred(data_frame)
# print(name.columns)

# model = load_model()
# print(model.predict(name))
# print(name)
# # print(data_frame)
