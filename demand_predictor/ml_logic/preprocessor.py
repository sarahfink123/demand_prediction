import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler, MinMaxScaler

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
