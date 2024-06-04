def preprocess_meal(df: pd.DataFrame) -> pd.DataFrame:

    df = df.drop(columns=['arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month',
                          'children', 'babies', 'previous_cancellations', 'previous_bookings_not_canceled',
                          'reserved_room_type', 'assigned_room_type', 'booking_changes', 'deposit_type', 'agent',
                          'days_in_waiting_list', 'customer_type', 'required_car_parking_spaces',
                          'total_of_special_requests', 'reservation_status_date', 'MO_YR'])

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Drop NaN's
    df.dropna(inplace=True)
    ########################################################################################################
    df['total_stays'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df.drop(columns= ['stays_in_weekend_nights','stays_in_week_nights'], inplace=True)

    ####### FILTERING ######################################################################################
    # Valid countries list
    valid_countries = ['PRT', 'GBR', 'ESP', 'FRA', 'DEU']

    # Keep only rows with valid countries
    df = df[df['country'].isin(valid_countries)]

    #Drop undefined

    df = df[df["meal"] != "Undefined"]
    df = df[df["market_segment"] != "Undefined"]
    df = df[df["distribution_channel"] != "Undefined"]

    ####### ENCODING ########################################################################################

    # One Hot Encode 'country'
    ohe = OneHotEncoder(sparse_output=False)
    columns_to_encode = ['market_segment', 'distribution_channel', 'reservation_status', 'country']
    # Fit and transform the data
    one_hot_encoded_data = ohe.fit_transform(df[columns_to_encode])
    # Convert encoded data to DataFrame
    one_hot_df = pd.DataFrame(one_hot_encoded_data, columns=ohe.get_feature_names_out(columns_to_encode))
    # Concatenate the encoded columns with the original DataFrame
    df = pd.concat([df.reset_index(drop=True), one_hot_df.reset_index(drop=True)], axis=1)
    # Drop the original columns that were encoded
    df.drop(columns=columns_to_encode, inplace=True)

    # Encode 'arrival_date_month' using sin and cos
    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df['arrival_date_month'] = df['arrival_date_month'].str.strip().map(month_mapping)

    df = encode_time(df, 'arrival_date_month', 12)
    df.drop(columns=['arrival_date_month'], inplace=True)
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
    features_to_robust = ['lead_time', 'arrival_date_month', 'total_stays', 'adults', 'adr', 'FUEL_PRCS']

    robust_scaler = RobustScaler()
    df[features_to_robust] = robust_scaler.fit_transform(df[features_to_robust])

    features_to_minmax = ['CPI_AVG', 'INFLATION', 'INFLATION_CHG', 'CSMR_SENT', 'UNRATE', 'INTRSRT', 'GDP', 'DIS_INC', 'CPI_HOTELS']

    minmax_scaler = MinMaxScaler()
    df[features_to_minmax] = minmax_scaler.fit_transform(df[features_to_minmax])
    #save the column order so the model can properly predict later
    column_order = df.columns
    # Save the scalers and encoder
    folder_path = os.path.dirname(__file__)

    # Create the full file paths
    robust_scaler_path = os.path.join(folder_path, 'robust_scaler_meal.pkl')
    minmax_scaler_path = os.path.join(folder_path, 'minmax_scaler_meal.pkl')
    onehot_encoder_path = os.path.join(folder_path, 'onehot_encoder_meal.pkl')
    column_order_path = os.path.join(folder_path, 'column_order_meal.pkl')

    # Save the scalers and encoder
    joblib.dump(robust_scaler, robust_scaler_path)
    joblib.dump(minmax_scaler, minmax_scaler_path)
    joblib.dump(ohe, onehot_encoder_path)
    joblib.dump(column_order, column_order_path)

    return df

def preprocess_meal_X_pred(df: pd.DataFrame) -> pd.DataFrame:
    #######################################################################################################
    # Load the scalers and encoders
    folder_path = os.path.dirname(__file__)
    r_scaler_path = os.path.join(folder_path, "robust_scaler_meal.pkl")
    m_scaler_path = os.path.join(folder_path, "minmax_scaler_meal.pkl")
    ohe_path = os.path.join(folder_path, "onehot_encoder_meal.pkl")
    column_order_path = os.path.join(folder_path, 'column_order_meal.pkl')

    robust_scaler = joblib.load(r_scaler_path)
    minmax_scaler = joblib.load(m_scaler_path)
    ohe = joblib.load(ohe_path)
    column_order = joblib.load(column_order_path)

    # Valid countries list
    valid_countries = ['PRT', 'GBR', 'ESP', 'FRA', 'DEU']

    ####### FILTERING ######################################################################################
    # Keep only rows with valid countries
    df = df[df['country'].isin(valid_countries)]

    df = df[df["meal"] != "Undefined"]
    df = df[df["market_segment"] != "Undefined"]
    df = df[df["distribution_channel"] != "Undefined"]

    ####### ENCODING ######################################################################################
    # Encode country
    columns_to_encode = ['market_segment', 'distribution_channel', 'reservation_status', 'country']
    # Fit and transform the data
    one_hot_encoded_data = ohe.fit_transform(df[columns_to_encode])
    # Convert encoded data to DataFrame
    one_hot_df = pd.DataFrame(one_hot_encoded_data, columns=ohe.get_feature_names_out(columns_to_encode))
    # Concatenate the encoded columns with the original DataFrame
    df = pd.concat([df.reset_index(drop=True), one_hot_df.reset_index(drop=True)], axis=1)
    # Drop the original columns that were encoded
    df.drop(columns=columns_to_encode, inplace=True)

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
    #Change hotel to binary
    hotel_mapping = {
        'City Hotel': 1,
        'Resort Hotel': 0
    }
    #Strip spaces
    df['hotel'] = df['hotel'].str.strip()
    # Replace month names with numbers
    df['hotel'] = df['hotel'].map(hotel_mapping)

    ####### SCALING ######################################################################################
    # Scaling
    features_to_robust = ['lead_time', 'arrival_date_month', 'total_stays', 'adults', 'adr', 'FUEL_PRCS']
    df[features_to_robust] = robust_scaler.transform(df[features_to_robust])

    # Ensure the order of features matches what was used during fitting
    features_to_minmax = ['CPI_AVG', 'INFLATION', 'INFLATION_CHG', 'CSMR_SENT', 'UNRATE', 'INTRSRT', 'GDP', 'DIS_INC', 'CPI_HOTELS']
    df[features_to_minmax] = minmax_scaler.transform(df[features_to_minmax])
    #sorts the DataFrame like the training data
    df = df[column_order]
    return df
