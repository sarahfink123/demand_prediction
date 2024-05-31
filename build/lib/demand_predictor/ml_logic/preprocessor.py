import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler, MinMaxScaler
import joblib
from registry import load_model

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
# columns = 'country', 'FUEL_PRCS', 'lead_time', 'adr', "arrival_date_month", 'stays_in_week_nights', 'INFLATION', 'is_canceled'

    df = df.drop(columns=['arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month',
                          'children', 'babies', 'previous_cancellations', 'previous_bookings_not_canceled',
                          'reserved_room_type', 'assigned_room_type', 'booking_changes', 'deposit_type', 'agent',
                          'days_in_waiting_list', 'customer_type', 'required_car_parking_spaces',
                          'total_of_special_requests', 'reservation_status_date', 'MO_YR', 'meal', 'market_segment',
                          'distribution_channel', 'reservation_status','stays_in_weekend_nights','adults',
                          'CPI_AVG',  'INFLATION_CHG', 'CSMR_SENT','UNRATE', 'INTRSRT', 'GDP', 'DIS_INC', 'CPI_HOTELS',
                           'is_repeated_guest','US_GINI','hotel'])

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Drop NaN's
    df.dropna(inplace=True)
    #######ENCODING########################################################################################
    # Encode country
    country_mapping = {'ABW': 0, 'AGO': 1, 'AIA': 2, 'ALB': 3, 'AND': 4, 'ARE': 5, 'ARG': 6, 'ARM': 7, 'ASM': 8,\
        'ATA': 9, 'ATF': 10, 'AUS': 11, 'AUT': 12, 'AZE': 13, 'BDI': 14, 'BEL': 15, 'BEN': 16, 'BFA': 17, 'BGD': 18,\
        'BGR': 19, 'BHR': 20, 'BHS': 21, 'BIH': 22, 'BLR': 23, 'BOL': 24, 'BRA': 25, 'BRB': 26, 'BWA': 27, 'CAF': 28,\
        'CHE': 29, 'CHL': 30, 'CHN': 31, 'CIV': 32, 'CMR': 33, 'CN': 34, 'COL': 35, 'COM': 36, 'CPV': 37, 'CRI': 38, \
        'CUB': 39, 'CYM': 40, 'CYP': 41, 'CZE': 42, 'DEU': 43, 'DJI': 44, 'DMA': 45, 'DNK': 46, 'DOM': 47, 'DZA': 48, \
        'ECU': 49, 'EGY': 50, 'ESP': 51, 'EST': 52, 'ETH': 53, 'FIN': 54, 'FJI': 55, 'FRA': 56, 'FRO': 57, 'GAB': 58, \
        'GBR': 59, 'GEO': 60, 'GGY': 61, 'GHA': 62, 'GIB': 63, 'GLP': 64, 'GNB': 65, 'GRC': 66, 'GTM': 67, 'GUY': 68, \
        'HKG': 69, 'HND': 70, 'HRV': 71, 'HUN': 72, 'IDN': 73, 'IMN': 74, 'IND': 75, 'IRL': 76, 'IRN': 77, 'IRQ': 78, \
        'ISL': 79, 'ISR': 80, 'ITA': 81, 'JAM': 82, 'JEY': 83, 'JOR': 84, 'JPN': 85, 'KAZ': 86, 'KEN': 87, 'KHM': 88, \
        'KIR': 89, 'KNA': 90, 'KOR': 91, 'KWT': 92, 'LAO': 93, 'LBN': 94, 'LBY': 95, 'LCA': 96, 'LIE': 97, 'LKA': 98, \
        'LTU': 99, 'LUX': 100, 'LVA': 101, 'MAC': 102, 'MAR': 103, 'MCO': 104, 'MDG': 105, 'MDV': 106, 'MEX': 107, \
        'MKD': 108, 'MLI': 109, 'MLT': 110, 'MMR': 111, 'MNE': 112, 'MOZ': 113, 'MRT': 114, 'MUS': 115, 'MWI': 116, \
        'MYS': 117, 'MYT': 118, 'NAM': 119, 'NCL': 120, 'NGA': 121, 'NIC': 122, 'NLD': 123, 'NOR': 124, 'NPL': 125, \
        'NZL': 126, 'OMN': 127, 'PAK': 128, 'PAN': 129, 'PER': 130, 'PHL': 131, 'PLW': 132, 'POL': 133, 'PRI': 134, \
        'PRT': 135, 'PRY': 136, 'PYF': 137, 'QAT': 138, 'ROU': 139, 'RUS': 140, 'RWA': 141, 'SAU': 142, 'SDN': 143, \
        'SEN': 144, 'SGP': 145, 'SLE': 146, 'SLV': 147, 'SMR': 148, 'SRB': 149, 'STP': 150, 'SUR': 151, 'SVK': 152, \
        'SVN': 153, 'SWE': 154, 'SYC': 155, 'SYR': 156, 'TGO': 157, 'THA': 158, 'TJK': 159, 'TMP': 160, 'TUN': 161, \
        'TUR': 162, 'TWN': 163, 'TZA': 164, 'UGA': 165, 'UKR': 166, 'UMI': 167, 'URY': 168, 'USA': 169, 'UZB': 170, \
        'VEN': 171, 'VGB': 172, 'VNM': 173, 'ZAF': 174, 'ZMB': 175, 'ZWE': 176}

    df["country"] = df["country"].map(country_mapping)


    # Change months to numbers
    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df['arrival_date_month'] = df['arrival_date_month'].str.strip().map(month_mapping)
    #######SCALING########################################################################################
    # Scaling
    features_to_robust = ['lead_time', 'arrival_date_month',
                          'stays_in_week_nights', 'adr', 'FUEL_PRCS']

    robust_scaler = RobustScaler()
    df[features_to_robust] = robust_scaler.fit_transform(df[features_to_robust])

    features_to_minmax = ['country',  'INFLATION']

    minmax_scaler = MinMaxScaler()
    df[features_to_minmax] = minmax_scaler.fit_transform(df[features_to_minmax])

     # Save the scalers
    joblib.dump(robust_scaler, 'robust_scaler.pkl')
    joblib.dump(minmax_scaler, 'minmax_scaler.pkl')

    X_processed = df

    return X_processed


def  preprocess_is_canceled_X_pred(df: pd.DataFrame) -> pd.DataFrame:
    #'country', 'FUEL_PRCS', 'lead_time', 'adr', "arrival_date_month", 'stays_in_week_nights', 'INFLATION'
    # ['lead_time', 'arrival_date_month','stays_in_week_nights', 'adr', 'FUEL_PRCS']
    # ['country',  'INFLATION']
    #######ENCODING########################################################################################
    # Encode country
    country_mapping = {'ABW': 0, 'AGO': 1, 'AIA': 2, 'ALB': 3, 'AND': 4, 'ARE': 5, 'ARG': 6, 'ARM': 7, 'ASM': 8,\
        'ATA': 9, 'ATF': 10, 'AUS': 11, 'AUT': 12, 'AZE': 13, 'BDI': 14, 'BEL': 15, 'BEN': 16, 'BFA': 17, 'BGD': 18,\
        'BGR': 19, 'BHR': 20, 'BHS': 21, 'BIH': 22, 'BLR': 23, 'BOL': 24, 'BRA': 25, 'BRB': 26, 'BWA': 27, 'CAF': 28,\
        'CHE': 29, 'CHL': 30, 'CHN': 31, 'CIV': 32, 'CMR': 33, 'CN': 34, 'COL': 35, 'COM': 36, 'CPV': 37, 'CRI': 38, \
        'CUB': 39, 'CYM': 40, 'CYP': 41, 'CZE': 42, 'DEU': 43, 'DJI': 44, 'DMA': 45, 'DNK': 46, 'DOM': 47, 'DZA': 48, \
        'ECU': 49, 'EGY': 50, 'ESP': 51, 'EST': 52, 'ETH': 53, 'FIN': 54, 'FJI': 55, 'FRA': 56, 'FRO': 57, 'GAB': 58, \
        'GBR': 59, 'GEO': 60, 'GGY': 61, 'GHA': 62, 'GIB': 63, 'GLP': 64, 'GNB': 65, 'GRC': 66, 'GTM': 67, 'GUY': 68, \
        'HKG': 69, 'HND': 70, 'HRV': 71, 'HUN': 72, 'IDN': 73, 'IMN': 74, 'IND': 75, 'IRL': 76, 'IRN': 77, 'IRQ': 78, \
        'ISL': 79, 'ISR': 80, 'ITA': 81, 'JAM': 82, 'JEY': 83, 'JOR': 84, 'JPN': 85, 'KAZ': 86, 'KEN': 87, 'KHM': 88, \
        'KIR': 89, 'KNA': 90, 'KOR': 91, 'KWT': 92, 'LAO': 93, 'LBN': 94, 'LBY': 95, 'LCA': 96, 'LIE': 97, 'LKA': 98, \
        'LTU': 99, 'LUX': 100, 'LVA': 101, 'MAC': 102, 'MAR': 103, 'MCO': 104, 'MDG': 105, 'MDV': 106, 'MEX': 107, \
        'MKD': 108, 'MLI': 109, 'MLT': 110, 'MMR': 111, 'MNE': 112, 'MOZ': 113, 'MRT': 114, 'MUS': 115, 'MWI': 116, \
        'MYS': 117, 'MYT': 118, 'NAM': 119, 'NCL': 120, 'NGA': 121, 'NIC': 122, 'NLD': 123, 'NOR': 124, 'NPL': 125, \
        'NZL': 126, 'OMN': 127, 'PAK': 128, 'PAN': 129, 'PER': 130, 'PHL': 131, 'PLW': 132, 'POL': 133, 'PRI': 134, \
        'PRT': 135, 'PRY': 136, 'PYF': 137, 'QAT': 138, 'ROU': 139, 'RUS': 140, 'RWA': 141, 'SAU': 142, 'SDN': 143, \
        'SEN': 144, 'SGP': 145, 'SLE': 146, 'SLV': 147, 'SMR': 148, 'SRB': 149, 'STP': 150, 'SUR': 151, 'SVK': 152, \
        'SVN': 153, 'SWE': 154, 'SYC': 155, 'SYR': 156, 'TGO': 157, 'THA': 158, 'TJK': 159, 'TMP': 160, 'TUN': 161, \
        'TUR': 162, 'TWN': 163, 'TZA': 164, 'UGA': 165, 'UKR': 166, 'UMI': 167, 'URY': 168, 'USA': 169, 'UZB': 170, \
        'VEN': 171, 'VGB': 172, 'VNM': 173, 'ZAF': 174, 'ZMB': 175, 'ZWE': 176}

    df["country"] = df["country"].map(country_mapping)


    # Change months to numbers
    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df['arrival_date_month'] = df['arrival_date_month'].str.strip().map(month_mapping)
    #######SCALING########################################################################################
    # Load the scalers
    folder_path = os.path.dirname(__file__)
    r_scaler_path = os.path.join(folder_path,"..", "robust_scaler.pkl")
    m_scaler_path = os.path.join(folder_path,"..", "minmax_scaler.pkl")


    robust_scaler = joblib.load(r_scaler_path)
    minmax_scaler = joblib.load(m_scaler_path)

    # Scaling
    features_to_robust = ['lead_time', 'arrival_date_month',
                          'stays_in_week_nights', 'adr', 'FUEL_PRCS']

    df[features_to_robust] = robust_scaler.transform(df[features_to_robust])

    features_to_minmax = ['country',  'INFLATION']

    df[features_to_minmax] = minmax_scaler.transform(df[features_to_minmax])

    X_processed = df

    return X_processed

data_frame = pd.DataFrame(
        {
    'lead_time': 342,
    'arrival_date_month': 'July',
    'stays_in_week_nights': 0,
    'adr': 0,
    'FUEL_PRCS': 194,
    'country': 'PRT',
    'INFLATION': 1.8
        }, index=[0])
name = preprocess_is_canceled_X_pred(
    data_frame)
model = load_model()
print(model.predict(name))
# print(name)
# # print(data_frame)
