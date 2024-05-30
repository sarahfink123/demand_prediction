import pickle
import pandas as pd
import numpy as np

from colorama import Fore, Style

from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

# Define custom transformers
class MonthsTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
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
        X = X.copy()  # Ensure we do not modify the original DataFrame
        X['arrival_date_month'] = X['arrival_date_month'].str.strip()
        X['arrival_date_month'] = X['arrival_date_month'].map(month_mapping)
        return X

class CountryTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.encoder = LabelEncoder()
        self.encoder.fit(X.squeeze())
        return self

    def transform(self, X, y=None):
        X = X.copy()  # Ensure we do not modify the original DataFrame
        # Handle unseen countries by assigning a value of -1
        X['country'] = X['country'].apply(lambda x: self.encoder.transform([x])[0] if x in self.encoder.classes_ else -1)
        return X

def create_pipeline_and_preprocess_features(X: pd.DataFrame, save_path: str = "pipeline.pkl") -> np.ndarray:
    def create_sklearn_preprocessor(numerical_features, categorical_features) -> ColumnTransformer:
        # preprocessor = ColumnTransformer(
        #     transformers=[
        #         ('num', Pipeline(steps=[
        #             ('imputer', SimpleImputer(strategy='median')),
        #             ('scaler', StandardScaler())
        #         ]), numerical_features),
        #         ('cat', Pipeline(steps=[
        #             ('imputer', SimpleImputer(strategy='most_frequent')),
        #             ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        #         ]), categorical_features)
        #     ]
        # )
        # Use the pipeline for our features from the notebook
        features_country = ["country"]
        features_months = ["arrival_date_month"]
        features_to_robust = ['lead_time', 'adr', 'stays_in_week_nights', 'FUEL_PRCS']
        features_to_minmax = ['INFLATION']

        # Define preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "country",
                    Pipeline(steps=[
                        ("country_mapping", CountryTransformer()),
                        ("scaler", MinMaxScaler())
                    ]),
                    features_country
                ),
                (
                    "arrival_date_month",
                    Pipeline(steps=[
                        ("arrival_date_month_mapping", MonthsTransformer()),
                        ("scaler", RobustScaler())
                    ]),
                    features_months
                ),
                (
                    "robust",
                    Pipeline(steps=[("scaler", RobustScaler())]),
                    features_to_robust
                ),
                (
                    "minmax",
                    Pipeline(steps=[("scaler", MinMaxScaler())]),
                    features_to_minmax
                ),
            ],
            remainder='passthrough'  # Keep other columns unchanged
        )

        return preprocessor

    print(Fore.BLUE + "\nPreprocessing features..." + Style.RESET_ALL)

    # Are now hardcoded
    # numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    # if not numerical_features and not categorical_features:
    #     raise ValueError("The input DataFrame has no numerical or categorical features to preprocess.")
    # preprocessor = create_sklearn_preprocessor(numerical_features, categorical_features)

    preprocessor = create_sklearn_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    print("✅ X_processed, with shape", X_processed.shape)

    with open(save_path, "wb") as fh:
        pickle.dump(preprocessor, fh)

    return X_processed


def preprocess_data_from_file(X: pd.DataFrame, pipeline_path: str = "pipeline.pkl"):
    with open(pipeline_path, "rb") as fh:
        preprocessor = pickle.load(fh)

    X_processed = preprocessor.transform(X)
    print("✅ X_processed, with shape", X_processed.shape)
    return X_processed
