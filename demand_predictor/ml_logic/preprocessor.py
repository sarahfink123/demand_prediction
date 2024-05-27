import pickle
import pandas as pd
import numpy as np

from colorama import Fore, Style

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    def create_sklearn_preprocessor(numerical_features, categorical_features) -> ColumnTransformer:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numerical_features),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
                ]), categorical_features)
            ]
        )

        return preprocessor

    print(Fore.BLUE + "\nPreprocessing features..." + Style.RESET_ALL)

    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    if not numerical_features and not categorical_features:
        raise ValueError("The input DataFrame has no numerical or categorical features to preprocess.")

    preprocessor = create_sklearn_preprocessor(numerical_features, categorical_features)
    X_processed = preprocessor.fit_transform(X)

    print("✅ X_processed, with shape", X_processed.shape)

    return X_processed
