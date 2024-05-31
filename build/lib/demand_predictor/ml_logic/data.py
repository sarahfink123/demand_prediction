import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - removing buggy or irrelevant transactions
    - maybe we can drop certain rows if there are major outliers...
    """

    # Remove buggy transactions
    df = df.drop_duplicates()  # TODO: handle whether data is consumed in chunks directly in the data source
    df = df.dropna(how='any', axis=0)

    print("✅ data cleaned")

    return df
