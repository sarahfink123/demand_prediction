import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from demand_predictor.ml_logic.registry import load_model  # Correct import for load_model
from demand_predictor.ml_logic.preprocessor import preprocess_is_canceled_X_pred  # Correct import for preprocess_features

app = FastAPI()
app.state.model = load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
#'country', 'FUEL_PRCS', 'lead_time', 'adr', "arrival_date_month", 'stays_in_week_nights', 'INFLATION'
@app.get("/predict")
def predict(
        lead_time: int,
        arrival_date_month: str,
        stays_in_week_nights: int,
        adr: float,
        FUEL_PRCS: float,
        country: str,
        INFLATION: float
    ):

    # ['lead_time', 'arrival_date_month','stays_in_week_nights', 'adr', 'FUEL_PRCS']
    # ['country',  'INFLATION']
    """
    Predict demand based on input features.
    """
    X_pred = pd.DataFrame(locals(), index=[0])

    model = app.state.model
    assert model is not None

    X_processed = preprocess_is_canceled_X_pred(X_pred)
    y_pred = model.predict(X_processed)

    return {"prediction": int(y_pred)}

@app.get("/")
def root():
    return {'message': 'Welcome to the Demand Predictor API'}
