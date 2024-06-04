import pandas as pd
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from demand_predictor.ml_logic.registry import load_model  # Correct import for load_model
from demand_predictor.ml_logic.preprocessor import preprocess_is_canceled_X_pred  # Correct import for preprocess_features

app = FastAPI()
app.state.model = load_model('is_canceled')

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
        adr: float,
        INFLATION: float,
        FUEL_PRCS: float,
        total_stay: int,
        country: str,
        arrival_date_month: str
    ):
    """
    Predict demand based on input features.
    """
    X_pred = pd.DataFrame(locals(), index=[0])

    model = app.state.model
    assert model is not None

    X_processed = preprocess_is_canceled_X_pred(X_pred)

    y_pred = model.predict(X_processed)

    probabilities = model.predict_proba(X_processed)
    proba = probabilities[0][1]

    return {"prediction": int(y_pred),
            "prediction probability": float(proba)}

@app.get("/")
def root():
    return {'message': 'Welcome to the Demand Predictor API'}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
