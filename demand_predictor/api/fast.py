import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from demand_predictor.ml_logic.registry import load_model
from demand_predictor.ml_logic.preprocessor import preprocess_features

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

@app.get("/predict")
def predict(
        feature1: float,
        feature2: float,
        feature3: float,
        feature4: float,
        feature5: float
    ):
    """
    Predict demand based on input features.
    """
    df = pd.DataFrame(dict(
        feature1=[feature1],
        feature2=[feature2],
        feature3=[feature3],
        feature4=[feature4],
        feature5=[feature5]
    ))

    df_processed = preprocess_features(df)

    y_pred = app.state.model.predict(df_processed)
    return {"prediction": float(y_pred)}

@app.get("/")
def root():
    return {'message': 'Welcome to the Demand Predictor API'}
