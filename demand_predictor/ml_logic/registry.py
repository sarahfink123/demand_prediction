import os
import time
import pickle
from demand_predictor.params import LOCAL_REGISTRY_PATH  # Adjust your import accordingly

def save_model(model, model_type: str) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{model_type}/{timestamp}.pkl"
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join(LOCAL_REGISTRY_PATH, "models", model_type)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, f"{timestamp}.pkl")
    with open(model_path, "wb") as file:
        pickle.dump(model, file)
    print(f"✅ Model saved locally at {model_path}")

def load_model(model_type: str):
    """
    Return the most recently saved model from the local directory f"{LOCAL_REGISTRY_PATH}/models/{model_type}/".
    """
    model_dir = os.path.join(LOCAL_REGISTRY_PATH, "models", model_type)
    if not os.path.exists(model_dir):
        print("❌ No model directory found.")
        return None

    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    if not model_files:
        print("❌ No model files found.")
        return None

    latest_model_path = max(model_files, key=lambda f: os.path.getmtime(os.path.join(model_dir, f)))
    latest_model_path = os.path.join(model_dir, latest_model_path)

    with open(latest_model_path, "rb") as file:
        model = pickle.load(file)
    print(f"✅ Model loaded from {latest_model_path}")
    return model

def save_results(params: dict, metrics: dict, model_type: str) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{model_type}/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{model_type}/{current_timestamp}.pickle"
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        params_dir = os.path.join(LOCAL_REGISTRY_PATH, "params", model_type)
        if not os.path.exists(params_dir):
            os.makedirs(params_dir)
        params_path = os.path.join(params_dir, timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_dir = os.path.join(LOCAL_REGISTRY_PATH, "metrics", model_type)
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
        metrics_path = os.path.join(metrics_dir, timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally at:", params_path, metrics_path)
