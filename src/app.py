from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib

app = FastAPI(title="AI Text Classifier")

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "clf.pkl"
if not MODEL_PATH.exists():
    raise FileNotFoundError("models/clf.pkl not found. Run train.py first.")


_bundle = joblib.load(MODEL_PATH)
_model, _names = _bundle["model"], _bundle["target_names"]


class Item(BaseModel):
    text: str


@app.post("/predict")
def predict(item: Item):
    y = _model.predict([item.text])[0]
    proba = getattr(_model, "predict_proba", None)
    p = float(proba([item.text])[0].max()) if callable(proba) else None
    return {"label_idx": int(y), "label": _names[y], "confidence": p}