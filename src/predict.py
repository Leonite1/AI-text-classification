import sys, joblib
from pathlib import Path

MODEL_PATH = (Path(__file__).resolve().parents[1] / "models" / "clf.pkl")

def predict(text: str):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("models/clf.pkl not found. Run train.py first.")
    bundle = joblib.load(MODEL_PATH)
    model, names = bundle["model"], bundle["target_names"]
    y = model.predict([text])[0]
    proba = getattr(model, "predict_proba", None)
    p = proba([text])[0].max() if callable(proba) else None
    return names[y], p

if __name__=="__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/predict.py "some text"'); sys.exit(1)
    label, p = predict(" ".join(sys.argv[1:]))
    print(f"Pred: {label}" if p is None else f"Pred: {label} | Confidence: {p:.3f}")
