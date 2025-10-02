from pathlib import Path
import joblib
from src.data import load_dataset
from sklearn.metrics import accuracy_score, classification_report

MODELS = (Path(__file__).resolve().parents[1] / "models")

if __name__=="__main__":
    f = MODELS / "clf.pkl"
    if not f.exists():
        raise FileNotFoundError("models/clf.pkl not found. Run train.py first.")
    bundle = joblib.load(f); clf, names = bundle["model"], bundle["target_names"]

    df, _=load_dataset("test")
    y_pred=clf.predict(df["text"])
    print(f"acc: {accuracy_score(df['target'], y_pred):.3f}")
    print(classification_report(df["target"], y_pred, target_names=names))