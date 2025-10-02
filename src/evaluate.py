from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from src.data import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ROOT / "models"

if __name__=="__main__":
    bundle = joblib.load(MODELS / "clf.pkl")
    model = bundle["model"]
    names = bundle["target_names"]

    df_test, _ = load_dataset("test")
    y_tr = df_test["target"]
    y_pred = model.predict(df_test["text"])

    acc = accuracy_score(y_tr, y_pred)
    print("Accuracy:", round(acc, 3))

    report = classification_report(y_tr, y_pred, target_names=names, output_dict=True)
    pd.DataFrame(report).to_csv(OUT / "classification_report.csv")

    cm = confusion_matrix(y_tr, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", annot=False, xticklabels=names, yticklabels=names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png")
    print("Saved metrics to outputs/")
