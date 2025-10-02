from pathlib import Path
import joblib
from src.data import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODELS = (Path(__file__).resolve().parents[1] / "models"); MODELS.mkdir(parents=True, exist_ok=True)

if __name__=="__main__":
    df, names = load_dataset("train")
    X_tr, X_va, y_tr, y_va = train_test_split(
        df["text"], df["target"], test_size=0.2, stratify=df["target"], random_state=7
    )
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=40000)),
        ("lr", LogisticRegression(max_iter=1000))
    ]).fit(X_tr, y_tr)

    print(f"val acc: {clf.score(X_va, y_va):.3f}")
    joblib.dump({"model": clf, "target_names": names}, MODELS / "clf.pkl")
    print("Saved: models/clf.pkl")