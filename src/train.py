from pathlib import Path
import joblib
import re
from src.data import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODELS = (Path(__file__).resolve().parent.parent / "models"); 
MODELS.mkdir(parents=True, exist_ok=True)

def simple_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text=text.lower()
    text=re.sub(r"http\S+|www\.\S+", " ", text)
    text=re.sub(r"[^a-z\s]", " ", text)
    text=re.sub(r"\s+", " ", text).strip()
    return text

if __name__=="__main__":
    print("Loading dataset...")
    df, target_names = load_dataset("train")
    print("Rows:", len(df))

    X_tr, X_va, y_tr, y_va = train_test_split(
        df["text"], df["target"], test_size=0.2, stratify=df["target"], random_state=42
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=simple_clean,
            stop_words="english",
            ngram_range=(1,2),
            min_df=2,
            max_df=0.9,
            max_features=100000
        )),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=None))
    ])

    pipe.fit(X_tr, y_tr)
    joblib.dump({"model": pipe, "target_names": target_names}, MODELS / "clf.pkl")
    print("Saved:", MODELS / "clf.pkl")
