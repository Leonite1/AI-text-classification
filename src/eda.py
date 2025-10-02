from pathlib import Path
import matplotlib.pyplot as plt
from src.data import load_dataset
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

OUT = Path(__file__).resolve().parent.parent / "outputs"; 
OUT.mkdir(parents=True, exist_ok=True)

def save_top_words(texts, top_k=30):
    def simple_clean(s: str):
        import re
        s=s.lower()
        s=re.sub(r"http\S+|www\.\S+", " ", s)
        s=re.sub(r"[a-z\s]", " ", s)
        s=re.sub(r"\s+", " ", s).strip()
        return s
    vec = CountVectorizer(stop_words="english", preprocessor=simple_clean)
    X = vec.fit_transform(texts)
    freqs = X.sum(axis=0).A1
    vocab = vec.get_feature_names_out()
    pairs = sorted(zip(vocab,freqs), key=lambda x: x[1], reverse=True)[:top_k]
    df_top = pd.DataFrame(pairs, columns=["word", "frequency"])
    df_top.to_csv(OUT / "top_words.csv", index=False)



if __name__=="__main__":
    df, target_names = load_dataset("train")

    counts = Counter(df["target"])
    labels = [target_names[i] for i in counts.keys()]
    values = [counts[i] for i in counts.keys()]
    plt.figure()
    plt.bar(labels, values)
    plt.title("Samples per class")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(OUT / "class_counts.png")

    df["chars"]=df["text"].str.len()
    df["words"]=df["text"].str.split().str.len()
    df[["chars", "words"]].describe().to_csv(OUT / "basic_stats.csv")

    save_top_words(df["text"], top_k=30)

    print("EDA u ruajt te:", OUT)
