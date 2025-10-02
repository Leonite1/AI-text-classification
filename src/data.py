from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

OUT = (Path(__file__).resolve().parents[1] / "outputs"); OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "sk_cache"; CACHE.mkdir(exist_ok=True)

def load_dataset(subset="train"):
    if subset not in {"train", "test"}:
        raise ValueError("subset must be 'train'or 'test'")
    b = fetch_20newsgroups(subset=subset, remove=("headers", "footers", "quotes"))
    df = pd.DataFrame({"text": b.data, "target": b.target})
    return df, list (b.target_names)