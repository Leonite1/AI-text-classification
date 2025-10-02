from pathlib import Path
import matplotlib.pyplot as plt
from src.data import load_dataset

OUT = (Path(__file__).resolve().parents[1] / "outputs"); OUT.mkdir(parents=True, exist_ok=True)

if __name__=="__main__":
    df, names = load_dataset("train")
    counts = df["target"].value_counts().sort_index()
    plt.bar([names[i] for i in counts.index], counts.values)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(OUT / "class_counts.png", dpi=150)
    print("Saved to outputs/class_counts.png")