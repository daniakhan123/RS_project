import joblib
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent

print("Loading model...")
art = joblib.load(BASE / "model" / "model.pkl")

print("Building user_history from dataset...")
df = pd.read_csv(BASE / "data" / "electronics_small.csv")
df.dropna(subset=["user", "item", "rating"], inplace=True)
df["rating"] = df["rating"].astype(int)

user_history = (
    df.groupby("user")
    .apply(lambda g: dict(zip(g["item"], g["rating"])))
    .to_dict()
)

art["user_history"] = user_history
print(f"Added user_history for {len(user_history):,} users")

joblib.dump(art, BASE / "model" / "model.pkl", compress=3)
print("Saved! Keys now:", list(art.keys()))
print("\nDone! Now run:  python -m streamlit run app_streamlit.py")