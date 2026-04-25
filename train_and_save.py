"""
CartIQ - Electronics Recommendation System
Training Script: Pure NumPy/SciPy SVD (no scikit-surprise, Python 3.12 safe)
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")

# Always resolve paths relative to this script file, not the working directory
BASE_DIR  = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "electronics_small.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"

# ─────────────────────────────────────────────
# 1. Load & Clean Data
# ─────────────────────────────────────────────
print("=" * 55)
print("  CartIQ — Electronics Recommendation System")
print("=" * 55)
print("\n[1/5] Loading dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"\n[ERROR] Dataset not found: {DATA_PATH}\nPlace electronics_small.csv inside the data/ folder next to train_model.py")

df = pd.read_csv(DATA_PATH)
df.dropna(subset=["user", "item", "rating"], inplace=True)
df["rating"]  = df["rating"].astype(int)
df["review"]  = df["review"].fillna("")
df["summary"] = df["summary"].fillna("")
df["text"]    = df["summary"] + " " + df["review"]

print(f"      Records loaded : {len(df):,}")
print(f"      Unique users   : {df['user'].nunique():,}")
print(f"      Unique items   : {df['item'].nunique():,}")
print(f"      Rating range   : {df['rating'].min()} - {df['rating'].max()}")

# ─────────────────────────────────────────────
# 2. Build User-Item Matrix & Train SVD
# ─────────────────────────────────────────────
print("\n[2/5] Training SVD (scipy) collaborative filtering model...")

user_ids = df["user"].unique()
item_ids = df["item"].unique()
user2idx = {u: i for i, u in enumerate(user_ids)}
item2idx = {it: i for i, it in enumerate(item_ids)}
idx2user = {i: u for u, i in user2idx.items()}
idx2item = {i: it for it, i in item2idx.items()}

df["u_idx"] = df["user"].map(user2idx)
df["i_idx"] = df["item"].map(item2idx)

n_users = len(user_ids)
n_items = len(item_ids)

global_mean = df["rating"].mean()
user_bias = df.groupby("u_idx")["rating"].mean() - global_mean
item_bias = df.groupby("i_idx")["rating"].mean() - global_mean

data_vals = (df["rating"].values
             - global_mean
             - df["u_idx"].map(user_bias).fillna(0).values
             - df["i_idx"].map(item_bias).fillna(0).values)

R = csr_matrix((data_vals, (df["u_idx"].values, df["i_idx"].values)),
               shape=(n_users, n_items))

K = 50
U, sigma, Vt = svds(R.astype(float), k=K)

R_hat = (np.dot(np.dot(U, np.diag(sigma)), Vt)
         + global_mean
         + user_bias.reindex(range(n_users)).fillna(0).values[:, None]
         + item_bias.reindex(range(n_items)).fillna(0).values[None, :])
R_hat = np.clip(R_hat, 1, 5)

# 3-Fold CV
print("      Running 3-fold cross-validation...")
kf = KFold(n_splits=3, shuffle=True, random_state=42)
rmse_scores, mae_scores = [], []

for train_idx, test_idx in kf.split(df):
    tr = df.iloc[train_idx]
    te = df.iloc[test_idx]
    gm = tr["rating"].mean()
    ub = tr.groupby("u_idx")["rating"].mean() - gm
    ib = tr.groupby("i_idx")["rating"].mean() - gm
    d = (tr["rating"].values - gm
         - tr["u_idx"].map(ub).fillna(0).values
         - tr["i_idx"].map(ib).fillna(0).values)
    R_tr = csr_matrix((d, (tr["u_idx"].values, tr["i_idx"].values)),
                      shape=(n_users, n_items))
    U_, s_, Vt_ = svds(R_tr.astype(float), k=K)
    Rh = (np.dot(np.dot(U_, np.diag(s_)), Vt_) + gm
          + ub.reindex(range(n_users)).fillna(0).values[:, None]
          + ib.reindex(range(n_items)).fillna(0).values[None, :])
    Rh = np.clip(Rh, 1, 5)
    preds = Rh[te["u_idx"].values, te["i_idx"].values]
    rmse_scores.append(np.sqrt(mean_squared_error(te["rating"].values, preds)))
    mae_scores.append(mean_absolute_error(te["rating"].values, preds))

rmse = float(np.mean(rmse_scores))
mae  = float(np.mean(mae_scores))
print(f"      Cross-Val RMSE : {rmse:.4f}")
print(f"      Cross-Val MAE  : {mae:.4f}")
print("      SVD model trained ✓")

# ─────────────────────────────────────────────
# 3. Content-Based Filtering — TF-IDF
# ─────────────────────────────────────────────
print("\n[3/5] Building content-based TF-IDF index...")

item_text = df.groupby("item")["text"].apply(lambda x: " ".join(x)).reset_index()
item_text.columns = ["item", "combined_text"]

tfidf = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(item_text["combined_text"])
print(f"      TF-IDF matrix  : {tfidf_matrix.shape[0]} items x {tfidf_matrix.shape[1]} features")

item_avg = df.groupby("item")["rating"].agg(["mean", "count"]).reset_index()
item_avg.columns = ["item", "avg_rating", "review_count"]

# Build human-readable product names from review summaries
import re as _re
_GENERIC = {
    "five stars","four stars","three stars","two stars","one star",
    "great","good","nice","excellent","perfect","love it","love this",
    "awesome","amazing","best","worst","bad","terrible","ok","okay",
    "works great","works fine","works well","highly recommend","not happy",
    "disappointed","waste of money","as described","as advertised","happy",
}
def _pick_name(summaries):
    for s in summaries:
        s = str(s).strip()
        if (len(s) > 15
                and s.lower().rstrip(".!?") not in _GENERIC
                and not _re.fullmatch(r"[\w\s]{1,12}", s)):
            return s[:70]
    return None

_raw = df.groupby("item")["summary"].apply(list).reset_index()
_raw["display_name"] = _raw["summary"].apply(_pick_name)
_raw["display_name"] = _raw["display_name"].fillna(
    _raw["item"].apply(lambda x: f"Product …{x[-6:]}")
)
item_names = dict(zip(_raw["item"], _raw["display_name"]))   # {item_id -> name}
named_pct  = _raw["display_name"].apply(lambda n: not n.startswith("Product")).mean()
print(f"      Product names  : {named_pct*100:.0f}% resolved from review text")

# ─────────────────────────────────────────────
# 4. User History Index
# ─────────────────────────────────────────────
print("\n[4/5] Building user-history index...")
user_history = (
    df.groupby("user")
    .apply(lambda g: dict(zip(g["item"], g["rating"])))
    .to_dict()
)
print(f"      Indexed {len(user_history):,} user profiles")

# ─────────────────────────────────────────────
# 5. Save Artifacts
# ─────────────────────────────────────────────
print("\n[5/5] Saving model artifacts -> model/model.pkl ...")

MODEL_DIR.mkdir(exist_ok=True)
artifact = {
    "R_hat":        R_hat,
    "user2idx":     user2idx,
    "item2idx":     item2idx,
    "idx2user":     idx2user,
    "idx2item":     idx2item,
    "global_mean":  global_mean,
    "tfidf":        tfidf,
    "tfidf_matrix": tfidf_matrix,
    "item_text_df": item_text,
    "item_avg":     item_avg,
    "item_names":   item_names,        # {item_id -> human-readable name}
    "all_items":    list(item_ids),
    "all_users":    list(user_ids),
    "metrics": {
        "rmse":           round(rmse, 4),
        "mae":            round(mae, 4),
        "total_records":  len(df),
        "unique_users":   df["user"].nunique(),
        "unique_items":   df["item"].nunique(),
        "latent_factors": K,
    },
}

joblib.dump(artifact, MODEL_PATH, compress=3)
size_mb = MODEL_PATH.stat().st_size / 1e6
print(f"      Saved model/model.pkl  ({size_mb:.1f} MB)")
print("\n Training complete!\n")