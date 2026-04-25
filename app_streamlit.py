import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CartIQ — Electronics Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.pkl"
DATA_PATH  = Path(__file__).resolve().parent / "data" / "electronics_small.csv"

# ── Load Model ───────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        st.error(f"model.pkl not found. Run train_and_save.py first.")
        st.stop()
    art = joblib.load(MODEL_PATH)

    # Load raw dataset for review text lookup
    df = pd.read_csv(DATA_PATH)
    df.dropna(subset=["user", "item", "rating"], inplace=True)
    df["rating"]  = df["rating"].astype(int)
    df["review"]  = df["review"].fillna("")
    df["summary"] = df["summary"].fillna("")

    # item -> best review snippet (highest rated, max 200 chars)
    item_review = (
        df.sort_values("rating", ascending=False)
        .groupby("item")
        .first()[["summary", "review"]]
        .reset_index()
    )
    item_review["snippet"] = item_review["review"].str[:180].str.strip() + "…"

    # user+item -> their specific review
    df_dedup = df.sort_values("rating", ascending=False).drop_duplicates(subset=["user", "item"])
    user_item_review = df_dedup.set_index(["user", "item"])[["rating", "summary", "review"]].to_dict("index")

    return art, item_review.set_index("item"), user_item_review

art, item_review_df, user_item_review = load_artifacts()

R_hat        = art["R_hat"]
user2idx     = art["user2idx"]
item2idx     = art["item2idx"]
idx2item     = art["idx2item"]
tfidf_matrix = art["tfidf_matrix"]
item_text_df = art["item_text_df"]
item_avg     = art["item_avg"]
user_history = art["user_history"]
all_items    = art["all_items"]
all_users    = art["all_users"]
metrics      = art["metrics"]

item_tfidf_idx = {row["item"]: i for i, row in item_text_df.iterrows()}

# ── Helpers ──────────────────────────────────────────────
def star_bar(rating, max_r=5):
    f = max(0, min(int(round(float(rating))), max_r))
    return "⭐" * f + "☆" * (max_r - f)

def get_avg(item_id):
    r = item_avg[item_avg["item"] == item_id]
    if r.empty:
        return 0.0, 0
    return float(r["avg_rating"].values[0]), int(r["review_count"].values[0])

def get_snippet(item_id):
    if item_id in item_review_df.index:
        row = item_review_df.loc[item_id]
        return row["snippet"]
    return "No review available."

def render_product_card(item_id, score_label, score_value, extra_col=None):
    """Render a clean product card with ID, rating, score, and review snippet."""
    avg, cnt = get_avg(item_id)
    snippet   = get_snippet(item_id)

    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**🏷️ Product ID:** `{item_id}`")
            st.markdown(f"**{score_label}:** `{score_value}`  &nbsp;&nbsp; **Community Rating:** {star_bar(avg)} `{avg:.2f}` ({cnt} reviews)")
            st.caption(f"💬 *\"{snippet}\"*")
        with col2:
            if extra_col:
                st.markdown(extra_col)

# ── Recommendation Logic ─────────────────────────────────
def cf_recommendations(user_id, top_n):
    u = user2idx.get(user_id)
    if u is None:
        return []
    rated = set(user_history.get(user_id, {}).keys())
    rows = [(idx2item[i], float(R_hat[u, i]))
            for i in range(len(idx2item)) if idx2item[i] not in rated]
    rows.sort(key=lambda x: -x[1])
    return rows[:top_n]

def cb_recommendations(item_id, top_n):
    idx = item_tfidf_idx.get(item_id)
    if idx is None:
        return []
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    top_idx = sims.argsort()[::-1][1:top_n + 1]
    return [(item_text_df.iloc[i]["item"], float(sims[i])) for i in top_idx]

def hybrid_recommendations(user_id, top_n, alpha):
    u = user2idx.get(user_id)
    if u is None:
        return []
    rated     = user_history.get(user_id, {})
    rated_set = set(rated.keys())

    svd_scores = {idx2item[i]: float(R_hat[u, i])
                  for i in range(len(idx2item)) if idx2item[i] not in rated_set}
    vals = np.array(list(svd_scores.values()))
    mn, mx = vals.min(), vals.max()
    svd_norm = {k: (v - mn) / (mx - mn + 1e-9) for k, v in svd_scores.items()}

    liked = [it for it, r in rated.items() if r >= 4 and it in item_tfidf_idx]
    cb_norm = {}
    if liked:
        liked_vecs = tfidf_matrix[[item_tfidf_idx[it] for it in liked]]
        cand_items = [it for it in svd_scores if it in item_tfidf_idx]
        cand_vecs  = tfidf_matrix[[item_tfidf_idx[it] for it in cand_items]]
        sims = cosine_similarity(liked_vecs, cand_vecs).mean(axis=0)
        for it, s in zip(cand_items, sims):
            cb_norm[it] = float(s)

    scored = [(it, alpha * svd_norm.get(it, 0) + (1 - alpha) * cb_norm.get(it, 0),
               svd_scores.get(it, 0))
              for it in svd_scores]
    scored.sort(key=lambda x: -x[1])
    return scored[:top_n]

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=64)
    st.title("CartIQ")
    st.caption("Electronics Recommendation System")
    st.divider()
    mode  = st.radio("🔍 Mode", ["Collaborative Filtering", "Content-Based", "Hybrid"], index=2)
    top_n = st.slider("📋 Results", 3, 15, 6)
    st.divider()
    st.subheader("📊 Stats")
    st.metric("Total Reviews", f"{metrics['total_records']:,}")
    st.metric("Unique Users",  f"{metrics['unique_users']:,}")
    st.metric("Unique Items",  f"{metrics['unique_items']:,}")
    st.metric("SVD RMSE",      str(metrics["rmse"]))
    st.metric("SVD MAE",       str(metrics["mae"]))

# ── Header ───────────────────────────────────────────────
st.markdown("<h1 style='margin-bottom:0'>🛒 CartIQ</h1>", unsafe_allow_html=True)
st.caption("Personalized E-Commerce Recommendation System · FAST-NUCES")
st.divider()

# ── Collaborative Filtering ──────────────────────────────
if mode == "Collaborative Filtering":
    st.subheader("👥 Collaborative Filtering (SVD)")
    st.markdown("Predicts ratings for products the user hasn't seen, based on similar users.")

    col1, col2 = st.columns([3, 1])
    with col1:
        user_id = st.selectbox("Select User ID", all_users[:500])
    with col2:
        st.metric("Their reviews", len(user_history.get(user_id, {})))

    if st.button("🚀 Recommend", type="primary"):
        # Show user's own review history first
        past = user_history.get(user_id, {})
        if past:
            with st.expander(f"📖 {user_id}'s Rating History ({len(past)} products)"):
                for item_id, rating in sorted(past.items(), key=lambda x: -x[1]):
                    key = (user_id, item_id)
                    rev = user_item_review.get(key, {})
                    summary = rev.get("summary", "")
                    review  = rev.get("review",  "")[:160] + "…"
                    with st.container(border=True):
                        st.markdown(f"**🏷️ Product ID:** `{item_id}` &nbsp;&nbsp; **Their Rating:** {star_bar(rating)} `{rating}/5`")
                        if summary:
                            st.markdown(f"**Review title:** *{summary}*")
                        st.caption(f"💬 *\"{review}\"*")

        st.markdown(f"### 🎯 Top {top_n} Recommendations for `{user_id}`")
        recs = cf_recommendations(user_id, top_n)
        if not recs:
            st.warning("User not found in model.")
        else:
            for item_id, pred in recs:
                render_product_card(item_id,
                                    "Predicted Rating",
                                    f"{pred:.2f} / 5  {star_bar(pred)}")

# ── Content-Based ────────────────────────────────────────
elif mode == "Content-Based":
    st.subheader("📝 Content-Based Filtering (TF-IDF + Cosine Similarity)")
    st.markdown("Finds products with the most similar review language to a chosen product.")

    item_id = st.selectbox("Select a Product ID", all_items[:500])

    # Show the selected product's info
    avg, cnt = get_avg(item_id)
    snippet  = get_snippet(item_id)
    st.info(f"**Selected:** `{item_id}` · Community Rating: {star_bar(avg)} `{avg:.2f}` ({cnt} reviews)\n\n💬 *\"{snippet}\"*")

    if st.button("🔍 Find Similar Products", type="primary"):
        recs = cb_recommendations(item_id, top_n)
        if not recs:
            st.warning("Item not in content index.")
        else:
            st.markdown(f"### 🎯 Products Similar to `{item_id}`")
            for rec_id, sim in recs:
                render_product_card(rec_id, "Similarity Score", f"{sim:.4f}")

# ── Hybrid ───────────────────────────────────────────────
else:
    st.subheader("🔀 Hybrid Recommender (SVD + Content-Based)")
    st.markdown("Blends personalised SVD predictions with content similarity to your liked products.")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        user_id = st.selectbox("Select User ID", all_users[:500], key="h_user")
    with col2:
        alpha = st.slider("SVD weight α", 0.0, 1.0, 0.6, 0.05)
    with col3:
        st.metric("Mix", f"{int(alpha*100)}% SVD\n{int((1-alpha)*100)}% Content")

    if st.button("⚡ Recommend", type="primary"):
        past = user_history.get(user_id, {})
        if past:
            with st.expander(f"📖 {user_id}'s Rating History ({len(past)} products)"):
                for item_id, rating in sorted(past.items(), key=lambda x: -x[1]):
                    key = (user_id, item_id)
                    rev = user_item_review.get(key, {})
                    summary = rev.get("summary", "")
                    review  = rev.get("review",  "")[:160] + "…"
                    with st.container(border=True):
                        st.markdown(f"**🏷️ Product ID:** `{item_id}` &nbsp;&nbsp; **Their Rating:** {star_bar(rating)} `{rating}/5`")
                        if summary:
                            st.markdown(f"**Review title:** *{summary}*")
                        st.caption(f"💬 *\"{review}\"*")

        st.markdown(f"### 🎯 Top {top_n} Hybrid Recommendations for `{user_id}`")
        recs = hybrid_recommendations(user_id, top_n, alpha)
        if not recs:
            st.warning("Not enough data.")
        else:
            for item_id, hybrid_score, svd_rating in recs:
                render_product_card(
                    item_id,
                    "Hybrid Score",
                    f"{hybrid_score:.4f}",
                    extra_col=f"**SVD Rating:** `{svd_rating:.2f}/5`\n\n{star_bar(svd_rating)}"
                )

# ── Footer ───────────────────────────────────────────────
st.divider()
with st.expander("📈 Community Rating Distribution"):
    dist = item_avg["avg_rating"].round(0).value_counts().sort_index()
    dist.index = dist.index.astype(int).astype(str) + " star"
    st.bar_chart(dist)
st.caption("CartIQ · Amazon Reviews Dataset · FAST-NUCES · Dania Khan & Tanisha Kataria")