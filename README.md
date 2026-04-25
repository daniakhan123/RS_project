# 🛒 CartIQ — Personalized Electronics Recommendation System

**FAST-NUCES · Dania Khan (23K-0072) · Tanisha Kataria (23K-0067)**

---

## Project Overview

CartIQ is an electronics recommendation system built on the Amazon Reviews 2018 dataset.
It combines three recommendation strategies into a live Streamlit demo.

| Technique | Description |
|---|---|
| **Collaborative Filtering (SVD)** | Matrix factorization on user–item ratings |
| **Content-Based (TF-IDF)** | Cosine similarity over review text embeddings |
| **Hybrid** | Weighted blend of SVD + content scores |

---

## Project Structure

```
cartiq/
├── data/
│   └── electronics_small.csv   ← reduced dataset (50k rows)
├── model/
│   └── model.pkl               ← trained artifacts (auto-generated)
├── train_model.py              ← trains & saves model.pkl
├── app.py                      ← Streamlit demo app
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```
This produces `model/model.pkl` containing the SVD model, TF-IDF vectorizer,
cosine-similarity matrix, and supporting lookup tables.

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

---

## Model Performance

| Metric | Value |
|---|---|
| Cross-Val RMSE | ~1.03 |
| Cross-Val MAE  | ~0.75 |
| CV Folds | 3 |

---

## Dataset

**Amazon Reviews 2018 — Electronics**  
Source: [Kaggle](https://www.kaggle.com/datasets/magdawjcicka/amazon-reviews-2018-electronics)

| Column  | Description |
|---|---|
| user    | Reviewer ID |
| item    | Product ASIN |
| rating  | Star rating 1–5 |
| review  | Full review text |
| summary | Short review title |

---

## App Features

- 🔵 **Collaborative Filtering tab** — pick any user, get SVD-predicted ratings
- 🟢 **Content-Based tab** — pick any item, find similar items by review text
- 🟠 **Hybrid tab** — blend both signals with an adjustable α slider
- 📊 **Rating history expander** — see what each user has already rated
- 📈 **Bar charts** — visual comparison of recommendation scores
