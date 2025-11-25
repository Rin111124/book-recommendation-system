## Book Recommendation System – End-to-End Guide

This repository contains everything needed to build a book recommendation platform using collaborative filtering, matrix factorization (SVD), and a hybrid ensemble. Follow the steps below to go from an empty workspace to a running Flask demo.

---

## 1. Prerequisites

- **Python**: 3.10 or 3.11 (scientific stack does not yet support 3.12+).
- **Pip / Virtualenv**: any environment manager is fine.
- **Kaggle account** with API access (Settings → Create New Token).
- **Git** (optional but recommended).

---

## 2. Repository Structure

```
book-recommendation-system/
├── config.py                 # Paths + hyperparameters
├── requirements.txt          # Python dependencies
├── data/
│   ├── raw/                  # Kaggle CSVs (Books/Users/Ratings)
│   ├── processed/            # Cleaned datasets
│   └── models/               # Serialized artifacts (future use)
├── notebooks/                # Guided notebooks 00–04
├── src/                      # Core Python modules (preprocessing, CF, SVD, hybrid, metrics)
└── web_app/                  # Flask UI + API
```

---

## 3. Setup & Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> Tip: If you see SciPy or Surprise errors, re-check that you are using Python 3.10/3.11 and that `scikit-surprise` installed correctly.

---

## 4. Configure Kaggle Access

1. Download `kaggle.json` from [Kaggle Account → Create New Token](https://www.kaggle.com/account/token).
2. Place the file in:
   - Windows: `%USERPROFILE%\.kaggle\kaggle.json`
   - macOS/Linux: `~/.kaggle/kaggle.json`
3. Ensure the file is only readable by you (on Unix: `chmod 600 ~/.kaggle/kaggle.json`).
4. Alternatively set env vars before running notebooks:
   ```bash
   set KAGGLE_USERNAME=your_username
   set KAGGLE_KEY=your_api_key
   ```

---

## 5. Download & Prepare Data

The fastest way is via the guided notebook:

1. Launch Jupyter (VS Code, Jupyter Lab, or `jupyter notebook`).
2. Open `notebooks/00_Kaggle_Download.ipynb`.
3. Run every cell in order:
   - installs `kaggle` CLI,
   - downloads `ra4u/book-crossing-dataset`,
   - saves `Books.csv`, `Users.csv`, `Ratings.csv` into `data/raw/`,
   - runs `src/data_preprocessing.preprocess_pipeline()` to create processed CSVs,
   - sanity-checks a quick collaborative filtering recommendation.

After completion you should see:

```
data/raw/Books.csv
data/raw/Ratings.csv
data/raw/Users.csv
data/processed/books_processed.csv
...
```

---

## 6. Explore & Train Models

Progress through the remaining notebooks:

1. **01_EDA.ipynb** – exploratory plots, rating distributions, descriptive stats.
2. **02_Collaborative_Filtering.ipynb** – user/item based CF, similarity matrices, evaluation.
3. **03_Matrix_Factorization.ipynb** – SVD training/evaluation with Surprise.
4. **04_Model_Evaluation.ipynb** – compare CF vs SVD vs hybrid, compute RMSE/MAE/Precision@K/etc.

All notebooks rely on the modules in `src/`, so you can move logic into scripts later if needed.

---

## 7. Reusable Python Modules

- `src/data_preprocessing.py` – load/clean/filter/split ratings, build matrices, save processed data.
- `src/collaborative_filtering.py` – user-based & item-based CF with cosine similarity.
- `src/matrix_factorization.py` – Surprise SVD wrapper + evaluation helpers.
- `src/hybrid_model.py` – weighted ensemble of CF+SVD, cold-start routing.
- `src/evaluation.py` – metrics (RMSE, MAE, precision@K, recall@K, NDCG, coverage, diversity).

Import these modules in your scripts/notebooks to avoid duplicating code.

---

## 8. Run the Flask Web App

1. Make sure data is preprocessed (Section 5).
2. Start the server:

   ```bash
   set FLASK_APP=web_app/app.py        # Windows (cmd)
   set FLASK_ENV=development
   flask run

   # macOS/Linux
   export FLASK_APP=web_app/app.py
   export FLASK_ENV=development
   flask run
   ```

3. Visit `http://127.0.0.1:5000`. Enter a user ID (default from `config.py`) and choose the algorithm (user CF, item CF, SVD, or hybrid). If the dataset is missing you’ll see a warning banner asking you to place the CSV files under `data/raw/`.

API usage:
```bash
curl -X POST http://127.0.0.1:5000/api/recommend ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\":276729,\"algorithm\":\"hybrid\",\"limit\":10}"
```

---

## 9. Next Steps / Customization Ideas

- Tune hyperparameters in `config.py`.
- Persist trained models into `data/models/` for faster web start-up.
- Add more evaluation metrics (coverage, diversity) using `src/evaluation.py`.
- Deploy the Flask app on Render/Railway/Heroku by containerizing the project.
- Extend the hybrid model with switching logic for cold-start users or integrate content metadata (authors, genres).

---

## Need Help?

Open an issue or extend the notebooks with additional experiments. Every artifact in this repo is meant to be a starting point—customize freely to match your recommendation goals.
