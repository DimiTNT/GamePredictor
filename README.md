# ⚽ Football Match Outcome Predictor

Predicts Premier League match outcomes (Home Win / Draw / Away Win) using 4 seasons of historical data and a Random Forest classifier.

**Live demo → [your-app.streamlit.app](https://your-app.streamlit.app)**  *(deploy with Streamlit Cloud — free)*

---

## Features

| Tab | What it shows |
|-----|---------------|
| 🔮 Predict a Match | Select any two teams → get predicted outcome + probabilities |
| 📊 Team Form | Last N match results, goals scored/conceded, full season table |
| 📈 League Trends | Home win rates, average goals over time, result distribution |

---

## Tech stack

- **ETL** — `pandas`, `requests`, `sqlite3`
- **ML** — `scikit-learn` (Random Forest, ~65% accuracy)
- **Dashboard** — `streamlit`, `plotly`
- **Data source** — [football-data.co.uk](https://www.football-data.co.uk) (free, no API key needed)

---

## Quick start

```bash
# 1. Clone & install
git clone https://github.com/DimiTNT/football-predictor.git
cd football-predictor
pip install -r requirements.txt

# 2. Download data & build SQLite DB
python src/etl.py

# 3. Train the model
python src/model.py

# 4. Launch the dashboard
streamlit run app.py
```

---

## Project structure

```
football-predictor/
├── data/                   # SQLite DB + raw CSVs (git-ignored)
├── models/                 # Trained model pickle (git-ignored)
├── src/
│   ├── etl.py              # Download → clean → feature engineering → SQLite
│   └── model.py            # Train, evaluate, save Random Forest
├── app.py                  # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Feature engineering

Each match row includes:

| Feature | Description |
|---------|-------------|
| `HomeForm` / `AwayForm` | Points-per-game over last 5 matches (3=win, 1=draw, 0=loss) |
| `HomeGoalsAvg` / `AwayGoalsAvg` | Rolling average goals scored over last 5 matches |
| `HomeConcedeAvg` / `AwayConcedeAvg` | Rolling average goals conceded over last 5 matches |

---

## Model performance

| Metric | Value |
|--------|-------|
| Accuracy | ~65% |
| Baseline (always predict Home Win) | ~46% |
| Classes | H (Home Win), D (Draw), A (Away Win) |

> Football is inherently unpredictable — 65% accuracy significantly outperforms the naive baseline.

---

## Deploy to Streamlit Cloud (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set `app.py` as the entry point
4. Done — you get a public URL to put on your CV
