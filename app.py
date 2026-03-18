"""
app.py
------
Streamlit dashboard for the Football Match Outcome Predictor.

Run:
    streamlit run app.py
"""

import sqlite3
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DB_PATH    = "data/football.db"
MODEL_PATH = "models/rf_model.pkl"
ENC_PATH   = "models/label_encoder.pkl"
FEATURES   = [
    "HomeForm", "AwayForm",
    "HomeGoalsAvg", "AwayGoalsAvg",
    "HomeConcedeAvg", "AwayConcedeAvg",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM matches", con)
    con.close()
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENC_PATH, "rb") as f:
        le = pickle.load(f)
    return model, le

def team_form_chart(df: pd.DataFrame, team: str, n: int = 10):
    mask = (df["HomeTeam"] == team) | (df["AwayTeam"] == team)
    team_df = df[mask].sort_values("Date").tail(n).copy()

    def result_for_team(row):
        if row["HomeTeam"] == team:
            return {"H": "Win", "D": "Draw", "A": "Loss"}[row["FTR"]]
        else:
            return {"A": "Win", "D": "Draw", "H": "Loss"}[row["FTR"]]

    team_df["Result"]    = team_df.apply(result_for_team, axis=1)
    team_df["Opponent"]  = team_df.apply(
        lambda r: r["AwayTeam"] if r["HomeTeam"] == team else r["HomeTeam"], axis=1
    )
    team_df["GoalsFor"]  = team_df.apply(
        lambda r: r["FTHG"] if r["HomeTeam"] == team else r["FTAG"], axis=1
    )
    team_df["GoalsAgainst"] = team_df.apply(
        lambda r: r["FTAG"] if r["HomeTeam"] == team else r["FTHG"], axis=1
    )

    color_map = {"Win": "#2ecc71", "Draw": "#f39c12", "Loss": "#e74c3c"}
    fig = px.bar(
        team_df, x="Date", y="GoalsFor",
        color="Result", color_discrete_map=color_map,
        hover_data=["Opponent", "GoalsFor", "GoalsAgainst", "Result"],
        title=f"{team} — Last {n} matches",
        labels={"GoalsFor": "Goals Scored"},
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f0f0f0",
        legend_title_text="Result",
    )
    return fig

def prediction_gauge(home_prob: float, draw_prob: float, away_prob: float,
                     home: str, away: str):
    labels = [f"{home} Win", "Draw", f"{away} Win"]
    values = [home_prob, draw_prob, away_prob]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    fig = go.Figure(go.Bar(
        x=labels, y=[v * 100 for v in values],
        marker_color=colors,
        text=[f"{v*100:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Predicted outcome probabilities",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f0f0f0",
    )
    return fig

def win_rate_chart(df: pd.DataFrame):
    home_wins  = df[df["FTR"] == "H"].groupby("HomeTeam").size().rename("Wins")
    home_total = df.groupby("HomeTeam").size().rename("Total")
    rate = (home_wins / home_total * 100).dropna().sort_values(ascending=False).head(15)
    fig = px.bar(
        rate.reset_index(), x="HomeTeam", y=0,
        labels={"HomeTeam": "Team", "0": "Home Win Rate (%)"},
        title="Top 15 Home Win Rates (all seasons)",
        color=rate.values, color_continuous_scale="Greens",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f0f0f0",
        coloraxis_showscale=False,
    )
    return fig

def goals_trend_chart(df: pd.DataFrame):
    df2 = df.copy()
    df2["Month"] = df2["Date"].dt.to_period("M").astype(str)
    monthly = df2.groupby("Month")[["FTHG", "FTAG"]].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["FTHG"],
                             mode="lines+markers", name="Home goals avg", line=dict(color="#2ecc71")))
    fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["FTAG"],
                             mode="lines+markers", name="Away goals avg", line=dict(color="#e74c3c")))
    fig.update_layout(
        title="Average goals per match over time",
        xaxis_title="Month", yaxis_title="Avg Goals",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f0f0f0",
    )
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="⚽ Football Predictor", layout="wide", page_icon="⚽")

st.title("⚽ Premier League — Match Outcome Predictor")
st.caption("4 seasons of data (2020–2024) · Random Forest classifier · football-data.co.uk")

df = load_data()
teams = sorted(df["HomeTeam"].unique().tolist())

# ── Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Predict a Match", "📊 Team Form", "📈 League Trends"])

# ── Tab 1 : Prediction
with tab1:
    st.subheader("Predict match outcome")
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("🏠 Home team", teams, index=teams.index("Man City") if "Man City" in teams else 0)
    with col2:
        away_options = [t for t in teams if t != home_team]
        away_team = st.selectbox("✈️ Away team", away_options)

    if st.button("Predict", type="primary"):
        try:
            model, le = load_model()

            def latest(team, home):
                prefix = "Home" if home else "Away"
                mask = (df["HomeTeam"] == team) if home else (df["AwayTeam"] == team)
                row = df[mask].sort_values("Date").iloc[-1]
                return {
                    f"{prefix}Form":       row[f"{prefix}Form"],
                    f"{prefix}GoalsAvg":   row[f"{prefix}GoalsAvg"],
                    f"{prefix}ConcedeAvg": row[f"{prefix}ConcedeAvg"],
                }

            X = pd.DataFrame([{**latest(home_team, True), **latest(away_team, False)}])[FEATURES]
            proba = model.predict_proba(X)[0]
            proba_dict = dict(zip(le.classes_, proba))
            pred = le.inverse_transform([proba.argmax()])[0]
            label = {"H": f"🏠 {home_team} wins", "D": "🤝 Draw", "A": f"✈️ {away_team} wins"}[pred]

            st.success(f"**Predicted outcome : {label}**")
            st.plotly_chart(
                prediction_gauge(proba_dict.get("H", 0), proba_dict.get("D", 0), proba_dict.get("A", 0),
                                 home_team, away_team),
                use_container_width=True
            )

            with st.expander("Show input features"):
                st.dataframe(X.T.rename(columns={0: "Value"}).round(3))

        except FileNotFoundError:
            st.error("Model not found — run `python src/model.py` first.")

# ── Tab 2 : Team form
with tab2:
    st.subheader("Team recent form")
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_team = st.selectbox("Select a team", teams)
    with col2:
        n_matches = st.slider("Last N matches", 5, 20, 10)

    st.plotly_chart(team_form_chart(df, selected_team, n_matches), use_container_width=True)

    # Season stats table
    season = st.selectbox("Filter by season", ["All"] + sorted(df["Season"].unique().tolist(), reverse=True))
    filtered = df if season == "All" else df[df["Season"] == season]
    mask = (filtered["HomeTeam"] == selected_team) | (filtered["AwayTeam"] == selected_team)
    team_matches = filtered[mask].copy()

    def result_label(row):
        if row["HomeTeam"] == selected_team:
            return {"H": "Win", "D": "Draw", "A": "Loss"}[row["FTR"]]
        return {"A": "Win", "D": "Draw", "H": "Loss"}[row["FTR"]]

    team_matches["Result"]   = team_matches.apply(result_label, axis=1)
    team_matches["Opponent"] = team_matches.apply(
        lambda r: r["AwayTeam"] if r["HomeTeam"] == selected_team else r["HomeTeam"], axis=1
    )
    st.dataframe(
        team_matches[["Date", "Opponent", "FTHG", "FTAG", "Result"]]
        .sort_values("Date", ascending=False)
        .rename(columns={"FTHG": "Home Goals", "FTAG": "Away Goals"})
        .reset_index(drop=True),
        use_container_width=True,
    )

# ── Tab 3 : League trends
with tab3:
    st.subheader("League-wide trends")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(win_rate_chart(df), use_container_width=True)
    with col2:
        st.plotly_chart(goals_trend_chart(df), use_container_width=True)

    # Result distribution
    result_counts = df["FTR"].value_counts().rename({"H": "Home Win", "D": "Draw", "A": "Away Win"})
    fig_pie = px.pie(
        values=result_counts.values,
        names=result_counts.index,
        title="Overall result distribution",
        color_discrete_sequence=["#2ecc71", "#f39c12", "#e74c3c"],
    )
    fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#f0f0f0")
    st.plotly_chart(fig_pie, use_container_width=True)
