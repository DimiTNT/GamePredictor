"""
app.py
------
Streamlit dashboard for the Football Match Outcome Predictor.

Run:
    streamlit run app.py
"""

import json
import sqlite3
import pickle
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DB_PATH        = "data/football.db"
MODEL_PATH     = "models/rf_model.pkl"
ENC_PATH       = "models/label_encoder.pkl"
TIMESTAMP_PATH = "data/last_updated.txt"
LOG_PATH       = "data/prediction_log.json"
CURRENT_SEASON = "2025-26"
REFRESH_DAYS   = 7

FEATURES = [
    "HomeForm", "AwayForm",
    "HomeGoalsAvg", "AwayGoalsAvg",
    "HomeConcedeAvg", "AwayConcedeAvg",
    "H2H_HomeWinRate", "H2H_AwayWinRate",
    "H2H_HomeGoalsAvg", "H2H_AwayGoalsAvg",
]

# FPL team name → database team name (only exceptions needed)
FPL_TO_DB = {
    "Man Utd":  "Man United",
    "Spurs":    "Tottenham",
}

STATUS_LABEL = {
    "i": "Injured",
    "d": "Doubtful",
    "s": "Suspended",
}

# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM matches", con)
    con.close()
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    return df

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENC_PATH, "rb") as f:
        le = pickle.load(f)
    return model, le

def last_updated() -> datetime | None:
    p = Path(TIMESTAMP_PATH)
    if p.exists():
        try:
            return datetime.fromisoformat(p.read_text().strip())
        except ValueError:
            return None
    return None

def run_refresh():
    result = subprocess.run([sys.executable, "etl.py"], capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr
    result = subprocess.run([sys.executable, "model.py"], capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr
    load_data.clear()
    load_model.clear()
    return True, ""

# ── Injury helpers ────────────────────────────────────────────────────────────

@st.cache_data(ttl=10800)  # refresh every 3 hours
def fetch_fpl_injuries() -> dict[str, list[dict]]:
    """
    Pulls live injury/suspension data from the Fantasy Premier League API.
    Only includes actual injuries (i), doubts (d), suspensions (s) —
    players on loan (status=u) are excluded.
    Returns {db_team_name: [player_dict, ...]}
    """
    try:
        resp = requests.get(
            "https://fantasy.premierleague.com/api/bootstrap-static/",
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {}

    team_map = {t["id"]: t["name"] for t in data["teams"]}
    result: dict[str, list[dict]] = {}

    for p in data["elements"]:
        if p["status"] not in ("i", "d", "s"):
            continue  # skip loans and available players

        fpl_name = team_map.get(p["team"], "")
        db_name  = FPL_TO_DB.get(fpl_name, fpl_name)

        chance = p.get("chance_of_playing_next_round")
        if chance is None:
            chance = p.get("chance_of_playing_this_round")

        try:
            ownership = float(p.get("selected_by_percent") or 0)
        except (ValueError, TypeError):
            ownership = 0.0

        try:
            minutes = int(p.get("minutes") or 0)
            total_pts = int(p.get("total_points") or 0)
            ppg = float(p.get("points_per_game") or 0)
        except (ValueError, TypeError):
            minutes, total_pts, ppg = 0, 0, 0.0

        result.setdefault(db_name, []).append({
            "name":       p["web_name"],
            "status":     p["status"],
            "label":      STATUS_LABEL.get(p["status"], p["status"]),
            "chance":     chance,
            "news":       p.get("news", ""),
            "ownership":  ownership,
            "minutes":    minutes,    # season minutes played — best proxy for starter
            "total_pts":  total_pts,  # season FPL points
            "ppg":        ppg,        # points per game
        })

    return result


# Weight each classification applies to the injury burden calculation
PLAYER_WEIGHT = {"Key player": 1.0, "Normal": 0.4, "Irrelevant": 0.0}


def auto_classify_player(player: dict) -> str:
    """
    Classify an injured player's importance using objective FPL season stats.

    Logic (in priority order):
      - minutes > 900  AND total_pts > 70  → Key player  (undisputed starter)
      - ppg >= 5.5     OR  total_pts > 55  → Key player  (high-impact player)
      - minutes > 300  AND total_pts > 25  → Normal      (squad regular)
      - otherwise                          → Irrelevant  (fringe / backup)

    This means a goalkeeper with 1800 min / 80 pts is Key, a bench striker
    with 120 min / 8 pts is Irrelevant — no human judgment required.
    """
    m  = player.get("minutes", 0) or 0
    tp = player.get("total_pts", 0) or 0
    pg = player.get("ppg", 0.0) or 0.0

    if m > 900 and tp > 70:
        return "Key player"
    if pg >= 5.5 or tp > 55:
        return "Key player"
    if m > 300 and tp > 25:
        return "Normal"
    return "Irrelevant"


# Keep old name as alias so the manual predict tab still works unchanged
default_classification = auto_classify_player


def injury_team_factor(players: list[dict], overrides: dict[str, str]) -> float:
    """
    Returns a multiplier (0.80–1.00) reflecting how much a team is weakened.
    Uses user-supplied classification overrides; falls back to ownership-based
    defaults for any player the user hasn't classified.
    """
    burden = 0.0
    for p in players:
        classification = overrides.get(p["name"], default_classification(p))
        weight = PLAYER_WEIGHT.get(classification, 0.4)
        if weight == 0.0:
            continue
        chance  = p["chance"] if p["chance"] is not None else 0
        unavail = (100 - chance) / 100
        burden += unavail * weight
    return max(0.80, 1.0 - 0.12 * burden)


def apply_injury_factor(feats: dict, prefix: str, factor: float) -> dict:
    feats = dict(feats)
    feats[f"{prefix}Form"]       *= factor
    feats[f"{prefix}GoalsAvg"]   *= factor
    feats[f"{prefix}ConcedeAvg"] /= factor
    return feats


def predict_fixture(home: str, away: str, df: pd.DataFrame,
                    model, le, all_injuries: dict) -> dict | None:
    """
    Run a full prediction for one fixture, auto-classifying injuries.
    Returns a result dict or None if teams not in DB.
    """
    home_mask = df["HomeTeam"] == home
    away_mask = df["AwayTeam"] == away
    if not home_mask.any() or not away_mask.any():
        return None

    def latest(team, home_side):
        prefix = "Home" if home_side else "Away"
        mask   = (df["HomeTeam"] == team) if home_side else (df["AwayTeam"] == team)
        row    = df[mask].sort_values("Date").iloc[-1]
        return {
            f"{prefix}Form":       row[f"{prefix}Form"],
            f"{prefix}GoalsAvg":   row[f"{prefix}GoalsAvg"],
            f"{prefix}ConcedeAvg": row[f"{prefix}ConcedeAvg"],
        }

    home_feats = latest(home, True)
    away_feats = latest(away, False)
    h2h        = h2h_stats(df, home, away)

    home_players = all_injuries.get(home, [])
    away_players = all_injuries.get(away, [])

    # Fully automatic — no overrides, uses auto_classify_player for all
    h_factor = injury_team_factor(home_players, {})
    a_factor = injury_team_factor(away_players, {})

    if h_factor < 1.0:
        home_feats = apply_injury_factor(home_feats, "Home", h_factor)
    if a_factor < 1.0:
        away_feats = apply_injury_factor(away_feats, "Away", a_factor)

    X     = pd.DataFrame([{**home_feats, **away_feats, **h2h}])[FEATURES]
    proba = model.predict_proba(X)[0]
    pdict = dict(zip(le.classes_, proba))
    pred  = le.inverse_transform([proba.argmax()])[0]

    return {
        "home": home, "away": away,
        "pred": pred,
        "prob_H": pdict.get("H", 0),
        "prob_D": pdict.get("D", 0),
        "prob_A": pdict.get("A", 0),
        "h_factor": h_factor,
        "a_factor": a_factor,
        "home_key_injured": [p["name"] for p in home_players
                             if auto_classify_player(p) == "Key player"],
        "away_key_injured": [p["name"] for p in away_players
                             if auto_classify_player(p) == "Key player"],
    }

# ── Chart helpers ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_pl_news() -> list[dict]:
    try:
        resp = requests.get(
            "https://feeds.bbci.co.uk/sport/football/premier-league/rss.xml",
            timeout=8,
        )
        resp.raise_for_status()
        root  = ET.fromstring(resp.content)
        items = root.findall(".//item")
        return [
            {
                "title": item.findtext("title", ""),
                "link":  item.findtext("link", ""),
                "date":  item.findtext("pubDate", ""),
                "desc":  item.findtext("description", ""),
            }
            for item in items[:12]
        ]
    except Exception:
        return []

def h2h_stats(df: pd.DataFrame, home: str, away: str, n: int = 10) -> dict:
    mask = (
        ((df["HomeTeam"] == home) & (df["AwayTeam"] == away)) |
        ((df["HomeTeam"] == away) & (df["AwayTeam"] == home))
    )
    past = df[mask].sort_values("Date").tail(n)
    if past.empty:
        return {"H2H_HomeWinRate": 0.5, "H2H_AwayWinRate": 0.5,
                "H2H_HomeGoalsAvg": 1.2, "H2H_AwayGoalsAvg": 1.2}
    home_wins = away_wins = 0
    home_goals, away_goals = [], []
    for _, m in past.iterrows():
        gh = m["FTHG"] if m["HomeTeam"] == home else m["FTAG"]
        ga = m["FTAG"] if m["HomeTeam"] == home else m["FTHG"]
        home_goals.append(gh); away_goals.append(ga)
        if (m["FTR"] == "H" and m["HomeTeam"] == home) or (m["FTR"] == "A" and m["HomeTeam"] == away):
            home_wins += 1
        elif (m["FTR"] == "A" and m["HomeTeam"] == home) or (m["FTR"] == "H" and m["HomeTeam"] == away):
            away_wins += 1
    total = len(past)
    return {
        "H2H_HomeWinRate":  home_wins / total,
        "H2H_AwayWinRate":  away_wins / total,
        "H2H_HomeGoalsAvg": sum(home_goals) / total,
        "H2H_AwayGoalsAvg": sum(away_goals) / total,
    }

def style_results_table(display: pd.DataFrame) -> object:
    """Apply green/yellow/red row background based on the Result column."""
    def row_color(row):
        c = {"Win": "#1a4d2e", "Draw": "#4a3d00", "Loss": "#4d1a1a"}.get(row["Result"], "")
        return [f"background-color: {c}; color: #f0f0f0" if c else ""] * len(row)
    return display.style.apply(row_color, axis=1)


def season_standings(df: pd.DataFrame, season: str) -> pd.DataFrame:
    s = df[df["Season"] == season]
    all_teams = sorted(set(s["HomeTeam"].tolist() + s["AwayTeam"].tolist()))
    rows = []
    for team in all_teams:
        home = s[s["HomeTeam"] == team]
        away = s[s["AwayTeam"] == team]
        w  = int((home["FTR"] == "H").sum() + (away["FTR"] == "A").sum())
        d  = int((home["FTR"] == "D").sum() + (away["FTR"] == "D").sum())
        l  = int((home["FTR"] == "A").sum() + (away["FTR"] == "H").sum())
        gf = int(home["FTHG"].sum() + away["FTAG"].sum())
        ga = int(home["FTAG"].sum() + away["FTHG"].sum())
        cs = int((home["FTAG"] == 0).sum() + (away["FTHG"] == 0).sum())
        rows.append({"Team": team, "P": w+d+l, "W": w, "D": d, "L": l,
                     "GF": gf, "GA": ga, "GD": gf-ga, "Pts": w*3+d, "CS": cs})
    tbl = pd.DataFrame(rows).sort_values("Pts", ascending=False).reset_index(drop=True)
    tbl.index += 1
    return tbl


def goals_attack_defense_chart(standings: pd.DataFrame):
    fig = px.scatter(
        standings, x="GF", y="GA", text="Team",
        title="Attack vs Defence (current season)",
        labels={"GF": "Goals Scored", "GA": "Goals Conceded"},
        color="Pts", color_continuous_scale="RdYlGn",
        size="Pts", size_max=20,
    )
    fig.update_traces(textposition="top center", textfont_size=10)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#f0f0f0", coloraxis_showscale=False)
    return fig


def clean_sheet_chart(standings: pd.DataFrame):
    s = standings.sort_values("CS", ascending=True)
    fig = px.bar(s, x="CS", y="Team", orientation="h",
                 title="Clean sheets (current season)",
                 labels={"CS": "Clean Sheets", "Team": ""},
                 color="CS", color_continuous_scale="Blues")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#f0f0f0", coloraxis_showscale=False, height=520)
    return fig


def home_away_chart(df: pd.DataFrame, season: str):
    s = df[df["Season"] == season]
    teams_s = sorted(set(s["HomeTeam"].tolist() + s["AwayTeam"].tolist()))
    rows = []
    for team in teams_s:
        home = s[s["HomeTeam"] == team]
        away = s[s["AwayTeam"] == team]
        hp = int((home["FTR"]=="H").sum()*3 + (home["FTR"]=="D").sum())
        ap = int((away["FTR"]=="A").sum()*3 + (away["FTR"]=="D").sum())
        rows.append({"Team": team, "Home pts": hp, "Away pts": ap})
    tbl = pd.DataFrame(rows).sort_values("Home pts", ascending=False)
    fig = go.Figure()
    fig.add_bar(name="Home pts", x=tbl["Team"], y=tbl["Home pts"], marker_color="#2ecc71")
    fig.add_bar(name="Away pts", x=tbl["Team"], y=tbl["Away pts"], marker_color="#3498db")
    fig.update_layout(barmode="group", title="Home vs Away points (current season)",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#f0f0f0", xaxis_tickangle=-45)
    return fig


def draw_tendency_chart(standings: pd.DataFrame):
    s = standings.copy()
    s["Draw %"] = (s["D"] / s["P"] * 100).round(1)
    s = s.sort_values("Draw %", ascending=False)
    fig = px.bar(s, x="Team", y="Draw %",
                 title="Draw tendency by team (current season)",
                 color="Draw %", color_continuous_scale="Oranges")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#f0f0f0", coloraxis_showscale=False, xaxis_tickangle=-45)
    return fig


def prediction_gauge(home_prob, draw_prob, away_prob, home, away):
    fig = go.Figure(go.Bar(
        x=[f"{home} Win", "Draw", f"{away} Win"],
        y=[home_prob * 100, draw_prob * 100, away_prob * 100],
        marker_color=["#2ecc71", "#f39c12", "#e74c3c"],
        text=[f"{v*100:.1f}%" for v in [home_prob, draw_prob, away_prob]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Predicted outcome probabilities",
        yaxis_title="Probability (%)", yaxis_range=[0, 100],
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#f0f0f0",
    )
    return fig


@st.cache_data(ttl=21600)
def fetch_upcoming_fixtures() -> list[dict]:
    """Fetch upcoming PL fixtures from FPL API (cached 6 h)."""
    try:
        r1 = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/",
                          timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r1.raise_for_status()
        team_map = {t["id"]: t["name"] for t in r1.json()["teams"]}

        r2 = requests.get("https://fantasy.premierleague.com/api/fixtures/?future=1",
                          timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r2.raise_for_status()
        out = []
        for f in r2.json():
            if f.get("finished") or f.get("started"):
                continue
            home_fpl = team_map.get(f["team_h"], "")
            away_fpl = team_map.get(f["team_a"], "")
            ko = f.get("kickoff_time")
            if ko:
                dt = datetime.fromisoformat(ko.replace("Z", "+00:00"))
                ko_str = dt.astimezone(tz=None).strftime("%d %b %Y %H:%M")
            else:
                ko_str = "TBC"
            out.append({
                "gw":        f.get("event") or 0,
                "home":      FPL_TO_DB.get(home_fpl, home_fpl),
                "away":      FPL_TO_DB.get(away_fpl, away_fpl),
                "kickoff":   ko_str,
                "home_diff": f.get("team_h_difficulty", 3),
                "away_diff": f.get("team_a_difficulty", 3),
            })
        return sorted(out, key=lambda x: (x["gw"], x["kickoff"]))
    except Exception:
        return []


def h2h_table(df: pd.DataFrame, team1: str, team2: str, n: int = 10) -> pd.DataFrame:
    mask = (
        ((df["HomeTeam"] == team1) & (df["AwayTeam"] == team2)) |
        ((df["HomeTeam"] == team2) & (df["AwayTeam"] == team1))
    )
    past = df[mask].sort_values("Date", ascending=False).head(n).copy()
    if past.empty:
        return pd.DataFrame()

    def t1_result(row):
        if row["HomeTeam"] == team1:
            return {"H": "Win", "D": "Draw", "A": "Loss"}[row["FTR"]]
        return {"A": "Win", "D": "Draw", "H": "Loss"}[row["FTR"]]

    past["Result"] = past.apply(t1_result, axis=1)
    past["Score"]  = past.apply(lambda r: f"{int(r['FTHG'])}-{int(r['FTAG'])}", axis=1)
    past["Date"]   = past["Date"].dt.strftime("%Y-%m-%d")
    return past[["Date", "HomeTeam", "Score", "AwayTeam", "Result"]].reset_index(drop=True)


# ── Prediction log ────────────────────────────────────────────────────────────

def load_prediction_log() -> list[dict]:
    p = Path(LOG_PATH)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_prediction(entry: dict) -> None:
    log = load_prediction_log()
    log.insert(0, entry)
    Path(LOG_PATH).write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")


def auto_fill_results(df: pd.DataFrame, log: list[dict]) -> tuple[list[dict], bool]:
    """Look up actual results for pending log entries."""
    changed = False
    for entry in log:
        if entry.get("actual"):
            continue
        mask = (
            (df["HomeTeam"] == entry["home"]) &
            (df["AwayTeam"] == entry["away"]) &
            (df["Date"] >= pd.Timestamp(entry["ts"][:10]))
        )
        match = df[mask].sort_values("Date")
        if not match.empty:
            entry["actual"] = match.iloc[0]["FTR"]
            changed = True
    return log, changed


def render_injury_list(players: list[dict]):
    """Render a compact injury table for a team."""
    if not players:
        st.caption("No injuries or suspensions reported.")
        return
    for p in sorted(players, key=lambda x: -(x["ownership"])):
        chance_txt = f"{p['chance']}% chance" if p["chance"] is not None else "0% chance"
        icon = {"i": "🔴", "d": "🟡", "s": "🟠"}.get(p["status"], "⚫")
        st.markdown(
            f"{icon} **{p['name']}** — {p['label']} · {chance_txt}"
            + (f"\n\n   _{p['news']}_" if p["news"] else ""),
            unsafe_allow_html=False,
        )

# ── Layout ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Football Predictor", layout="wide", page_icon="⚽")
st.title("⚽ Premier League — Match Outcome Predictor")
st.caption("5 seasons of data (2021–2026) · Random Forest · football-data.co.uk · injuries via FPL API")

df    = load_data()
teams = sorted(df["HomeTeam"].unique().tolist())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data")
    lu = last_updated()
    if lu:
        age = datetime.now() - lu
        st.caption(f"Last updated: {lu.strftime('%d %b %Y %H:%M')}")
        if age > timedelta(days=REFRESH_DAYS):
            st.warning(f"Data is {age.days} days old — refresh recommended.")
    else:
        st.warning("No update timestamp found.")

    if st.button("Refresh data & retrain", type="primary"):
        with st.spinner("Downloading latest matches and retraining model..."):
            ok, err = run_refresh()
        if ok:
            st.success("Done! Reloading...")
            st.rerun()
        else:
            st.error(f"Refresh failed:\n{err}")

    st.caption(f"Current season: **{CURRENT_SEASON}**")

# Pre-fetch injuries once (cached)
all_injuries = fetch_fpl_injuries()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔮 Predict", "📊 Team Form", "📈 League Trends",
    "🗓️ Fixtures", "📋 Prediction Log", "🏥 Injuries & News",
])

# ── Tab 1 : Prediction ────────────────────────────────────────────────────────
with tab1:
    st.subheader("Predict match outcome")
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("🏠 Home team", teams,
                                 index=teams.index("Man City") if "Man City" in teams else 0)
    with col2:
        away_options = [t for t in teams if t != home_team]
        away_team = st.selectbox("✈️ Away team", away_options)

    home_players = all_injuries.get(home_team, [])
    away_players = all_injuries.get(away_team, [])

    # ── Injury classification — shown before Predict so user can adjust
    CLASSES = ["Key player", "Normal", "Irrelevant"]
    home_overrides: dict[str, str] = {}
    away_overrides: dict[str, str] = {}

    if home_players or away_players:
        st.markdown("**Classify injured / suspended players** _(adjust before predicting)_")
        inj_c1, inj_c2 = st.columns(2)

        with inj_c1:
            if home_players:
                st.markdown(f"🏠 **{home_team}**")
                for p in sorted(home_players, key=lambda x: -x["ownership"]):
                    icon = {"i": "🔴", "d": "🟡", "s": "🟠"}.get(p["status"], "⚫")
                    chance_txt = f"{p['chance']}% chance" if p["chance"] is not None else "0%"
                    default = default_classification(p)
                    chosen = st.selectbox(
                        f"{icon} {p['name']} — {p['label']} ({chance_txt})",
                        CLASSES,
                        index=CLASSES.index(default),
                        key=f"home_{p['name']}",
                    )
                    home_overrides[p["name"]] = chosen

        with inj_c2:
            if away_players:
                st.markdown(f"✈️ **{away_team}**")
                for p in sorted(away_players, key=lambda x: -x["ownership"]):
                    icon = {"i": "🔴", "d": "🟡", "s": "🟠"}.get(p["status"], "⚫")
                    chance_txt = f"{p['chance']}% chance" if p["chance"] is not None else "0%"
                    default = default_classification(p)
                    chosen = st.selectbox(
                        f"{icon} {p['name']} — {p['label']} ({chance_txt})",
                        CLASSES,
                        index=CLASSES.index(default),
                        key=f"away_{p['name']}",
                    )
                    away_overrides[p["name"]] = chosen

    if st.button("Predict", type="primary"):
        try:
            model, le = load_model()

            def latest(team, home):
                prefix = "Home" if home else "Away"
                mask   = (df["HomeTeam"] == team) if home else (df["AwayTeam"] == team)
                row    = df[mask].sort_values("Date").iloc[-1]
                return {
                    f"{prefix}Form":       row[f"{prefix}Form"],
                    f"{prefix}GoalsAvg":   row[f"{prefix}GoalsAvg"],
                    f"{prefix}ConcedeAvg": row[f"{prefix}ConcedeAvg"],
                }

            home_feats = latest(home_team, True)
            away_feats = latest(away_team, False)
            h2h        = h2h_stats(df, home_team, away_team)

            h_factor = injury_team_factor(home_players, home_overrides)
            a_factor = injury_team_factor(away_players, away_overrides)

            if h_factor < 1.0:
                home_feats = apply_injury_factor(home_feats, "Home", h_factor)
            if a_factor < 1.0:
                away_feats = apply_injury_factor(away_feats, "Away", a_factor)

            X     = pd.DataFrame([{**home_feats, **away_feats, **h2h}])[FEATURES]
            proba = model.predict_proba(X)[0]
            proba_dict = dict(zip(le.classes_, proba))
            pred  = le.inverse_transform([proba.argmax()])[0]
            label = {"H": f"🏠 {home_team} wins", "D": "🤝 Draw",
                     "A": f"✈️ {away_team} wins"}[pred]

            if h_factor < 1.0 or a_factor < 1.0:
                parts = []
                if h_factor < 1.0:
                    parts.append(f"{home_team} -{(1-h_factor)*100:.0f}%")
                if a_factor < 1.0:
                    parts.append(f"{away_team} -{(1-a_factor)*100:.0f}%")
                st.info(f"Injury adjustment applied: {', '.join(parts)}")

            st.success(f"**Predicted outcome: {label}**")
            st.plotly_chart(
                prediction_gauge(proba_dict.get("H", 0), proba_dict.get("D", 0),
                                 proba_dict.get("A", 0), home_team, away_team),
                use_container_width=True,
            )

            # ── Injury summary below the chart
            if home_players or away_players:
                st.divider()
                sum_c1, sum_c2 = st.columns(2)
                with sum_c1:
                    st.markdown(f"**🏥 {home_team} absentees**")
                    render_injury_list(home_players)
                with sum_c2:
                    st.markdown(f"**🏥 {away_team} absentees**")
                    render_injury_list(away_players)

            # ── Save to prediction log
            save_prediction({
                "ts":       datetime.now().isoformat(timespec="seconds"),
                "home":     home_team,
                "away":     away_team,
                "pred":     pred,
                "prob_H":   round(float(proba_dict.get("H", 0)), 3),
                "prob_D":   round(float(proba_dict.get("D", 0)), 3),
                "prob_A":   round(float(proba_dict.get("A", 0)), 3),
                "inj_home": round(h_factor, 3),
                "inj_away": round(a_factor, 3),
                "actual":   None,
            })

            with st.expander("Show model input features"):
                st.dataframe(X.T.rename(columns={0: "Value"}).round(3))

        except FileNotFoundError:
            st.error("Model not found — run `python model.py` first.")

# ── Tab 2 : Team form ─────────────────────────────────────────────────────────
with tab2:
    st.subheader("Team recent form")
    selected_team = st.selectbox("Select a team", teams)

    season_df = df[df["Season"] == CURRENT_SEASON]
    if season_df.empty:
        season_df = df[df["Season"] == sorted(df["Season"].unique())[-1]]
    mask = (season_df["HomeTeam"] == selected_team) | (season_df["AwayTeam"] == selected_team)
    team_matches = season_df[mask].copy()

    def result_label(row):
        if row["HomeTeam"] == selected_team:
            return {"H": "Win", "D": "Draw", "A": "Loss"}[row["FTR"]]
        return {"A": "Win", "D": "Draw", "H": "Loss"}[row["FTR"]]

    team_matches["Result"]    = team_matches.apply(result_label, axis=1)
    team_matches["Opponent"]  = team_matches.apply(
        lambda r: r["AwayTeam"] if r["HomeTeam"] == selected_team else r["HomeTeam"], axis=1)
    team_matches["Venue"]     = team_matches.apply(
        lambda r: "Home" if r["HomeTeam"] == selected_team else "Away", axis=1)
    team_matches["GF"]        = team_matches.apply(
        lambda r: r["FTHG"] if r["HomeTeam"] == selected_team else r["FTAG"], axis=1)
    team_matches["GA"]        = team_matches.apply(
        lambda r: r["FTAG"] if r["HomeTeam"] == selected_team else r["FTHG"], axis=1)

    display = team_matches[["Date", "Venue", "Opponent", "GF", "GA", "Result"]].copy()
    display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
    display = display.sort_values("Date", ascending=False).reset_index(drop=True)

    st.caption(f"Season {CURRENT_SEASON} — {len(display)} matches played")
    st.dataframe(
        style_results_table(display),
        use_container_width=True,
        hide_index=True,
    )

    # ── H2H section
    st.divider()
    st.markdown("**Head-to-head history**")

    upcoming = fetch_upcoming_fixtures()
    next_match = next(
        (f for f in upcoming if f["home"] == selected_team or f["away"] == selected_team), None
    )

    if next_match:
        next_opp = next_match["away"] if next_match["home"] == selected_team else next_match["home"]
        next_venue = "Home" if next_match["home"] == selected_team else "Away"
        st.caption(f"Next fixture: **{next_venue}** vs **{next_opp}** — GW{next_match['gw']} · {next_match['kickoff']}")
    else:
        next_opp = None
        st.caption("No upcoming fixture found — select opponent manually below")

    all_opponents = sorted([t for t in teams if t != selected_team])
    default_idx = all_opponents.index(next_opp) if next_opp and next_opp in all_opponents else 0
    h2h_opp = st.selectbox("Compare H2H vs", all_opponents, index=default_idx, key="h2h_opp")

    h2h_df = h2h_table(df, selected_team, h2h_opp)
    if h2h_df.empty:
        st.info(f"No H2H history found between {selected_team} and {h2h_opp}.")
    else:
        wins   = (h2h_df["Result"] == "Win").sum()
        draws  = (h2h_df["Result"] == "Draw").sum()
        losses = (h2h_df["Result"] == "Loss").sum()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Matches", len(h2h_df))
        m2.metric(f"{selected_team} wins", wins)
        m3.metric("Draws", draws)
        m4.metric(f"{h2h_opp} wins", losses)
        st.dataframe(style_results_table(h2h_df), use_container_width=True, hide_index=True)

# ── Tab 3 : League trends ─────────────────────────────────────────────────────
with tab3:
    st.subheader(f"League stats — {CURRENT_SEASON}")
    standings = season_standings(df, CURRENT_SEASON)

    st.markdown("**Current season standings**")
    st.dataframe(standings, use_container_width=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(goals_attack_defense_chart(standings), use_container_width=True)
    with c2:
        st.plotly_chart(clean_sheet_chart(standings), use_container_width=True)

    st.plotly_chart(home_away_chart(df, CURRENT_SEASON), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(draw_tendency_chart(standings), use_container_width=True)
    with c4:
        result_counts = df[df["Season"] == CURRENT_SEASON]["FTR"].value_counts().rename(
            {"H": "Home Win", "D": "Draw", "A": "Away Win"})
        fig_pie = px.pie(values=result_counts.values, names=result_counts.index,
                         title=f"Result distribution ({CURRENT_SEASON})",
                         color_discrete_sequence=["#2ecc71", "#f39c12", "#e74c3c"])
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#f0f0f0")
        st.plotly_chart(fig_pie, use_container_width=True)

# ── Tab 4 : Fixtures & GW Predictions ────────────────────────────────────────
with tab4:
    st.subheader("Upcoming fixtures & predictions")

    col_r1, col_r2 = st.columns([1, 1])
    with col_r1:
        if st.button("Refresh fixtures"):
            fetch_upcoming_fixtures.clear()
            st.rerun()

    fixtures = fetch_upcoming_fixtures()
    if not fixtures:
        st.warning("Could not fetch fixtures. Check your internet connection.")
    else:
        gws  = sorted({f["gw"] for f in fixtures})
        # Selector: which GW to predict
        sel_gw = st.selectbox("Gameweek", gws, index=0)
        gw_fix = [f for f in fixtures if f["gw"] == sel_gw]

        diff_icon = lambda d: "🟢" if d <= 2 else ("🟡" if d == 3 else ("🟠" if d == 4 else "🔴"))

        # ── Auto-predict all fixtures in selected GW
        try:
            model_gw, le_gw = load_model()
            results = []
            for f in gw_fix:
                res = predict_fixture(f["home"], f["away"], df,
                                      model_gw, le_gw, all_injuries)
                if res is None:
                    continue

                pred_label = {"H": f["home"], "D": "Draw", "A": f["away"]}[res["pred"]]
                conf = max(res["prob_H"], res["prob_D"], res["prob_A"])

                inj_notes = []
                if res["home_key_injured"]:
                    inj_notes.append(f"🔴 {f['home']}: {', '.join(res['home_key_injured'])}")
                if res["away_key_injured"]:
                    inj_notes.append(f"🔴 {f['away']}: {', '.join(res['away_key_injured'])}")

                results.append({
                    "Kick-off":    f["kickoff"],
                    "Home":        f["home"],
                    "Away":        f["away"],
                    "Prediction":  pred_label,
                    "H %":         f"{res['prob_H']*100:.0f}",
                    "D %":         f"{res['prob_D']*100:.0f}",
                    "A %":         f"{res['prob_A']*100:.0f}",
                    "Confidence":  f"{conf*100:.0f}%",
                    "Home diff":   diff_icon(f["home_diff"]),
                    "Away diff":   diff_icon(f["away_diff"]),
                    "Key injuries": " | ".join(inj_notes) if inj_notes else "—",
                    # internal fields used for logging
                    "_pred_raw":   res["pred"],
                    "_prob_H":     res["prob_H"],
                    "_prob_D":     res["prob_D"],
                    "_prob_A":     res["prob_A"],
                    "_h_factor":   res["h_factor"],
                    "_a_factor":   res["a_factor"],
                })

            if results:
                # Save to prediction log (skip duplicates by home+away+date)
                log_existing = {(e["home"], e["away"], e["ts"][:10]) for e in load_prediction_log()}
                today = datetime.now().strftime("%Y-%m-%d")
                for r in results:
                    key = (r["Home"], r["Away"], today)
                    if key not in log_existing:
                        save_prediction({
                            "ts":       datetime.now().isoformat(timespec="seconds"),
                            "home":     r["Home"],
                            "away":     r["Away"],
                            "pred":     r["_pred_raw"],
                            "prob_H":   round(r["_prob_H"], 3),
                            "prob_D":   round(r["_prob_D"], 3),
                            "prob_A":   round(r["_prob_A"], 3),
                            "inj_home": round(r["_h_factor"], 3),
                            "inj_away": round(r["_a_factor"], 3),
                            "actual":   None,
                        })

                # Display table (hide internal cols)
                display_cols = ["Kick-off", "Home", "Away", "Prediction",
                                "H %", "D %", "A %", "Confidence",
                                "Home diff", "Away diff", "Key injuries"]
                gw_df = pd.DataFrame(results)[display_cols]

                def color_gw_row(row):
                    pred = row["Prediction"]
                    if pred == row["Home"]:
                        c = "#1a3d4d"
                    elif pred == "Draw":
                        c = "#4a3d00"
                    else:
                        c = "#2d1a4d"
                    return [f"background-color: {c}; color: #f0f0f0"] * len(row)

                st.dataframe(
                    gw_df.style.apply(color_gw_row, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "Predictions are fully automatic. Injuries are auto-classified using "
                    "FPL minutes played + season points (no manual input needed). "
                    "All predictions are saved to the Prediction Log."
                )

        except FileNotFoundError:
            st.error("Model not found — run `python model.py` first.")

        # ── Other gameweeks: fixtures only
        other_gws = [g for g in gws if g != sel_gw]
        if other_gws:
            with st.expander("Other upcoming gameweeks (fixtures only)"):
                for gw in other_gws[:4]:
                    st.markdown(f"**Gameweek {gw}**")
                    rows = [{"Kick-off": f["kickoff"], "Home": f["home"],
                             "Away": f["away"],
                             "Home diff": diff_icon(f["home_diff"]),
                             "Away diff": diff_icon(f["away_diff"])}
                            for f in fixtures if f["gw"] == gw]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Tab 5 : Prediction Log ────────────────────────────────────────────────────
with tab5:
    st.subheader("Prediction log")

    log = load_prediction_log()
    # Auto-fill actual results from DB
    log, changed = auto_fill_results(df, log)
    if changed:
        Path(LOG_PATH).write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    if not log:
        st.info("No predictions yet — run a prediction in the Predict tab.")
    else:
        if st.button("Clear all predictions"):
            Path(LOG_PATH).write_text("[]", encoding="utf-8")
            st.rerun()

        rows = []
        for e in log:
            pred_label  = {"H": "Home win", "D": "Draw", "A": "Away win"}.get(e["pred"], e["pred"])
            actual      = e.get("actual")
            actual_label = {"H": "Home win", "D": "Draw", "A": "Away win"}.get(actual, "Pending") if actual else "Pending"
            correct     = (e["pred"] == actual) if actual else None
            rows.append({
                "Date":      e["ts"][:10],
                "Home":      e["home"],
                "Away":      e["away"],
                "Predicted": pred_label,
                "H %":       f"{e['prob_H']*100:.0f}",
                "D %":       f"{e['prob_D']*100:.0f}",
                "A %":       f"{e['prob_A']*100:.0f}",
                "Actual":    actual_label,
                "Correct":   "✅" if correct is True else ("❌" if correct is False else "⏳"),
            })

        log_df = pd.DataFrame(rows)

        # Summary metrics
        completed = [e for e in log if e.get("actual")]
        if completed:
            accuracy = sum(e["pred"] == e["actual"] for e in completed) / len(completed)
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Total predictions", len(log))
            mc2.metric("Completed", len(completed))
            mc3.metric("Accuracy", f"{accuracy:.0%}")

        def color_log_row(row):
            if row["Correct"] == "✅":
                return ["background-color: #1a4d2e; color: #f0f0f0"] * len(row)
            elif row["Correct"] == "❌":
                return ["background-color: #4d1a1a; color: #f0f0f0"] * len(row)
            return [""] * len(row)

        st.dataframe(
            log_df.style.apply(color_log_row, axis=1),
            use_container_width=True,
            hide_index=True,
        )

# ── Tab 6 : Injuries & News ───────────────────────────────────────────────────
with tab6:
    col_inj, col_news = st.columns([1, 1])

    with col_inj:
        st.subheader("🏥 Current Injuries & Suspensions")
        st.caption("Live data from Fantasy Premier League API — updated every 3 hours")

        if st.button("Force refresh injuries"):
            fetch_fpl_injuries.clear()
            st.rerun()

        if all_injuries:
            inj_team = st.selectbox("View team", sorted(all_injuries.keys()), key="inj_team_select")
            render_injury_list(all_injuries.get(inj_team, []))

            with st.expander("All teams summary"):
                rows = []
                for team, players in sorted(all_injuries.items()):
                    for p in players:
                        rows.append({
                            "Team": team,
                            "Player": p["name"],
                            "Status": p["label"],
                            "Chance": f"{p['chance']}%" if p["chance"] is not None else "0%",
                            "News": p["news"],
                        })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.warning("Could not fetch injury data. Check your internet connection.")

    with col_news:
        st.subheader("📰 Premier League News")
        st.caption("Latest headlines from BBC Sport (refreshed every hour)")

        if st.button("Refresh news"):
            fetch_pl_news.clear()
            st.rerun()

        news = fetch_pl_news()
        if news:
            for item in news:
                st.markdown(f"**[{item['title']}]({item['link']})**")
                if item["desc"]:
                    st.markdown(f"<small>{item['desc'][:160]}...</small>",
                                unsafe_allow_html=True)
                if item["date"]:
                    st.caption(item["date"])
                st.markdown("---")
        else:
            st.warning("Could not fetch news. Check your internet connection.")
