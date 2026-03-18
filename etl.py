"""
etl.py
------
Downloads Premier League match data from football-data.co.uk,
cleans it, engineers features, and stores everything in SQLite.

Usage:
    python src/etl.py
"""

import os
import sqlite3
import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────

SEASONS = {
    "2021-22": "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
    "2022-23": "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    "2023-24": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "2024-25": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "2025-26": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",  # current season
}

CURRENT_SEASON = "2025-26"
TIMESTAMP_PATH = "data/last_updated.txt"

COLS = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",          # Full-time goals & result (H/D/A)
    "HS", "AS",                      # Shots
    "HST", "AST",                    # Shots on target
    "HC", "AC",                      # Corners
    "HY", "AY",                      # Yellow cards
    "HR", "AR",                      # Red cards
]

DB_PATH = "data/football.db"
RAW_PATH = "data/raw_matches.csv"


# ── Download ──────────────────────────────────────────────────────────────────

def download_seasons() -> pd.DataFrame:
    frames = []
    for season, url in SEASONS.items():
        print(f"  Downloading {season}...")
        df = pd.read_csv(url, usecols=lambda c: c in COLS, on_bad_lines="skip")
        df["Season"] = season
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ── Clean ─────────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["FTR", "HomeTeam", "AwayTeam"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Fill missing stats with column median
    stat_cols = ["HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR"]
    for col in stat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def add_form(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    For each match, compute each team's points-per-game over their last N matches.
    Points: win=3, draw=1, loss=0.
    """
    df = df.copy()
    df["HomeForm"] = 0.0
    df["AwayForm"] = 0.0

    # Build a per-team match history as we iterate chronologically
    history: dict[str, list[float]] = {}

    for idx, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]

        # Look up recent form BEFORE this match
        h_pts = history.get(home, [])
        a_pts = history.get(away, [])
        df.at[idx, "HomeForm"] = sum(h_pts[-n:]) / min(len(h_pts[-n:]), n) if h_pts else 1.0
        df.at[idx, "AwayForm"] = sum(a_pts[-n:]) / min(len(a_pts[-n:]), n) if a_pts else 1.0

        # Update history with this match result
        result = row["FTR"]
        history.setdefault(home, []).append(3 if result == "H" else (1 if result == "D" else 0))
        history.setdefault(away, []).append(3 if result == "A" else (1 if result == "D" else 0))

    return df


def add_goal_avg(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Rolling average goals scored/conceded over last N matches."""
    df = df.copy()
    for col in ["HomeGoalsAvg", "AwayGoalsAvg", "HomeConcedeAvg", "AwayConcedeAvg"]:
        df[col] = 0.0

    goals_scored: dict[str, list[float]] = {}
    goals_conceded: dict[str, list[float]] = {}

    for idx, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]

        for team, scored_key, conceded_key, avg_scored_col, avg_conceded_col, goals_scored_val, goals_conceded_val in [
            (home, home, home, "HomeGoalsAvg", "HomeConcedeAvg", row["FTHG"], row["FTAG"]),
            (away, away, away, "AwayGoalsAvg",  "AwayConcedeAvg",  row["FTAG"], row["FTHG"]),
        ]:
            gs = goals_scored.get(team, [])
            gc = goals_conceded.get(team, [])
            df.at[idx, avg_scored_col]   = sum(gs[-n:]) / min(len(gs[-n:]), n) if gs else 1.2
            df.at[idx, avg_conceded_col] = sum(gc[-n:]) / min(len(gc[-n:]), n) if gc else 1.2
            goals_scored.setdefault(team,   []).append(goals_scored_val)
            goals_conceded.setdefault(team, []).append(goals_conceded_val)

    return df


def add_h2h(df: pd.DataFrame, n: int = 7) -> pd.DataFrame:
    """
    For each match, compute head-to-head stats from the last N encounters
    between the same two teams (any venue), BEFORE this match.
    Features are always expressed from the current home team's perspective.
    """
    df = df.copy()
    df["H2H_HomeWinRate"]  = 0.5
    df["H2H_AwayWinRate"]  = 0.5
    df["H2H_HomeGoalsAvg"] = 1.2
    df["H2H_AwayGoalsAvg"] = 1.2

    # history key = frozenset of the two teams; stores past match dicts
    history: dict[frozenset, list[dict]] = {}

    for idx, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        key  = frozenset([home, away])
        past = history.get(key, [])[-n:]

        if past:
            home_wins = away_wins = 0
            home_goals_list: list[float] = []
            away_goals_list: list[float] = []

            for m in past:
                # Goals from perspective of current home/away team
                g_home = m["fthg"] if m["ht"] == home else m["ftag"]
                g_away = m["ftag"] if m["ht"] == home else m["fthg"]
                home_goals_list.append(g_home)
                away_goals_list.append(g_away)

                # Win from current home team's perspective
                home_won = (m["ftr"] == "H" and m["ht"] == home) or \
                           (m["ftr"] == "A" and m["ht"] == away)
                away_won = (m["ftr"] == "A" and m["ht"] == home) or \
                           (m["ftr"] == "H" and m["ht"] == away)
                if home_won:
                    home_wins += 1
                elif away_won:
                    away_wins += 1

            total = len(past)
            df.at[idx, "H2H_HomeWinRate"]  = home_wins / total
            df.at[idx, "H2H_AwayWinRate"]  = away_wins / total
            df.at[idx, "H2H_HomeGoalsAvg"] = sum(home_goals_list) / total
            df.at[idx, "H2H_AwayGoalsAvg"] = sum(away_goals_list) / total

        history.setdefault(key, []).append({
            "ht": home, "fthg": row["FTHG"], "ftag": row["FTAG"], "ftr": row["FTR"],
        })

    return df


# ── Store ─────────────────────────────────────────────────────────────────────

def to_sqlite(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    # Store dates as ISO strings so they load correctly (avoids int64 epoch issue)
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df.to_sql("matches", con, if_exists="replace", index=False)

    # Useful analytical views
    con.execute("""
        CREATE VIEW IF NOT EXISTS team_stats AS
        SELECT HomeTeam AS team,
               COUNT(*)                                      AS matches,
               SUM(CASE WHEN FTR='H' THEN 1 ELSE 0 END)    AS wins,
               SUM(CASE WHEN FTR='D' THEN 1 ELSE 0 END)    AS draws,
               SUM(CASE WHEN FTR='A' THEN 1 ELSE 0 END)    AS losses,
               ROUND(AVG(FTHG), 2)                          AS avg_goals_scored,
               ROUND(AVG(FTAG), 2)                          AS avg_goals_conceded
        FROM matches
        GROUP BY HomeTeam
    """)
    con.commit()
    con.close()
    print(f"  Saved {len(df)} rows -> {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("Downloading data...")
    df = download_seasons()

    print("Cleaning...")
    df = clean(df)

    print("Engineering features...")
    df = add_form(df)
    df = add_goal_avg(df)
    df = add_h2h(df, n=10)

    print("Saving raw CSV...")
    os.makedirs("data", exist_ok=True)
    df.to_csv(RAW_PATH, index=False)

    print("Writing to SQLite...")
    to_sqlite(df, DB_PATH)

    # Write timestamp so the app knows when data was last refreshed
    from datetime import datetime
    with open(TIMESTAMP_PATH, "w") as f:
        f.write(datetime.now().isoformat())

    print(f"\nDone -- {len(df)} matches across {df['Season'].nunique()} seasons")
    print(f"   Teams: {df['HomeTeam'].nunique()}")
    print(f"   Date range: {df['Date'].min()} -> {df['Date'].max()}")


if __name__ == "__main__":
    run()
