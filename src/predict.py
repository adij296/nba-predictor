import joblib
import numpy as np
import pandas as pd

# --- Part 1: settings (must match the Elo code from notebook 05) ---
K = 20
HOME_ADV = 70
CARRY = 0.75
GAP_DAYS = 60

FEATURES = ["HOME_PTS_ROLL10", "HOME_PTS_ALLOWED_ROLL10", "HOME_WIN_ROLL10", "HOME_REST_DAYS",
            "AWAY_PTS_ROLL10", "AWAY_PTS_ALLOWED_ROLL10", "AWAY_WIN_ROLL10", "AWAY_REST_DAYS",
            "ELO_DIFF"]


# --- Part 2: summarize every team's current situation ---
def build_state(games):
    games = games.sort_values("GAME_DATE", kind="stable")

    ratings = {}
    prev_date = None
    for date, ht, at, hp, ap in zip(games["GAME_DATE"], games["HOME_TEAM"], games["AWAY_TEAM"],
                                    games["HOME_PTS"], games["AWAY_PTS"]):
        if prev_date is not None and (date - prev_date).days > GAP_DAYS:
            for t in ratings:
                ratings[t] = 1500 + CARRY * (ratings[t] - 1500)
        prev_date = date

        h = ratings.get(ht, 1500)
        a = ratings.get(at, 1500)
        diff = h + HOME_ADV - a
        expected = 1 / (1 + 10 ** (-diff / 400))
        margin = hp - ap
        home_won = 1 if margin > 0 else 0
        winner_diff = diff if home_won else -diff
        mov = (abs(margin) + 3) ** 0.8 / (7.5 + 0.006 * winner_diff)
        change = K * mov * (home_won - expected)
        ratings[ht] = h + change
        ratings[at] = a - change

    home = pd.DataFrame({"TEAM": games["HOME_TEAM"], "DATE": games["GAME_DATE"],
                         "PTS": games["HOME_PTS"], "ALLOWED": games["AWAY_PTS"]})
    away = pd.DataFrame({"TEAM": games["AWAY_TEAM"], "DATE": games["GAME_DATE"],
                         "PTS": games["AWAY_PTS"], "ALLOWED": games["HOME_PTS"]})
    long = pd.concat([home, away]).sort_values(["TEAM", "DATE"], kind="stable")
    long["WIN"] = (long["PTS"] > long["ALLOWED"]).astype(int)

    teams = {}
    for team, g in long.groupby("TEAM"):
        last10 = g.tail(10)
        teams[team] = {
            "elo": ratings[team],
            "pts_roll10": last10["PTS"].mean(),
            "allowed_roll10": last10["ALLOWED"].mean(),
            "win_roll10": last10["WIN"].mean(),
            "last_game": g["DATE"].max(),
        }
    return {"teams": teams, "last_date": prev_date}


# --- Part 3: turn a matchup into the model's 9 inputs ---
def make_features(state, home, away, date):
    date = pd.Timestamp(date)
    teams = state["teams"]
    if home == away:
        raise ValueError("Home and away teams must be different.")
    for t in (home, away):
        if t not in teams:
            raise ValueError(f"Unknown team: {t}")

    h, a = teams[home], teams[away]
    elo_h, elo_a = h["elo"], a["elo"]
    if (date - state["last_date"]).days > GAP_DAYS:
        elo_h = 1500 + CARRY * (elo_h - 1500)
        elo_a = 1500 + CARRY * (elo_a - 1500)

    rest_h = (date - h["last_game"]).days
    rest_a = (date - a["last_game"]).days
    if rest_h < 0 or rest_a < 0:
        raise ValueError("That date is before a team's last game in the data.")

    row = {
        "HOME_PTS_ROLL10": h["pts_roll10"], "HOME_PTS_ALLOWED_ROLL10": h["allowed_roll10"],
        "HOME_WIN_ROLL10": h["win_roll10"], "HOME_REST_DAYS": min(rest_h, 7),
        "AWAY_PTS_ROLL10": a["pts_roll10"], "AWAY_PTS_ALLOWED_ROLL10": a["allowed_roll10"],
        "AWAY_WIN_ROLL10": a["win_roll10"], "AWAY_REST_DAYS": min(rest_a, 7),
        "ELO_DIFF": elo_h - elo_a,
    }
    return pd.DataFrame([row], columns=FEATURES)


# --- Part 4: predict, and load the saved files ---
def predict_game(model, state, home, away, date):
    X = make_features(state, home, away, date)
    return float(model.predict_proba(X)[0, 1])


def load_artifacts(folder="models"):
    model = joblib.load(f"{folder}/final_model.joblib")
    state = joblib.load(f"{folder}/team_state.joblib")
    return model, state