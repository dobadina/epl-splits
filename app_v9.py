"""
Premier League splits explorer.

Needs the CSVs from league_event_splits_v6.py for chances created,
total duels, box-location breakdowns and results.

Reads the CSVs produced by league_event_splits_v4.py and frames them as football
questions rather than metric dumps. Every figure is either a rate, a share or a
per-match number, and every figure sits next to the league average and a rank.

Run it with:  streamlit run app_v3.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(os.environ.get("SPLITS_DATA_DIR",
                               Path(__file__).parent / "output"))
FOCUS_TEAM_HINT = "Man Utd"

TIME_ORDER = ["0-15", "16-30", "31-45", "46-60", "61-75", "76+"]
STATE_ORDER = ["-2 or worse", "-1", "Level", "+1", "+2 or better"]
# Below this much football in a split, the numbers are one match's noise.
# Not a user setting. Change it here if you ever need to.
TOUCH_FLOOR = 600
MIN_PLAYER_MINUTES = 90
CLAUDE_MODEL = "claude-sonnet-5"

st.set_page_config(page_title="Premier League splits", page_icon="⚽",
                   layout="wide")


# --------------------------------------------------------------------------
# METRIC DEFINITIONS
# --------------------------------------------------------------------------
# name: (label, lower_is_better, decimals, one-line explanation)
METRICS = {
    # Control
    "possession_pct": ("Possession %", False, 1,
                       "Share of all touches in their matches."),
    "field_tilt_pct": ("Field tilt %", False, 1,
                       "Share of final-third passes in their matches. Territory, not just the ball."),
    "box_touch_share_pct": ("Box touch share %", False, 1,
                            "Share of all penalty-area touches in their matches."),
    "passes_opp_half_pm": ("Passes in opposition half per match", False, 0,
                           "Completed passes beyond halfway."),
    "touches_pm": ("Touches per match", False, 0, "Raw involvement with the ball."),

    # Results
    "points_pm": ("Points per match", False, 2, "League points earned."),
    "clean_sheet_pct": ("Clean sheet %", False, 1, "Share of matches without conceding."),
    "win_pct": ("Win %", False, 1, "Share of matches won."),

    # Creation
    "shots_pm": ("Shots per match", False, 1, "Attempts on goal."),
    "chances_pm": ("Chances created per match", False, 1,
                   "Passes that lead directly to a shot."),
    "shots_per_chance": ("Shots per chance created", True, 2,
                         "How many attempts each created chance produces."),
    "shot_share_pct": ("Shot share %", False, 1,
                       "Share of all shots in their matches. Attacking dominance."),
    "box_touches_pm": ("Touches in opposition box per match", False, 1,
                       "How often they reach dangerous areas."),
    "box_touches_per_shot": ("Box touches per shot", True, 1,
                             "How much box presence it takes to produce one attempt."),
    "shots_in_box_pct": ("Shots from inside the box %", False, 1,
                         "Share of attempts taken from good positions."),

    # Finishing
    "on_target_pct": ("Shooting accuracy %", False, 1,
                      "Shots on target as a share of unblocked attempts, which "
                      "is how Opta defines it. A block says nothing about "
                      "whether the shot was on target."),
    "blocked_share_pct": ("Shots blocked %", True, 1,
                          "Share of attempts that hit a defender before the keeper."),
    "conversion_pct": ("Conversion %", False, 1,
                       "Goals from their own shots, per shot. Own goals excluded."),
    "goals_per_sot_pct": ("Goals per shot on target %", False, 1,
                          "Finishing once the shot is on target."),
    "shots_per_goal": ("Shots per goal", True, 1, "Attempts needed to score once."),
    "chances_per_goal": ("Chances created per goal", True, 1,
                         "Created chances needed to score once."),
    "goals_in_box_pct": ("Goals from inside the box %", False, 1,
                         "Share of their goals scored from close range."),
    "goals_pm": ("Goals per match", False, 2, "Goals scored."),

    # Route to goal
    "crosses_per_shot": ("Crosses per shot", True, 2,
                         "How cross-dependent the attack is."),
    "crosses_pm": ("Crosses per match", False, 1, "Open-play crosses attempted."),
    "cross_accuracy_pct": ("Cross completion %", False, 1, "Crosses that find a teammate."),
    "through_balls_pm": ("Through balls per match", False, 1,
                         "Passes played between defenders."),
    "forward_pass_share_pct": ("Forward pass share %", False, 1,
                               "Forward passes as a share of forward, sideways and backward."),
    "long_ball_share_pct": ("Long ball share %", False, 1,
                            "Long passes as a share of all passes."),
    "pass_accuracy_pct": ("Pass accuracy %", False, 1,
                          "Completed passes. Crosses, throw-ins and keeper "
                          "throws are excluded, as Opta excludes them."),

    # Duels
    "duel_win_pct": ("Duel win %", False, 1,
                     "Opta's five duel types combined: aerials, take-ons, "
                     "tackles, smothers and fouls."),
    "duels_contested_pm": ("Duels contested per match", False, 1,
                           "How often they enter a direct contest for the ball."),
    "duels_pm": ("Duels per match", False, 1, "How often they contest the ball directly."),
    "aerial_success_pct": ("Aerial win %", False, 1, "Headers won."),
    "ground_duel_success_pct": ("Ground duel win %", False, 1, "Ground contests won."),
    "take_on_success_pct": ("Take-on success %", False, 1, "Dribbles past an opponent."),
    "take_ons_pm": ("Take-ons attempted per match", False, 1, "Attempts to beat a man."),
    "fouls_won_pm": ("Fouls won per match", False, 1, "Free kicks earned."),
    "fouls_pm": ("Fouls committed per match", True, 1, "Free kicks given away."),

    # Defence
    "shots_faced_pm": ("Shots faced per match", True, 1, "Opponent attempts."),
    "sot_faced_pm": ("Shots on target faced per match", True, 1,
                     "Opponent attempts that reach the keeper."),
    "conceded_pm": ("Goals conceded per match", True, 2, "Goals against."),
    "goals_per_sot_faced_pct": ("Goals per shot on target faced %", True, 1,
                                "How often an opponent's on-target shot goes in."),
    "shots_faced_per_goal": ("Shots faced per goal conceded", False, 1,
                             "How much work an opponent needs to score."),
    "box_shots_faced_pct": ("Shots faced from inside the box %", True, 1,
                            "Share of opponent attempts from close range."),
    "sot_faced_in_box_pct": ("On-target shots faced from the box %", True, 1,
                             "Share of opponent shots on target taken from close range."),
    "conceded_in_box_pct": ("Goals conceded from inside the box %", True, 1,
                            "Share of goals against scored from close range."),
    "ppda": ("Opponent passes per defensive action", True, 1,
             "Lower means they press higher and harder."),
    "def_actions_pm": ("Defensive actions per match", False, 1,
                       "Tackles, interceptions, clearances and recoveries."),
    "goal_diff": ("Goal difference", False, 0, "Scored minus conceded."),
}

SECTIONS = {
    "What has it produced?": [
        "points_pm", "win_pct", "clean_sheet_pct", "goals_pm", "conceded_pm",
        "goal_diff"],
    "Do they control the game?": [
        "possession_pct", "field_tilt_pct", "box_touch_share_pct",
        "passes_opp_half_pm", "pass_accuracy_pct"],
    "Do they create?": [
        "chances_pm", "shots_pm", "shot_share_pct", "box_touches_pm",
        "box_touches_per_shot", "shots_in_box_pct", "shots_per_chance"],
    "Do they finish?": [
        "on_target_pct", "blocked_share_pct", "conversion_pct", "goals_per_sot_pct",
        "shots_per_goal", "chances_per_goal", "goals_in_box_pct", "goals_pm"],
    "How do they attack?": [
        "crosses_per_shot", "crosses_pm", "cross_accuracy_pct", "through_balls_pm",
        "forward_pass_share_pct", "long_ball_share_pct"],
    "Do they win the ball?": [
        "duel_win_pct", "duels_contested_pm", "aerial_success_pct", "ground_duel_success_pct",
        "take_on_success_pct", "fouls_won_pm", "fouls_pm"],
    "Do they defend?": [
        "shots_faced_pm", "sot_faced_pm", "conceded_pm", "goals_per_sot_faced_pct",
        "shots_faced_per_goal", "box_shots_faced_pct", "sot_faced_in_box_pct",
        "conceded_in_box_pct", "clean_sheet_pct", "ppda", "def_actions_pm"],
}

PRESETS = {
    "Results": [
        "points_pm", "win_pct", "clean_sheet_pct", "goals_pm", "conceded_pm",
        "goal_diff"],
    "Attack: volume against quality": [
        "chances_pm", "shots_pm", "on_target_pct", "blocked_share_pct",
        "conversion_pct", "shots_per_goal", "goals_pm"],
    "Control and territory": [
        "possession_pct", "field_tilt_pct", "box_touch_share_pct",
        "passes_opp_half_pm", "pass_accuracy_pct"],
    "Route to goal": [
        "crosses_per_shot", "crosses_pm", "through_balls_pm",
        "forward_pass_share_pct", "long_ball_share_pct", "box_touches_per_shot"],
    "Duels and physicality": [
        "duel_win_pct", "aerial_success_pct", "ground_duel_success_pct",
        "take_on_success_pct", "duels_pm", "fouls_pm"],
    "Defence: what they allow": [
        "shots_faced_pm", "sot_faced_pm", "box_shots_faced_pct", "sot_faced_in_box_pct",
        "conceded_pm", "clean_sheet_pct", "goals_per_sot_faced_pct", "ppda"],
    "Both ends: efficiency": [
        "shots_per_goal", "conversion_pct", "shots_faced_per_goal",
        "goals_per_sot_faced_pct", "shot_share_pct", "goal_diff"],
}

HEADLINE_PROFILE = ["possession_pct", "field_tilt_pct", "shot_share_pct", "shots_pm",
                    "chances_pm", "box_touches_pm", "on_target_pct", "conversion_pct",
                    "crosses_per_shot", "forward_pass_share_pct", "duel_win_pct",
                    "shots_faced_pm", "conceded_pm", "clean_sheet_pct", "ppda"]


SCATTERS = {
    "Chances created against conversion": ("chances_pm", "conversion_pct"),
    "Shot volume against shot quality": ("shots_pm", "on_target_pct"),
    "Possession against shot share": ("possession_pct", "shot_share_pct"),
    "Territory against goals": ("field_tilt_pct", "goals_pm"),
    "Crossing against conversion": ("crosses_per_shot", "conversion_pct"),
    "Pressing against goals conceded": ("ppda", "conceded_pm"),
    "Box presence against goals": ("box_touches_pm", "goals_pm"),
}

PLAYER_METRICS = {
    "chances_created_for": "Chances created", "shots_for": "Shots",
    "goals_for": "Goals", "crosses_attempted_for": "Crosses",
    "take_ons_for": "Take-ons", "take_ons_won_for": "Take-ons completed",
    "duels_contested_for": "Duels", "duels_won_for": "Duels won",
    "aerials_won_for": "Aerials won", "passes_attempted_for": "Passes",
    "passes_forward_for": "Forward passes",
    "touches_opp_box_for": "Touches in opposition box",
    "tackles_for": "Tackles", "interceptions_for": "Interceptions",
    "clearances_for": "Clearances", "fouls_committed_for": "Fouls",
}


# Player rates: key -> (label, numerator, denominator, minimum denominator, as %)
PLAYER_RATES = {
    "shot_accuracy": ("Shot accuracy %", "shots_on_target_for", "shots_for", 8, True),
    "conversion": ("Conversion %", "goals_for", "shots_for", 8, True),
    "goals_per_sot": ("Goals per shot on target %", "goals_for",
                      "shots_on_target_for", 4, True),
    "cross_completion": ("Cross completion %", "crosses_completed_for",
                         "crosses_attempted_for", 10, True),
    "take_on_success": ("Take-on success %", "take_ons_won_for", "take_ons_for", 8, True),
    "duel_win": ("Duel win %", "duels_won_for", "duels_contested_for", 20, True),
    "aerial_win": ("Aerial win %", "aerials_won_for", "aerials_for", 10, True),
    "pass_accuracy": ("Pass accuracy %", "passes_completed_for",
                      "passes_attempted_for", 100, True),
    "box_touches_per_shot": ("Box touches per shot", "touches_opp_box_for",
                             "shots_for", 8, False),
    "shots_per_chance": ("Shots per chance created", "shots_for",
                         "chances_created_for", 5, False),
}

# When someone ranks by a volume metric, these rates are worth seeing beside it.
RELATED_RATES = {
    "shots_for": ["shot_accuracy", "conversion"],
    "goals_for": ["conversion", "goals_per_sot"],
    "crosses_attempted_for": ["cross_completion"],
    "take_ons_for": ["take_on_success"],
    "take_ons_won_for": ["take_on_success"],
    "duels_contested_for": ["duel_win"],
    "duels_won_for": ["duel_win"],
    "aerials_won_for": ["aerial_win"],
    "passes_attempted_for": ["pass_accuracy"],
    "chances_created_for": ["shots_per_chance"],
    "touches_opp_box_for": ["box_touches_per_shot"],
}


def lbl(m: str) -> str:
    return METRICS[m][0] if m in METRICS else m


def lower_better(m: str) -> bool:
    return METRICS[m][1] if m in METRICS else False


def dec(m: str) -> int:
    return METRICS[m][2] if m in METRICS else 1


# --------------------------------------------------------------------------
# DATA AND DERIVED METRICS
# --------------------------------------------------------------------------
def col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def div(a: pd.Series, b: pd.Series) -> pd.Series:
    b = b.replace(0, np.nan)
    return a / b


def derive(df: pd.DataFrame, rank: bool = True) -> pd.DataFrame:
    """Turn raw counts into rates, shares and per-match figures."""
    d = df.copy()
    m = col(d, "matches_played")

    shots, sot = col(d, "shots_for"), col(d, "shots_on_target_for")
    blocked, goals = col(d, "shots_blocked_for"), col(d, "goals_for")
    faced, sot_faced = col(d, "shots_against"), col(d, "shots_on_target_against")
    conceded = col(d, "goals_against")

    # Goals include own goals the opponent scored, which belong in the scoreline
    # but are not a finish. Shot efficiency uses only goals from the team's own
    # shots. Older files without that column fall back to the goal total.
    shot_goals = col(d, "goals_from_shots_for")
    if shot_goals.isna().all():
        shot_goals = goals
    shot_goals_against = col(d, "goals_from_shots_against")
    if shot_goals_against.isna().all():
        shot_goals_against = conceded

    d["goal_diff"] = goals - conceded
    d["points_pm"] = div(col(d, "points"), m)
    d["clean_sheet_pct"] = div(col(d, "clean_sheets"), m) * 100
    d["win_pct"] = div(col(d, "wins"), m) * 100
    d["goals_pm"] = div(goals, m)
    d["conceded_pm"] = div(conceded, m)
    d["touches_pm"] = div(col(d, "touches_for"), m)

    # Control
    d["field_tilt_pct"] = div(col(d, "passes_opp_half_for"),
                              col(d, "passes_opp_half_for") + col(d, "passes_opp_half_against")) * 100
    d["box_touch_share_pct"] = div(col(d, "touches_opp_box_for"),
                                   col(d, "touches_opp_box_for") + col(d, "touches_opp_box_against")) * 100
    d["passes_opp_half_pm"] = div(col(d, "passes_opp_half_for"), m)
    d["pass_accuracy_pct"] = col(d, "pass_accuracy_pct_for")

    # Creation
    d["shots_pm"] = div(shots, m)
    d["shot_share_pct"] = div(shots, shots + faced) * 100
    d["box_touches_pm"] = div(col(d, "touches_opp_box_for"), m)
    d["box_touches_per_shot"] = div(col(d, "touches_opp_box_for"), shots)
    d["shots_in_box_pct"] = div(col(d, "shots_in_box_for"), shots) * 100
    chances = col(d, "chances_created_for")
    d["chances_pm"] = div(chances, m)
    d["shots_per_chance"] = div(shots, chances)
    d["chances_per_goal"] = div(chances, goals)

    # Finishing
    # Opta excludes blocked shots from shooting accuracy. Conversion keeps them,
    # so the two deliberately use different denominators.
    d["on_target_pct"] = div(sot, shots - blocked) * 100
    d["blocked_share_pct"] = div(blocked, shots) * 100
    d["conversion_pct"] = div(shot_goals, shots) * 100
    d["goals_per_sot_pct"] = div(shot_goals, sot) * 100
    d["shots_per_goal"] = div(shots, shot_goals)
    d["goals_in_box_pct"] = div(col(d, "goals_in_box_for"), goals) * 100

    # Route
    crosses = col(d, "crosses_attempted_for")
    d["crosses_pm"] = div(crosses, m)
    d["crosses_per_shot"] = div(crosses, shots)
    d["cross_accuracy_pct"] = col(d, "cross_accuracy_pct_for")
    d["through_balls_pm"] = div(col(d, "through_balls_for"), m)
    fwd, back, side = (col(d, "passes_forward_for"), col(d, "passes_backward_for"),
                       col(d, "passes_sideways_for"))
    d["forward_pass_share_pct"] = div(fwd, fwd + back + side) * 100
    d["long_ball_share_pct"] = div(col(d, "long_passes_attempted_for"),
                                   col(d, "passes_attempted_for")) * 100

    # Duels
    aerials, aerials_won = col(d, "aerials_for"), col(d, "aerials_won_for")
    ground, ground_won = col(d, "ground_duels_for"), col(d, "ground_duels_won_for")
    contested = col(d, "duels_contested_for")
    won = col(d, "duels_won_for")
    contested = contested.fillna(aerials + ground)
    won = won.fillna(aerials_won + ground_won)
    d["duels_pm"] = div(contested, m)
    d["duels_contested_pm"] = d["duels_pm"]
    d["duel_win_pct"] = div(won, contested) * 100
    d["aerial_success_pct"] = col(d, "aerial_success_pct_for")
    d["ground_duel_success_pct"] = col(d, "ground_duel_success_pct_for")
    d["take_on_success_pct"] = col(d, "take_on_success_pct_for")
    d["take_ons_pm"] = div(col(d, "take_ons_for"), m)
    d["fouls_won_pm"] = div(col(d, "fouls_won_for"), m)
    d["fouls_pm"] = div(col(d, "fouls_committed_for"), m)

    # Defence
    d["shots_faced_pm"] = div(faced, m)
    d["sot_faced_pm"] = div(sot_faced, m)
    d["goals_per_sot_faced_pct"] = div(shot_goals_against, sot_faced) * 100
    d["shots_faced_per_goal"] = div(faced, shot_goals_against)
    d["box_shots_faced_pct"] = div(col(d, "shots_in_box_against"), faced) * 100
    d["sot_faced_in_box_pct"] = div(col(d, "shots_on_target_in_box_against"), sot_faced) * 100
    d["conceded_in_box_pct"] = div(col(d, "goals_in_box_against"), conceded) * 100

    tackles, ints = col(d, "tackles_for"), col(d, "interceptions_for")
    clears, recov = col(d, "clearances_for"), col(d, "ball_recoveries_for")
    actions = tackles + ints + clears + recov
    d["def_actions_pm"] = div(actions, m)
    d["ppda"] = div(col(d, "passes_attempted_against"),
                    tackles + ints + col(d, "fouls_committed_for"))

    if not rank:
        return d.copy()

    # Ranks within each split. 1 is always the best, direction aware.
    present = [m for m in METRICS if m in d.columns]
    ranks = pd.concat(
        [d.groupby("split_value", observed=True)[m]
         .rank(ascending=lower_better(m), method="min").rename(f"rank__{m}")
         for m in present], axis=1)
    return pd.concat([d, ranks], axis=1).copy()


@st.cache_data(show_spinner=False)
def load_data(folder: str) -> dict:
    base = Path(folder)
    out = {}
    for key, name in [("totals", "league_totals.csv"),
                      ("time", "league_by_time_block.csv"),
                      ("state", "league_by_game_state.csv")]:
        path = base / name
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run the pipeline first.")
        out[key] = derive(pd.read_csv(path))

    # One row per match, rebuilt from the long file.
    out["matches"] = pd.DataFrame()
    long_path = base / "match_level_long.csv"
    if long_path.exists():
        long_df = pd.read_csv(long_path)
        sub = long_df[long_df["split_type"] == "total"]
        metric_cols = [c for c in sub.columns if c not in
                       {"team", "split_type", "match_id", "side", "split_value"}]
        wide = sub.pivot_table(index=["team", "match_id"], columns="side",
                               values=metric_cols, aggfunc="sum")
        wide.columns = [f"{a}_{b}" for a, b in wide.columns]
        wide = wide.reset_index()
        wide["matches_played"] = 1
        wide["split_value"] = "Season"
        pairs = wide[["match_id", "team"]].copy()
        opp = pairs.merge(pairs, on="match_id")
        opp = opp[opp["team_x"] != opp["team_y"]][["match_id", "team_x", "team_y"]]
        opp.columns = ["match_id", "team", "opponent"]
        wide = wide.merge(opp, on=["match_id", "team"], how="left")
        sched = base / "schedule.csv"
        if sched.exists():
            sc = pd.read_csv(sched)
            if {"game_id", "date"}.issubset(sc.columns):
                wide = wide.merge(sc[["game_id", "date"]].rename(
                    columns={"game_id": "match_id"}), on="match_id", how="left")
        out["matches"] = derive(wide, rank=False)

    out["players"] = pd.DataFrame()
    ppath = base / "league_players.csv"
    if ppath.exists():
        out["players"] = pd.read_csv(ppath)
    return out


def order_splits(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    present = [v for v in order if v in set(df["split_value"])]
    d = df.copy()
    d["split_value"] = pd.Categorical(d["split_value"], categories=present, ordered=True)
    return d.sort_values("split_value")


# --------------------------------------------------------------------------
# TABLES
# --------------------------------------------------------------------------
def thin_sample(touches: float) -> bool:
    return pd.isna(touches) or touches < TOUCH_FLOOR


def thin_note(touches: float) -> None:
    mins = int(touches / 13) if touches else 0
    st.info(f"This covers roughly {mins} minutes of football all season. The numbers "
            "below will move around a lot from week to week, so there is not enough "
            "here yet to read anything into.")


def profile_table(row: pd.Series, peers: pd.DataFrame, metrics: list[str],
                  compact: bool = False) -> pd.DataFrame:
    n = len(peers)
    out = []
    for metric in metrics:
        if metric not in peers.columns or pd.isna(row.get(metric)):
            continue
        avg = peers[metric].mean()
        rank = row.get(f"rank__{metric}")
        entry = {
            "Metric": lbl(metric),
            "Value": round(float(row[metric]), dec(metric)),
            "League avg": round(float(avg), dec(metric)),
            "Rank": f"{int(rank)}/{n}" if pd.notna(rank) else "",
        }
        if not compact:
            entry["Gap"] = round(float(row[metric] - avg), dec(metric))
            entry["What it measures"] = METRICS[metric][3]
        out.append(entry)
    return pd.DataFrame(out)


def notable_teams(d: pd.DataFrame, x: str, y: str, highlight: str, n: int = 3) -> set:
    """The teams worth naming: the highlighted one and the extremes on each axis."""
    picks = {highlight}
    for axis in (x, y):
        picks |= set(d.nlargest(n, axis)["team"])
        picks |= set(d.nsmallest(n, axis)["team"])
    return picks


def scatter(df: pd.DataFrame, x: str, y: str, highlight: str,
            label_all: bool = False) -> alt.Chart:
    d = df[["team", x, y]].dropna().copy()
    d["Highlight"] = np.where(d["team"] == highlight, highlight, "Other teams")
    shown = set(d["team"]) if label_all else notable_teams(d, x, y, highlight)
    d["Label"] = np.where(d["team"].isin(shown), d["team"], "")
    base = alt.Chart(d)
    colour = alt.Color("Highlight:N", legend=None,
                       scale=alt.Scale(domain=[highlight, "Other teams"],
                                       range=["#E4572E", "#8A9BA8"]))
    points = base.mark_circle(size=150, opacity=0.85).encode(
        x=alt.X(f"{x}:Q", title=lbl(x), scale=alt.Scale(zero=False)),
        y=alt.Y(f"{y}:Q", title=lbl(y), scale=alt.Scale(zero=False)),
        color=colour,
        tooltip=["team", alt.Tooltip(f"{x}:Q", format=".2f", title=lbl(x)),
                 alt.Tooltip(f"{y}:Q", format=".2f", title=lbl(y))])
    text = base.mark_text(align="left", baseline="middle", dx=10, fontSize=11).encode(
        x=f"{x}:Q", y=f"{y}:Q", text="Label:N", color=colour)
    vline = alt.Chart(pd.DataFrame({"v": [d[x].mean()]})).mark_rule(
        strokeDash=[4, 4], color="#B0BEC5").encode(x="v:Q")
    hline = alt.Chart(pd.DataFrame({"v": [d[y].mean()]})).mark_rule(
        strokeDash=[4, 4], color="#B0BEC5").encode(y="v:Q")
    return (points + text + vline + hline).properties(height=480).interactive()


def whatsapp_text(team: str, row: pd.Series, totals: pd.DataFrame) -> str:
    n = len(totals)
    lines = [f"*{team.upper()}: SEASON SO FAR*",
             f"_{int(row.get('matches_played', 0))} matches. Rank out of {n}, "
             "1 is best._", ""]
    for section, metrics in SECTIONS.items():
        rows = [m for m in metrics if m in totals.columns and pd.notna(row.get(m))]
        if not rows:
            continue
        lines.append(f"*{section}*")
        for m in rows:
            rank = row.get(f"rank__{m}")
            rank_txt = f" ({int(rank)}/{n})" if pd.notna(rank) else ""
            lines.append(f"• {lbl(m)}: {row[m]:.{dec(m)}f}{rank_txt}")
        lines.append("")
    return "\n".join(lines).strip()


def section_read(row: pd.Series, peers: pd.DataFrame, metrics: list[str]) -> str:
    """One sentence naming the strongest and weakest thing in this section."""
    n = len(peers)
    ranked = [(m, row[f"rank__{m}"]) for m in metrics
              if f"rank__{m}" in row and pd.notna(row.get(f"rank__{m}"))]
    if not ranked:
        return ""
    ranked.sort(key=lambda kv: kv[1])
    best, best_r = ranked[0]
    worst, worst_r = ranked[-1]
    if best == worst:
        return f"{lbl(best)}: {int(best_r)} of {n}."
    return (f"Best here: {lbl(best)}, {int(best_r)} of {n}. "
            f"Weakest here: {lbl(worst)}, {int(worst_r)} of {n}.")


def preset_table(df: pd.DataFrame, metrics: list[str], sort_by: str) -> pd.DataFrame:
    cols = [m for m in metrics if m in df.columns]
    table = df[["team"] + cols].copy()
    table = table.sort_values(sort_by, ascending=lower_better(sort_by))
    for m in cols:
        table[m] = table[m].round(dec(m))
    avg = {"team": "League average"}
    for m in cols:
        avg[m] = round(float(df[m].mean()), dec(m))
    table = pd.concat([table, pd.DataFrame([avg])], ignore_index=True)
    table = table.rename(columns={"team": "Team", **{m: lbl(m) for m in cols}})
    return table


# --------------------------------------------------------------------------
# CLAUDE
# --------------------------------------------------------------------------
def get_api_key() -> str | None:
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


def claude_available() -> bool:
    return bool(get_api_key())


@st.cache_data(show_spinner=False, ttl=3600)
def claude_analysis(payload_json: str) -> str:
    try:
        from anthropic import Anthropic
    except ImportError:
        return "Install the anthropic package to enable written analysis."
    key = get_api_key()
    if not key:
        return ""

    prompt = f"""You are a football data analyst briefing someone who follows the game
but does not read stats sheets. Here are the figures.

{payload_json}

Write four to six sentences.

Rules:
- Use only the numbers given. Never estimate, infer or invent a figure.
- Lead with the single most interesting thing, not a summary of everything.
- Connect metrics to each other where the data supports it. Say what kind of team
  this is and where the numbers disagree with each other.
- Plain declarative sentences. No em dashes anywhere. No hedging filler.
- If the sample is small, one short sentence at the end saying so.
- No headings, no bullets, no preamble."""

    try:
        client = Anthropic(api_key=key)
        msg = client.messages.create(model=CLAUDE_MODEL, max_tokens=600,
                                     messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if b.type == "text").strip()
    except Exception as exc:
        return f"Analysis unavailable: {type(exc).__name__}."


def payload_for(team: str, row: pd.Series, peers: pd.DataFrame,
                metrics: list[str], context: str) -> str:
    n = len(peers)
    body = {}
    for m in metrics:
        if m not in peers.columns or pd.isna(row.get(m)):
            continue
        body[lbl(m)] = {
            "value": round(float(row[m]), dec(m)),
            "league_average": round(float(peers[m].mean()), dec(m)),
            "rank_where_1_is_best": int(row[f"rank__{m}"]) if pd.notna(row.get(f"rank__{m}")) else None,
            "teams_ranked": n,
        }
    return json.dumps({"team": team, "context": context,
                       "matches_played": int(row.get("matches_played", 0)),
                       "metrics": body}, indent=1)


# --------------------------------------------------------------------------
# PAGES
# --------------------------------------------------------------------------
def page_season(data: dict, team: str, compact: bool, use_ai: bool) -> None:
    totals = data["totals"]
    rows = totals[totals["team"] == team]
    if rows.empty:
        st.info("No data for this team yet.")
        return
    row = rows.iloc[0]

    st.header(f"{team}, season so far")
    st.caption(f"{int(row.get('matches_played', 0))} matches. Every figure is a rate, a "
               f"share or a per-match number, so it is comparable across teams. "
               f"Rank 1 is always the best of {len(totals)}, whichever direction the metric runs.")

    cards = [("Points per match", "points_pm", "{:.2f}", False),
             ("Shots per match", "shots_pm", "{:.1f}", False),
             ("Chances per match", "chances_pm", "{:.1f}", False),
             ("Conversion", "conversion_pct", "{:.1f}%", False),
             ("Conceded per match", "conceded_pm", "{:.2f}", True)]
    cols = st.columns(2 if compact else 5)
    for i, (name, key, fmt, invert) in enumerate(cards):
        if key not in totals.columns or pd.isna(row.get(key)):
            continue
        gap = row[key] - totals[key].mean()
        cols[i % len(cols)].metric(name, fmt.format(row[key]), f"{gap:+.2f} vs league",
                                   delta_color="inverse" if invert else "normal")

    st.divider()
    for section, metrics in SECTIONS.items():
        st.subheader(section)
        table = profile_table(row, totals, metrics, compact)
        if table.empty:
            continue
        st.dataframe(table, hide_index=True, width="stretch")
        read = section_read(row, totals, metrics)
        if read:
            st.markdown(f"*{read}*")

    if use_ai:
        st.divider()
        st.subheader("Analysis")
        with st.spinner("Writing the analysis"):
            text = claude_analysis(payload_for(team, row, totals, HEADLINE_PROFILE,
                                               "whole season to date"))
        if text:
            st.write(text)

    st.divider()
    with st.expander("Send this to WhatsApp"):
        st.caption("Copy the block below and paste it into a chat. The asterisks "
                   "turn into bold formatting.")
        st.code(whatsapp_text(team, row, totals), language=None)


def page_splits(data: dict, team: str, kind: str, compact: bool, use_ai: bool) -> None:
    key = "time" if kind == "time" else "state"
    order = TIME_ORDER if key == "time" else STATE_ORDER
    heading = "through the match" if key == "time" else "by scoreline"
    unit = "time block" if key == "time" else "scoreline"

    league = data[key]
    rows = order_splits(league[league["team"] == team], order)
    if rows.empty:
        st.info("No data for this team yet.")
        return

    st.header(f"{team} {heading}")

    view = st.radio("Look at", ["Attack", "Defence", "Route to goal"], horizontal=True)
    sets = {
        "Attack": ["chances_pm", "shots_pm", "shot_share_pct", "on_target_pct",
                   "conversion_pct", "box_touches_pm", "goals_pm"],
        "Defence": ["shots_faced_pm", "sot_faced_pm", "box_shots_faced_pct",
                    "conceded_pm", "goals_per_sot_faced_pct", "possession_pct"],
        "Route to goal": ["crosses_per_shot", "crosses_pm", "through_balls_pm",
                          "forward_pass_share_pct", "box_touches_per_shot"],
    }
    metrics = [m for m in sets[view] if m in rows.columns]
    if compact:
        metrics = metrics[:4]

    league_avg = league.groupby("split_value", observed=True)[metrics].mean()
    pick = st.selectbox("Compare against the league", metrics, format_func=lbl)
    st.bar_chart(pd.DataFrame({team: rows.set_index("split_value")[pick],
                               "League average": league_avg[pick]}).round(dec(pick)))

    with st.expander("All the numbers", expanded=not compact):
        table = rows[["split_value"] + metrics].copy()
        for m in metrics:
            table[m] = table[m].round(dec(m))
        st.dataframe(table.rename(columns={"split_value": unit.capitalize(),
                                           **{m: lbl(m) for m in metrics}}),
                     hide_index=True, width="stretch")

    st.divider()
    st.subheader(f"One {unit} at a time")
    choice = st.selectbox(f"Choose a {unit}", list(rows["split_value"]))
    row = rows[rows["split_value"] == choice].iloc[0]
    peers = league[league["split_value"] == choice]

    touches = row.get("touches_for", 0)
    if thin_sample(touches):
        thin_note(touches)

    st.dataframe(profile_table(row, peers, HEADLINE_PROFILE, compact),
                 hide_index=True, width="stretch")

    if use_ai and not thin_sample(touches):
        with st.spinner("Writing the analysis"):
            text = claude_analysis(payload_for(team, row, peers, HEADLINE_PROFILE,
                                               f"{unit}: {choice}"))
        if text:
            st.subheader("Analysis")
            st.write(text)


def page_focus(data: dict, team: str, compact: bool, use_ai: bool) -> None:
    totals = data["totals"]
    rows = totals[totals["team"] == team]
    if rows.empty:
        st.info("No data for this team yet.")
        return
    row, n = rows.iloc[0], len(totals)

    st.header(f"{team} against the other nineteen")

    ranked = [(m, int(row[f"rank__{m}"])) for m in HEADLINE_PROFILE
              if f"rank__{m}" in row and pd.notna(row.get(f"rank__{m}"))]
    ranked.sort(key=lambda kv: kv[1])

    cols = st.columns(1 if compact else 2)
    with cols[0]:
        st.subheader("Strengths")
        for m, r in [x for x in ranked if x[1] <= 6][:6]:
            st.markdown(f"- {lbl(m)}: **{row[m]:.{dec(m)}f}**, {r} of {n} "
                        f"(league average {totals[m].mean():.{dec(m)}f})")
    with cols[-1]:
        st.subheader("Weaknesses")
        for m, r in [x for x in ranked if x[1] >= n - 5][-6:]:
            st.markdown(f"- {lbl(m)}: **{row[m]:.{dec(m)}f}**, {r} of {n} "
                        f"(league average {totals[m].mean():.{dec(m)}f})")

    st.subheader("Rank across the headline measures")
    prof = pd.DataFrame({"Rank": [r for _, r in ranked]},
                        index=[lbl(m) for m, _ in ranked])
    st.bar_chart(prof)
    st.caption("Shorter is better. Rank 1 is the best in the league on that measure.")

    st.divider()
    st.subheader("Where volume and output disagree")
    time_df = data["time"]
    t = order_splits(time_df[time_df["team"] == team], TIME_ORDER)
    avg = time_df.groupby("split_value", observed=True)[["shots_pm", "on_target_pct"]].mean()
    comp = pd.DataFrame({
        "Shots per match": t.set_index("split_value")["shots_pm"].round(1),
        "League average shots": avg["shots_pm"].round(1),
        "On target %": t.set_index("split_value")["on_target_pct"].round(1),
        "League on target %": avg["on_target_pct"].round(1),
    })
    comp["Quality gap"] = (comp["League on target %"] - comp["On target %"]).round(1)
    st.dataframe(comp.reset_index().rename(columns={"split_value": "Time block"}),
                 hide_index=True, width="stretch")
    worst = comp["Quality gap"].idxmax()
    w = comp.loc[worst]
    st.markdown(
        f"- **{worst}** is the widest gap. {w['Shots per match']} shots per match against "
        f"a league average of {w['League average shots']}, but only {w['On target %']}% "
        f"on target against the league's {w['League on target %']}%."
    )

    states = order_splits(data["state"][data["state"]["team"] == team], STATE_ORDER)
    if not states.empty:
        st.subheader("How the season has actually been spent")
        share = states.set_index("split_value")["touches_for"]
        st.bar_chart(share.rename("Touches"))
        total, level = share.sum(), share.get("Level", 0)
        st.markdown(f"- {int(level)} of {int(total)} touches come with the scores level, "
                    f"{round(level / total * 100, 1)}% of everything they have done.")
        thin = [str(i) for i, v in share.items() if v < TOUCH_FLOOR]
        if thin:
            st.markdown(f"- Barely any football yet at: {', '.join(thin)}.")

    if use_ai:
        st.divider()
        st.subheader("Analysis")
        with st.spinner("Writing the analysis"):
            text = claude_analysis(payload_for(team, row, totals, HEADLINE_PROFILE,
                                               "season profile against the league"))
        if text:
            st.write(text)


def page_matches(data: dict, team: str, compact: bool) -> None:
    matches = data["matches"]
    st.header(f"{team}, match by match")
    if matches.empty:
        st.info("No match-level file found. Re-run the pipeline to create it.")
        return

    mine = matches[matches["team"] == team].copy()
    if "date" in mine.columns:
        mine["date"] = pd.to_datetime(mine["date"], errors="coerce")
        mine = mine.sort_values("date")
    st.caption("Season aggregates hide whether a number is a habit or one freak match. "
               "Same metrics, one row per game.")

    view = st.selectbox("Look at", ["Attack", "Defence", "Control", "Route to goal"])
    sets = {"Attack": ["chances_pm", "shots_pm", "on_target_pct", "conversion_pct",
                       "goals_pm"],
            "Defence": ["shots_faced_pm", "sot_faced_pm", "conceded_pm",
                        "box_shots_faced_pct"],
            "Control": ["possession_pct", "field_tilt_pct", "shot_share_pct",
                        "box_touches_pm"],
            "Route to goal": ["crosses_per_shot", "crosses_pm", "through_balls_pm",
                              "forward_pass_share_pct"]}
    metrics = [m for m in sets[view] if m in mine.columns]
    if compact:
        metrics = metrics[:3]

    show = mine.copy()
    show["Match"] = show["opponent"] if "opponent" in show.columns else ""
    if "date" in show.columns:
        show["Date"] = show["date"].dt.strftime("%d %b")
    show["Score"] = (show["goals_for"].fillna(0).astype(int).astype(str) + " - "
                     + show["goals_against"].fillna(0).astype(int).astype(str))
    cols = (["Date"] if "Date" in show.columns else []) + ["Match", "Score"] + metrics
    table = show[cols].copy()
    for m in metrics:
        table[m] = table[m].round(dec(m))
    st.dataframe(table.rename(columns={m: lbl(m) for m in metrics}),
                 hide_index=True, width="stretch")

    pick = st.selectbox("Chart across the season", metrics, format_func=lbl)
    st.bar_chart(show.set_index("Match")[pick].rename(lbl(pick)))
    avg, lo, hi = show[pick].mean(), show[pick].min(), show[pick].max()
    st.markdown(f"- Season average {avg:.{dec(pick)}f}, ranging from {lo:.{dec(pick)}f} "
                f"to {hi:.{dec(pick)}f}.")
    if avg and (hi - lo) > abs(avg):
        st.markdown("- That spread is wider than the average itself, so the season "
                    "figure is being pulled around by individual matches.")


def page_map(data: dict, team: str) -> None:
    st.header("The league on two axes")
    st.caption("Tables hide outliers. Dashed lines are the league average, so the "
               "quadrant a team sits in is the story.")

    where = st.selectbox("Cover", ["Season total", "One time block", "One scoreline"])
    if where == "Season total":
        df, choice = data["totals"], "Season"
    elif where == "One time block":
        df = data["time"]
        choice = st.selectbox("Time block",
                              [v for v in TIME_ORDER if v in set(df["split_value"])])
    else:
        df = data["state"]
        choice = st.selectbox("Scoreline",
                              [v for v in STATE_ORDER if v in set(df["split_value"])])
    sub = df[df["split_value"] == choice]

    preset = st.selectbox("Compare", list(SCATTERS))
    x, y = SCATTERS[preset]
    if x not in sub.columns or y not in sub.columns:
        st.info("Those metrics are not in this file.")
        return

    label_all = st.toggle("Name every team", value=False,
                          help="Off means only the extremes and your team are "
                               "labelled, so the names do not sit on top of each "
                               "other. Hover any dot for its numbers.")
    st.altair_chart(scatter(sub, x, y, team, label_all), width="stretch")

    if team in set(sub["team"]):
        r = sub[sub["team"] == team].iloc[0]
        xm, ym = sub[x].mean(), sub[y].mean()
        st.markdown(
            f"- {team}: {lbl(x)} {r[x]:.{dec(x)}f}, "
            f"{'above' if r[x] > xm else 'below'} the league average of {xm:.{dec(x)}f}. "
            f"{lbl(y)} {r[y]:.{dec(y)}f}, "
            f"{'above' if r[y] > ym else 'below'} the average of {ym:.{dec(y)}f}.")
    st.caption(f"{lbl(x)}: {METRICS[x][3]}  |  {lbl(y)}: {METRICS[y][3]}")


def page_compare(data: dict, team: str, compact: bool) -> None:
    totals = data["totals"]
    teams = sorted(totals["team"].unique())
    st.header("Head to head")

    c1, c2 = st.columns(2)
    a = c1.selectbox("Team", teams, index=teams.index(team) if team in teams else 0)
    b = c2.selectbox("Against", [t for t in teams if t != a], index=0)

    preset = st.selectbox("Compare on", list(PRESETS))
    metrics = [m for m in PRESETS[preset] if m in totals.columns]
    ra = totals[totals["team"] == a].iloc[0]
    rb = totals[totals["team"] == b].iloc[0]

    table = pd.DataFrame({
        "Metric": [lbl(m) for m in metrics],
        a: [round(float(ra[m]), dec(m)) for m in metrics],
        b: [round(float(rb[m]), dec(m)) for m in metrics],
        "League avg": [round(float(totals[m].mean()), dec(m)) for m in metrics],
        f"{a} rank": [int(ra[f"rank__{m}"]) if pd.notna(ra.get(f"rank__{m}")) else None
                      for m in metrics],
        f"{b} rank": [int(rb[f"rank__{m}"]) if pd.notna(rb.get(f"rank__{m}")) else None
                      for m in metrics]})
    if compact:
        table = table[["Metric", a, b, "League avg"]]
    st.dataframe(table, hide_index=True, width="stretch")

    pick = st.selectbox("Chart one", metrics, format_func=lbl)
    st.bar_chart(pd.DataFrame({lbl(pick): [ra[pick], rb[pick], totals[pick].mean()]},
                              index=[a, b, "League average"]).round(dec(pick)))

    both = [m for m in metrics if pd.notna(ra.get(f"rank__{m}"))
            and pd.notna(rb.get(f"rank__{m}"))]
    wins = sum(1 for m in both if ra[f"rank__{m}"] < rb[f"rank__{m}"])
    if both:
        st.markdown(f"- {a} ranks higher than {b} on {wins} of {len(both)} measures here.")


def player_rate(df: pd.DataFrame, key: str) -> pd.Series:
    """One rate for every player, blank where the sample is too small."""
    label, num, den, min_den, as_pct = PLAYER_RATES[key]
    if num not in df.columns or den not in df.columns:
        return pd.Series(np.nan, index=df.index)
    d = pd.to_numeric(df[den], errors="coerce")
    n = pd.to_numeric(df[num], errors="coerce")
    out = n / d.replace(0, np.nan)
    if as_pct:
        out = out * 100
    return out.where(d >= min_den).round(1 if as_pct else 2)


def page_players(data: dict, team: str, compact: bool) -> None:
    players = data["players"]
    st.header("Players")
    if players.empty:
        st.info("No player file found. Re-run the pipeline with version 6 or later.")
        return

    scope = st.radio("Show", [team, "Whole league"], horizontal=True)
    df = players if scope == "Whole league" else players[players["team"] == team]
    df = df[df.get("minutes", pd.Series(0, index=df.index)) >= MIN_PLAYER_MINUTES].copy()
    if df.empty:
        st.info(f"Nobody with {MIN_PLAYER_MINUTES} minutes yet.")
        return

    basis = st.radio("Rank by", ["Volume", "Rate"], horizontal=True,
                     help="Volume is how much a player does. Rate is how well.")

    if basis == "Rate":
        available = {k: v[0] for k, v in PLAYER_RATES.items()
                     if v[1] in df.columns and v[2] in df.columns}
        if not available:
            st.info("The player file has no rate inputs in it.")
            return
        key = st.selectbox("Rate", list(available), format_func=lambda k: available[k])
        rate_label, num, den, min_den, _ = PLAYER_RATES[key]
        d = df.copy()
        d["value"] = player_rate(d, key)
        d = d.dropna(subset=["value"])
        if d.empty:
            st.info(f"Nobody has reached {min_den} {lbl(den).lower()} yet, so this rate "
                    "would be noise.")
            return
        ascending = key == "box_touches_per_shot"
        d = d.sort_values("value", ascending=ascending).head(15)

        cols = {"player": "Player", "team": "Team", "minutes": "Minutes",
                num: lbl(num) if num in METRICS else num.replace("_for", "").replace("_", " "),
                den: den.replace("_for", "").replace("_", " "),
                "value": rate_label}
        keep = ["player"] + ([] if scope != "Whole league" else ["team"]) + \
               (["minutes"] if not compact else []) + [num, den, "value"]
        show = d[[c for c in keep if c in d.columns]].copy()
        if "minutes" in show:
            show["minutes"] = show["minutes"].round(0)
        st.dataframe(show.rename(columns=cols), hide_index=True, width="stretch")
        st.bar_chart(d.set_index("player")["value"].rename(rate_label))
        st.caption(f"Minimum {min_den} to qualify, so one lucky attempt cannot top the "
                   "table. A rate with no volume behind it is not an achievement, so "
                   "read this next to the volume view.")
        return

    available = {k: v for k, v in PLAYER_METRICS.items() if k in df.columns}
    if not available:
        st.info("The player file has no recognised metrics in it.")
        return
    metric = st.selectbox("Metric", list(available), format_func=lambda k: available[k])
    per = st.radio("Per", ["90 minutes", "Total"], horizontal=True)

    d = df.copy()
    d["per90"] = (d[metric] / d["minutes"].replace(0, np.nan) * 90).round(2)
    d["total"] = d[metric].round(0)
    d["value"] = d["per90"] if per == "90 minutes" else d["total"]

    # Share of what the team produced, which is the context a raw count lacks.
    team_totals = players.groupby("team")[metric].sum()
    d["share"] = (d[metric] / d["team"].map(team_totals) * 100).round(1)

    d = d.dropna(subset=["value"]).sort_values("value", ascending=False).head(15)

    rates = [k for k in RELATED_RATES.get(metric, [])
             if PLAYER_RATES[k][1] in d.columns and PLAYER_RATES[k][2] in d.columns]
    for k in rates:
        d[k] = player_rate(d, k)

    value_label = f"{available[metric]} {'per 90' if per == '90 minutes' else 'total'}"
    cols = {"player": "Player", "team": "Team", "appearances": "Apps",
            "minutes": "Minutes", "value": value_label,
            "share": "Share of team %", **{k: PLAYER_RATES[k][0] for k in rates}}
    keep = ["player"] + ([] if scope != "Whole league" else ["team"]) + \
           ([] if compact else ["appearances", "minutes"]) + ["value", "share"] + rates
    show = d[[c for c in keep if c in d.columns]].copy()
    if "minutes" in show:
        show["minutes"] = show["minutes"].round(0)
    st.dataframe(show.rename(columns=cols), hide_index=True, width="stretch")
    st.bar_chart(d.set_index("player")["value"].rename(available[metric]))

    notes = ["Share of team shows how concentrated this is in one player."]
    if rates:
        notes.append("The rate columns say whether the volume is any good.")
    if per == "90 minutes":
        notes.append("Anyone under about 200 minutes will show wild per-90 numbers, "
                     "so check the minutes column before believing one.")
    st.caption(" ".join(notes))
    st.caption(f"Players with at least {MIN_PLAYER_MINUTES} minutes. Minutes are worked "
               "out from the substitution events, so they are close but not official.")


def page_league(data: dict, team: str, compact: bool) -> None:
    st.header("League tables")
    st.caption("Themed views rather than one metric at a time, so each number has "
               "something next to it that explains it.")

    where = st.radio("Cover", ["Season total", "One time block", "One scoreline"],
                     horizontal=True)
    if where == "Season total":
        df, choice = data["totals"], "Season"
    elif where == "One time block":
        df = data["time"]
        choice = st.selectbox("Time block",
                              [v for v in TIME_ORDER if v in set(df["split_value"])])
    else:
        df = data["state"]
        choice = st.selectbox("Scoreline",
                              [v for v in STATE_ORDER if v in set(df["split_value"])])

    sub = df[df["split_value"] == choice]
    preset = st.selectbox("View", list(PRESETS))
    metrics = [m for m in PRESETS[preset] if m in sub.columns]
    sort_by = st.selectbox("Sort by", metrics, format_func=lbl)
    shown = metrics[:3] if compact else metrics
    if sort_by not in shown:
        shown = [sort_by] + shown[:-1]

    st.dataframe(preset_table(sub, shown, sort_by), hide_index=True, width="stretch")
    st.caption("Sorted best first. " + " | ".join(
        f"{lbl(m)}: {METRICS[m][3]}" for m in shown))

    st.subheader(f"{lbl(sort_by)} across the league")
    chart = sub.set_index("team")[sort_by].sort_values(
        ascending=lower_better(sort_by)).round(dec(sort_by))
    st.bar_chart(chart.rename(lbl(sort_by)))

    if team in set(sub["team"]):
        r = sub[sub["team"] == team].iloc[0]
        rank = r.get(f"rank__{sort_by}")
        if pd.notna(rank):
            st.markdown(f"- {team}: **{r[sort_by]:.{dec(sort_by)}f}**, ranked "
                        f"{int(rank)} of {len(sub)}, against a league average of "
                        f"{sub[sort_by].mean():.{dec(sort_by)}f}.")


# --------------------------------------------------------------------------
# ACCESS
# --------------------------------------------------------------------------
def password_ok() -> bool:
    """True when the viewer may see the app.

    With no APP_PASSWORD in secrets there is no gate at all, which is what you
    want when running it locally.
    """
    try:
        wanted = st.secrets.get("APP_PASSWORD")
    except Exception:
        wanted = None
    if not wanted:
        return True
    if st.session_state.get("access_granted"):
        return True

    st.title("Premier League splits")
    st.write("This one is not public. Enter the password from the message.")
    given = st.text_input("Password", type="password",
                          label_visibility="collapsed", placeholder="Password")
    if given:
        if given == wanted:
            st.session_state["access_granted"] = True
            st.rerun()
        else:
            st.error("Not that one.")
    return False


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main() -> None:
    if not password_ok():
        return

    try:
        data = load_data(str(DATA_DIR))
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    teams = sorted(data["totals"]["team"].unique())
    default_idx = next((i for i, t in enumerate(teams)
                        if FOCUS_TEAM_HINT.lower() in t.lower()), 0)

    with st.sidebar:
        st.header("Premier League splits")
        team = st.selectbox("Team", teams, index=default_idx)
        page = st.radio("Show", ["Season so far", "Match by match", "Through the match",
                                 "By scoreline", "Against the league", "League map",
                                 "Head to head", "Players", "League tables"])
        st.divider()
        compact = st.toggle("Phone layout", value=True,
                            help="Fewer columns and tucked-away tables, for small screens.")
        use_ai = st.toggle("Written analysis", value=claude_available(),
                           disabled=not claude_available())
        if not claude_available():
            st.caption("Add ANTHROPIC_API_KEY to secrets to switch this on.")
        st.divider()
        st.caption(f"Season to date, up to "
                   f"{int(data['totals']['matches_played'].max())} matches per team.")

    if page == "Season so far":
        page_season(data, team, compact, use_ai)
    elif page == "Match by match":
        page_matches(data, team, compact)
    elif page == "Through the match":
        page_splits(data, team, "time", compact, use_ai)
    elif page == "By scoreline":
        page_splits(data, team, "state", compact, use_ai)
    elif page == "Against the league":
        page_focus(data, team, compact, use_ai)
    elif page == "League map":
        page_map(data, team)
    elif page == "Head to head":
        page_compare(data, team, compact)
    elif page == "Players":
        page_players(data, team, compact)
    else:
        page_league(data, team, compact)


if __name__ == "__main__":
    main()
