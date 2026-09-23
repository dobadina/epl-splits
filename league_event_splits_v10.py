"""
Premier League event-data pipeline: every team, split by time block and game state.

Version 10: Opta definitions, phase 2
-------------------------------------
Adds the columns the app needs to replace metrics I invented with published
ones, all from coordinates already in the cache.

* Final-third passes, by origin and by entry, which also lets field tilt use
  its standard definition instead of the opposition half.
* Progressive passes to Opta's definition: a completed pass in the attacking
  two-thirds that moves the ball at least 25% closer to the goal.
* The two halves of PPDA as Opta defines it: opposition passes outside their
  own defensive third, and defensive actions outside ours.
* Pressing actions and recovery actions separated, since a tackle high up and a
  clearance in your own box are opposite things.
* Missed tackles: challenges lost plus fouls committed while attempting a tackle.

Version 9 changes: Opta definitions, phase 1
-------------------------------------------
Brings four metrics into line with Opta's published definitions, so the numbers
can be compared with anyone else's.

* Shooting accuracy now excludes blocked shots from the denominator, which is
  how Opta defines it. Conversion keeps all shots, which is also Opta's
  definition, so the two use different denominators on purpose.
* Crosses, throw-ins and keeper throws no longer count as passes. Opta treats
  crossing separately, so including it was dragging pass accuracy down for the
  teams that cross most. Deliveries are still counted, under crosses_all.
* Duels rebuilt to Opta's five pairs. Fouls and smothers are duels and were
  missing entirely, which is why the contested counts looked low.
* A tackle is a duel won whatever its outcome. Opta records won and lost
  tackles as tackles, and the duel is with the player who lost the ball.

Version 8 change
----------------
* Own goals are now credited to the team that benefits. Before, a Goal event
  tagged as an own goal was excluded from the scorer's total and never added to
  the opponent's, so the goal vanished. That made scorelines, wins, draws, clean
  sheets and points wrong in any match containing one.
* Shot conversion still uses goals from the team's own shots, since an opponent's
  own goal is not a finish. That figure is kept as goals_from_shots.

Version 7 change
----------------
* The placeholder cleanup now checks file size rather than row count. The old
  check saw every cached match as empty, so it deleted the whole cache and
  re-fetched all of it on every run.

Version 6 changes
-----------------
* Adds player-level output: league_players.csv, with every metric aggregated per
  player per team, plus appearances and minutes played worked out from the
  substitution events. Everything else is unchanged.

Version 5 changes
-----------------
* Adds chances created, assists, total duels contested and won, shots and goals
  split by inside or outside the box, and shots on target by box location.
* Adds match results: clean sheets, wins, draws, losses and points, which only
  appear in the season totals file.

Version 4 changes
-----------------
* Fixtures that have not kicked off are never requested. Version 3 asked about
  all 380 calendar entries, spending most of a two-hour run on empty games.
* A match with no events is no longer cached. Version 3 wrote an empty file,
  which made the script treat that fixture as done forever, so its real data
  would never be fetched once the match was played.
* Empty files left by earlier versions are cleared automatically at startup.
* The cross definition excludes corners and free kicks, matching published
  cross counts.

Weekly refresh: just run it again. Only newly played matches are fetched.

Version 3 change
----------------
* The qualifier reader now accepts numpy arrays and JSON text, not just lists.
  Parquet returns that column as an array, so version 2 read no qualifiers at
  all: no crosses, no corners, no long balls, and every saved shot counted as
  on target because blocked shots could not be identified.

Version 2 changes
-----------------
* Team names now come from the event data itself, not the fixture list. The two
  sources spell clubs differently ("Man Utd" vs "Manchester United"), which
  silently wiped out every "for" metric in version 1.
* The fixture list is fetched once, not before every match. Version 1 spent
  roughly four minutes per game re-scraping the calendar. This cuts it to well
  under a minute.

Run it with:  python league_event_splits.py
Configure it in the CONFIG block below.
"""

from __future__ import annotations

import difflib
import json
import time
import traceback
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
LEAGUE = "ENG-Premier League"
SEASON = "2627"
OUTPUT_DIR = Path("output")
CACHE_DIR = Path("match_cache")
FOCUS_TEAM = "Manchester United"   # matched loosely against the event data names

MAX_MATCHES = None                 # set to e.g. 5 for a test run
PAUSE_SECONDS = 2
RETRY_FAILED = True
DOWNLOAD = False                    # set False to re-process the cache without scraping

SKIP_UNPLAYED = True               # never request fixtures that have not kicked off
MIN_HOURS_AFTER_KICKOFF = 3        # wait this long after kick-off before fetching
CLEAN_PLACEHOLDERS = True          # delete empty cache files left by earlier versions

BOX_X_MIN, BOX_Y_MIN, BOX_Y_MAX = 83.0, 21.1, 78.9
DIRECTION_THRESHOLD = 5.0

# Thirds of the pitch on Opta's 0-100 grid, always in the attacking direction.
FINAL_THIRD = 66.7
OWN_THIRD = 33.3
# Opta's progressive pass: at least 25% closer to the goal.
PROGRESS_SHARE = 0.25

TIME_BINS = [0, 15, 30, 45, 60, 75, 1000]
TIME_LABELS = ["0-15", "16-30", "31-45", "46-60", "61-75", "76+"]
GAME_STATE_LABELS = ["-2 or worse", "-1", "Level", "+1", "+2 or better"]
SHOT_TYPES = {"MissedShots", "ShotOnPost", "SavedShot", "Goal"}


# --------------------------------------------------------------------------
# SHARED HELPERS
# --------------------------------------------------------------------------
def flatten_display(value):
    if isinstance(value, dict):
        return value.get("displayName", value.get("value"))
    return value


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    renames = {
        "outcomeType": "outcome_type", "isTouch": "is_touch", "isShot": "is_shot",
        "isGoal": "is_goal", "endX": "end_x", "endY": "end_y",
        "expandedMinute": "expanded_minute", "eventId": "event_id", "teamId": "team_id",
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})

    for col in ("type", "outcome_type", "period"):
        if col in df.columns:
            df[col] = df[col].map(flatten_display)

    for col in ("is_touch", "is_shot", "is_goal"):
        df[col] = df[col].fillna(False).astype(bool) if col in df.columns else False

    for col in ("x", "y", "end_x", "end_y", "minute", "second"):
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.NA

    return df


def qualifier_names(qualifiers) -> set:
    """Qualifier display names for one event.

    Tolerant of how the column arrives: a list, a numpy array (what parquet
    gives back), a single dict, or JSON text.
    """
    names: set = set()
    if qualifiers is None:
        return names
    if isinstance(qualifiers, float):   # NaN
        return names

    if isinstance(qualifiers, str):
        try:
            qualifiers = json.loads(qualifiers)
        except (ValueError, TypeError):
            return {qualifiers}

    if isinstance(qualifiers, dict):
        qualifiers = [qualifiers]

    try:
        items = list(qualifiers)
    except TypeError:
        return names

    for q in items:
        if isinstance(q, str):
            names.add(q)
            continue
        if not hasattr(q, "get"):
            continue
        name = flatten_display(q.get("type", q))
        if name is None:
            name = q.get("displayName")
        if name is not None:
            names.add(str(name))
    return names


def has_q(qset: set, *wanted: str) -> bool:
    lowered = {q.lower() for q in qset}
    return any(w.lower() in lowered for w in wanted)


def add_time_block(df: pd.DataFrame) -> pd.DataFrame:
    df["time_block"] = pd.cut(df["minute"], bins=TIME_BINS, labels=TIME_LABELS,
                              include_lowest=True, right=True)
    first_half = df["period"].astype(str).str.lower().str.contains("first", na=False)
    df.loc[first_half & (df["minute"] > 45), "time_block"] = "31-45"
    return df


def sort_events(df: pd.DataFrame) -> pd.DataFrame:
    period_rank = {"FirstHalf": 1, "SecondHalf": 2, "FirstPeriodOfExtraTime": 3,
                   "SecondPeriodOfExtraTime": 4, "PenaltyShootout": 5}
    df = df.copy()
    df["_pr"] = df["period"].map(period_rank).fillna(9).astype(int)
    return df.sort_values(["_pr", "minute", "second"]).drop(columns="_pr")


def running_goal_diff(df: pd.DataFrame, ref_team: str) -> pd.Series:
    """Goal difference from ref_team's point of view, BEFORE each event."""
    gd, out = 0, []
    for _, row in df.iterrows():
        out.append(gd)
        if row["type"] == "Goal":
            own_goal = has_q(row["_qset"], "OwnGoal")
            scored_by_ref = (row["team"] == ref_team)
            credit_ref = scored_by_ref != own_goal   # XOR handles own goals
            gd += 1 if credit_ref else -1
    return pd.Series(out, index=df.index)


def player_minutes(ev: pd.DataFrame) -> pd.DataFrame:
    """Minutes played per player, from the substitution events.

    A player with no substitution event either side is treated as playing the
    whole match. Someone brought on has their minutes counted from that point,
    and someone taken off up to that point.
    """
    match_end = float(pd.to_numeric(ev["minute"], errors="coerce").max() or 90)
    on = (ev[ev["type"].eq("SubstitutionOn")]
          .groupby(["team", "player"])["minute"].min())
    off = (ev[ev["type"].eq("SubstitutionOff")]
           .groupby(["team", "player"])["minute"].min())

    played = (ev[ev["player"].notna()][["team", "player"]]
              .drop_duplicates().set_index(["team", "player"]))
    played["start"] = on.reindex(played.index).fillna(0.0)
    played["end"] = off.reindex(played.index).fillna(match_end)
    played["minutes"] = (played["end"] - played["start"]).clip(lower=0)
    return played.reset_index()[["team", "player", "minutes"]]


def build_flags(df: pd.DataFrame) -> pd.DataFrame:
    t, ok, q = df["type"], df["outcome_type"].eq("Successful"), df["_qset"]

    is_cross = q.apply(lambda s: has_q(s, "Cross"))
    is_corner = q.apply(lambda s: has_q(s, "CornerTaken"))
    is_longball = q.apply(lambda s: has_q(s, "Longball", "LongBall"))
    is_through = q.apply(lambda s: has_q(s, "ThroughBall"))
    is_blocked = q.apply(lambda s: has_q(s, "Blocked"))
    is_own_goal = q.apply(lambda s: has_q(s, "OwnGoal"))
    dead_ball = q.apply(lambda s: has_q(s, "ThrowIn", "GoalKick", "FreekickTaken"))

    # Any pass-type event, crosses included. Used for qualifier-based metrics
    # such as chances created, where the delivery type does not matter.
    pass_event = t.eq("Pass")
    is_throw_in = q.apply(lambda s: has_q(s, "ThrowIn"))
    is_keeper_throw = q.apply(lambda s: has_q(s, "KeeperThrow"))

    # Opta's pass: crosses, keeper throws and throw-ins do not count.
    is_pass = pass_event & ~is_cross & ~is_throw_in & ~is_keeper_throw
    dx = df["end_x"] - df["x"]

    # Thirds. Origin says where the play was, end says where it reached.
    from_final_third = df["x"].ge(FINAL_THIRD)
    into_final_third = df["end_x"].ge(FINAL_THIRD) & ~from_final_third
    outside_own_third = df["x"].ge(OWN_THIRD)

    # Progressive pass: completed, starting in the attacking two-thirds, and
    # closing at least a quarter of the remaining distance to goal.
    dist_before = (100 - df["x"]).clip(lower=0)
    dist_after = (100 - df["end_x"]).clip(lower=0)
    progressive = (is_pass & ok & outside_own_third
                   & (dist_after <= dist_before * (1 - PROGRESS_SHARE)))
    shot = t.isin(SHOT_TYPES)
    in_box = df["x"].ge(BOX_X_MIN) & df["y"].between(BOX_Y_MIN, BOX_Y_MAX)

    # A chance created is a pass that leads directly to a shot.
    is_key_pass = q.apply(lambda s: has_q(s, "KeyPass", "ShotAssist"))
    is_assist = q.apply(lambda s: has_q(s, "IntentionalAssist", "Assist"))

    take_on, aerial, tackle = t.eq("TakeOn"), t.eq("Aerial"), t.eq("Tackle")
    challenge, dispossessed = t.eq("Challenge"), t.eq("Dispossessed")
    smother = t.eq("Smother")
    foul_won = t.eq("Foul") & ok
    foul_conceded = t.eq("Foul") & ~ok
    attempted_tackle = q.apply(lambda s: has_q(s, "AttemptedTackle", "Attempted Tackle"))

    # Opta's PPDA counts fouls, tackles, interceptions, challenges and blocked
    # passes as defensive actions.
    press_action = (t.eq("Tackle") | t.eq("Interception") | t.eq("Challenge")
                    | t.eq("BlockedPass") | foul_conceded)
    bad_touch = t.eq("BallTouch") & ~ok

    # Opta's five duel pairs: aerial, take-on against challenge, tackle against
    # unsuccessful take-on or dispossessed, smother, and foul won against foul
    # conceded. A tackle wins its duel whatever the tackle's own outcome.
    ground_duel = (take_on | tackle | challenge | dispossessed | smother
                   | foul_won | foul_conceded)
    ground_duel_won = (take_on & ok) | tackle | smother | foul_won

    on_target = (t.eq("SavedShot") & ~is_blocked) | t.eq("Goal")
    scored = t.eq("Goal") & ~is_own_goal
    own_goal_event = t.eq("Goal") & is_own_goal
    in_box_mirrored = (df["x"].le(100 - BOX_X_MIN)
                       & df["y"].between(BOX_Y_MIN, BOX_Y_MAX))

    return pd.DataFrame({
        "touches": df["is_touch"].astype(int),
        "touches_opp_box": (df["is_touch"] & in_box).astype(int),

        "passes_attempted": is_pass.astype(int),
        "passes_completed": (is_pass & ok).astype(int),
        "open_play_passes": (is_pass & ~is_corner & ~dead_ball).astype(int),
        "passes_forward": (is_pass & (dx > DIRECTION_THRESHOLD)).astype(int),
        "passes_backward": (is_pass & (dx < -DIRECTION_THRESHOLD)).astype(int),
        "passes_sideways": (is_pass & dx.abs().le(DIRECTION_THRESHOLD)).astype(int),
        "passes_opp_half": (is_pass & ok & df["x"].ge(50)).astype(int),
        "passes_own_half": (is_pass & ok & df["x"].lt(50)).astype(int),

        # Open play crosses, which is what published cross counts show.
        "crosses_attempted": (pass_event & is_cross & ~is_corner & ~dead_ball).astype(int),
        "crosses_completed": (pass_event & is_cross & ~is_corner
                              & ~dead_ball & ok).astype(int),
        # Every cross including set pieces, so no delivery is lost when crosses
        # are taken out of the pass count.
        "crosses_all": (pass_event & is_cross).astype(int),
        "crosses_all_completed": (pass_event & is_cross & ok).astype(int),
        "long_passes_attempted": (is_pass & is_longball).astype(int),
        "long_passes_completed": (is_pass & is_longball & ok).astype(int),
        "through_balls": (is_pass & is_through).astype(int),
        "passes_final_third": (is_pass & from_final_third).astype(int),
        "passes_final_third_completed": (is_pass & from_final_third & ok).astype(int),
        "passes_into_final_third": (is_pass & into_final_third & ok).astype(int),
        "progressive_passes": progressive.astype(int),
        # One half of PPDA: passes this team was allowed outside their own third.
        "passes_outside_own_third": (is_pass & outside_own_third).astype(int),
        "corners": (pass_event & is_corner).astype(int),
        # Chances created counts the delivery whatever its type, so this uses
        # every pass event rather than Opta's narrower pass definition.
        "chances_created": (pass_event & (is_key_pass | is_assist)).astype(int),
        "assists": (pass_event & is_assist).astype(int),

        "shots": shot.astype(int),
        "shots_on_target": on_target.astype(int),
        "shots_off_target": t.isin({"MissedShots", "ShotOnPost"}).astype(int),
        "shots_blocked": (t.eq("SavedShot") & is_blocked).astype(int),
        "shots_in_box": (shot & in_box).astype(int),
        "shots_outside_box": (shot & ~in_box).astype(int),
        "shots_on_target_in_box": (on_target & in_box).astype(int),
        "shots_on_target_outside_box": (on_target & ~in_box).astype(int),
        "goals": scored.astype(int),
        "goals_in_box": (scored & in_box).astype(int),
        "goals_outside_box": (scored & ~in_box).astype(int),
        # Goals from this team's own shots, kept apart so conversion stays honest
        # once opponents' own goals are added to the goal totals.
        "goals_from_shots": scored.astype(int),
        # Own goals sit with the player who scored them. The box test is mirrored
        # because the event is recorded facing that team's attacking direction.
        "own_goals": own_goal_event.astype(int),
        "own_goals_in_box": (own_goal_event & in_box_mirrored).astype(int),
        "own_goals_outside_box": (own_goal_event & ~in_box_mirrored).astype(int),

        "take_ons": take_on.astype(int),
        "take_ons_won": (take_on & ok).astype(int),
        "aerials": aerial.astype(int),
        "aerials_won": (aerial & ok).astype(int),
        "ground_duels": ground_duel.astype(int),
        "ground_duels_won": ground_duel_won.astype(int),
        "duels_contested": (ground_duel | aerial).astype(int),
        "duels_won": (ground_duel_won | (aerial & ok)).astype(int),
        "smothers": smother.astype(int),

        "tackles": tackle.astype(int),
        "interceptions": t.eq("Interception").astype(int),
        "clearances": t.eq("Clearance").astype(int),
        "ball_recoveries": t.eq("BallRecovery").astype(int),
        "blocked_passes": t.eq("BlockedPass").astype(int),
        # Trying to win the ball back, as opposed to getting it clear.
        "pressing_actions": press_action.astype(int),
        "recovery_actions": (t.eq("Clearance") | t.eq("BallRecovery")).astype(int),
        # The other half of PPDA: pressing done outside our own defensive third.
        "pressing_actions_outside_own_third": (press_action
                                               & outside_own_third).astype(int),
        "missed_tackles": (challenge | (foul_conceded & attempted_tackle)).astype(int),

        "fouls_committed": (t.eq("Foul") & ~ok).astype(int),
        "fouls_won": (t.eq("Foul") & ok).astype(int),

        "possession_lost": ((is_pass & ~ok) | (take_on & ~ok)
                            | dispossessed | bad_touch).astype(int),
    }, index=df.index)


# --------------------------------------------------------------------------
# PHASE 1: DOWNLOAD, WITH RESUME AND A SINGLE FIXTURE-LIST FETCH
# --------------------------------------------------------------------------
def clean_placeholders() -> None:
    """Remove empty cache files written by earlier versions for unplayed games.

    Without this, a fixture cached as empty is treated as done forever and its
    real data is never fetched once the match is played.
    """
    removed = 0
    for f in CACHE_DIR.glob("*.parquet"):
        try:
            # Size, not row count. Reading with no columns returns zero rows on
            # some pandas versions, which made this delete every good file.
            if f.stat().st_size < 10_000:
                f.unlink()
                removed += 1
        except Exception:
            f.unlink()          # unreadable file, drop it and fetch again
            removed += 1
    if removed:
        print(f"Cleared {removed} empty or unreadable cache files.")


def download_all() -> pd.DataFrame:
    import soccerdata as sd

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if CLEAN_PLACEHOLDERS:
        clean_placeholders()

    print(f"Connecting to WhoScored: {LEAGUE} {SEASON}")
    ws = sd.WhoScored(leagues=LEAGUE, seasons=SEASON)

    print("Fetching the fixture list once. This takes a couple of minutes.")
    schedule = ws.read_schedule().reset_index()
    schedule = schedule.dropna(subset=["game_id"]).copy()
    schedule["game_id"] = schedule["game_id"].astype(int)

    # Never ask about fixtures that have not kicked off. This is what turns a
    # weekly refresh from two hours into a few minutes.
    if SKIP_UNPLAYED and "date" in schedule.columns:
        kickoff = pd.to_datetime(schedule["date"], errors="coerce", utc=True)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=MIN_HOURS_AFTER_KICKOFF)
        played = kickoff.notna() & (kickoff < cutoff)
        print(f"Fixtures in calendar: {len(schedule)}. "
              f"Played: {played.sum()}. Skipping {(~played).sum()} not yet played.")
        schedule = schedule[played]

    if MAX_MATCHES:
        schedule = schedule.head(MAX_MATCHES)
    schedule.to_csv(OUTPUT_DIR / "schedule.csv", index=False)

    failed_path = OUTPUT_DIR / "failed_matches.txt"
    previously_failed = set()
    if failed_path.exists():
        previously_failed = {int(x) for x in failed_path.read_text().split() if x.strip()}

    still_failed = []
    total = len(schedule)
    print(f"{total} matches in the fixture list.")

    for i, row in enumerate(schedule.itertuples(), start=1):
        gid = row.game_id
        out_file = CACHE_DIR / f"{gid}.parquet"

        if out_file.exists():
            continue
        if gid in previously_failed and not RETRY_FAILED:
            continue

        label = f"[{i}/{total}] {row.home_team} v {row.away_team}"
        started = time.time()
        try:
            # force_cache=True stops it re-scraping the whole calendar every match.
            ev = ws.read_events(match_id=[gid], force_cache=True).reset_index()
            if ev.empty:
                # No placeholder file, so the match is fetched properly once played.
                print(f"{label}  no events yet, not cached")
                time.sleep(PAUSE_SECONDS)
                continue
            ev["schedule_home"] = row.home_team
            ev["schedule_away"] = row.away_team
            ev["game_id"] = gid
            ev.to_parquet(out_file, index=False)
            print(f"{label}  ok ({len(ev):,} events, {time.time() - started:.0f}s)")
        except Exception as exc:
            still_failed.append(gid)
            print(f"{label}  FAILED: {type(exc).__name__}: {exc}")
        time.sleep(PAUSE_SECONDS)

    failed_path.write_text("\n".join(str(g) for g in still_failed))
    cached = len(list(CACHE_DIR.glob("*.parquet")))
    print(f"\nCached matches: {cached}/{total}. Failed this run: {len(still_failed)}.")
    if still_failed:
        print("Re-run the script to retry the failures. Cached matches are skipped.")
    return schedule


# --------------------------------------------------------------------------
# PHASE 2: PROCESS EACH MATCH FROM BOTH TEAMS' POINTS OF VIEW
# --------------------------------------------------------------------------
def summarise_match(path: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    ev = pd.read_parquet(path)
    if ev.empty:
        return None
    ev = normalise_columns(ev)
    ev["_qset"] = (ev["qualifiers"].apply(qualifier_names)
                   if "qualifiers" in ev.columns else [set()] * len(ev))
    ev = sort_events(ev)
    ev = add_time_block(ev)

    # Team names come from the events, so nothing can mismatch.
    teams = [t for t in pd.unique(ev["team"].dropna())]
    if len(teams) != 2:
        raise ValueError(f"{path.name}: expected 2 teams, found {teams}")

    ref = teams[0]
    ref_gd = running_goal_diff(ev, ref)
    flags = build_flags(ev)

    parts = []
    for team in teams:
        gd = ref_gd if team == ref else -ref_gd
        state = pd.cut(gd, bins=[-99, -2, -1, 0, 1, 99],
                       labels=GAME_STATE_LABELS, right=True)
        side = ev["team"].eq(team).map({True: "for", False: "against"})

        splits = {
            "time_block": ev["time_block"],
            "game_state": state,
            "total": pd.Series(["Season"] * len(ev), index=ev.index),
        }
        for split_type, series in splits.items():
            g = flags.groupby([side, series], observed=False).sum().reset_index()
            g.columns = ["side", "split_value"] + list(flags.columns)
            g.insert(0, "team", team)
            g.insert(1, "split_type", split_type)
            g.insert(2, "match_id", ev["game_id"].iloc[0])
            parts.append(g)

    # Match result per team, for clean sheets and points.
    match_rows = []
    for team in teams:
        is_team = ev["team"].eq(team)
        scored = int(flags.loc[is_team, "goals"].sum()
                     + flags.loc[~is_team, "own_goals"].sum())
        conceded = int(flags.loc[~is_team, "goals"].sum()
                       + flags.loc[is_team, "own_goals"].sum())
        match_rows.append({
            "team": team,
            "match_id": ev["game_id"].iloc[0],
            "scored": scored,
            "conceded": conceded,
            "clean_sheets": int(conceded == 0),
            "wins": int(scored > conceded),
            "draws": int(scored == conceded),
            "losses": int(scored < conceded),
            "points": 3 if scored > conceded else (1 if scored == conceded else 0),
        })

    # Per-player totals. Only actions the player performed, so no "against" side.
    players = pd.DataFrame()
    if "player" in ev.columns:
        has_player = ev["player"].notna()
        if has_player.any():
            keys = ev.loc[has_player, ["team", "player"]]
            players = (flags.loc[has_player]
                       .groupby([keys["team"], keys["player"]], observed=True)
                       .sum().reset_index())
            players.columns = ["team", "player"] + list(flags.columns)
            mins = player_minutes(ev)
            players = players.merge(mins, on=["team", "player"], how="left")
            players["appearances"] = 1
            players.insert(0, "match_id", ev["game_id"].iloc[0])

    return (pd.concat(parts, ignore_index=True), pd.DataFrame(match_rows), players)


GOAL_PAIRS = [("goals", "own_goals"),
              ("goals_in_box", "own_goals_in_box"),
              ("goals_outside_box", "own_goals_outside_box")]


def credit_own_goals(long_df: pd.DataFrame) -> pd.DataFrame:
    """Move own goals to the team that benefits.

    Done once here, on the long frame, so every file written downstream carries
    the corrected numbers: the season tables, the splits and the per-match file.
    """
    key = ["team", "match_id", "split_type", "split_value"]
    own_cols = [own for _, own in GOAL_PAIRS if own in long_df.columns]
    if not own_cols or "side" not in long_df.columns:
        return long_df

    lookup = long_df.pivot_table(index=key, columns="side", values=own_cols,
                                 aggfunc="sum", observed=False).fillna(0)
    lookup.columns = [f"{a}__{b}" for a, b in lookup.columns]
    out = long_df.merge(lookup.reset_index(), on=key, how="left")

    for goal_col, own_col in GOAL_PAIRS:
        if goal_col not in out.columns:
            continue
        for side, other in (("for", "against"), ("against", "for")):
            src = f"{own_col}__{other}"
            if src not in out.columns:
                continue
            mask = out["side"].eq(side)
            out.loc[mask, goal_col] = (out.loc[mask, goal_col]
                                       + out.loc[mask, src].fillna(0))
    return out.drop(columns=[c for c in out.columns if "__" in c])


def process_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = sorted(CACHE_DIR.glob("*.parquet"))
    print(f"\nProcessing {len(files)} cached matches.")
    rows, results, players = [], [], []
    for i, f in enumerate(files, start=1):
        try:
            result = summarise_match(f)
            if result is not None:
                rows.append(result[0])
                results.append(result[1])
                if len(result) > 2 and not result[2].empty:
                    players.append(result[2])
        except Exception:
            print(f"  problem processing {f.name}:")
            traceback.print_exc(limit=1)
        if i % 25 == 0:
            print(f"  {i}/{len(files)} done")
    if not rows:
        raise SystemExit("No matches could be processed.")
    player_df = (pd.concat(players, ignore_index=True) if players
                 else pd.DataFrame(columns=["team", "player"]))
    return (pd.concat(rows, ignore_index=True),
            pd.concat(results, ignore_index=True), player_df)


# --------------------------------------------------------------------------
# PHASE 3: AGGREGATE, DERIVE RATES, RANK
# --------------------------------------------------------------------------
def add_rates(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    def pct(a, b):
        num, den = df[f"{a}{suffix}"], df[f"{b}{suffix}"]
        return (num / den * 100).round(2).where(den > 0)

    df[f"pass_accuracy_pct{suffix}"] = pct("passes_completed", "passes_attempted")
    df[f"cross_accuracy_pct{suffix}"] = pct("crosses_completed", "crosses_attempted")
    df[f"long_pass_accuracy_pct{suffix}"] = pct("long_passes_completed",
                                                "long_passes_attempted")
    # Opta excludes blocked shots from shooting accuracy, since a block says
    # nothing about whether the shot was on target. Conversion keeps them.
    blocked = f"shots_blocked{suffix}"
    shots = f"shots{suffix}"
    if blocked in df.columns and shots in df.columns:
        unblocked = (df[shots] - df[blocked]).replace(0, float("nan"))
        df[f"shooting_accuracy_pct{suffix}"] = (
            df[f"shots_on_target{suffix}"] / unblocked * 100).round(2)
    else:
        df[f"shooting_accuracy_pct{suffix}"] = pct("shots_on_target", "shots")
    df[f"cross_accuracy_all_pct{suffix}"] = pct("crosses_all_completed", "crosses_all")
    df[f"final_third_pass_accuracy_pct{suffix}"] = pct("passes_final_third_completed",
                                                       "passes_final_third")
    df[f"conversion_pct{suffix}"] = pct("goals_from_shots", "shots")
    df[f"take_on_success_pct{suffix}"] = pct("take_ons_won", "take_ons")
    df[f"aerial_success_pct{suffix}"] = pct("aerials_won", "aerials")
    df[f"ground_duel_success_pct{suffix}"] = pct("ground_duels_won", "ground_duels")
    df[f"duel_success_pct{suffix}"] = pct("duels_won", "duels_contested")
    return df


def build_table(long_df: pd.DataFrame, split_type: str,
                matches_played: pd.Series) -> pd.DataFrame:
    sub = long_df[long_df["split_type"] == split_type]
    metric_cols = [c for c in sub.columns
                   if c not in {"team", "split_type", "match_id", "side", "split_value"}]

    agg = sub.groupby(["team", "split_value", "side"], observed=False)[metric_cols].sum()
    wide = agg.unstack("side")
    wide.columns = [f"{m}_{s}" for m, s in wide.columns]
    wide = wide.fillna(0).reset_index()

    wide = add_rates(wide, "_for")
    wide = add_rates(wide, "_against")

    touches_total = wide["touches_for"] + wide["touches_against"]
    wide.insert(2, "possession_pct",
                (wide["touches_for"] / touches_total * 100).round(2)
                .where(touches_total > 0))
    wide.insert(2, "matches_played", wide["team"].map(matches_played))

    numeric = [c for c in wide.columns
               if c not in {"team", "split_value", "matches_played"}
               and pd.api.types.is_numeric_dtype(wide[c])]
    ranks = (wide.groupby("split_value", observed=False)[numeric]
             .rank(ascending=False, method="min"))
    ranks.columns = [f"rank_{c}" for c in ranks.columns]

    out = pd.concat([wide, ranks], axis=1)
    return out.sort_values(["split_value", "team"]).reset_index(drop=True)


def find_team(names, wanted: str):
    """Loose match, so 'Manchester United' finds 'Man Utd'."""
    lowered = {str(n).lower(): n for n in names}
    for low, original in lowered.items():
        if wanted.lower() in low or low in wanted.lower():
            return original
    close = difflib.get_close_matches(wanted.lower(), list(lowered), n=1, cutoff=0.4)
    return lowered[close[0]] if close else None


def main() -> None:
    if DOWNLOAD:
        download_all()

    long_df, match_results, player_rows = process_all()
    long_df = credit_own_goals(long_df)

    matches_played = (long_df[long_df["split_type"] == "total"]
                      .groupby("team")["match_id"].nunique())

    totals = build_table(long_df, "total", matches_played)

    # Results context, which only makes sense at season level.
    record = (match_results.groupby("team")[
        ["clean_sheets", "wins", "draws", "losses", "points"]].sum().reset_index())
    totals = totals.merge(record, on="team", how="left")
    for c in ["clean_sheets", "wins", "draws", "losses", "points"]:
        totals[f"rank_{c}"] = totals[c].rank(ascending=False, method="min")
    by_time = build_table(long_df, "time_block", matches_played)
    by_state = build_table(long_df, "game_state", matches_played)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    totals.to_csv(OUTPUT_DIR / "league_totals.csv", index=False)
    by_time.to_csv(OUTPUT_DIR / "league_by_time_block.csv", index=False)
    by_state.to_csv(OUTPUT_DIR / "league_by_game_state.csv", index=False)
    long_df.to_csv(OUTPUT_DIR / "match_level_long.csv", index=False)

    if not player_rows.empty:
        metric_cols = [c for c in player_rows.columns
                       if c not in {"match_id", "team", "player"}]
        players = (player_rows.groupby(["team", "player"], observed=True)[metric_cols]
                   .sum().reset_index())
        players = add_rates(players.rename(columns={c: f"{c}_for" for c in metric_cols
                                                    if c not in {"minutes", "appearances"}}),
                            "_for")
        players = players.rename(columns={"minutes_for": "minutes",
                                          "appearances_for": "appearances"})
        players["minutes"] = players.get("minutes", 0)
        players.to_csv(OUTPUT_DIR / "league_players.csv", index=False)
        print(f"Player rows written: {len(players)}")

    # Crosses, corners and blocked shots all depend on qualifiers. If the whole
    # league shows none of them, the qualifier reader failed rather than the data.
    for check in ("crosses_attempted_for", "corners_for", "shots_blocked_for"):
        if check in totals.columns and totals[check].sum() == 0:
            print(f"WARNING: {check} is zero across every team. "
                  "The qualifier tags are not being read.")

    names = sorted(totals["team"].unique())
    print(f"\nTeams found in the event data ({len(names)}):")
    print(", ".join(str(n) for n in names))

    team = find_team(names, FOCUS_TEAM)
    if team:
        cols = ["matches_played", "shots_for", "rank_shots_for",
                "shots_on_target_for", "shots_blocked_for", "goals_for",
                "conversion_pct_for", "possession_pct",
                "pass_accuracy_pct_for", "crosses_attempted_for",
                "shots_against", "goals_against"]
        row = totals[totals["team"] == team]
        print(f"\nTotals for {team}:")
        print(row[[c for c in cols if c in row.columns]].T.to_string(header=False))
    else:
        print(f"\nCould not match '{FOCUS_TEAM}' to any team above.")

    print(f"\nFiles written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
