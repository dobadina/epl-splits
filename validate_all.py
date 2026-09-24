"""
Validate the numbers, not just the scores.

Four layers:

1. Invariants        Figures that must agree with each other, whatever the data
                     says. Shots equal on target plus off target plus blocked,
                     and so on. These catch aggregation bugs.
2. Consistency       Time blocks must sum to the season total. So must game
                     states. What one team records "for" must equal what their
                     opponent records "against".
3. Independent count Key metrics recounted straight from the cached events with
                     different code from the pipeline, then compared.
4. Scorelines        Computed goals against the official scores, with own goals
                     flagged, since those are a known trap.

Layer 5 cannot be automated: whether the definitions match the published ones.
The script prints one match's numbers at the end so you can compare them against
that match's page on the source site by eye.

Run it with:  python validate_all.py
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT = Path("output")
CACHE = Path("match_cache")
SAMPLE_MATCHES = 8          # how many matches to recount independently
TOLERANCE = 1e-6

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"  {detail}" if detail and not ok else ""))


# --------------------------------------------------------------------------
# DATA
# --------------------------------------------------------------------------
def load_long() -> pd.DataFrame:
    path = OUTPUT / "match_level_long.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run the pipeline first.")
    return pd.read_csv(path)


def wide_totals(long_df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match, with _for and _against columns."""
    sub = long_df[long_df["split_type"] == "total"]
    metrics = [c for c in sub.columns if c not in
               {"team", "split_type", "match_id", "side", "split_value"}]
    wide = sub.pivot_table(index=["team", "match_id"], columns="side",
                           values=metrics, aggfunc="sum")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    return wide.reset_index()


def display_name(v):
    return v.get("displayName") if isinstance(v, dict) else v


def qualifier_set(q) -> set:
    out = set()
    if q is None or isinstance(q, float):
        return out
    try:
        items = list(q)
    except TypeError:
        return out
    for item in items:
        if hasattr(item, "get"):
            label = display_name(item.get("type", item)) or item.get("displayName")
            if label is not None:
                out.add(str(label))
        elif isinstance(item, str):
            out.add(item)
    return out


# --------------------------------------------------------------------------
# LAYER 1: INVARIANTS
# --------------------------------------------------------------------------
def check_invariants(w: pd.DataFrame) -> None:
    print("\n1. INVARIANTS: figures that must agree with each other")

    def has(*cols) -> bool:
        return all(c in w.columns for c in cols)

    def equal(name, left_cols, right_cols):
        if not has(*left_cols, *right_cols):
            record(name, True, "columns absent, skipped")
            return
        left = w[list(left_cols)].sum(axis=1)
        right = w[list(right_cols)].sum(axis=1)
        bad = w[(left - right).abs() > TOLERANCE]
        detail = ""
        if len(bad):
            r = bad.iloc[0]
            detail = (f"{len(bad)} row(s), e.g. {r['team']} match {int(r['match_id'])}: "
                      f"{left.loc[bad.index[0]]:g} vs {right.loc[bad.index[0]]:g}")
        record(name, len(bad) == 0, detail)

    def at_most(name, small, big):
        if not has(small, big):
            record(name, True, "columns absent, skipped")
            return
        bad = w[w[small] > w[big] + TOLERANCE]
        detail = ""
        if len(bad):
            r = bad.iloc[0]
            detail = (f"{len(bad)} row(s), e.g. {r['team']} match {int(r['match_id'])}: "
                      f"{r[small]:g} > {r[big]:g}")
        record(name, len(bad) == 0, detail)

    for side in ("for", "against"):
        equal(f"Shots = on target + off target + blocked ({side})",
              [f"shots_{side}"],
              [f"shots_on_target_{side}", f"shots_off_target_{side}",
               f"shots_blocked_{side}"])
        equal(f"Shots = in box + outside box ({side})",
              [f"shots_{side}"],
              [f"shots_in_box_{side}", f"shots_outside_box_{side}"])
        equal(f"On target = in box + outside box ({side})",
              [f"shots_on_target_{side}"],
              [f"shots_on_target_in_box_{side}", f"shots_on_target_outside_box_{side}"])
        equal(f"Goals = in box + outside box ({side})",
              [f"goals_{side}"],
              [f"goals_in_box_{side}", f"goals_outside_box_{side}"])
        equal(f"Duels contested = aerial + ground ({side})",
              [f"duels_contested_{side}"],
              [f"aerials_{side}", f"ground_duels_{side}"])
        equal(f"Duels won = aerials won + ground won ({side})",
              [f"duels_won_{side}"],
              [f"aerials_won_{side}", f"ground_duels_won_{side}"])
        # Goals from the team's own shots cannot exceed its shots on target.
        # Total goals can, because an opponent's own goal counts for you.
        shot_goals = (f"goals_from_shots_{side}"
                      if f"goals_from_shots_{side}" in w.columns else f"goals_{side}")
        at_most(f"Goals from shots never exceed shots on target ({side})",
                shot_goals, f"shots_on_target_{side}")
        if f"shots_blocked_{side}" in w.columns:
            at_most(f"Blocked shots never exceed shots ({side})",
                    f"shots_blocked_{side}", f"shots_{side}")
        if f"goals_from_shots_{side}" in w.columns:
            at_most(f"Goals from shots never exceed total goals ({side})",
                    f"goals_from_shots_{side}", f"goals_{side}")
        at_most(f"Passes completed never exceed attempted ({side})",
                f"passes_completed_{side}", f"passes_attempted_{side}")
        at_most(f"Crosses completed never exceed attempted ({side})",
                f"crosses_completed_{side}", f"crosses_attempted_{side}")
        at_most(f"Long passes completed never exceed attempted ({side})",
                f"long_passes_completed_{side}", f"long_passes_attempted_{side}")
        at_most(f"Take-ons won never exceed attempted ({side})",
                f"take_ons_won_{side}", f"take_ons_{side}")
        at_most(f"Aerials won never exceed contested ({side})",
                f"aerials_won_{side}", f"aerials_{side}")
        at_most(f"Shots in box never exceed shots ({side})",
                f"shots_in_box_{side}", f"shots_{side}")
        at_most(f"Duels won never exceed duels contested ({side})",
                f"duels_won_{side}", f"duels_contested_{side}")
        at_most(f"Chances created never exceed passes ({side})",
                f"chances_created_{side}", f"passes_attempted_{side}")


# --------------------------------------------------------------------------
# LAYER 2: CONSISTENCY
# --------------------------------------------------------------------------
def check_duel_balance(w: pd.DataFrame) -> None:
    """Every duel has a winner and a loser, so across the league the duels won
    must come to about half the duels contested. A long way off means the two
    sides of some pairing are not both being counted."""
    if "duels_won_for" not in w.columns or "duels_contested_for" not in w.columns:
        record("League-wide duels won is about half of duels contested", True,
               "columns absent, skipped")
        return
    won = w["duels_won_for"].sum()
    contested = w["duels_contested_for"].sum()
    share = won / contested if contested else 0
    ok = 0.45 <= share <= 0.55
    record("League-wide duels won is about half of duels contested", ok,
           f"{share:.1%} of {int(contested)} contested")


def check_consistency(long_df: pd.DataFrame) -> None:
    print("\n2. CONSISTENCY: splits, perspectives and players")

    metrics = [c for c in long_df.columns if c not in
               {"team", "split_type", "match_id", "side", "split_value"}]

    for split in ("time_block", "game_state"):
        piece = long_df[long_df["split_type"] == split]
        total = long_df[long_df["split_type"] == "total"]
        if piece.empty:
            record(f"{split} sums to the season total", True, "split absent, skipped")
            continue
        a = piece.groupby(["team", "match_id", "side"])[metrics].sum()
        b = total.groupby(["team", "match_id", "side"])[metrics].sum()
        joined = a.join(b, lsuffix="_split", rsuffix="_total", how="inner")
        worst_metric, worst_gap = "", 0.0
        for m in metrics:
            gap = (joined[f"{m}_split"] - joined[f"{m}_total"]).abs().max()
            if gap > worst_gap:
                worst_metric, worst_gap = m, gap
        record(f"{split} sums to the season total", worst_gap <= TOLERANCE,
               f"largest gap {worst_gap:g} on {worst_metric}")

    total = long_df[long_df["split_type"] == "total"]
    mine = total[total["side"] == "for"].set_index(["match_id", "team"])[metrics]
    theirs = total[total["side"] == "against"].set_index(["match_id", "team"])[metrics]
    pairs = total[["match_id", "team"]].drop_duplicates()
    opp = pairs.merge(pairs, on="match_id")
    opp = opp[opp["team_x"] != opp["team_y"]]

    gaps = []
    for r in opp.itertuples():
        key_me = (r.match_id, r.team_x)
        key_them = (r.match_id, r.team_y)
        if key_me in mine.index and key_them in theirs.index:
            gaps.append((mine.loc[key_me] - theirs.loc[key_them]).abs().max())
    worst = max(gaps) if gaps else 0.0
    record("What one team does 'for' equals what the other records 'against'",
           worst <= TOLERANCE, f"largest gap {worst:g}")

    ppath = OUTPUT / "league_players.csv"
    if not ppath.exists():
        record("Player rows account for the team's actions", True, "no player file")
        return
    players = pd.read_csv(ppath)
    team_totals = (total[total["side"] == "for"].groupby("team")[metrics].sum())
    worst_name, worst_share = "", 1.0
    for m in ["shots", "passes_attempted", "chances_created", "crosses_attempted"]:
        pcol = f"{m}_for"
        if pcol not in players.columns or m not in team_totals.columns:
            continue
        got = players.groupby("team")[pcol].sum()
        want = team_totals[m]
        share = (got / want.replace(0, np.nan)).min()
        if pd.notna(share) and share < worst_share:
            worst_name, worst_share = m, share
    ok = worst_share >= 0.95
    record("Player rows account for at least 95% of team actions", ok,
           f"lowest {worst_share:.1%} on {worst_name}")


# --------------------------------------------------------------------------
# LAYER 3: INDEPENDENT RECOUNT
# --------------------------------------------------------------------------
def recount(path: Path) -> pd.DataFrame:
    """Count a few metrics from the raw events, deliberately not reusing the
    pipeline's code."""
    ev = pd.read_parquet(path)
    if ev.empty:
        return pd.DataFrame()
    types = ev["type"].map(display_name)
    outcomes = (ev["outcome_type"].map(display_name) if "outcome_type" in ev.columns
                else pd.Series("", index=ev.index))
    quals = ev["qualifiers"].apply(qualifier_set) if "qualifiers" in ev.columns else \
        pd.Series([set()] * len(ev))

    rows = []
    for team in ev["team"].dropna().unique():
        m = ev["team"] == team
        shot_types = {"MissedShots", "ShotOnPost", "SavedShot", "Goal"}
        blocked = quals.apply(lambda s: "Blocked" in s)
        own = quals.apply(lambda s: "OwnGoal" in s)
        not_a_pass = quals.apply(
            lambda s: bool({"Cross", "ThrowIn", "KeeperThrow"} & s))
        rows.append({
            "match_id": int(ev["game_id"].iloc[0]),
            "team": team,
            # Own goals are not shots, which is how the pipeline counts them.
            "shots": int((m & types.isin(shot_types) & ~own).sum()),
            "goals_from_shots": int((m & types.eq("Goal") & ~own).sum()),
            # Opta's pass: crosses, throw-ins and keeper throws do not count.
            "passes_attempted": int((m & types.eq("Pass") & ~not_a_pass).sum()),
            "passes_completed": int((m & types.eq("Pass") & ~not_a_pass &
                                     outcomes.eq("Successful")).sum()),
            "shots_blocked": int((m & types.eq("SavedShot") & blocked).sum()),
            "aerials": int((m & types.eq("Aerial")).sum()),
            "corners": int((m & types.eq("Pass") &
                            quals.apply(lambda s: "CornerTaken" in s)).sum()),
        })
    return pd.DataFrame(rows)


def check_recount(long_df: pd.DataFrame) -> None:
    print(f"\n3. INDEPENDENT RECOUNT: {SAMPLE_MATCHES} matches counted again from source")

    files = sorted(f for f in CACHE.glob("*.parquet") if f.stat().st_size > 10_000)
    if not files:
        record("Recount from cached events", True, "no cache found, skipped")
        return
    random.seed(0)
    picked = random.sample(files, min(SAMPLE_MATCHES, len(files)))

    theirs = pd.concat([recount(f) for f in picked], ignore_index=True)
    total = long_df[(long_df["split_type"] == "total") & (long_df["side"] == "for")]
    mine = total.groupby(["match_id", "team"]).sum(numeric_only=True).reset_index()

    joined = theirs.merge(mine, on=["match_id", "team"], suffixes=("_recount", "_pipeline"))
    if joined.empty:
        record("Recount matches the pipeline", False,
               "could not line up the recount with the pipeline output")
        return

    for metric in ["shots", "goals_from_shots", "passes_attempted", "passes_completed",
                   "shots_blocked", "aerials", "corners"]:
        a, b = f"{metric}_recount", f"{metric}_pipeline"
        if a not in joined.columns or b not in joined.columns:
            continue
        gap = (joined[a] - joined[b]).abs()
        bad = joined[gap > TOLERANCE]
        detail = ""
        if len(bad):
            r = bad.iloc[0]
            detail = (f"{len(bad)} of {len(joined)} differ, e.g. {r['team']} "
                      f"match {int(r['match_id'])}: recount {r[a]:g}, "
                      f"pipeline {r[b]:g}")
        record(f"Recount agrees on {metric}", len(bad) == 0, detail)


# --------------------------------------------------------------------------
# LAYER 4: SCORELINES
# --------------------------------------------------------------------------
def official_scores() -> pd.DataFrame | None:
    path = OUTPUT / "schedule.csv"
    if not path.exists():
        return None
    s = pd.read_csv(path)
    home = next((c for c in s.columns if re.fullmatch(r"home_?score", c, re.I)), None)
    away = next((c for c in s.columns if re.fullmatch(r"away_?score", c, re.I)), None)
    if home and away:
        out = s[["game_id", home, away]].copy()
        out.columns = ["match_id", "home_goals", "away_goals"]
    else:
        col = next((c for c in s.columns if "score" in c.lower()), None)
        if not col:
            return None
        parsed = s[col].astype(str).str.extract(r"(\d+)\s*[:\-]\s*(\d+)")
        out = pd.DataFrame({"match_id": s["game_id"],
                            "home_goals": pd.to_numeric(parsed[0], errors="coerce"),
                            "away_goals": pd.to_numeric(parsed[1], errors="coerce")})
    return out.dropna(subset=["home_goals", "away_goals"])


def check_scores(long_df: pd.DataFrame) -> None:
    print("\n4. SCORELINES: computed goals against the official result")

    total = long_df[(long_df["split_type"] == "total") & (long_df["side"] == "for")]
    comp = (total.groupby(["match_id", "team"])["goals"].sum().reset_index())
    per_match = comp.groupby("match_id")["goals"].agg(list).reset_index()

    own = []
    for f in sorted(CACHE.glob("*.parquet")):
        if f.stat().st_size < 10_000:
            continue
        try:
            ev = pd.read_parquet(f)
        except Exception:
            continue
        types = ev["type"].map(display_name)
        goals = ev[types == "Goal"]
        if goals.empty:
            continue
        count = sum(1 for q in goals.get("qualifiers", [])
                    if "OwnGoal" in qualifier_set(q))
        own.append({"match_id": int(ev["game_id"].iloc[0]), "own_goals": count})
    own_df = pd.DataFrame(own)
    if not own_df.empty:
        print(f"  Own goals in the cache: {int(own_df['own_goals'].sum())} "
              f"across {int((own_df['own_goals'] > 0).sum())} match(es)")
        per_match = per_match.merge(own_df, on="match_id", how="left")

    off = official_scores()
    if off is None:
        record("Scorelines match the official result", True,
               "no scores in the fixture list, skipped")
        return

    merged = per_match.merge(off, on="match_id", how="inner")
    merged["official"] = merged.apply(
        lambda r: sorted([int(r["home_goals"]), int(r["away_goals"])]), axis=1)
    merged["computed"] = merged["goals"].apply(sorted)
    bad = merged[merged.apply(lambda r: r["computed"] != r["official"], axis=1)]

    record(f"Scorelines match the official result ({len(merged)} checked)",
           len(bad) == 0, f"{len(bad)} wrong")
    if len(bad):
        for r in bad.head(10).itertuples():
            og = f", own goals {int(r.own_goals)}" if hasattr(r, "own_goals") \
                and pd.notna(r.own_goals) else ""
            print(f"      match {r.match_id}: counted {r.computed}, "
                  f"official {r.official}{og}")
        if hasattr(bad, "own_goals"):
            explained = int((bad["own_goals"].fillna(0) > 0).sum())
            print(f"      {explained} of {len(bad)} involve an own goal")


# --------------------------------------------------------------------------
# LAYER 5: FOR THE HUMAN
# --------------------------------------------------------------------------
def print_spot_check(long_df: pd.DataFrame) -> None:
    print("\n5. BY EYE: definitions can only be checked against the source page")
    total = long_df[(long_df["split_type"] == "total") & (long_df["side"] == "for")]
    if total.empty:
        return
    match_id = total["match_id"].iloc[0]
    one = total[total["match_id"] == match_id]

    show = ["goals", "shots", "shots_on_target", "shots_blocked", "corners",
            "passes_attempted", "passes_completed", "crosses_attempted",
            "crosses_completed", "aerials", "aerials_won", "tackles",
            "interceptions", "clearances", "fouls_committed", "chances_created"]
    cols = [c for c in show if c in one.columns]
    table = one.set_index("team")[cols].T

    print(f"\n  Match {match_id}, as this pipeline counts it. Open the same match on")
    print("  the source site and compare. Small gaps mean a definition differs,")
    print("  large gaps mean something is wrong.\n")
    print(table.to_string())


# --------------------------------------------------------------------------
def main() -> None:
    long_df = load_long()
    w = wide_totals(long_df)
    print(f"Validating {w['match_id'].nunique()} matches, "
          f"{w['team'].nunique()} teams, {len(w)} team-match rows.")

    check_invariants(w)
    check_duel_balance(w)
    check_consistency(long_df)
    check_recount(long_df)
    check_scores(long_df)

    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed
    print(f"\n{'=' * 60}")
    print(f"{passed} checks passed, {failed} failed.")
    if failed:
        print("\nFailed checks:")
        for name, ok, detail in results:
            if not ok:
                print(f"  {name}: {detail}")
    print(f"{'=' * 60}")

    print_spot_check(long_df)


if __name__ == "__main__":
    main()
