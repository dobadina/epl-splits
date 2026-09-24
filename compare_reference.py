"""
Compare our season totals against Opta's published figures.

Reads a CSV saved by fetch_reference.py, lines it up with our league_totals.csv
by club, and reports the gap on every metric we can map. Anything more than a
couple of per cent out is a definition problem worth chasing.

Set REFERENCE_FILE and fill in COLUMN_MAP once you have seen the headings the
scraper found.

Run it with:  python compare_reference.py
"""

from __future__ import annotations

import difflib
from pathlib import Path

import pandas as pd

OUTPUT = Path("output")
REFERENCE = Path("reference")
REFERENCE_FILE = "table_0.csv"     # whichever table held the club names

# Their column heading -> our column in league_totals.csv.
# Fill this in once the scraper has shown you the real headings.
COLUMN_MAP: dict[str, str] = {
    # "Goals": "goals_for",
    # "Shots": "shots_for",
    # "Shots on target": "shots_on_target_for",
    # "Passes": "passes_attempted_for",
    # "Crosses": "crosses_attempted_for",
    # "Corners": "corners_for",
    # "Duels won": "duels_won_for",
    # "Aerial duels won": "aerials_won_for",
    # "Tackles": "tackles_for",
    # "Interceptions": "interceptions_for",
    # "Clearances": "clearances_for",
    # "Fouls": "fouls_committed_for",
    # "Chances created": "chances_created_for",
}

# How far apart before it is worth investigating.
TOLERANCE_PCT = 2.0


def find_team_column(df: pd.DataFrame) -> str:
    """The column holding club names."""
    clubs = {"arsenal", "liverpool", "chelsea", "everton", "tottenham"}
    best, best_hits = df.columns[0], -1
    for col in df.columns:
        values = {str(v).strip().lower() for v in df[col]}
        hits = len(values & clubs)
        if hits > best_hits:
            best, best_hits = col, hits
    return best


def match_teams(ours: list[str], theirs: list[str]) -> dict[str, str]:
    """Line up 'Man Utd' with 'Manchester United' and so on."""
    pairs = {}
    for name in theirs:
        hit = difflib.get_close_matches(str(name).lower(),
                                        [o.lower() for o in ours], n=1, cutoff=0.4)
        if hit:
            pairs[name] = next(o for o in ours if o.lower() == hit[0])
        else:
            # Fall back to the longest shared word, which catches Man Utd.
            words = [w for w in str(name).lower().split() if len(w) > 3]
            for o in ours:
                if any(w[:4] in o.lower() for w in words):
                    pairs[name] = o
                    break
    return pairs


def to_number(v) -> float:
    try:
        return float(str(v).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    ours_path = OUTPUT / "league_totals.csv"
    ref_path = REFERENCE / REFERENCE_FILE
    if not ours_path.exists():
        raise SystemExit(f"Missing {ours_path}. Run the pipeline first.")
    if not ref_path.exists():
        raise SystemExit(f"Missing {ref_path}. Run fetch_reference.py first.")
    if not COLUMN_MAP:
        raise SystemExit("COLUMN_MAP is empty. Fill it in with the headings the "
                         "scraper found, then run this again.")

    ours = pd.read_csv(ours_path)
    theirs = pd.read_csv(ref_path)

    team_col = find_team_column(theirs)
    pairs = match_teams(list(ours["team"]), list(theirs[team_col]))
    print(f"Matched {len(pairs)} of {len(theirs)} clubs by name.\n")
    unmatched = [t for t in theirs[team_col] if t not in pairs]
    if unmatched:
        print("Could not match:", ", ".join(str(u) for u in unmatched), "\n")

    rows = []
    for their_col, our_col in COLUMN_MAP.items():
        if their_col not in theirs.columns:
            print(f"skipping {their_col}: not in their table")
            continue
        if our_col not in ours.columns:
            print(f"skipping {their_col}: {our_col} not in ours")
            continue
        for _, r in theirs.iterrows():
            team = pairs.get(r[team_col])
            if team is None:
                continue
            mine = ours.loc[ours["team"] == team, our_col]
            if mine.empty:
                continue
            a, b = to_number(r[their_col]), float(mine.iloc[0])
            if pd.isna(a) or a == 0:
                continue
            rows.append({"metric": their_col, "team": team,
                         "theirs": a, "ours": b,
                         "gap_pct": round((b - a) / a * 100, 1)})

    if not rows:
        raise SystemExit("Nothing could be compared. Check COLUMN_MAP.")

    df = pd.DataFrame(rows)
    summary = (df.groupby("metric")
               .agg(teams=("team", "count"),
                    mean_gap_pct=("gap_pct", lambda s: round(s.mean(), 2)),
                    worst_gap_pct=("gap_pct", lambda s: round(s.abs().max(), 2)))
               .reset_index()
               .sort_values("worst_gap_pct"))

    print("How our numbers compare with theirs, by metric:\n")
    print(summary.to_string(index=False))

    bad = df[df["gap_pct"].abs() > TOLERANCE_PCT]
    print(f"\n{len(df) - len(bad)} of {len(df)} comparisons within {TOLERANCE_PCT}%.")
    if len(bad):
        print("\nWorth investigating, largest first:\n")
        print(bad.reindex(bad["gap_pct"].abs().sort_values(ascending=False).index)
              .head(20).to_string(index=False))
    else:
        print("Every metric agrees with the published figures.")

    out = REFERENCE / "comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nFull comparison written to {out}")


if __name__ == "__main__":
    main()
