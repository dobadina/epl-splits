"""
Check our player numbers against Opta's published player table.

This is the only check that tests the minutes logic, which works out how long
each player was on the pitch from the substitution events and was never
validated against anything real.

It also brings back expected goals, which we cannot compute ourselves. Once the
players are matched, it reports each club's xG next to the goals we counted.

Run it with:  python compare_players.py
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

import pandas as pd

OUTPUT = Path("output")
REFERENCE = Path("reference")
REFERENCE_FILE = "players_table_0.csv"    # whichever file holds the player table

# Their heading -> ours in league_players.csv.
COLUMN_MAP = {
    "apps": "appearances",
    "mins": "minutes",
    "goals": "goals_for",
    "shots": "shots_for",
    "sot": "shots_on_target_for",
}

TOLERANCE_PCT = 3.0
NAME_CUTOFF = 0.82        # how close two names must be to count as the same player


def to_number(v) -> float:
    try:
        return float(str(v).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return float("nan")


def simplify(name: str) -> str:
    """Strip accents and punctuation so 'Lisandro Martínez' matches."""
    import unicodedata
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]", "", text.lower()).strip()


def match_players(theirs: list[str], ours: list[str]) -> dict[str, str]:
    lookup = {simplify(o): o for o in ours}
    keys = list(lookup)
    pairs = {}
    for name in theirs:
        plain = simplify(name)
        if plain in lookup:
            pairs[name] = lookup[plain]
            continue
        hit = difflib.get_close_matches(plain, keys, n=1, cutoff=NAME_CUTOFF)
        if hit:
            pairs[name] = lookup[hit[0]]
            continue
        # Surname fallback, which catches shortened first names.
        surname = plain.split()[-1] if plain.split() else ""
        if len(surname) > 3:
            for k in keys:
                if k.split() and k.split()[-1] == surname:
                    pairs[name] = lookup[k]
                    break
    return pairs


def main() -> None:
    ours_path = OUTPUT / "league_players.csv"
    ref_path = REFERENCE / REFERENCE_FILE
    if not ours_path.exists():
        raise SystemExit(f"Missing {ours_path}. Run the pipeline first.")
    if not ref_path.exists():
        candidates = sorted(REFERENCE.glob("*table*.csv"))
        raise SystemExit(f"Missing {ref_path}. Files available: "
                         + (", ".join(c.name for c in candidates) or "none"))

    ours = pd.read_csv(ours_path)
    theirs = pd.read_csv(ref_path)

    name_col = next((c for c in theirs.columns if str(c).lower() in
                     {"name", "player", "player name"}), theirs.columns[0])
    pairs = match_players(list(theirs[name_col]), list(ours["player"]))
    print(f"Matched {len(pairs)} of {len(theirs)} players by name.")
    missed = [n for n in theirs[name_col] if n not in pairs]
    if missed:
        print("Not matched:", ", ".join(str(m) for m in missed[:12]),
              "..." if len(missed) > 12 else "")

    rows = []
    for _, r in theirs.iterrows():
        player = pairs.get(r[name_col])
        if player is None:
            continue
        mine = ours[ours["player"] == player]
        if mine.empty:
            continue
        mine = mine.iloc[0]
        for their_col, our_col in COLUMN_MAP.items():
            if their_col not in theirs.columns or our_col not in ours.columns:
                continue
            a, b = to_number(r[their_col]), to_number(mine[our_col])
            if pd.isna(a) or pd.isna(b) or a == 0:
                continue
            rows.append({"metric": their_col, "player": player,
                         "theirs": a, "ours": b,
                         "gap_pct": round((b - a) / a * 100, 1)})

    if not rows:
        raise SystemExit("Nothing could be compared. Check the column names.")

    df = pd.DataFrame(rows)
    summary = (df.groupby("metric")
               .agg(players=("player", "count"),
                    mean_gap_pct=("gap_pct", lambda s: round(s.mean(), 2)),
                    worst_gap_pct=("gap_pct", lambda s: round(s.abs().max(), 2)))
               .reset_index().sort_values("worst_gap_pct"))
    print("\nHow our player numbers compare with theirs:\n")
    print(summary.to_string(index=False))

    bad = df[df["gap_pct"].abs() > TOLERANCE_PCT]
    print(f"\n{len(df) - len(bad)} of {len(df)} comparisons within {TOLERANCE_PCT}%.")
    if len(bad):
        print("\nWorth investigating, largest first:\n")
        print(bad.reindex(bad["gap_pct"].abs().sort_values(ascending=False).index)
              .head(20).to_string(index=False))

    mins = df[df["metric"] == "mins"]
    if not mins.empty:
        print(f"\nMinutes: average gap {mins['gap_pct'].mean():+.1f}%, "
              f"worst {mins['gap_pct'].abs().max():.1f}%.")
        print("A small positive bias is expected, since our clock runs to the last")
        print("event of the half rather than the referee's whistle.")

    # Expected goals, which we cannot work out ourselves.
    xg_col = next((c for c in theirs.columns if str(c).lower() in {"xg", "x g"}), None)
    if xg_col:
        team_of = dict(zip(ours["player"], ours["team"]))
        xg = theirs.copy()
        xg["team"] = xg[name_col].map(lambda n: team_of.get(pairs.get(n, ""), None))
        xg["xg_value"] = xg[xg_col].map(to_number)
        by_team = (xg.dropna(subset=["team"]).groupby("team")["xg_value"]
                   .sum().round(2).sort_values(ascending=False))
        if len(by_team):
            print("\nExpected goals from the players listed, by club:\n")
            totals = pd.read_csv(OUTPUT / "league_totals.csv")
            for team, value in by_team.items():
                scored = totals.loc[totals["team"] == team, "goals_for"]
                got = f"{scored.iloc[0]:.0f}" if len(scored) else "?"
                print(f"  {team:20} xG {value:6.2f}   goals {got}")
            print("\nOnly the players in their table are included, so treat these as")
            print("a floor rather than the club's full xG.")

    out = REFERENCE / "player_comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nFull comparison written to {out}")


if __name__ == "__main__":
    main()
