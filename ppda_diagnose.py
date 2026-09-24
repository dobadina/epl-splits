"""
Work out which defensive actions Opta count in PPDA.

We know their answer for one club, so this counts every candidate action in the
pressing zone, tries every combination of them as the denominator, and reports
which combinations land on the published figure.

Set TEAM and TARGET_PPDA to a club and figure you have from their site.

Run it with:  python ppda_diagnose.py
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

CACHE = Path("match_cache")
# Every club's published PPDA, read off their pressing table. Fitting one club
# proves nothing, since a dozen combinations can hit a single number. Only the
# real definition fits all twenty.
TARGETS = {
    "Brighton": 7.4, "Tottenham": 8.8, "Liverpool": 9.1, "Man City": 9.9,
    "Sunderland": 10.4, "Brentford": 11.2, "Newcastle": 11.2, "Bournemouth": 11.3,
    "Arsenal": 11.7, "Leeds": 11.9, "Man Utd": 12.1, "Nottingham Forest": 12.1,
    "Ipswich": 13.5, "Fulham": 13.6, "Everton": 14.6, "Chelsea": 14.8,
    "Crystal Palace": 14.8, "Aston Villa": 15.2, "Hull": 17.2, "Coventry": 19.8,
}

OWN_THIRD = 33.3          # our defensive third ends here
FINAL_THIRD = 66.7        # their passes beyond this are inside our defensive third

CANDIDATES = ["Tackle", "Interception", "Challenge", "BlockedPass", "Foul",
              "BallRecovery", "Clearance", "Aerial"]


def display_name(v):
    return v.get("displayName") if isinstance(v, dict) else v


def qualifier_set(q) -> set:
    if q is None or isinstance(q, float):
        return set()
    try:
        items = list(q)
    except TypeError:
        return set()
    out = set()
    for item in items:
        if hasattr(item, "get"):
            label = display_name(item.get("type", item)) or item.get("displayName")
            if label is not None:
                out.add(str(label))
    return out


def gather() -> dict:
    """Count the passes allowed and each candidate action, for every club."""
    files = [f for f in sorted(CACHE.glob("*.parquet")) if f.stat().st_size > 10_000]
    if not files:
        raise SystemExit("No cached matches found.")

    data: dict[str, dict] = {}
    for f in files:
        ev = pd.read_parquet(f)
        if ev.empty:
            continue
        ev["t"] = ev["type"].map(display_name)
        ev["o"] = ev["outcome_type"].map(display_name)
        ev["q"] = ev["qualifiers"].apply(qualifier_set)
        x = pd.to_numeric(ev["x"], errors="coerce")
        not_a_pass = ev["q"].apply(
            lambda s: bool({"Cross", "ThrowIn", "KeeperThrow"} & s))

        for team in [t for t in ev["team"].dropna().unique()]:
            d = data.setdefault(team, {"narrow": 0, "all": 0, "matches": 0,
                                       **{c: 0 for c in CANDIDATES}})
            d["matches"] += 1
            opp = ev["team"].ne(team) & ev["team"].notna()
            mine = ev["team"].eq(team)

            zone_pass = opp & ev["t"].eq("Pass") & x.lt(FINAL_THIRD)
            d["all"] += int(zone_pass.sum())
            d["narrow"] += int((zone_pass & ~not_a_pass).sum())

            zone_action = mine & x.ge(OWN_THIRD)
            for c in CANDIDATES:
                if c == "Foul":
                    d[c] += int((zone_action & ev["t"].eq("Foul")
                                 & ev["o"].ne("Successful")).sum())
                elif c == "Aerial":
                    d[c] += int((zone_action & ev["t"].eq("Aerial")
                                 & ev["o"].eq("Successful")).sum())
                else:
                    d[c] += int((zone_action & ev["t"].eq(c)).sum())
    return data


def main() -> None:
    data = gather()
    clubs = [t for t in TARGETS if t in data]
    missing = [t for t in TARGETS if t not in data]
    print(f"Testing against {len(clubs)} clubs with a published figure.")
    if missing:
        print("No data for:", ", ".join(missing))
    print(f"Clubs found in the cache: {', '.join(sorted(data))}\n")

    results = []
    for size in range(1, len(CANDIDATES) + 1):
        for combo in combinations(CANDIDATES, size):
            for label in ("narrow", "all"):
                errors = []
                for team in clubs:
                    d = data[team]
                    denom = sum(d[c] for c in combo)
                    if denom == 0:
                        errors = None
                        break
                    ppda = d[label] / denom
                    errors.append(abs(ppda - TARGETS[team]) / TARGETS[team] * 100)
                if errors:
                    results.append((sum(errors) / len(errors), max(errors),
                                    label, combo))

    results.sort()
    print("Best fits across every club, by average error:\n")
    print(f"  {'avg err':>8} {'worst':>7}  {'numerator':<14} actions")
    for avg, worst, label, combo in results[:10]:
        name = "all deliveries" if label == "all" else "narrow passes"
        print(f"  {avg:7.1f}% {worst:6.1f}%  {name:<14} {' + '.join(combo)}")

    best_avg, best_worst, best_label, best_combo = results[0]
    print(f"\nBest: {' + '.join(best_combo)} with "
          f"{'all deliveries' if best_label == 'all' else 'narrow passes'}, "
          f"average error {best_avg:.1f}%.")
    if best_avg > 5:
        print("That is not close enough to call it their definition. The zone is the")
        print("next suspect: they may use 60% of the pitch rather than a third.")
    else:
        print("\nPer club with that combination:\n")
        for team in sorted(clubs, key=lambda t: TARGETS[t]):
            d = data[team]
            ppda = d[best_label] / sum(d[c] for c in best_combo)
            print(f"  {team:20} ours {ppda:5.1f}   theirs {TARGETS[team]:5.1f}")


if __name__ == "__main__":
    main()
