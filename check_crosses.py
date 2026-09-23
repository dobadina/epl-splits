"""Break down one match's cross count, to see what is inflating it.

Usage:  python check_crosses.py 1983548 "Man Utd"
"""

import sys
from pathlib import Path

import pandas as pd


def display_name(value):
    if isinstance(value, dict):
        return value.get("displayName", value.get("value"))
    return value


def qnames(qualifiers) -> set:
    if qualifiers is None or isinstance(qualifiers, float):
        return set()
    try:
        items = list(qualifiers)
    except TypeError:
        return set()
    out = set()
    for q in items:
        if hasattr(q, "get"):
            name = display_name(q.get("type", q)) or q.get("displayName")
            if name is not None:
                out.add(str(name))
        elif isinstance(q, str):
            out.add(q)
    return out


match_id = sys.argv[1] if len(sys.argv) > 1 else "1983548"
team = sys.argv[2] if len(sys.argv) > 2 else None

df = pd.read_parquet(Path("match_cache") / f"{match_id}.parquet")
df["type_name"] = df["type"].map(display_name)
df["outcome"] = df["outcome_type"].map(display_name) if "outcome_type" in df else None
df["q"] = df["qualifiers"].apply(qnames)

if team:
    df = df[df["team"] == team]

passes = df[df["type_name"] == "Pass"]
crosses = passes[passes["q"].apply(lambda s: "Cross" in s)]


def count(tag):
    return crosses["q"].apply(lambda s: tag in s).sum()


print(f"Match {match_id}, team: {team or 'both'}")
print(f"Total passes:            {len(passes)}")
print(f"Tagged as Cross:         {len(crosses)}")
print(f"  also CornerTaken:      {count('CornerTaken')}")
print(f"  also FreekickTaken:    {count('FreekickTaken')}")
print(f"  also ThrowIn:          {count('ThrowIn')}")
print(f"  also GoalKick:         {count('GoalKick')}")
print(f"  also Longball:         {count('Longball')}")
print(f"  also Chipped:          {count('Chipped')}")

set_piece = crosses["q"].apply(
    lambda s: bool({"CornerTaken", "FreekickTaken", "ThrowIn", "GoalKick"} & s)
)
print(f"\nOpen play crosses only:  {(~set_piece).sum()}")
print(f"Successful open play:    {((~set_piece) & crosses['outcome'].eq('Successful')).sum()}")

print("\nMost common qualifier tags on crosses:")
tally = {}
for s in crosses["q"]:
    for tag in s:
        tally[tag] = tally.get(tag, 0) + 1
for tag, n in sorted(tally.items(), key=lambda kv: -kv[1])[:15]:
    print(f"  {tag:<22} {n}")
