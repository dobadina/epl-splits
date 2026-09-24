"""
Look inside the page fetch_reference.py saved.

Works offline on reference/page.html. Saves every table it finds, shows the
headings and the first rows, and lists the buttons, tabs and dropdown options on
the page, which is how we find the switch from player stats to team stats.

Run it with:  python inspect_reference.py
"""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

import pandas as pd

REFERENCE = Path("reference")
PAGE = REFERENCE / "page.html"


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def controls(html: str) -> None:
    """Anything that looks like a way to change what the page shows."""
    print("\nCONTROLS ON THE PAGE")

    buttons = {clean(re.sub(r"<[^>]+>", " ", m))
               for m in re.findall(r"<button[^>]*>(.*?)</button>", html, re.S)}
    buttons = {b for b in buttons if 0 < len(b) < 40}
    if buttons:
        print(f"\n  buttons ({len(buttons)}):")
        for b in sorted(buttons)[:40]:
            print(f"    {b}")

    options = {clean(m) for m in re.findall(r"<option[^>]*>(.*?)</option>", html, re.S)}
    options = {o for o in options if 0 < len(o) < 40}
    if options:
        print(f"\n  dropdown options ({len(options)}):")
        for o in sorted(options)[:60]:
            print(f"    {o}")

    # Tabs are often links or list items carrying the word team or player.
    hits = set()
    for m in re.findall(r'<(?:a|li|span|div)[^>]*>([^<]{2,30})</(?:a|li|span|div)>',
                        html):
        text = clean(m)
        if re.fullmatch(r"(?i)(teams?|players?|goalkeep\w*|attacking|defending|"
                        r"passing|possession|discipline|summary|overview|shooting|"
                        r"duels?|sequences?)", text):
            hits.add(text)
    if hits:
        print(f"\n  words that look like tabs ({len(hits)}):")
        for h in sorted(hits):
            print(f"    {h}")

    for word in ["team", "Team", "TEAM"]:
        count = html.count(f">{word}<")
        if count:
            print(f"\n  '{word}' appears as its own element {count} time(s)")
            break


def main() -> None:
    if not PAGE.exists():
        raise SystemExit(f"Missing {PAGE}. Run fetch_reference.py first.")
    html = PAGE.read_text(encoding="utf-8", errors="ignore")
    print(f"Read {len(html):,} characters from {PAGE}")

    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        tables = []

    print(f"\nTABLES: {len(tables)} found")
    for i, df in enumerate(tables):
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            continue
        path = REFERENCE / f"table_{i}.csv"
        df.to_csv(path, index=False)
        print(f"\n  table {i}: {df.shape[0]} rows x {df.shape[1]} cols  ->  {path}")
        print(f"  columns: {', '.join(str(c) for c in df.columns)}")
        print(df.head(3).to_string(index=False))

    controls(html)

    print("\nNEXT")
    print("  If a control above looks like the switch to team stats, tell me its")
    print("  exact wording and I will make the scraper click it. If the tables")
    print("  above are players, they are still worth checking: minutes and")
    print("  appearances would test the substitution logic, and xG is something")
    print("  we do not have at all.")


if __name__ == "__main__":
    main()
