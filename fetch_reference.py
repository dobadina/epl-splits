"""
Pull Opta's own published Premier League totals, to check ours against.

The page renders through JavaScript, so this drives the same browser the match
scraper uses. On the first run it is reconnaissance: it saves the rendered page,
lists every table it found with its headings, and writes each one to CSV. Once
we know which table holds what, compare_reference.py does the checking.

Run it with:  python fetch_reference.py
"""

from __future__ import annotations

import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd

URL = "https://theanalyst.com/competition/premier-league/stats"
OUT = Path("reference")
WAIT_SECONDS = 12          # give the tables time to render
SCROLLS = 6                # lazy-loaded sections need scrolling into view

# The page opens on player stats. These are the controls that change the view,
# found by inspecting the saved page.
VIEWS = [("players", None),          # what loads by default
         ("teams", "teams"),         # click this button for team stats
         ("teams_more", "More")]     # then this one, which reveals more columns


def click_text(driver, text: str) -> bool:
    """Click the first clickable element whose text matches, ignoring case."""
    lower = text.lower()
    xpath = (f"//*[self::button or self::a or self::span or self::div or self::li]"
             f"[translate(normalize-space(text()),"
             f"'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')='{lower}']")
    try:
        elements = driver.find_elements("xpath", xpath)
    except Exception:
        return False
    for el in elements:
        try:
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            time.sleep(0.5)
            driver.execute_script("arguments[0].click();", el)
            return True
        except Exception:
            continue
    return False


def settle(driver) -> None:
    """Let the page draw, including anything that loads on scroll."""
    time.sleep(WAIT_SECONDS)
    for _ in range(SCROLLS):
        driver.execute_script("window.scrollBy(0, window.innerHeight);")
        time.sleep(1.2)
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(1.2)


def render(url: str) -> dict[str, str]:
    """Return the rendered HTML for each view of the page."""
    from seleniumbase import Driver

    pages: dict[str, str] = {}
    driver = Driver(uc=True)
    try:
        driver.get(url)
        settle(driver)
        for name, control in VIEWS:
            if control is not None:
                clicked = click_text(driver, control)
                print(f"  clicking '{control}': {'done' if clicked else 'not found'}")
                if not clicked:
                    continue
                time.sleep(4)
                settle(driver)
            pages[name] = driver.page_source
            print(f"  captured view '{name}' ({len(pages[name]):,} characters)")
        return pages
    finally:
        try:
            driver.quit()
        except Exception:
            pass


def looks_like_teams(df: pd.DataFrame) -> bool:
    """A table worth keeping has a column full of club names."""
    clubs = {"arsenal", "liverpool", "chelsea", "everton", "tottenham", "fulham",
             "newcastle", "brentford", "brighton", "leeds", "sunderland",
             "bournemouth", "ipswich", "coventry", "hull"}
    for col in df.columns:
        values = {str(v).strip().lower() for v in df[col].head(25)}
        if len(values & clubs) >= 3:
            return True
    return False


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"Opening {URL}")
    print("A browser window will appear. Leave it alone while it works.\n")
    try:
        pages = render(URL)
    except Exception as exc:
        sys.exit(f"Could not render the page: {type(exc).__name__}: {exc}\n"
                 "If this mentions Chrome, close any open Chrome windows and retry.")

    print()
    for view, html in pages.items():
        raw = OUT / f"page_{view}.html"
        raw.write_text(html, encoding="utf-8")
        try:
            tables = pd.read_html(StringIO(html))
        except ValueError:
            tables = []

        print(f"VIEW '{view}': {len(tables)} table(s), page saved to {raw}")
        for i, df in enumerate(tables):
            df = df.dropna(how="all").dropna(axis=1, how="all")
            if df.empty:
                continue
            kind = "teams" if looks_like_teams(df) else "players or other"
            path = OUT / f"{view}_table_{i}.csv"
            df.to_csv(path, index=False)
            print(f"  table {i} [{kind}]: {df.shape[0]} rows x {df.shape[1]} cols "
                  f"-> {path}")
            print(f"    columns: {', '.join(str(c) for c in df.columns)}")
        print()

    print("Send me the column headings above and I will wire up the comparison.")


if __name__ == "__main__":
    main()
