"""
Push the refreshed CSVs to GitHub, which makes the deployed app redeploy itself.

Run it straight after the pipeline. The scheduled task can chain them:
    python league_event_splits_v7.py && python push_data.py

It only commits the output folder, so code changes you are part way through are
left alone. If nothing changed it does nothing and says so.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).parent
FILES = ["output/league_totals.csv",
         "output/league_by_time_block.csv",
         "output/league_by_game_state.csv",
         "output/league_players.csv",
         "output/match_level_long.csv"]


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def main() -> None:
    check = run("rev-parse", "--is-inside-work-tree")
    if check.returncode != 0:
        sys.exit("This folder is not a git repository yet. Set that up first.")

    present = [f for f in FILES if (REPO / f).exists()]
    if not present:
        sys.exit("No CSVs found in output/. Run the pipeline first.")

    add = run("add", *present)
    if add.returncode != 0:
        sys.exit(f"git add failed: {add.stderr.strip()}")

    staged = run("diff", "--cached", "--name-only")
    if not staged.stdout.strip():
        print("Data unchanged since the last push. Nothing to do.")
        return

    stamp = datetime.now().strftime("%d %b %Y")
    commit = run("commit", "-m", f"Data refresh, {stamp}")
    if commit.returncode != 0:
        sys.exit(f"git commit failed: {commit.stderr.strip() or commit.stdout.strip()}")

    push = run("push")
    if push.returncode != 0:
        sys.exit(f"git push failed: {push.stderr.strip()}\n"
                 "If this mentions authentication, run 'git push' by hand once to "
                 "sign in, then the scheduled runs will work.")

    changed = [line for line in staged.stdout.strip().splitlines()]
    print(f"Pushed {len(changed)} file(s). The app will redeploy in a minute or two.")


if __name__ == "__main__":
    main()
