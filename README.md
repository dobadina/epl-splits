# Premier League splits

Team and player metrics for the Premier League, split by time block and by
scoreline, built from match event data and checked against Opta's published
figures.

Published stats give you season totals. They cannot tell you what a side does
between the 16th and 30th minute, or while a goal ahead. That needs event data
with timestamps, which is what this collects.

## What it produces

For every club and every player:

- **Shooting**: shots, on target, blocked, by box location, conversion
- **Passing**: completion, final-third passes, progressive passes, crosses,
  through balls, long balls
- **Possession and territory**: possession share, field tilt, box touches
- **Duels**: Opta's five duel types, aerial and ground split out
- **Defending**: shots faced, goals conceded by location, PPDA, pressing and
  recovery actions, clean sheets
- **Sequences**: build-up attacks, direct attacks, high turnovers, passes per
  passage, direct speed, start distance

Each of these is available as a season total, split into six time blocks, and
split by scoreline from two goals down to two goals up.

## Running it

Everything is driven by one script. It is scheduled weekly and needs nothing
from you.

```
python league_event_splits_v13.py     # collect and process
python validate_all.py                # 47 checks
python push_data.py                   # send the CSVs to GitHub
streamlit run app.py                  # the app, locally
```

Two switches at the top of the pipeline matter. `DOWNLOAD` fetches new matches
when true and re-processes the cache when false. `SEASON` is a two-digit pair,
so 2627 is the 2026/27 season.

The scheduled task chains the pipeline and the push. The app redeploys itself
when the data lands in GitHub.

## Checking the numbers

Four layers, in increasing order of how much they prove.

`validate_all.py` runs every time. It checks figures that must agree with each
other, that the splits sum to the season totals, that one team's "for" equals
their opponent's "against", recounts a sample of matches with different code,
and compares every scoreline against the official result.

`fetch_reference.py` pulls Opta's own published figures. `compare_reference.py`
and `compare_players.py` line ours up against theirs.

`ppda_diagnose.py` works out which defensive actions a published PPDA figure
implies, by testing every combination against all twenty clubs at once.

## What matches Opta exactly

Shots, shots faced, conversion, passes, passes completed, pass accuracy,
crosses, through balls, final-third passes and their completions, fouls
conceded, scorelines.

## What does not, and why

- **Possession** is approximated from share of touches, since events carry no
  duration. It runs about three points below the published figure.
- **Shots on target** can read one lower for a club that has scored an own goal.
  Opta appear to keep it in their on-target total while leaving it out of shots.
  We leave it out of both so that shots still equal on target plus off target
  plus blocked.
- **PPDA** uses Opta's documented action list and fits their published figures
  to about 4% across all twenty clubs.
- **Sequence metrics** are within about 6% on passes per passage and sequence
  time. Opta publish what a sequence is but not the rules for ending one, so
  ours splits play slightly more often than theirs. Sequence counts, start
  distance, direct attacks and pressed sequences differ by more.
- **Chances created** counts openings and not their quality. Expected goals
  cannot be computed from this data.

## Reading the metrics

Two kinds. **Quality** metrics have a better and a worse value, and are ranked.
**Style** metrics describe how a side plays and are shown without a rank: a team
can cross constantly or barely press and be excellent at either.

Splits need enough football behind them. Below a set floor the app shows the
numbers but will not narrate them, because a few hundred touches is one match's
noise rather than a pattern.

## Layout

```
league_event_splits_v13.py   collect, process, write the CSVs
app.py                       the Streamlit app
push_data.py                 commit and push refreshed data
validate_all.py              the checks
fetch_reference.py           pull Opta's published figures
inspect_reference.py         look inside a saved page
compare_reference.py         our team totals against theirs
compare_players.py           our player numbers against theirs
ppda_diagnose.py             work out a published PPDA definition
output/                      the CSVs the app reads
match_cache/                 scraped events, one file per match, not in git
reference/                   scraped published figures, not in git
```

## Notes

Event data is scraped for personal analysis. The scraped data is not in this
repository and should not be redistributed. The pipeline drives a real browser,
so it runs on a desktop machine rather than a server.
