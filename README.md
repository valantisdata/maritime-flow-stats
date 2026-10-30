# maritime-flow-stats

Statistical analysis of ship traffic through the world's maritime chokepoints
(Suez Canal, Panama Canal, Bab el-Mandeb, Strait of Hormuz, Cape of Good Hope
and others) and through individual ports such as Piraeus, using live daily
data from **IMF PortWatch**.

About 80% of world trade by volume moves by sea, and a large share of it passes
through a handful of narrow routes. When one of them is disrupted, by a grounded
ship, a drought or a conflict, the effect shows up in daily vessel counts within
days. This project downloads those counts, stores them in a SQL database and
answers concrete questions with statistical methods:

| Question | Method |
|---|---|
| When did the traffic level actually change? | PELT change-point detection on seasonally adjusted weekly data |
| How large was the effect of an event, judged by the unit's own past? | interrupted time series (quasi-Poisson), counterfactual simulation, block bootstrap |
| How large was it compared with similar unaffected units? | synthetic control with placebo-in-space inference |
| Did traffic move to an alternative route? | route share, Welch t-test, correlation of weekly changes |
| How did a port's trade respond? | port calls and import/export estimates, Piraeus as case study |
| Which days were unusual? | robust z-scores on seasonal-model residuals |
| What happens over the next four weeks? | three models in a rolling-origin backtest (MASE, Diebold-Mariano) |

Output: an HTML report per analysis, CSV tables, a SQLite/PostgreSQL database
with ready-made SQL queries, and an interactive Streamlit dashboard.

<!-- After the first live run: copy reports/red-sea/figures/overview.png to docs/img/ and uncomment
![Daily transits](docs/img/overview.png)
-->

## Quick start

```bash
git clone https://github.com/<your-username>/maritime-flow-stats.git
cd maritime-flow-stats
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,app]"

maritime-flow demo                       # everything on synthetic data, no network
maritime-flow list                       # chokepoints on PortWatch
maritime-flow list --kind port --country Greece
maritime-flow -v analyze --target Suez --comparator "Good Hope" --event red_sea --out reports/red-sea
maritime-flow dashboard                  # interactive app in the browser
```

Open `reports/red-sea/report.html`. Tables are saved as CSV in
`reports/red-sea/tables/` and the headline numbers in `results.json`.

### Commands

| Command | What it does |
|---|---|
| `maritime-flow list [--kind port] [--country C]` | chokepoints or ports known to PortWatch |
| `maritime-flow fetch NAME ... [--kind port]` | download, clean and save a daily panel; prints a data-quality report |
| `maritime-flow analyze --target NAME [--kind port] [--comparator NAME] [--donors NAME ...] [--event DATE\|KEY]` | full analysis and report |
| `maritime-flow db load NAME ... [--kind port]` / `db load --demo` | upsert data into the database |
| `maritime-flow db query list` / `db query NAME [--metric M] [--event E]` | run a named SQL query, or your own `SELECT` |
| `maritime-flow analyze ... --from-db` | analyse what is in the database instead of downloading |
| `maritime-flow dashboard` | Streamlit dashboard |
| `maritime-flow demo` | chokepoint and port reports on synthetic data |

Names match loosely (`suez`, `hormuz`, `good hope`). `--event` takes a date
(`2023-11-19`) or a key from `src/maritime_flow/events.py`. `--column` picks the
series (default `n_total` for chokepoints, `portcalls` for ports). `--offline`
uses the local cache only; downloads are cached for 24 hours.

## Case studies

### The Red Sea crisis and the Suez Canal

From November 2023 attacks on merchant ships in the Red Sea led many carriers
to avoid the Bab el-Mandeb / Suez route and sail around Africa instead.

```bash
maritime-flow analyze --target Suez --comparator "Good Hope" --event red_sea \
    --donors <unaffected chokepoints> --out reports/red-sea
```

<!-- Replace with your own findings after the live run: 2-3 sentences with numbers. -->

### Piraeus

```bash
maritime-flow list --kind port --country Greece
maritime-flow analyze --kind port --target Piraeus --event red_sea \
    --donors <comparable ports> --out reports/piraeus
```

<!-- Replace with your own findings after the live run. -->

**Choosing donors.** Synthetic control assumes the donors were not affected by
the event. Routes that gained traffic (Cape of Good Hope) or ports on the same
disrupted trade lane are poor donors. The choice has to be argued, and the
report's placebo and leave-one-out checks show how much it matters. See
[docs/methodology.md](docs/methodology.md#5b-impact-of-an-event-synthetic-control).

## Database and SQL

```bash
maritime-flow db load Suez "Good Hope" Malacca Panama        # SQLite at data/maritime.db
maritime-flow db load --kind port Piraeus
maritime-flow db query list
maritime-flow db query event_ranking --metric n_total --event red_sea
maritime-flow db query monthly_yoy --metric portcalls --csv yoy.csv
```

PostgreSQL: `pip install -e ".[db]"` and pass
`--db-url postgresql://user:pass@host/dbname` (or set `MARITIME_FLOW_DB_URL`).
The schema and queries are in `src/maritime_flow/storage.py` and
`src/maritime_flow/sql/`.

## Dashboard

`maritime-flow dashboard` opens a Streamlit app with five tabs: overview, event
impact, synthetic control, forecast and a SQL console. It works on demo data,
live PortWatch data or the local database.

To publish it: push the repository to GitHub, then create an app on
[Streamlit Community Cloud](https://streamlit.io/cloud) pointing at
`streamlit_app.py`. Dependencies are read from `requirements.txt`.

## Project structure

```
src/maritime_flow/
├── config.py              endpoints, attribution text, column names
├── events.py              reference events with sources
├── data/
│   ├── portwatch.py       API client: paging, retries with backoff, disk cache
│   ├── cleaning.py        validation, identities, missing days, quality report
│   └── synthetic.py       data with known ground truth (tests, demo, CI)
├── analysis/
│   ├── models.py          OLS + Newey-West, Poisson IRLS (quasi), harmonic design
│   ├── descriptive.py     summaries, dispersion, weekday test, vessel mix, trade
│   ├── changepoint.py     PELT, seasonal adjustment
│   ├── impact.py          block bootstrap, interrupted time series, counterfactual
│   ├── synthetic_control.py  weights, placebo inference, robustness checks
│   ├── diversion.py       route share and correlation of changes
│   ├── anomaly.py         robust z-scores
│   └── forecast.py        models, rolling-origin backtest, DM test, intervals
├── storage.py             SQLite / PostgreSQL schema, upserts, queries
├── sql/*.sql              named analytical queries (window functions)
├── app/                   Streamlit dashboard (data.py is tested, UI is thin)
├── viz/plots.py           matplotlib figures (every figure carries the source)
├── report/html.py         self-contained HTML report
├── pipeline.py            wires it together
└── cli.py                 command line
tests/                     pytest suite, offline
docs/methodology.md        the statistics, step by step
```

Design choices:

- **Statistics are pure functions** that take pandas objects and return numbers
  or DataFrames. No I/O inside, so they are easy to test.
- **Estimators are written with numpy/scipy**, so every assumption is visible.
  Tests check them against scipy reference results and against simulated data
  with a known answer: a 50% drop must be recovered within its confidence
  interval, pure noise must not trigger a change point, and a contaminated donor
  must visibly bias the synthetic control.
- **SQL is checked against pandas.** Weekly totals and moving averages from the
  database must equal the same numbers computed in pandas.
- **Missing data stays missing.** Short gaps are interpolated, long gaps are not
  filled, and everything that was changed is reported.

## Testing

```bash
pytest          # offline, about 20 seconds
ruff check src tests
```

CI runs the tests on Python 3.10-3.12 and a demo run on every push. A scheduled
workflow rebuilds the Red Sea report from live data every week and stores it as
a build artifact.

## Data and licence

Data: **International Monetary Fund, IMF PortWatch**
(<https://portwatch.imf.org>), based on UN Global Platform AIS data. Reports,
figures and the dashboard show transformed and modelled estimates, not official
IMF statistics. Raw data is not redistributed in this repository; it is
downloaded by the user at run time. Details and the IMF terms are in
[DATA_SOURCES.md](DATA_SOURCES.md).

Code: MIT licence, see [LICENSE](LICENSE).

## Limitations

- AIS coverage is not complete and PortWatch's newest days are provisional.
- Port trade volumes are PortWatch model estimates, not customs statistics.
- Impact estimates describe how unusual traffic is relative to its own past or
  to comparable units. Causal reading rests on assumptions (no other shock at the
  same time, unaffected donors) that the data cannot fully verify.
