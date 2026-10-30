# Changelog

## 0.2.0

- Ports: daily port calls and trade estimates (`--kind port`), import/export charts,
  Piraeus as a worked case study.
- Synthetic control with placebo-in-space p-values, in-time placebo and
  leave-one-out checks.
- SQL layer: SQLite by default, PostgreSQL optional; idempotent upserts and six
  named analytical queries (`maritime-flow db ...`).
- Streamlit dashboard (`maritime-flow dashboard`), deployable to Streamlit Community Cloud.
- Demo data now has a shared "world trade" factor and eight donor units.

## 0.1.0

- PortWatch client with paging, retries and cache; data cleaning and quality report.
- Change points (PELT), interrupted time series, counterfactual, block bootstrap.
- Route diversion, anomalies, forecasting with rolling-origin backtest.
- HTML report, CLI, CI.
