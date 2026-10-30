# Data sources and terms of use

## IMF PortWatch

All real data used by this project comes from **IMF PortWatch**
(<https://portwatch.imf.org>), an open platform from the International Monetary
Fund developed with the University of Oxford. Daily chokepoint transit calls
and port activity are estimated from AIS (Automatic Identification System)
vessel positions provided by the UN Global Platform.

Endpoints used (public ArcGIS feature services, no API key):

| Dataset | Layer |
|---|---|
| Daily chokepoint transits | `Daily_Chokepoints_Data/FeatureServer/0` |
| Chokepoint metadata | `PortWatch_chokepoints_database/FeatureServer/0` |
| Daily port activity and trade estimates | `Daily_Trade_Data/FeatureServer/0` |
| Port metadata | `PortWatch_ports_database/FeatureServer/0` |

The same rules apply to data stored in a local database (`maritime-flow db load`) or
shown in the dashboard: the database file is git-ignored, and the dashboard
shows the attribution on every page.

Base URL: `https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/`

### Terms

The IMF's [copyright and terms of use](https://www.imf.org/en/About/copyright-and-terms)
contain special terms for published statistical data. In short:

- Data may be downloaded, copied, transformed and published **with attribution
  to the IMF as the source**.
- If the data is **materially transformed**, this must be stated together with
  the source citation.
- Commercial reuse requires permission from the IMF (copyright@imf.org).

How this repository complies:

- Every figure carries a source line and every report ends with the full
  attribution and a statement that the numbers are transformed, modelled
  estimates (see `ATTRIBUTION` in `src/maritime_flow/config.py`).
- **No raw IMF data is committed to this repository.** Data is downloaded at
  run time by the user and cached locally (`~/.cache/maritime-flow`); `data/` and
  `reports/` are git-ignored.
- The project is non-commercial.
- Requests are paged, cached for 24 h by default and retried with backoff to
  keep load on the free public service low.

Suggested citation:

> Source: International Monetary Fund, IMF PortWatch, https://portwatch.imf.org
> (accessed YYYY-MM-DD). Data aggregated and modelled by the author.

## Reference events

Event dates used to annotate charts are listed in `src/maritime_flow/events.py`,
each with a public source link.

## Synthetic data

`maritime-flow demo` and the test suite use synthetic data produced by
`src/maritime_flow/data/synthetic.py`. It mimics the PortWatch schema but is
randomly generated and says nothing about real shipping.

## Methods references

- Killick, R., Fearnhead, P. & Eckley, I. A. (2012). Optimal detection of
  changepoints with a linear computational cost. *JASA*, 107(500).
- Newey, W. K. & West, K. D. (1987, 1994). HAC covariance estimation and
  automatic lag selection.
- Diebold, F. X. & Mariano, R. S. (1995); Harvey, Leybourne & Newbold (1997)
  small-sample correction.
- Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary
  observations (moving-block bootstrap).
- Iglewicz, B. & Hoaglin, D. (1993). *How to Detect and Handle Outliers*
  (modified z-score).
- Hyndman, R. J. & Koehler, A. B. (2006). Another look at measures of forecast
  accuracy (MASE).
- Bernal, J. L., Cummins, S. & Gasparrini, A. (2017). Interrupted time series
  regression for the evaluation of public health interventions. *IJE*, 46(1).
- Abadie, A. & Gardeazabal, J. (2003). The economic costs of conflict: a case
  study of the Basque Country. *AER*, 93(1).
- Abadie, A., Diamond, A. & Hainmueller, J. (2010). Synthetic control methods
  for comparative case studies. *JASA*, 105(490).
- Abadie, A. (2021). Using synthetic controls: feasibility, data requirements,
  and methodological aspects. *Journal of Economic Literature*, 59(2).
- Duan, N. (1983). Smearing estimate: a nonparametric retransformation method.
