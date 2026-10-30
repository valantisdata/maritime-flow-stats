# Methodology

This note explains each statistical step, why it was chosen and what it
assumes. Notation: $y_t$ is the number of vessel transits on day $t$.

## 1. Data preparation

PortWatch reports one row per chokepoint per day. Cleaning
(`data/cleaning.py`):

- dates from epoch milliseconds (UTC) to calendar days;
- duplicate `(portid, date)` rows: last one kept, number reported;
- negative values set to missing;
- consistency checks `n_total = n_cargo + n_tanker` and
  `n_cargo = container + dry_bulk + general_cargo + roro`, reported, not altered;
- the series is re-indexed to a complete daily calendar. **Missing days stay
  missing.** Filling them with zero would invent days with no traffic.
  Gaps of up to three days are linearly interpolated so that weekly models can
  run; longer gaps are left as they are;
- the last few days are dropped if they fall far below the recent median,
  because the newest PortWatch data is provisional.

## 2. Description

- Mean, median, standard deviation, coefficient of variation and the
  **dispersion index** $\mathrm{Var}(y)/\bar y$. For Poisson counts this is 1;
  ship counts are typically well above 1 (over-dispersion), which is the reason
  for the quasi-Poisson models below.
- **Weekday effect.** Each day is divided by its centred 28-day moving average
  (so a crisis cannot look like a weekday effect), then a Kruskal-Wallis test
  compares the seven weekdays. Kruskal-Wallis is rank-based and does not need
  normality.

## 3. Seasonality model

Most models share one design matrix (`analysis/models.py`):

$$
\log \mu_t = \beta_0 + \beta_1 \cdot \text{years}_t
+ \sum_{d=\text{Tue}}^{\text{Sun}} \gamma_d \, \mathbb 1[\text{dow}_t = d]
+ \sum_{k=1}^{K} \big(a_k \sin \tfrac{2\pi k t}{365.25} + b_k \cos \tfrac{2\pi k t}{365.25}\big)
$$

Six weekday dummies capture the weekly cycle and $K = 3$ Fourier pairs give a
smooth annual cycle with only six parameters, instead of 365 day-of-year
effects. The Fourier terms use absolute time, so in-sample and forecast rows
share the same phase.

## 4. Structural breaks: PELT

Goal: find the dates where the average level changed, without choosing them by
eye.

1. Remove seasonality: fit the model above (no trend) on $\log(1 + y_t)$ and
   divide the series by the estimated seasonal factor.
2. Average by week and take logs. Count noise grows with the level; on the log
   scale it is roughly constant, which is what a Gaussian cost assumes.
3. Minimise
   $\sum_{\text{segments}} \sum_{t \in s} (x_t - \bar x_s)^2 + \beta \cdot (\text{number of breaks})$
   with **PELT** (Killick et al., 2012). Segment costs come from cumulative sums
   in O(1); pruning keeps run time near-linear.
4. Penalty $\beta = 2\hat\sigma^2 \log n$ (BIC-type). $\hat\sigma$ is estimated
   from the MAD of first differences, which is not inflated by the level shifts.
5. Repeat once with the found regimes added as dummies to the seasonal model,
   so a large shift is not partly absorbed into the annual Fourier terms.

The tests check both directions: a known 50% drop is found within a week, and
twenty simulated series with no break produce no false alarms.

## 5. Impact of an event

Three estimates with different assumptions. If they agree, the result does not
depend on one modelling choice.

**(a) Before / after means with a moving-block bootstrap.** The same length of
time before and after the event. Daily values are autocorrelated, so an ordinary
bootstrap would understate uncertainty; resampling 14-day blocks keeps the
dependence. A Mann-Whitney test is reported alongside.

**(b) Interrupted time series (segmented quasi-Poisson regression).**

$$
\log \mu_t = \text{seasonality}_t + \beta_1 \text{years}_t + \delta \, \mathbb 1[t \ge T_0] + \lambda \, (t - T_0)_+ / 30.44
$$

$e^\delta$ is the **rate ratio** at the event (0.6 means 40% fewer transits than
the pre-event pattern implies) and $e^\lambda$ is the extra month-on-month change
afterwards. Fitted by IRLS; standard errors are multiplied by the Pearson
dispersion $\hat\phi = \frac{1}{n-p}\sum (y_t - \hat\mu_t)^2/\hat\mu_t$
(quasi-Poisson), which corrects for over-dispersion.

**(c) Counterfactual.** Fit the seasonal model on the two years before the
event only, and project it forward: "what would traffic have looked like?". No
trend by default, because extrapolating a linear trend for a year is fragile.
Uncertainty is simulated: draw coefficients from their estimated sampling
distribution, then draw daily counts from a negative binomial with variance
$\hat\phi\mu$. The shortfall is expected minus observed, with a 90% interval.

Caveat: none of this proves causation. Other things happen at the same time
(freight rates, fuel prices, seasonal trade). The estimates describe how
unusual post-event traffic is relative to its own history.

## 5b. Impact of an event: synthetic control

The three estimates above use only the unit's own past. Synthetic control
(Abadie & Gardeazabal, 2003; Abadie, Diamond & Hainmueller, 2010) uses other
units instead: it builds a "synthetic twin" of the treated unit as a weighted
average of donors that the event did not affect.

1. Weekly means for the treated unit and $J$ donors, each rescaled so its
   pre-event mean is 100 (units of very different size become comparable).
2. Weights solve

```math
\min_{w}\; \sum_{t < T_0} \Big(y_{1t} - \sum_{j=1}^{J} w_j y_{jt}\Big)^2
\quad \text{s.t.}\quad w_j \ge 0,\; \sum_j w_j = 1
```

   (SLSQP). The constraints keep the twin an interpolation of real units, never
   an extrapolation.
3. Effect = average post-event gap divided by the synthetic level.
4. **Placebo-in-space inference.** Each donor is treated as if it had been hit
   and the procedure is rerun. The statistic is the ratio of post-event to
   pre-event RMSPE (a large post gap only counts if the pre-event fit was good).
   The p-value is the treated unit's rank among all $J+1$ ratios, so the
   smallest attainable p-value is $1/(J+1)$. With 8 donors that is 0.11: a
   small pool simply cannot give "significance".
5. **In-time placebo.** The whole procedure is rerun with a fake event one year
   earlier, using only pre-event data. The "effect" should be near zero.
6. **Leave-one-out.** The effect is recomputed without each donor that has
   weight; a result driven by one donor is fragile.

The assumption that cannot be tested from data: donors were not affected. For
the Red Sea crisis, the Cape of Good Hope is a bad donor for Suez (traffic
moved there), and most Mediterranean ports are exposed too. The test suite
shows the consequence: adding a contaminated donor biases the estimate.

## 6. Alternative routes

If traffic avoids route A, does it appear on route B?

- **Share** of B in A + B per week, compared before and after the event with
  Welch's t-test (unequal variances).
- **Correlation of weekly changes** (Pearson and Spearman). Levels are not
  correlated directly: two series that both trend will correlate even if they
  are unrelated (spurious correlation). Differencing removes that.
- **Cross-correlation** at lags of up to 8 weeks, with an approximate 95% band
  of $\pm 1.96/\sqrt n$.

## 7. Unusual days

Residuals of the seasonal model on log counts, minus a centred 29-day rolling
median (the local level), scored with the modified z-score
$z = 0.6745 (r - \tilde r)/\mathrm{MAD}$. Days with $|z| > 3.5$ are flagged
(Iglewicz & Hoaglin, 1993). Median and MAD are used instead of mean and
standard deviation because the outliers would otherwise inflate the threshold
used to find them.

## 8. Forecasting

Models:

| Model | Idea |
|---|---|
| Seasonal naive | tomorrow = same weekday last week (benchmark) |
| Harmonic regression | OLS on $\log(1+y)$ with the seasonal design, last two years; Duan smearing for the back-transform |
| Holt-Winters | additive level, damped trend and weekly season; smoothing weights chosen by minimising one-step squared error |

**Evaluation** is a rolling-origin backtest: train up to an origin, forecast 28
days, move forward 28 days, repeat 10 times. Windows do not overlap, and the
model never sees the data it is scored on.

- **MASE** = MAE / in-sample MAE of the weekly naive forecast. Below 1 = better
  than the benchmark. Scale-free, so chokepoints of different size compare.
- **Diebold-Mariano** test on squared errors, with a Newey-West long-run
  variance ($h-1$ lags) and the Harvey-Leybourne-Newbold small-sample
  correction. A negative statistic means the model beats the benchmark.

Prediction intervals are empirical: quantiles of backtest errors, pooled by
forecast week, added to the point forecast. They assume the next month behaves
like the backtest months; they will be too narrow at the start of a new crisis.

## 9. SQL layer

Cleaned data can be loaded into SQLite (default) or PostgreSQL. A single long
table `observations(portid, obs_date, metric, value)` holds every metric of
every location; calendar keys (week, month, year) are computed at load so the
same SQL runs on both engines. Loads are upserts, so revised PortWatch days
overwrite old values. The named queries in `src/maritime_flow/sql/` use window
functions (`LAG`, moving averages with `ROWS BETWEEN`, `RANK`) and the tests
check them against the equivalent pandas computation.

## 10. Why no statsmodels?

The estimators used here are short to write with numpy/scipy (OLS, HAC
covariance, Poisson IRLS, PELT, Holt-Winters). Implementing them keeps the
install light and makes every assumption visible in code. The tests compare
them against scipy reference results and against simulated data with a known
answer.
