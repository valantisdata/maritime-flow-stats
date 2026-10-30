-- Monthly totals per location and change versus the same month a year earlier
-- params: :metric
WITH monthly AS (
    SELECT l.portname,
           o.month_start,
           SUM(o.value) AS total,
           AVG(o.value) AS daily_mean,
           COUNT(*)     AS days
    FROM observations o
    JOIN locations l ON l.portid = o.portid
    WHERE o.metric = :metric
    GROUP BY l.portname, o.month_start
),
with_lag AS (
    SELECT portname,
           month_start,
           total,
           daily_mean,
           days,
           LAG(daily_mean, 12) OVER (PARTITION BY portname ORDER BY month_start)  AS daily_mean_ly,
           LAG(month_start, 12) OVER (PARTITION BY portname ORDER BY month_start) AS month_ly
    FROM monthly
)
SELECT portname,
       month_start,
       total,
       daily_mean,
       daily_mean_ly,
       -- daily means rather than totals, so months with a few missing days still compare fairly
       CASE WHEN daily_mean_ly > 0 THEN daily_mean / daily_mean_ly - 1 END AS yoy_change,
       month_ly
FROM with_lag
ORDER BY portname, month_start;
