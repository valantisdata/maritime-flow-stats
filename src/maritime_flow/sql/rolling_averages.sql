-- Daily values for one location with 7-day and 28-day moving averages
-- params: :metric, :portname
SELECT o.obs_date,
       o.value,
       AVG(o.value) OVER (ORDER BY o.obs_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)  AS ma_7d,
       AVG(o.value) OVER (ORDER BY o.obs_date ROWS BETWEEN 27 PRECEDING AND CURRENT ROW) AS ma_28d
FROM observations o
JOIN locations l ON l.portid = o.portid
WHERE o.metric = :metric
  AND l.portname = :portname
ORDER BY o.obs_date;
