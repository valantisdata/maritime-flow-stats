-- Weekly totals per location with week-over-week change (complete weeks only)
-- params: :metric
WITH weekly AS (
    SELECT l.portname,
           o.week_start,
           SUM(o.value) AS total,
           COUNT(*)     AS days
    FROM observations o
    JOIN locations l ON l.portid = o.portid
    WHERE o.metric = :metric
    GROUP BY l.portname, o.week_start
)
SELECT portname,
       week_start,
       total,
       total - LAG(total) OVER (PARTITION BY portname ORDER BY week_start) AS wow_change
FROM weekly
WHERE days = 7
ORDER BY portname, week_start;
