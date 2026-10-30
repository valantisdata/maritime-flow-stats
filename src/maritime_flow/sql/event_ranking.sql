-- Rank all locations by the change in average daily value after an event
-- params: :metric, :pre_start, :event, :post_end
WITH windows AS (
    SELECT l.portname,
           l.kind,
           AVG(CASE WHEN o.obs_date >= :pre_start AND o.obs_date < :event THEN o.value END) AS mean_before,
           AVG(CASE WHEN o.obs_date >= :event AND o.obs_date < :post_end THEN o.value END)  AS mean_after,
           SUM(CASE WHEN o.obs_date >= :pre_start AND o.obs_date < :event THEN 1 ELSE 0 END) AS days_before,
           SUM(CASE WHEN o.obs_date >= :event AND o.obs_date < :post_end THEN 1 ELSE 0 END)  AS days_after
    FROM observations o
    JOIN locations l ON l.portid = o.portid
    WHERE o.metric = :metric
    GROUP BY l.portname, l.kind
)
SELECT portname,
       kind,
       mean_before,
       mean_after,
       mean_after / mean_before - 1 AS pct_change,
       days_before,
       days_after,
       RANK() OVER (ORDER BY mean_after / mean_before) AS most_affected_rank
FROM windows
WHERE mean_before > 0 AND mean_after IS NOT NULL
ORDER BY pct_change;
