-- Data coverage per location: first and last day, days present, days expected
-- params: :metric
SELECT l.portname,
       l.kind,
       MIN(o.obs_date) AS first_day,
       MAX(o.obs_date) AS last_day,
       COUNT(*)        AS days_present,
       COUNT(DISTINCT o.week_start) AS weeks_present
FROM observations o
JOIN locations l ON l.portid = o.portid
WHERE o.metric = :metric
GROUP BY l.portname, l.kind
ORDER BY l.kind, l.portname;
