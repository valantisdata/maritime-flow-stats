-- Yearly share of each vessel type in total chokepoint transits
-- params: none
WITH yearly AS (
    SELECT l.portname,
           o.obs_year,
           SUM(CASE WHEN o.metric = 'n_container'     THEN o.value ELSE 0 END) AS container,
           SUM(CASE WHEN o.metric = 'n_dry_bulk'      THEN o.value ELSE 0 END) AS dry_bulk,
           SUM(CASE WHEN o.metric = 'n_general_cargo' THEN o.value ELSE 0 END) AS general_cargo,
           SUM(CASE WHEN o.metric = 'n_roro'          THEN o.value ELSE 0 END) AS roro,
           SUM(CASE WHEN o.metric = 'n_tanker'        THEN o.value ELSE 0 END) AS tanker,
           SUM(CASE WHEN o.metric = 'n_total'         THEN o.value ELSE 0 END) AS total
    FROM observations o
    JOIN locations l ON l.portid = o.portid
    WHERE l.kind = 'chokepoint'
    GROUP BY l.portname, o.obs_year
)
SELECT portname,
       obs_year,
       total,
       container / total     AS share_container,
       dry_bulk / total      AS share_dry_bulk,
       general_cargo / total AS share_general_cargo,
       roro / total          AS share_roro,
       tanker / total        AS share_tanker
FROM yearly
WHERE total > 0
ORDER BY portname, obs_year;
