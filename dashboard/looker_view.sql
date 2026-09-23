-- Looker Studio source: hourly cost / latency / throughput over the request_metrics table.
-- Replace PROJECT.DATASET, run once, then point a Looker Studio BigQuery connector at the view.
CREATE OR REPLACE VIEW `PROJECT.DATASET.request_metrics_hourly` AS
SELECT
  TIMESTAMP_TRUNC(ts, HOUR)                                   AS hour,
  COUNT(*)                                                    AS requests,
  COUNTIF(status != 'ok')                                     AS errors,
  SUM(cost_usd)                                               AS total_cost_usd,
  SAFE_DIVIDE(SUM(cost_usd), COUNT(*))                        AS cost_per_request_usd,
  APPROX_QUANTILES(latency_s, 100)[OFFSET(50)]                AS p50_latency_s,
  APPROX_QUANTILES(latency_s, 100)[OFFSET(95)]                AS p95_latency_s,
  AVG(ttft_s)                                                 AS mean_ttft_s,
  AVG(tokens_per_sec)                                         AS mean_tokens_per_sec,
  SUM(total_tokens)                                           AS total_tokens
FROM `PROJECT.DATASET.request_metrics`
GROUP BY hour;
