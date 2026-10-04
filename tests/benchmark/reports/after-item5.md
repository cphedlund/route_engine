# Recommendation benchmark report

Generated 2026-10-04 20:25 UTC. Engine path: rules-based NLQ (LLM disabled), top 5 per query. Flat = gain <= max(300 ft, 60 ft/mi x route distance).

| Metric | Value | Cases | Target |
|---|---|---|---|
| Constraint-violation rate | 0.0% | 0/86 | 0% |
| Empty-when-feasible rate | 0.0% | 0/98 | 0% |
| Hit rate @5 | 100.0% | 67/67 | maximize |
| Latency p50 / p95 | 1.5 ms / 3.2 ms | 102 | report |
| Cases passed | 102/102 | | |

## Per category

| Category | Pass/Total | Violation | Empty-when-feasible | Hit@5 | p50 ms |
|---|---|---|---|---|---|
| bike | 6/6 | 0/6 | 0/6 | 5/5 | 1.4 |
| combined | 6/6 | 0/6 | 0/6 | 4/4 | 1.5 |
| distance | 6/6 | 0/6 | 0/6 | 5/5 | 1.7 |
| dog | 5/5 | 0/5 | 0/5 | 3/3 | 1.8 |
| elevation | 4/4 | 0/4 | 0/4 | 4/4 | 1.7 |
| flat | 5/5 | 0/5 | 0/5 | 5/5 | 1.4 |
| infeasible | 3/3 | 0/0 | 0/0 | 0/0 | 1.5 |
| location | 1/1 | 0/1 | 0/1 | 1/1 | 1.6 |
| nonsense | 12/12 | 0/0 | 0/12 | 0/0 | 3.0 |
| park | 3/3 | 0/3 | 0/3 | 3/3 | 1.4 |
| park-alias | 17/17 | 0/17 | 0/17 | 13/13 | 1.4 |
| proximity | 6/6 | 0/5 | 0/5 | 2/2 | 2.2 |
| regression-false-empty | 6/6 | 0/6 | 0/6 | 4/4 | 1.8 |
| regression-flat-elevation | 11/11 | 0/11 | 0/11 | 9/9 | 1.5 |
| shape | 4/4 | 0/4 | 0/4 | 4/4 | 1.5 |
| wheelchair | 7/7 | 0/7 | 0/7 | 5/5 | 1.6 |

## Failing cases

| Case | Category | Owner | Query | Failure |
|---|---|---|---|---|
