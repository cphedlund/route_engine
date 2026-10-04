# Recommendation benchmark report

Generated 2026-10-04 19:49 UTC. Engine path: rules-based NLQ (LLM disabled), top 5 per query. Flat = gain <= max(300 ft, 60 ft/mi x route distance).

| Metric | Value | Cases | Target |
|---|---|---|---|
| Constraint-violation rate | 1.2% | 1/83 | 0% |
| Empty-when-feasible rate | 0.0% | 0/95 | 0% |
| Hit rate @5 | 100.0% | 64/64 | maximize |
| Latency p50 / p95 | 1.5 ms / 3.2 ms | 102 | report |
| Cases passed | 98/102 | | |

## Per category

| Category | Pass/Total | Violation | Empty-when-feasible | Hit@5 | p50 ms |
|---|---|---|---|---|---|
| bike | 5/5 | 0/5 | 0/5 | 4/4 | 1.6 |
| combined | 6/6 | 0/6 | 0/6 | 4/4 | 1.4 |
| distance | 6/6 | 0/6 | 0/6 | 5/5 | 1.6 |
| dog | 5/5 | 0/5 | 0/5 | 3/3 | 1.9 |
| elevation | 4/4 | 0/4 | 0/4 | 4/4 | 1.7 |
| flat | 5/5 | 0/5 | 0/5 | 5/5 | 1.4 |
| infeasible | 3/6 | 0/0 | 0/0 | 0/0 | 1.4 |
| location | 1/1 | 0/1 | 0/1 | 1/1 | 1.5 |
| nonsense | 12/12 | 0/0 | 0/12 | 0/0 | 3.0 |
| park | 2/2 | 0/2 | 0/2 | 2/2 | 1.4 |
| park-alias | 17/17 | 0/17 | 0/17 | 13/13 | 1.4 |
| proximity | 6/6 | 0/5 | 0/5 | 2/2 | 2.2 |
| regression-false-empty | 5/6 | 1/6 | 0/6 | 4/4 | 1.7 |
| regression-flat-elevation | 11/11 | 0/11 | 0/11 | 9/9 | 1.5 |
| shape | 4/4 | 0/4 | 0/4 | 4/4 | 1.5 |
| wheelchair | 6/6 | 0/6 | 0/6 | 4/4 | 1.5 |

## Failing cases

| Case | Category | Owner | Query | Failure |
|---|---|---|---|---|
| reg-empty-001 | regression-false-empty | atlas-route-engine | 'flat 5 mile hike' | 1/5 returned routes violate hard constraints |
| inf-001 | infeasible | atlas-route-engine | 'bike ride at Almaden Quicksilver' | case marked infeasible but 5 routes satisfy hard constraints (bad case); infeasible query: expected relaxed results plus an explanation field (got 5 routes, exp |
| inf-005 | infeasible | atlas-route-engine | 'hike at Anderson Lake' | case marked infeasible but 2 routes satisfy hard constraints (bad case); infeasible query: expected relaxed results plus an explanation field (got 2 routes, exp |
| inf-006 | infeasible | atlas-route-engine | '15 mile wheelchair accessible flat walk with dog' | infeasible query: expected relaxed results plus an explanation field (got 1 routes, explanation=None, keys=['has_more', 'min_conformity', 'notice', 'progressive |
