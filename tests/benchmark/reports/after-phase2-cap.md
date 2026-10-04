# Recommendation benchmark report

Generated 2026-10-04 19:44 UTC. Engine path: rules-based NLQ (LLM disabled), top 5 per query. Flat = gain <= max(300 ft, 60 ft/mi x route distance).

| Metric | Value | Cases | Target |
|---|---|---|---|
| Constraint-violation rate | 10.8% | 9/83 | 0% |
| Empty-when-feasible rate | 0.0% | 0/95 | 0% |
| Hit rate @5 | 92.2% | 59/64 | maximize |
| Latency p50 / p95 | 1.5 ms / 3.1 ms | 102 | report |
| Cases passed | 90/102 | | |

## Per category

| Category | Pass/Total | Violation | Empty-when-feasible | Hit@5 | p50 ms |
|---|---|---|---|---|---|
| bike | 5/5 | 0/5 | 0/5 | 4/4 | 1.5 |
| combined | 6/6 | 0/6 | 0/6 | 4/4 | 1.3 |
| distance | 5/6 | 1/6 | 0/6 | 5/5 | 1.6 |
| dog | 5/5 | 0/5 | 0/5 | 3/3 | 1.7 |
| elevation | 2/4 | 2/4 | 0/4 | 2/4 | 1.9 |
| flat | 4/5 | 1/5 | 0/5 | 5/5 | 1.3 |
| infeasible | 3/6 | 0/0 | 0/0 | 0/0 | 1.3 |
| location | 1/1 | 0/1 | 0/1 | 1/1 | 1.4 |
| nonsense | 12/12 | 0/0 | 0/12 | 0/0 | 3.0 |
| park | 2/2 | 0/2 | 0/2 | 2/2 | 1.3 |
| park-alias | 17/17 | 0/17 | 0/17 | 13/13 | 1.4 |
| proximity | 5/6 | 1/5 | 0/5 | 2/2 | 2.2 |
| regression-false-empty | 5/6 | 1/6 | 0/6 | 4/4 | 1.7 |
| regression-flat-elevation | 8/11 | 3/11 | 0/11 | 6/9 | 1.4 |
| shape | 4/4 | 0/4 | 0/4 | 4/4 | 1.4 |
| wheelchair | 6/6 | 0/6 | 0/6 | 4/4 | 1.3 |

## Failing cases

| Case | Category | Owner | Query | Failure |
|---|---|---|---|---|
| reg-empty-001 | regression-false-empty | atlas-route-engine | 'flat 5 mile hike' | 1/5 returned routes violate hard constraints |
| reg-flat-006 | regression-flat-elevation | atlas-nlq | '5 mile hike with no hills' | 5/5 returned routes violate hard constraints; no must_include_any route in top 5 |
| reg-flat-007 | regression-flat-elevation | atlas-nlq | '5 mile hike with minimal elevation' | 5/5 returned routes violate hard constraints; no must_include_any route in top 5 |
| reg-flat-008 | regression-flat-elevation | atlas-nlq | '5 mile hike under 300 ft gain' | 5/5 returned routes violate hard constraints; no must_include_any route in top 5 |
| flat-005 | flat | atlas-nlq | 'gentle 4 mile stroll' | 1/5 returned routes violate hard constraints |
| dist-005 | distance | atlas-nlq | 'a flat five mile hike' | 4/5 returned routes violate hard constraints |
| elev-003 | elevation | atlas-nlq | '3 mile hike with less than 500 ft of elevation gai...' | 5/5 returned routes violate hard constraints; no must_include_any route in top 5 |
| elev-004 | elevation | atlas-nlq | '6 mile hike with at least 2000 ft of climbing' | 3/5 returned routes violate hard constraints; no must_include_any route in top 5 |
| prox-003 | proximity | atlas-nlq | 'hike within 5 miles of me' | 4/5 returned routes violate hard constraints |
| inf-001 | infeasible | atlas-route-engine | 'bike ride at Almaden Quicksilver' | case marked infeasible but 5 routes satisfy hard constraints (bad case); infeasible query: expected relaxed results plus an explanation field (got 5 routes, exp |
| inf-005 | infeasible | atlas-route-engine | 'hike at Anderson Lake' | case marked infeasible but 2 routes satisfy hard constraints (bad case); infeasible query: expected relaxed results plus an explanation field (got 2 routes, exp |
| inf-006 | infeasible | atlas-route-engine | '15 mile wheelchair accessible flat walk with dog' | infeasible query: expected relaxed results plus an explanation field (got 1 routes, explanation=None, keys=['has_more', 'min_conformity', 'notice', 'progressive |
