# Recommendation benchmark report

Generated 2026-10-04 00:57 UTC. Engine path: rules-based NLQ (LLM disabled), top 5 per query. Flat = gain <= max(300 ft, 60 ft/mi x route distance).

| Metric | Value | Cases | Target |
|---|---|---|---|
| Constraint-violation rate | 12.0% | 10/83 | 0% |
| Empty-when-feasible rate | 0.0% | 0/95 | 0% |
| Hit rate @5 | 90.6% | 58/64 | maximize |
| Latency p50 / p95 | 1.4 ms / 3.0 ms | 102 | report |
| Cases passed | 92/102 | | |

## Per category

| Category | Pass/Total | Violation | Empty-when-feasible | Hit@5 | p50 ms |
|---|---|---|---|---|---|
| bike | 5/5 | 0/5 | 0/5 | 4/4 | 1.3 |
| combined | 6/6 | 0/6 | 0/6 | 4/4 | 1.2 |
| distance | 4/6 | 2/6 | 0/6 | 4/5 | 1.7 |
| dog | 5/5 | 0/5 | 0/5 | 3/3 | 1.6 |
| elevation | 2/4 | 2/4 | 0/4 | 2/4 | 1.7 |
| flat | 4/5 | 1/5 | 0/5 | 5/5 | 1.2 |
| infeasible | 6/6 | 0/0 | 0/0 | 0/0 | 1.4 |
| location | 1/1 | 0/1 | 0/1 | 1/1 | 1.3 |
| nonsense | 12/12 | 0/0 | 0/12 | 0/0 | 3.0 |
| park | 2/2 | 0/2 | 0/2 | 2/2 | 1.2 |
| park-alias | 17/17 | 0/17 | 0/17 | 13/13 | 1.3 |
| proximity | 5/6 | 1/5 | 0/5 | 2/2 | 2.1 |
| regression-false-empty | 5/6 | 1/6 | 0/6 | 4/4 | 1.6 |
| regression-flat-elevation | 8/11 | 3/11 | 0/11 | 6/9 | 1.2 |
| shape | 4/4 | 0/4 | 0/4 | 4/4 | 1.3 |
| wheelchair | 6/6 | 0/6 | 0/6 | 4/4 | 1.2 |

## Failing cases

| Case | Category | Owner | Query | Failure |
|---|---|---|---|---|
| reg-empty-001 | regression-false-empty | atlas-route-engine | 'flat 5 mile hike' | 1/5 returned routes violate hard constraints |
| reg-flat-006 | regression-flat-elevation | atlas-nlq | '5 mile hike with no hills' | 5/5 returned routes violate hard constraints; no must_include_any route in top 5 |
| reg-flat-007 | regression-flat-elevation | atlas-nlq | '5 mile hike with minimal elevation' | 5/5 returned routes violate hard constraints; no must_include_any route in top 5 |
| reg-flat-008 | regression-flat-elevation | atlas-nlq | '5 mile hike under 300 ft gain' | 5/5 returned routes violate hard constraints; no must_include_any route in top 5 |
| flat-005 | flat | atlas-nlq | 'gentle 4 mile stroll' | 1/5 returned routes violate hard constraints |
| dist-004 | distance | atlas-route-engine | 'half marathon trail' | 5/5 returned routes violate hard constraints; no must_include_any route in top 5 |
| dist-005 | distance | atlas-nlq | 'a flat five mile hike' | 4/5 returned routes violate hard constraints |
| elev-003 | elevation | atlas-nlq | '3 mile hike with less than 500 ft of elevation gai...' | 5/5 returned routes violate hard constraints; no must_include_any route in top 5 |
| elev-004 | elevation | atlas-nlq | '6 mile hike with at least 2000 ft of climbing' | 3/5 returned routes violate hard constraints; no must_include_any route in top 5 |
| prox-003 | proximity | atlas-nlq | 'hike within 5 miles of me' | 3/5 returned routes violate hard constraints |
