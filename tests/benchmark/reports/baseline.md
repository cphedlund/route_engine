# Recommendation benchmark report

Generated 2026-10-04 00:15 UTC. Engine path: rules-based NLQ (LLM disabled), top 5 per query. Flat = gain <= max(300 ft, 60 ft/mi x route distance).

| Metric | Value | Cases | Target |
|---|---|---|---|
| Constraint-violation rate | 32.1% | 26/81 | 0% |
| Empty-when-feasible rate | 6.5% | 6/93 | 0% |
| Hit rate @5 | 84.1% | 53/63 | maximize |
| Latency p50 / p95 | 1.5 ms / 2.9 ms | 100 | report |
| Cases passed | 61/100 | | |

## Per category

| Category | Pass/Total | Violation | Empty-when-feasible | Hit@5 | p50 ms |
|---|---|---|---|---|---|
| bike | 5/5 | 0/5 | 0/5 | 4/4 | 1.3 |
| combined | 4/6 | 2/6 | 0/6 | 4/4 | 1.3 |
| distance | 4/6 | 2/6 | 0/6 | 4/5 | 1.7 |
| dog | 2/5 | 3/5 | 0/5 | 3/3 | 1.6 |
| elevation | 2/4 | 2/4 | 0/4 | 2/4 | 1.6 |
| flat | 2/5 | 3/5 | 0/5 | 5/5 | 1.6 |
| infeasible | 0/6 | 0/0 | 0/0 | 0/0 | 1.2 |
| location | 0/1 | 1/1 | 0/1 | 1/1 | 1.8 |
| nonsense | 12/12 | 0/0 | 0/12 | 0/0 | 2.8 |
| park | 2/2 | 0/2 | 0/2 | 2/2 | 1.2 |
| park-alias | 15/17 | 2/17 | 0/17 | 13/13 | 1.2 |
| proximity | 3/5 | 1/4 | 0/4 | 2/2 | 1.7 |
| regression-false-empty | 0/6 | 0/6 | 6/6 | 0/4 | 1.4 |
| regression-flat-elevation | 5/11 | 6/11 | 0/11 | 6/9 | 1.7 |
| shape | 4/4 | 0/4 | 0/4 | 4/4 | 1.3 |
| wheelchair | 1/5 | 4/5 | 0/5 | 3/3 | 1.2 |

## Failing cases

| Case | Category | Owner | Query | Failure |
|---|---|---|---|---|
| reg-empty-001 | regression-false-empty | atlas-route-engine | 'flat 5 mile hike' | empty result but 5 feasible routes exist; no must_include_any route in top 5 |
| reg-empty-002 | regression-false-empty | atlas-route-engine | 'hike near me' | empty result but 143 feasible routes exist |
| reg-empty-003 | regression-false-empty | atlas-route-engine | 'dog friendly 3 mile hike' | empty result but 49 feasible routes exist; no must_include_any route in top 5 |
| reg-empty-004 | regression-false-empty | atlas-route-engine | 'wheelchair accessible walk' | empty result but 36 feasible routes exist |
| reg-empty-005 | regression-false-empty | atlas-route-engine | '5 mile loop' | empty result but 30 feasible routes exist; no must_include_any route in top 5 |
| reg-empty-006 | regression-false-empty | atlas-route-engine | 'flat 3 mile walk' | empty result but 8 feasible routes exist; no must_include_any route in top 5 |
| loc-001 | location | atlas-route-engine | 'flat 5 mile hike' | 2/5 returned routes violate hard constraints |
| reg-flat-004 | regression-flat-elevation | atlas-route-engine | 'flat 5 mile trail run' | 2/5 returned routes violate hard constraints |
| reg-flat-005 | regression-flat-elevation | atlas-route-engine | 'nice flat 5 mi walk with views' | 1/5 returned routes violate hard constraints |
| reg-flat-006 | regression-flat-elevation | atlas-nlq | '5 mile hike with no hills' | 5/5 returned routes violate hard constraints; no must_include_any route in top 5 |
| reg-flat-007 | regression-flat-elevation | atlas-nlq | '5 mile hike with minimal elevation' | 5/5 returned routes violate hard constraints; no must_include_any route in top 5 |
| reg-flat-008 | regression-flat-elevation | atlas-nlq | '5 mile hike under 300 ft gain' | 5/5 returned routes violate hard constraints; no must_include_any route in top 5 |
| reg-flat-010 | regression-flat-elevation | atlas-route-engine | 'flat 3 mile walk' | 3/6 returned routes violate hard constraints |
| flat-002 | flat | atlas-route-engine | 'flat 8 mile hike' | 1/5 returned routes violate hard constraints |
| flat-004 | flat | atlas-route-engine | '10k flat run' | 2/5 returned routes violate hard constraints |
| flat-005 | flat | atlas-nlq | 'gentle 4 mile stroll' | 1/5 returned routes violate hard constraints |
| dist-004 | distance | atlas-route-engine | 'half marathon trail' | 5/5 returned routes violate hard constraints; no must_include_any route in top 5 |
| dist-005 | distance | atlas-nlq | 'a flat five mile hike' | 4/5 returned routes violate hard constraints |
| elev-003 | elevation | atlas-nlq | '3 mile hike with less than 500 ft of elevation gai...' | 5/5 returned routes violate hard constraints; no must_include_any route in top 5 |
| elev-004 | elevation | atlas-nlq | '6 mile hike with at least 2000 ft of climbing' | 3/5 returned routes violate hard constraints; no must_include_any route in top 5 |
| dog-002 | dog | atlas-route-engine | 'dogs allowed 8 mile hike' | 1/5 returned routes violate hard constraints |
| dog-004 | dog | atlas-route-engine | 'hike with my dog at Almaden Quicksilver' | 1/5 returned routes violate hard constraints |
| dog-005 | dog | atlas-route-engine | 'dog friendly hike' | 3/5 returned routes violate hard constraints |
| wheel-001 | wheelchair | atlas-route-engine | 'wheelchair accessible walk' | 5/5 returned routes violate hard constraints |
| wheel-002 | wheelchair | atlas-route-engine | 'wheelchair accessible 3 mile' | 2/5 returned routes violate hard constraints |
| wheel-004 | wheelchair | atlas-route-engine | 'paved wheelchair accessible 5 mile flat' | 2/5 returned routes violate hard constraints |
| wheel-005 | wheelchair | atlas-route-engine | 'ADA accessible trail' | 5/5 returned routes violate hard constraints |
| park-017 | park-alias | atlas-route-engine | 'hike at Sierra Azul' | 3/5 returned routes violate hard constraints |
| park-018 | park-alias | atlas-route-engine | 'hike at Castle Rock State Park' | 5/5 returned routes violate hard constraints |
| prox-003 | proximity | atlas-nlq | 'hike within 5 miles of me' | 3/5 returned routes violate hard constraints |
| prox-005 | proximity | atlas-route-engine | 'hike near me' | infeasible query: expected relaxed results plus an explanation field (got 0 routes, explanation=None, keys=['has_more', 'min_conformity', 'progressive_relax_lev |
| comb-002 | combined | atlas-route-engine | 'flat 4 mile wheelchair accessible walk' | 4/5 returned routes violate hard constraints |
| comb-003 | combined | atlas-route-engine | 'flat 6 mile bike ride with my dog' | 1/5 returned routes violate hard constraints |
| inf-001 | infeasible | atlas-route-engine | 'bike ride at Almaden Quicksilver' | infeasible query: expected relaxed results plus an explanation field (got 0 routes, explanation=None, keys=['has_more', 'min_conformity', 'progressive_relax_lev |
| inf-002 | infeasible | atlas-route-engine | 'wheelchair accessible walk at Uvas Canyon' | infeasible query: expected relaxed results plus an explanation field (got 0 routes, explanation=None, keys=['has_more', 'min_conformity', 'progressive_relax_lev |
| inf-003 | infeasible | atlas-route-engine | 'flat 50 mile hike' | infeasible query: expected relaxed results plus an explanation field (got 5 routes, explanation=None, keys=['has_more', 'min_conformity', 'progressive_relax_lev |
| inf-004 | infeasible | atlas-route-engine | 'flat 10 mile hike at Sanborn' | infeasible query: expected relaxed results plus an explanation field (got 5 routes, explanation=None, keys=['has_more', 'min_conformity', 'progressive_relax_lev |
| inf-005 | infeasible | atlas-route-engine | 'hike at Anderson Lake' | infeasible query: expected relaxed results plus an explanation field (got 0 routes, explanation=None, keys=['has_more', 'min_conformity', 'progressive_relax_lev |
| inf-006 | infeasible | atlas-route-engine | '15 mile wheelchair accessible flat walk with dog' | case marked infeasible but 1 routes satisfy hard constraints (bad case); infeasible query: expected relaxed results plus an explanation field (got 5 routes, exp |
