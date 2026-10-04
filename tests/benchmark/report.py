from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.benchmark import harness as H

TARGETS = {"constraint_violation_rate": 0.0, "empty_when_feasible_rate": 0.0}


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def metrics_table(summary: Dict[str, Any]) -> List[str]:
    return [
        "| Metric | Value | Cases | Target |",
        "|---|---|---|---|",
        f"| Constraint-violation rate | {_pct(summary['constraint_violation_rate'])} | {summary['violation_cases']} | 0% |",
        f"| Empty-when-feasible rate | {_pct(summary['empty_when_feasible_rate'])} | {summary['empty_cases']} | 0% |",
        f"| Hit rate @5 | {_pct(summary['hit_rate_at_5'])} | {summary['hit_cases']} | maximize |",
        f"| Latency p50 / p95 | {summary['latency_p50_ms']} ms / {summary['latency_p95_ms']} ms | {summary['cases']} | report |",
        f"| Cases passed | {summary['passed']}/{summary['cases']} | | |",
    ]


def render_markdown(results: List[Dict[str, Any]]) -> str:
    summary = H.summarize(results)
    cats = H.by_category(results)
    out: List[str] = []
    out.append("# Recommendation benchmark report")
    out.append("")
    out.append(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Engine path: rules-based NLQ (LLM disabled), top {H.TOP_K} per query. Flat = gain <= max({H.FLAT_MAX_GAIN_FT:.0f} ft, {H.FLAT_GAIN_FT_PER_MILE:.0f} ft/mi x route distance).")
    out.append("")
    out.extend(metrics_table(summary))
    out.append("")
    out.append("## Per category")
    out.append("")
    out.append("| Category | Pass/Total | Violation | Empty-when-feasible | Hit@5 | p50 ms |")
    out.append("|---|---|---|---|---|---|")
    for k, s in cats.items():
        out.append(f"| {k} | {s['passed']}/{s['cases']} | {s['violation_cases']} | {s['empty_cases']} | {s['hit_cases']} | {s['latency_p50_ms']} |")
    out.append("")
    out.append("## Failing cases")
    out.append("")
    out.append("| Case | Category | Owner | Query | Failure |")
    out.append("|---|---|---|---|---|")
    for r in results:
        if not r["passed"]:
            q = (r["query"][:50] + "...") if len(r["query"]) > 50 else r["query"]
            out.append(f"| {r['id']} | {r['category']} | {r['owner']} | {q!r} | {'; '.join(r['failures'])[:160]} |")
    return "\n".join(out) + "\n"


def main() -> int:
    results = H.run_all()
    H.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    name = sys.argv[1] if len(sys.argv) > 1 else "latest"
    (H.REPORTS_DIR / f"{name}.json").write_text(json.dumps({"summary": H.summarize(results), "by_category": H.by_category(results), "results": results}, indent=2, default=str), encoding="utf-8")
    md = render_markdown(results)
    (H.REPORTS_DIR / f"{name}.md").write_text(md, encoding="utf-8")
    sys.stdout.buffer.write(md.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
