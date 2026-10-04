import json

import pytest

from tests.benchmark import harness as H


@pytest.fixture(scope="session")
def benchmark_results():
    return {}


@pytest.fixture(scope="session", autouse=True)
def _store(benchmark_results, request):
    request.config._benchmark_results = benchmark_results
    yield


def pytest_configure(config):
    config.addinivalue_line("markers", "regression: known-bug regression case")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    results = list(getattr(config, "_benchmark_results", {}).values())
    if not results:
        return
    from tests.benchmark.report import metrics_table
    s = H.summarize(results)
    cats = H.by_category(results)
    tr = terminalreporter
    tr.write_sep("=", "recommendation benchmark metrics")
    for line in metrics_table(s):
        tr.write_line(line)
    tr.write_line("")
    for k, c in cats.items():
        tr.write_line(f"{k:28s} pass {c['passed']:>2}/{c['cases']:<2}  violation {c['violation_cases']:<6} empty-feasible {c['empty_cases']:<6} hit@5 {c['hit_cases']}")
    H.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (H.REPORTS_DIR / "last_pytest_run.json").write_text(json.dumps({"summary": s, "by_category": cats}, indent=2), encoding="utf-8")
