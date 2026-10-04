import os

import pytest

from tests.benchmark import harness as H

pytestmark = pytest.mark.llm

ENABLED = os.environ.get("ATLAS_BENCH_LLM") == "1" and bool(os.environ.get("OPENAI_API_KEY"))

REGRESSION = [c for c in H.load_cases() if "regression" in (c.get("tags") or [])]


@pytest.mark.skipif(not ENABLED, reason="set ATLAS_BENCH_LLM=1 and OPENAI_API_KEY to run the LLM-on suite")
@pytest.mark.parametrize("case", [pytest.param(c, id=c["id"]) for c in REGRESSION])
def test_regression_case_llm_on(case):
    result = H.evaluate_case(case)
    assert result["passed"], "; ".join(result["failures"])
