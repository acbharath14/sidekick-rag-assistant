"""Shared test isolation for the API suite.

The rate limiter is process-wide and the TestClient always comes from the
same client IP, so without a reset between tests the suite's own earlier
upload/chat hits would 429 later tests once a bucket fills (the upload
bucket is 10/min by default). Reset the limiter before (and after) every
test so no test starts with a dirty bucket. Within-test accumulation is
untouched: the rate-limit tests still prove the limits with their own
`_limited` helper in test_api.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from rag_demo import rate_limit as _rate_limit


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    _rate_limit.reset_rate_limiter()
    yield
    _rate_limit.reset_rate_limiter()
