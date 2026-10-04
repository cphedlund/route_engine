import os

import pytest
import requests

import app as appmod
import atlas_mapbox


def test_no_mapbox_token_in_tests():
    assert os.environ.get("MAPBOX_TOKEN", "") == ""


def test_http_is_blocked_in_tests():
    with pytest.raises(requests.ConnectionError):
        atlas_mapbox.requests.get("https://api.mapbox.com/")


def test_pdf_cache_default_is_16_mb():
    assert appmod.MAP_PDF_CACHE_MAX_BYTES == 16 * 1024 * 1024 or "MAP_PDF_CACHE_MAX_BYTES" in os.environ
