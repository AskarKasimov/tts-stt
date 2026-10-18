import pytest

import api


@pytest.fixture(autouse=True)
def no_background_model_downloads(monkeypatch):
    """HTTP unit tests supply fake models explicitly; never download weights."""
    monkeypatch.setattr(api, "_start_runtime", lambda: None)
