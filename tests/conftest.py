"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.fixtures import build_stub_llm, load_sample  # noqa: E402


@pytest.fixture
def stub_llm():
    return build_stub_llm()


@pytest.fixture
def sample_text():
    """Function fixture: load a sample by relative path."""
    return load_sample
