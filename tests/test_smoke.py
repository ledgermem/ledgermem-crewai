"""Smoke test: the CrewAI memory provider talks to a mocked LedgerMem SDK."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _install_fake_ledgermem() -> None:
    if "ledgermem" in sys.modules:
        return
    fake = types.ModuleType("ledgermem")

    class LedgerMem:
        def __init__(self, *a, **k):
            pass

        def search(self, query, limit=5):
            return types.SimpleNamespace(hits=[])

        def add(self, content, metadata=None):
            return types.SimpleNamespace(id="mem_1")

        def delete(self, memory_id):
            return None

        def list(self, limit=20, cursor=None):
            return types.SimpleNamespace(items=[], next_cursor=None)

    class AsyncLedgerMem(LedgerMem):
        pass

    fake.LedgerMem = LedgerMem
    fake.AsyncLedgerMem = AsyncLedgerMem
    sys.modules["ledgermem"] = fake


_install_fake_ledgermem()

from ledgermem import LedgerMem  # noqa: E402
from ledgermem_crewai import LedgerMemLongTermMemory  # noqa: E402


def test_imports() -> None:
    assert LedgerMemLongTermMemory is not None


def test_save_attaches_metadata() -> None:
    client = LedgerMem()
    client.add = MagicMock(return_value=None)
    memory = LedgerMemLongTermMemory(client=client, agent_id="researcher")
    memory.save("Found a paper on RAG.", metadata={"task": "research"})
    args, kwargs = client.add.call_args
    assert args[0] == "Found a paper on RAG."
    assert kwargs["metadata"]["agent_id"] == "researcher"
    assert kwargs["metadata"]["task"] == "research"
    assert kwargs["metadata"]["source"] == "crewai"


def test_search_threshold_filter() -> None:
    client = LedgerMem()
    hits = [
        type("Hit", (), {"id": "a", "content": "high", "metadata": {}, "score": 0.9})(),
        type("Hit", (), {"id": "b", "content": "low", "metadata": {}, "score": 0.2})(),
    ]
    client.search = MagicMock(return_value=type("R", (), {"hits": hits})())
    memory = LedgerMemLongTermMemory(client=client)
    results = memory.search("anything", limit=5, score_threshold=0.5)
    assert len(results) == 1
    assert results[0]["id"] == "a"
