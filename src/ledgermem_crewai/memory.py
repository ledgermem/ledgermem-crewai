"""CrewAI long-term memory provider backed by LedgerMem.

CrewAI memory backends expose three primary methods used by the framework:
``save``, ``search``, and ``reset``. We implement that contract directly so the
class can be plugged into any Crew via ``Crew(memory=True, long_term_memory=...)``
or used as a standalone ``MemoryStorage`` replacement.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ledgermem import LedgerMem


class LedgerMemLongTermMemory:
    """A CrewAI-compatible long-term memory provider."""

    def __init__(self, client: LedgerMem, agent_id: str | None = None) -> None:
        self._client = client
        self._agent_id = agent_id

    # --- CrewAI memory interface ------------------------------------------

    def save(
        self,
        value: Any,
        metadata: dict[str, Any] | None = None,
        agent: str | None = None,
    ) -> None:
        """Persist ``value`` to LedgerMem with optional metadata."""
        # Caller metadata is merged FIRST so trusted server-controlled fields
        # (source, agent_id, saved_at) cannot be overwritten by untrusted input.
        merged: dict[str, Any] = dict(metadata) if metadata else {}
        merged["source"] = "crewai"
        merged["saved_at"] = datetime.now(timezone.utc).isoformat()
        agent_id = agent or self._agent_id
        if agent_id:
            merged["agent_id"] = agent_id
        content = value if isinstance(value, str) else str(value)
        self._client.add(content, metadata=merged)

    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search returning CrewAI-shaped result dicts."""
        response = self._client.search(query, limit=limit)
        out: list[dict[str, Any]] = []
        for hit in getattr(response, "hits", []) or []:
            score = getattr(hit, "score", None)
            if score_threshold is not None and score is not None and score < score_threshold:
                continue
            out.append(
                {
                    "id": getattr(hit, "id", None),
                    "context": getattr(hit, "content", None) or getattr(hit, "text", ""),
                    "metadata": dict(getattr(hit, "metadata", {}) or {}),
                    "score": score,
                }
            )
        return out

    def reset(self) -> None:
        """Delete every memory in the workspace this client points at."""
        cursor: str | None = None
        while True:
            page = self._client.list(limit=100, cursor=cursor)
            items = getattr(page, "items", []) or getattr(page, "memories", []) or []
            for item in items:
                memory_id = getattr(item, "id", None)
                if memory_id is not None:
                    self._client.delete(memory_id)
            cursor = getattr(page, "next_cursor", None)
            if not cursor:
                return
