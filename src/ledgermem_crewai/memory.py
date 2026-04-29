"""CrewAI long-term memory provider backed by Mnemo.

CrewAI memory backends expose three primary methods used by the framework:
``save``, ``search``, and ``reset``. We implement that contract directly so the
class can be plugged into any Crew via ``Crew(memory=True, long_term_memory=...)``
or used as a standalone ``MemoryStorage`` replacement.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from getmnemo import Mnemo


class MnemoLongTermMemory:
    """A CrewAI-compatible long-term memory provider."""

    def __init__(self, client: Mnemo, agent_id: str | None = None) -> None:
        self._client = client
        self._agent_id = agent_id

    # --- CrewAI memory interface ------------------------------------------

    def save(
        self,
        value: Any,
        metadata: dict[str, Any] | None = None,
        agent: str | None = None,
    ) -> None:
        """Persist ``value`` to Mnemo with optional metadata."""
        # Caller metadata is merged FIRST so trusted server-controlled fields
        # (source, agent_id, saved_at) cannot be overwritten by untrusted input.
        merged: dict[str, Any] = dict(metadata) if metadata else {}
        merged["source"] = "crewai"
        merged["saved_at"] = datetime.now(timezone.utc).isoformat()
        agent_id = agent or self._agent_id
        if agent_id:
            merged["agent_id"] = agent_id
        # Serialize dict/list values as JSON so they round-trip through
        # search results — str(dict) produces "{'k': 'v'}" with single
        # quotes, which is not valid JSON and breaks any downstream parser.
        if isinstance(value, str):
            content = value
        elif isinstance(value, (dict, list)):
            content = json.dumps(value, default=str)
        else:
            content = str(value)
        self._client.add(content, metadata=merged)

    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search returning CrewAI-shaped result dicts."""
        # When a score_threshold is set we have to ask the server for more
        # than `limit` rows because the threshold filter runs *after*
        # retrieval — without over-fetching we'd return fewer than `limit`
        # eligible hits whenever the bottom of the page is below threshold.
        fetch_limit = max(limit * 4, 20) if score_threshold is not None else limit
        response = self._client.search(query, limit=fetch_limit)
        out: list[dict[str, Any]] = []
        for hit in getattr(response, "hits", []) or []:
            score = getattr(hit, "score", None)
            # When a caller asks for a score threshold they want a quality
            # gate — a hit without any score should be dropped, not
            # silently let through. The previous ``score is not None``
            # guard meant any backend that omitted scores returned every
            # row regardless of the threshold.
            if score_threshold is not None:
                if score is None or score < score_threshold:
                    continue
            out.append(
                {
                    "id": getattr(hit, "id", None),
                    "context": getattr(hit, "content", None) or getattr(hit, "text", ""),
                    "metadata": dict(getattr(hit, "metadata", {}) or {}),
                    "score": score,
                }
            )
            if len(out) >= limit:
                break
        return out

    def reset(self) -> None:
        """Delete every memory in the workspace this client points at."""
        # Snapshot every id BEFORE deleting. Deleting during pagination
        # mutates the underlying collection — depending on the backend this
        # either skips rows (server-side cursors that compact) or loops
        # forever (offset-based cursors that re-shift on delete).
        ids: list[str] = []
        cursor: str | None = None
        while True:
            page = self._client.list(limit=100, cursor=cursor)
            items = getattr(page, "items", []) or getattr(page, "memories", []) or []
            for item in items:
                memory_id = getattr(item, "id", None)
                if memory_id is not None:
                    ids.append(memory_id)
            cursor = getattr(page, "next_cursor", None)
            if not cursor:
                break
        for memory_id in ids:
            self._client.delete(memory_id)
