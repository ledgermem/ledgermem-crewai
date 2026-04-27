# ledgermem-crewai

CrewAI long-term memory provider backed by [LedgerMem](https://github.com/ledgermem/ledgermem-python). Give your crew persistent, searchable memory across runs in three lines.

## Install

```bash
pip install ledgermem-crewai
```

## Quickstart

```python
from crewai import Agent, Crew, Task
from crewai.memory import LongTermMemory
from ledgermem import LedgerMem
from ledgermem_crewai import LedgerMemLongTermMemory

mem = LedgerMemLongTermMemory(
    client=LedgerMem(api_key="lm_...", workspace_id="ws_..."),
    agent_id="researcher",
)

researcher = Agent(role="researcher", goal="...", backstory="...")
task = Task(description="Find recent RAG benchmarks", agent=researcher)

crew = Crew(
    agents=[researcher],
    tasks=[task],
    memory=True,
    long_term_memory=LongTermMemory(storage=mem),
)

crew.kickoff()
```

## Direct API

```python
mem.save("The user prefers async APIs.", metadata={"task": "preferences"})
hits = mem.search("what does the user prefer?", limit=5, score_threshold=0.6)
for hit in hits:
    print(hit["score"], hit["context"])
mem.reset()  # wipe the workspace
```

## License

MIT — see [LICENSE](./LICENSE).
