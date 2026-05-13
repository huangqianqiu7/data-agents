# RAG 与记忆模块架构 v2 实施计划（M0–M3）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 `2026-05-13-rag-memory-architecture-design-v2.md` 落地结构化 Memory MVP（`dataset_knowledge` + `tool_playbook`），并为 Corpus RAG 预留 `memory/rag/` 占位边界。

**Architecture:** Store / Retriever / Writer 三层解耦；`AppConfig` 新增 `MemoryConfig`；`RunState` 仅新增 `memory_hits: Annotated[list[MemoryHit], operator.add]`；`tool_node` 成功后写、`planner_node` / `model_node` 入口召回；`prompts.py` 白名单渲染；`observability.events.dispatch_observability_event("memory_recall", ...)` 审计。

**Tech Stack:** Python 3.11+，dataclasses（`frozen=True, slots=True`），LangGraph 0.4，LangChain core，pytest，typer CLI，JSONL append-only。

---

## File Structure

新增文件：

- `src/data_agent_langchain/memory/base.py` —— Protocol、`MemoryRecord`、`RetrievalResult`。
- `src/data_agent_langchain/memory/records.py` —— `DatasetKnowledgeRecord`、`ToolPlaybookRecord`。
- `src/data_agent_langchain/memory/types.py` —— `MemoryHit`（写入 `RunState`）。
- `src/data_agent_langchain/memory/stores/__init__.py`
- `src/data_agent_langchain/memory/stores/jsonl.py` —— `JsonlMemoryStore`。
- `src/data_agent_langchain/memory/retrievers/__init__.py`
- `src/data_agent_langchain/memory/retrievers/exact.py` —— `ExactNamespaceRetriever`。
- `src/data_agent_langchain/memory/writers/__init__.py`
- `src/data_agent_langchain/memory/writers/store_backed.py` —— `StoreBackedMemoryWriter`。
- `src/data_agent_langchain/memory/factory.py` —— `build_store` / `build_retriever` / `build_writer`。
- `src/data_agent_langchain/memory/rag/__init__.py` —— RAG 占位，仅抛 `NotImplementedError`。
- 测试文件均放在 `tests/` 顶层，沿用项目现有 `test_phaseX_*` 命名约定。

修改文件：

- `src/data_agent_langchain/memory/__init__.py` —— 新增导出。
- `src/data_agent_langchain/config.py` —— 新增 `MemoryConfig`，挂到 `AppConfig`，`to_dict`/`from_dict` 加分支。
- `src/data_agent_langchain/runtime/state.py` —— `RunState` 新增 `memory_hits` 字段。
- `src/data_agent_langchain/agents/prompts.py` —— 新增 `render_dataset_facts`，在 `build_*_messages` 中可选追加 facts 段。
- `src/data_agent_langchain/agents/tool_node.py` —— 成功路径写 `DatasetKnowledgeRecord` / `ToolPlaybookRecord`。
- `src/data_agent_langchain/agents/planner_node.py` —— 召回 + 注入 + `memory_recall` 事件。
- `src/data_agent_langchain/agents/model_node.py` —— 同上（仅 ReAct / execution 阶段）。
- `src/data_agent_langchain/cli.py` —— 新增 `--memory-mode` 选项。
- `src/data_agent_langchain/submission.py` —— 评测路径默认值。

---

## Conventions（所有任务通用）

- 每个 TDD 步骤必须按 RED → GREEN → COMMIT 推进；不要把多个测试合并到同一次提交。
- 测试名都用 `tests/test_phase9_memory_*.py` 前缀（沿用项目 `test_phaseX_*` 模式；Memory 视作 Phase 9）。
- 提交信息格式：`feat(memory): <task>` / `test(memory): <task>` / `refactor(memory): <task>`。
- 跑测试：`pytest tests/test_phase9_memory_*.py -v`；想跑单测：`pytest tests/test_phase9_memory_records.py::test_name -v`。
- 任何 dataclass 新增字段都保持 `frozen=True, slots=True` 与 picklable。
- 不允许在写入路径接收 `question` / `answer` / `approach` / `hint` / `summary` 等字段。

---

# Milestone M0：协议与 record 类型

## Task M0.1：`MemoryRecord` 与 `RetrievalResult`

**Files:**
- Create: `src/data_agent_langchain/memory/base.py`
- Test: `tests/test_phase9_memory_base.py`

- [ ] **Step 1: 写失败测试**

`tests/test_phase9_memory_base.py`：

```python
from datetime import datetime

import pytest

from data_agent_langchain.memory.base import MemoryRecord, RetrievalResult


def test_memory_record_is_frozen_and_picklable():
    import pickle

    rec = MemoryRecord(
        id="dk:ds:transactions.csv",
        namespace="dataset:ds",
        kind="dataset_knowledge",
        payload={"file_path": "transactions.csv"},
    )
    # frozen
    with pytest.raises(Exception):
        rec.id = "x"  # type: ignore[misc]
    # picklable
    assert pickle.loads(pickle.dumps(rec)) == rec
    assert isinstance(rec.created_at, datetime)


def test_retrieval_result_carries_reason():
    rec = MemoryRecord(
        id="x", namespace="dataset:ds", kind="dataset_knowledge", payload={}
    )
    res = RetrievalResult(record=rec, score=1.0, reason="exact_namespace")
    assert res.reason == "exact_namespace"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_phase9_memory_base.py -v`
Expected: FAIL（`ImportError: cannot import name 'MemoryRecord'`）

- [ ] **Step 3: 最小实现**

`src/data_agent_langchain/memory/base.py`：

```python
"""Memory 层协议与基础数据结构。"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable


RecordKind = Literal["working", "dataset_knowledge", "tool_playbook", "corpus"]


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """跨 task 记忆的通用 KV 信封。"""
    id: str
    namespace: str
    kind: RecordKind
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    record: MemoryRecord
    score: float
    reason: str


@runtime_checkable
class MemoryStore(Protocol):
    def put(self, record: MemoryRecord) -> None: ...
    def get(self, namespace: str, record_id: str) -> MemoryRecord | None: ...
    def list(self, namespace: str, *, limit: int = 100) -> list[MemoryRecord]: ...
    def delete(self, namespace: str, record_id: str) -> None: ...


@runtime_checkable
class Retriever(Protocol):
    def retrieve(
        self, query: str, *, namespace: str, k: int = 5
    ) -> list[RetrievalResult]: ...


@runtime_checkable
class MemoryWriter(Protocol):
    def write_dataset_knowledge(self, dataset: str, record: Any) -> None: ...
    def write_tool_playbook(
        self, dataset: str, tool_name: str, record: Any
    ) -> None: ...


__all__ = [
    "MemoryRecord",
    "MemoryStore",
    "MemoryWriter",
    "RecordKind",
    "Retriever",
    "RetrievalResult",
]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_phase9_memory_base.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/data_agent_langchain/memory/base.py tests/test_phase9_memory_base.py
git commit -m "feat(memory): add MemoryRecord, RetrievalResult, and Protocols"
```

---

## Task M0.2：`DatasetKnowledgeRecord` 与 `ToolPlaybookRecord`

**Files:**
- Create: `src/data_agent_langchain/memory/records.py`
- Test: `tests/test_phase9_memory_records.py`

- [ ] **Step 1: 写失败测试**

`tests/test_phase9_memory_records.py`：

```python
import pytest

from data_agent_langchain.memory.records import (
    DatasetKnowledgeRecord,
    ToolPlaybookRecord,
)


def test_dataset_knowledge_record_fields():
    rec = DatasetKnowledgeRecord(
        file_path="transactions.csv",
        file_kind="csv",
        schema={"date": "string", "amount": "float"},
        row_count_estimate=12450,
    )
    assert rec.file_path == "transactions.csv"
    assert rec.encoding is None
    assert rec.sample_columns == []


def test_dataset_knowledge_record_rejects_freetext_fields():
    """字段层面禁止 question / answer / approach / hint / summary。"""
    with pytest.raises(TypeError):
        DatasetKnowledgeRecord(  # type: ignore[call-arg]
            file_path="x.csv",
            file_kind="csv",
            schema={},
            row_count_estimate=None,
            question="leaked",
        )


def test_tool_playbook_record_fields():
    rec = ToolPlaybookRecord(
        tool_name="read_csv",
        input_template={"max_rows": 20},
        preconditions=["preview_done"],
    )
    assert rec.typical_failures == []


def test_tool_playbook_record_rejects_freetext_fields():
    with pytest.raises(TypeError):
        ToolPlaybookRecord(  # type: ignore[call-arg]
            tool_name="read_csv",
            input_template={},
            preconditions=[],
            example_answer="leaked",
        )
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_phase9_memory_records.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 最小实现**

`src/data_agent_langchain/memory/records.py`：

```python
"""跨 task 结构化记忆 record 类型。

字段经过审计，刻意不含 question / answer / approach / hint / summary 等
自由文本字段；如需扩展必须修改本文件并通过 PR review。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


FileKind = Literal["csv", "json", "doc", "sqlite", "other"]


@dataclass(frozen=True, slots=True)
class DatasetKnowledgeRecord:
    file_path: str
    file_kind: FileKind
    schema: dict[str, str]
    row_count_estimate: int | None
    encoding: str | None = None
    sample_columns: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ToolPlaybookRecord:
    tool_name: str
    input_template: dict[str, Any]
    preconditions: list[str]
    typical_failures: list[str] = field(default_factory=list)


__all__ = ["DatasetKnowledgeRecord", "FileKind", "ToolPlaybookRecord"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_phase9_memory_records.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/data_agent_langchain/memory/records.py tests/test_phase9_memory_records.py
git commit -m "feat(memory): add DatasetKnowledgeRecord and ToolPlaybookRecord with whitelisted fields"
```

---

## Task M0.3：`MemoryHit` 轻量摘要

**Files:**
- Create: `src/data_agent_langchain/memory/types.py`
- Test: `tests/test_phase9_memory_hit.py`

- [ ] **Step 1: 写失败测试**

```python
import pickle

from data_agent_langchain.memory.types import MemoryHit


def test_memory_hit_is_picklable_and_frozen():
    hit = MemoryHit(
        record_id="dk:ds:x.csv",
        namespace="dataset:ds",
        score=1.0,
        summary="File: x.csv  Columns: ['a']",
    )
    assert pickle.loads(pickle.dumps(hit)) == hit
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_phase9_memory_hit.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 最小实现**

`src/data_agent_langchain/memory/types.py`：

```python
"""写入 RunState 的轻量摘要类型。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MemoryHit:
    record_id: str
    namespace: str
    score: float
    summary: str


__all__ = ["MemoryHit"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_phase9_memory_hit.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/data_agent_langchain/memory/types.py tests/test_phase9_memory_hit.py
git commit -m "feat(memory): add MemoryHit summary type"
```

---

# Milestone M1：JSONL 后端、Retriever、Writer、Factory、`MemoryConfig`

## Task M1.1：`MemoryConfig` 接入 `AppConfig`

**Files:**
- Modify: `src/data_agent_langchain/config.py`
- Test: `tests/test_phase9_memory_config.py`

- [ ] **Step 1: 写失败测试**

```python
from data_agent_langchain.config import AppConfig, MemoryConfig, default_app_config


def test_memory_config_defaults_safe_for_eval():
    cfg = MemoryConfig()
    assert cfg.mode == "disabled"
    assert cfg.store_backend == "jsonl"
    assert cfg.retriever_type == "exact"
    assert cfg.retrieval_max_results == 5


def test_app_config_round_trip_with_memory():
    cfg = default_app_config()
    assert isinstance(cfg.memory, MemoryConfig)
    payload = cfg.to_dict()
    assert payload["memory"]["mode"] == "disabled"
    restored = AppConfig.from_dict(payload)
    assert restored.memory == cfg.memory


def test_memory_config_picklable():
    import pickle

    cfg = MemoryConfig(mode="full")
    assert pickle.loads(pickle.dumps(cfg)) == cfg
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_phase9_memory_config.py -v`
Expected: FAIL（`ImportError: cannot import name 'MemoryConfig'`）

- [ ] **Step 3: 修改 `config.py`**

在 `src/data_agent_langchain/config.py` 中，紧邻 `EvaluationConfig` 之后新增 `MemoryConfig`：

```python
@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """跨 task 记忆配置（v2 设计 §4.2）。"""
    mode: str = "disabled"                          # disabled | read_only_dataset | full
    store_backend: str = "jsonl"                    # jsonl | sqlite (后续)
    path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "artifacts" / "memory"
    )
    retriever_type: str = "exact"
    retrieval_max_results: int = 5
```

修改 `AppConfig`（同文件）：

```python
@dataclass(frozen=True, slots=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    run: RunConfig = field(default_factory=lambda: RunConfig())
    observability: ObservabilityConfig = field(default_factory=lambda: ObservabilityConfig())
    evaluation: EvaluationConfig = field(default_factory=lambda: EvaluationConfig())
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain_dict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AppConfig":
        return cls(
            dataset=_dataclass_from_dict(DatasetConfig, payload.get("dataset", {})),
            agent=_dataclass_from_dict(AgentConfig, payload.get("agent", {})),
            tools=_dataclass_from_dict(ToolsConfig, payload.get("tools", {})),
            run=_dataclass_from_dict(RunConfig, payload.get("run", {})),
            observability=_dataclass_from_dict(
                ObservabilityConfig, payload.get("observability", {})
            ),
            evaluation=_dataclass_from_dict(
                EvaluationConfig, payload.get("evaluation", {})
            ),
            memory=_dataclass_from_dict(MemoryConfig, payload.get("memory", {})),
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_phase9_memory_config.py tests/test_phase5_config.py -v`
Expected: PASS（包括既有 config 测试）

- [ ] **Step 5: 提交**

```bash
git add src/data_agent_langchain/config.py tests/test_phase9_memory_config.py
git commit -m "feat(memory): add MemoryConfig to AppConfig with safe defaults"
```

---

## Task M1.2：`JsonlMemoryStore`

**Files:**
- Create: `src/data_agent_langchain/memory/stores/__init__.py`
- Create: `src/data_agent_langchain/memory/stores/jsonl.py`
- Test: `tests/test_phase9_memory_jsonl_store.py`

- [ ] **Step 1: 写失败测试**

```python
from pathlib import Path

from data_agent_langchain.memory.base import MemoryRecord
from data_agent_langchain.memory.stores.jsonl import JsonlMemoryStore


def test_put_then_list(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    rec = MemoryRecord(
        id="dk:ds:a.csv",
        namespace="dataset:ds",
        kind="dataset_knowledge",
        payload={"file_path": "a.csv"},
    )
    store.put(rec)
    results = store.list("dataset:ds", limit=5)
    assert len(results) == 1
    assert results[0].id == "dk:ds:a.csv"
    assert results[0].payload["file_path"] == "a.csv"


def test_list_returns_recent_first(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    for i in range(3):
        store.put(MemoryRecord(
            id=f"x{i}", namespace="dataset:ds",
            kind="dataset_knowledge", payload={"i": i},
        ))
    results = store.list("dataset:ds", limit=2)
    assert [r.id for r in results] == ["x2", "x1"]


def test_namespace_isolated(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    store.put(MemoryRecord(
        id="a", namespace="dataset:ds1",
        kind="dataset_knowledge", payload={},
    ))
    assert store.list("dataset:ds2") == []


def test_empty_namespace_returns_empty(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    assert store.list("dataset:none") == []


def test_get_returns_latest_or_none(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    store.put(MemoryRecord(
        id="a", namespace="dataset:ds",
        kind="dataset_knowledge", payload={"v": 1},
    ))
    store.put(MemoryRecord(
        id="a", namespace="dataset:ds",
        kind="dataset_knowledge", payload={"v": 2},
    ))
    rec = store.get("dataset:ds", "a")
    assert rec is not None
    assert rec.payload["v"] == 2
    assert store.get("dataset:ds", "missing") is None


def test_delete_writes_tombstone(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    store.put(MemoryRecord(
        id="a", namespace="dataset:ds",
        kind="dataset_knowledge", payload={},
    ))
    store.delete("dataset:ds", "a")
    assert store.list("dataset:ds") == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_phase9_memory_jsonl_store.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 最小实现**

`src/data_agent_langchain/memory/stores/__init__.py`：

```python
from data_agent_langchain.memory.stores.jsonl import JsonlMemoryStore

__all__ = ["JsonlMemoryStore"]
```

`src/data_agent_langchain/memory/stores/jsonl.py`：

```python
"""append-only JSONL 后端的 MemoryStore 实现。

每个 namespace 一个文件，文件名通过 ``_safe_filename`` 把 ``:`` / ``/``
等替换为 ``__``，避免路径越界。删除通过写入 tombstone 行实现。
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from data_agent_langchain.memory.base import MemoryRecord, RecordKind


def _safe_filename(namespace: str) -> str:
    return namespace.replace(":", "__").replace("/", "__") + ".jsonl"


def _record_to_json(rec: MemoryRecord) -> str:
    data = asdict(rec)
    data["created_at"] = rec.created_at.isoformat()
    return json.dumps(data, ensure_ascii=False, default=str)


def _record_from_json(line: str) -> MemoryRecord | None:
    obj: dict[str, Any] = json.loads(line)
    if "_tombstone" in obj:
        return None
    created_at = datetime.fromisoformat(obj["created_at"])
    return MemoryRecord(
        id=obj["id"],
        namespace=obj["namespace"],
        kind=obj["kind"],
        payload=obj.get("payload", {}),
        metadata=obj.get("metadata", {}),
        created_at=created_at,
    )


class JsonlMemoryStore:
    """按 namespace 分文件的 append-only JSONL store。"""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str) -> Path:
        return self._root / _safe_filename(namespace)

    def put(self, record: MemoryRecord) -> None:
        with self._path(record.namespace).open("a", encoding="utf-8") as fp:
            fp.write(_record_to_json(record) + "\n")

    def _iter_active(self, namespace: str) -> list[MemoryRecord]:
        path = self._path(namespace)
        if not path.exists():
            return []
        tombstoned: set[str] = set()
        records: list[MemoryRecord] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            obj = json.loads(raw)
            if "_tombstone" in obj:
                tombstoned.add(obj["_tombstone"])
                continue
            rec = _record_from_json(raw)
            if rec is not None:
                records.append(rec)
        # tombstoned IDs 不出现在结果里
        return [r for r in records if r.id not in tombstoned]

    def get(self, namespace: str, record_id: str) -> MemoryRecord | None:
        latest: MemoryRecord | None = None
        for rec in self._iter_active(namespace):
            if rec.id == record_id:
                latest = rec
        return latest

    def list(self, namespace: str, *, limit: int = 100) -> list[MemoryRecord]:
        records = self._iter_active(namespace)
        # 同 id 取最近一次写入，其余视作历史
        seen: dict[str, MemoryRecord] = {}
        for rec in records:
            seen[rec.id] = rec
        ordered = sorted(seen.values(), key=lambda r: r.created_at, reverse=True)
        return ordered[:limit]

    def delete(self, namespace: str, record_id: str) -> None:
        with self._path(namespace).open("a", encoding="utf-8") as fp:
            fp.write(json.dumps({"_tombstone": record_id}) + "\n")


__all__ = ["JsonlMemoryStore"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_phase9_memory_jsonl_store.py -v`
Expected: PASS（6 个测试全过）

- [ ] **Step 5: 提交**

```bash
git add src/data_agent_langchain/memory/stores tests/test_phase9_memory_jsonl_store.py
git commit -m "feat(memory): add append-only JsonlMemoryStore"
```

---

## Task M1.3：`ExactNamespaceRetriever`

**Files:**
- Create: `src/data_agent_langchain/memory/retrievers/__init__.py`
- Create: `src/data_agent_langchain/memory/retrievers/exact.py`
- Test: `tests/test_phase9_memory_retriever_exact.py`

- [ ] **Step 1: 写失败测试**

```python
from pathlib import Path

from data_agent_langchain.memory.base import MemoryRecord
from data_agent_langchain.memory.retrievers.exact import ExactNamespaceRetriever
from data_agent_langchain.memory.stores.jsonl import JsonlMemoryStore


def _seed(store: JsonlMemoryStore, n: int) -> None:
    for i in range(n):
        store.put(MemoryRecord(
            id=f"r{i}", namespace="dataset:ds",
            kind="dataset_knowledge", payload={"i": i},
        ))


def test_retrieve_returns_up_to_k(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    _seed(store, 5)
    retriever = ExactNamespaceRetriever(store)
    results = retriever.retrieve("ignored", namespace="dataset:ds", k=2)
    assert len(results) == 2
    assert all(r.reason == "exact_namespace" for r in results)
    assert all(r.score == 1.0 for r in results)


def test_retrieve_empty_namespace_returns_empty(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    retriever = ExactNamespaceRetriever(store)
    assert retriever.retrieve("q", namespace="dataset:none", k=5) == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_phase9_memory_retriever_exact.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 最小实现**

`src/data_agent_langchain/memory/retrievers/__init__.py`：

```python
from data_agent_langchain.memory.retrievers.exact import ExactNamespaceRetriever

__all__ = ["ExactNamespaceRetriever"]
```

`src/data_agent_langchain/memory/retrievers/exact.py`：

```python
"""按 namespace 精确召回的 Retriever。"""
from __future__ import annotations

from data_agent_langchain.memory.base import MemoryStore, RetrievalResult


class ExactNamespaceRetriever:
    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def retrieve(
        self, query: str, *, namespace: str, k: int = 5
    ) -> list[RetrievalResult]:
        records = self._store.list(namespace, limit=k)
        return [
            RetrievalResult(record=rec, score=1.0, reason="exact_namespace")
            for rec in records
        ]


__all__ = ["ExactNamespaceRetriever"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_phase9_memory_retriever_exact.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/data_agent_langchain/memory/retrievers tests/test_phase9_memory_retriever_exact.py
git commit -m "feat(memory): add ExactNamespaceRetriever"
```

---

## Task M1.4：`StoreBackedMemoryWriter`

**Files:**
- Create: `src/data_agent_langchain/memory/writers/__init__.py`
- Create: `src/data_agent_langchain/memory/writers/store_backed.py`
- Test: `tests/test_phase9_memory_writer.py`

- [ ] **Step 1: 写失败测试**

```python
from pathlib import Path

from data_agent_langchain.memory.records import (
    DatasetKnowledgeRecord,
    ToolPlaybookRecord,
)
from data_agent_langchain.memory.stores.jsonl import JsonlMemoryStore
from data_agent_langchain.memory.writers.store_backed import StoreBackedMemoryWriter


def _writer(tmp_path: Path, mode: str) -> tuple[StoreBackedMemoryWriter, JsonlMemoryStore]:
    store = JsonlMemoryStore(root=tmp_path)
    return StoreBackedMemoryWriter(store, mode=mode), store


def _dk() -> DatasetKnowledgeRecord:
    return DatasetKnowledgeRecord(
        file_path="a.csv", file_kind="csv",
        schema={"a": "string"}, row_count_estimate=10,
    )


def _tp() -> ToolPlaybookRecord:
    return ToolPlaybookRecord(
        tool_name="read_csv", input_template={"max_rows": 20},
        preconditions=["preview_done"],
    )


def test_full_mode_writes_both(tmp_path: Path):
    w, store = _writer(tmp_path, mode="full")
    w.write_dataset_knowledge("ds", _dk())
    w.write_tool_playbook("ds", "read_csv", _tp())
    assert len(store.list("dataset:ds")) == 1
    assert len(store.list("dataset:ds/tool:read_csv")) == 1


def test_read_only_dataset_blocks_tool_playbook(tmp_path: Path):
    w, store = _writer(tmp_path, mode="read_only_dataset")
    w.write_dataset_knowledge("ds", _dk())
    w.write_tool_playbook("ds", "read_csv", _tp())
    assert len(store.list("dataset:ds")) == 1
    assert store.list("dataset:ds/tool:read_csv") == []


def test_disabled_writes_nothing(tmp_path: Path):
    w, store = _writer(tmp_path, mode="disabled")
    w.write_dataset_knowledge("ds", _dk())
    w.write_tool_playbook("ds", "read_csv", _tp())
    assert store.list("dataset:ds") == []
    assert store.list("dataset:ds/tool:read_csv") == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_phase9_memory_writer.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 最小实现**

`src/data_agent_langchain/memory/writers/__init__.py`：

```python
from data_agent_langchain.memory.writers.store_backed import StoreBackedMemoryWriter

__all__ = ["StoreBackedMemoryWriter"]
```

`src/data_agent_langchain/memory/writers/store_backed.py`：

```python
"""Mode-aware MemoryWriter（v2 §4.7）。"""
from __future__ import annotations

from dataclasses import asdict

from data_agent_langchain.memory.base import MemoryRecord, MemoryStore
from data_agent_langchain.memory.records import (
    DatasetKnowledgeRecord,
    ToolPlaybookRecord,
)


class StoreBackedMemoryWriter:
    def __init__(self, store: MemoryStore, *, mode: str) -> None:
        self._store = store
        self._mode = mode

    def write_dataset_knowledge(
        self, dataset: str, record: DatasetKnowledgeRecord
    ) -> None:
        if self._mode == "disabled":
            return
        self._store.put(MemoryRecord(
            id=f"dk:{dataset}:{record.file_path}",
            namespace=f"dataset:{dataset}",
            kind="dataset_knowledge",
            payload=asdict(record),
        ))

    def write_tool_playbook(
        self, dataset: str, tool_name: str, record: ToolPlaybookRecord
    ) -> None:
        if self._mode in {"disabled", "read_only_dataset"}:
            return
        self._store.put(MemoryRecord(
            id=f"tp:{dataset}:{tool_name}",
            namespace=f"dataset:{dataset}/tool:{tool_name}",
            kind="tool_playbook",
            payload=asdict(record),
        ))


__all__ = ["StoreBackedMemoryWriter"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_phase9_memory_writer.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/data_agent_langchain/memory/writers tests/test_phase9_memory_writer.py
git commit -m "feat(memory): add mode-aware StoreBackedMemoryWriter"
```

---

## Task M1.5：`factory.build_store / build_retriever / build_writer`

**Files:**
- Create: `src/data_agent_langchain/memory/factory.py`
- Modify: `src/data_agent_langchain/memory/__init__.py`
- Test: `tests/test_phase9_memory_factory.py`

- [ ] **Step 1: 写失败测试**

```python
from pathlib import Path

from data_agent_langchain.config import MemoryConfig
from data_agent_langchain.memory.factory import (
    build_retriever,
    build_store,
    build_writer,
)
from data_agent_langchain.memory.retrievers.exact import ExactNamespaceRetriever
from data_agent_langchain.memory.stores.jsonl import JsonlMemoryStore
from data_agent_langchain.memory.writers.store_backed import StoreBackedMemoryWriter


def test_build_store_jsonl(tmp_path: Path):
    cfg = MemoryConfig(mode="full", store_backend="jsonl", path=tmp_path)
    store = build_store(cfg)
    assert isinstance(store, JsonlMemoryStore)


def test_build_retriever_exact(tmp_path: Path):
    cfg = MemoryConfig(mode="full", retriever_type="exact", path=tmp_path)
    store = build_store(cfg)
    retriever = build_retriever(cfg, store=store)
    assert isinstance(retriever, ExactNamespaceRetriever)


def test_build_writer_carries_mode(tmp_path: Path):
    cfg = MemoryConfig(mode="read_only_dataset", path=tmp_path)
    store = build_store(cfg)
    writer = build_writer(cfg, store=store)
    assert isinstance(writer, StoreBackedMemoryWriter)
    assert writer._mode == "read_only_dataset"  # noqa: SLF001
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_phase9_memory_factory.py -v`
Expected: FAIL

- [ ] **Step 3: 最小实现**

`src/data_agent_langchain/memory/factory.py`：

```python
"""Memory 子系统工厂；只在子进程入口 / 节点入口调用。"""
from __future__ import annotations

from data_agent_langchain.config import MemoryConfig
from data_agent_langchain.memory.base import MemoryStore, Retriever
from data_agent_langchain.memory.retrievers.exact import ExactNamespaceRetriever
from data_agent_langchain.memory.stores.jsonl import JsonlMemoryStore
from data_agent_langchain.memory.writers.store_backed import StoreBackedMemoryWriter


def build_store(cfg: MemoryConfig) -> MemoryStore:
    if cfg.store_backend == "jsonl":
        return JsonlMemoryStore(root=cfg.path)
    raise ValueError(f"unsupported store_backend: {cfg.store_backend!r}")


def build_retriever(cfg: MemoryConfig, *, store: MemoryStore) -> Retriever:
    if cfg.retriever_type == "exact":
        return ExactNamespaceRetriever(store)
    raise ValueError(f"unsupported retriever_type: {cfg.retriever_type!r}")


def build_writer(cfg: MemoryConfig, *, store: MemoryStore) -> StoreBackedMemoryWriter:
    return StoreBackedMemoryWriter(store, mode=cfg.mode)


__all__ = ["build_retriever", "build_store", "build_writer"]
```

修改 `src/data_agent_langchain/memory/__init__.py`，在末尾追加导出：

```python
from data_agent_langchain.memory.base import MemoryRecord, RetrievalResult
from data_agent_langchain.memory.factory import (
    build_retriever,
    build_store,
    build_writer,
)
from data_agent_langchain.memory.records import (
    DatasetKnowledgeRecord,
    ToolPlaybookRecord,
)
from data_agent_langchain.memory.types import MemoryHit

__all__ += [
    "DatasetKnowledgeRecord",
    "MemoryHit",
    "MemoryRecord",
    "RetrievalResult",
    "ToolPlaybookRecord",
    "build_retriever",
    "build_store",
    "build_writer",
]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_phase9_memory_factory.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/data_agent_langchain/memory/factory.py src/data_agent_langchain/memory/__init__.py tests/test_phase9_memory_factory.py
git commit -m "feat(memory): add factories for store/retriever/writer"
```

---

## Task M1.6：`memory/rag/` 占位边界

**Files:**
- Create: `src/data_agent_langchain/memory/rag/__init__.py`
- Test: `tests/test_phase9_memory_rag_placeholder.py`

- [ ] **Step 1: 写失败测试**

```python
import pytest


def test_rag_module_is_placeholder():
    with pytest.raises(NotImplementedError):
        from data_agent_langchain.memory import rag  # noqa: F401
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_phase9_memory_rag_placeholder.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现占位**

`src/data_agent_langchain/memory/rag/__init__.py`：

```python
"""Corpus RAG 占位边界（v2 §4.11）。

Phase M4 才会落地；当前导入即抛错，避免被误用。
"""
raise NotImplementedError(
    "Corpus RAG is reserved for Phase M4 (see "
    "src/计划/RAG记忆架构演示/2026-05-13-rag-memory-architecture-design-v2.md §4.11)."
)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_phase9_memory_rag_placeholder.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/data_agent_langchain/memory/rag tests/test_phase9_memory_rag_placeholder.py
git commit -m "feat(memory): reserve memory/rag boundary with NotImplementedError"
```

---

# Milestone M2：节点接入

## Task M2.1：`RunState` 新增 `memory_hits`

**Files:**
- Modify: `src/data_agent_langchain/runtime/state.py`
- Test: `tests/test_phase9_runstate_memory.py`

- [ ] **Step 1: 写失败测试**

```python
import pickle

from data_agent_langchain.memory.types import MemoryHit
from data_agent_langchain.runtime.state import RunState


def test_runstate_accepts_memory_hits_and_is_picklable():
    state: RunState = {
        "task_id": "t1",
        "question": "q",
        "memory_hits": [
            MemoryHit(record_id="dk:ds:a", namespace="dataset:ds", score=1.0, summary="s")
        ],
    }
    data = pickle.dumps(state)
    restored = pickle.loads(data)
    assert restored["memory_hits"][0].record_id == "dk:ds:a"


def test_memory_hits_uses_operator_add_reducer():
    import typing

    hints = typing.get_type_hints(RunState, include_extras=True)
    annotation = hints["memory_hits"]
    metadata = getattr(annotation, "__metadata__", ())
    import operator

    assert operator.add in metadata
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_phase9_runstate_memory.py -v`
Expected: FAIL（`KeyError: 'memory_hits'`）

- [ ] **Step 3: 修改 `state.py`**

在 `RunState` 中（与 `steps` 同区块）新增：

```python
    # ----- Memory 召回摘要（v2 §4.9） -----
    memory_hits: Annotated[list["MemoryHit"], operator.add]
```

文件顶部 import：

```python
from data_agent_langchain.memory.types import MemoryHit
```

为避免循环依赖，可放在 `if TYPE_CHECKING` 段之外 —— `memory/types.py` 不反向依赖 `runtime/`，直接 import 安全。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_phase9_runstate_memory.py tests/test_phase1_runstate.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/data_agent_langchain/runtime/state.py tests/test_phase9_runstate_memory.py
git commit -m "feat(memory): add memory_hits field to RunState"
```

---

## Task M2.2：`render_dataset_facts` 白名单渲染

**Files:**
- Modify: `src/data_agent_langchain/agents/prompts.py`
- Test: `tests/test_phase9_prompts_dataset_facts.py`

- [ ] **Step 1: 写失败测试**

```python
from data_agent_langchain.agents.prompts import render_dataset_facts
from data_agent_langchain.memory.types import MemoryHit


def test_render_emits_only_whitelisted_summary_text():
    hits = [
        MemoryHit(
            record_id="dk:ds:a.csv",
            namespace="dataset:ds",
            score=1.0,
            summary="File: a.csv  Kind: csv  Columns: ['a', 'b']  Rows~: 10",
        )
    ]
    text = render_dataset_facts(hits)
    assert "a.csv" in text
    # 仅依赖 summary，不会泄露未授权字段
    assert "MemoryHit" not in text


def test_render_empty_hits_returns_empty_string():
    assert render_dataset_facts([]) == ""
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_phase9_prompts_dataset_facts.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 修改 `prompts.py`**

在 `agents/prompts.py` 顶部加 import：

```python
from data_agent_langchain.memory.types import MemoryHit
```

末尾加函数：

```python
def render_dataset_facts(hits: list[MemoryHit]) -> str:
    """把 MemoryHit 列表白名单渲染为 prompt 片段。

    仅使用 ``summary`` 字段，确保未授权的 payload 字段不会进 prompt。
    """
    if not hits:
        return ""
    lines = ["## Dataset facts (from prior runs, informational only)"]
    for hit in hits:
        lines.append(f"- {hit.summary}")
    return "\n".join(lines)
```

更新 `__all__` 加入 `render_dataset_facts`。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_phase9_prompts_dataset_facts.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/data_agent_langchain/agents/prompts.py tests/test_phase9_prompts_dataset_facts.py
git commit -m "feat(memory): add render_dataset_facts whitelist renderer"
```

---

## Task M2.3：`tool_node` 成功后写 `DatasetKnowledgeRecord`

**Files:**
- Modify: `src/data_agent_langchain/agents/tool_node.py`
- Test: `tests/test_phase9_tool_node_memory_write.py`

- [ ] **Step 1: 写失败测试**

```python
from pathlib import Path
from unittest.mock import patch

from data_agent_langchain.config import MemoryConfig, default_app_config
from data_agent_langchain.memory.factory import build_store, build_writer
from data_agent_langchain.runtime.state import RunState


def _state_for_read_csv() -> RunState:
    return {
        "task_id": "t1",
        "question": "q",
        "action": "read_csv",
        "action_input": {"file_path": "transactions.csv", "max_rows": 5},
        "step_index": 1,
        "skip_tool": False,
        "last_error_kind": None,
    }


def test_tool_node_writes_dataset_knowledge_on_success(tmp_path: Path, monkeypatch):
    from data_agent_langchain.agents import tool_node as tn

    # 构造一个 read_csv ok 的 fake 结果
    def fake_tool_node_body(state, config=None):
        # 直接走辅助函数 —— 实现里把写入封装为 _maybe_write_dataset_knowledge
        cfg = default_app_config()
        cfg_with_mem = type(cfg)(  # rebuild with full mode
            **{**{f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()},
               "memory": MemoryConfig(mode="full", path=tmp_path)}
        )
        store = build_store(cfg_with_mem.memory)
        writer = build_writer(cfg_with_mem.memory, store=store)
        tn._maybe_write_dataset_knowledge(
            writer=writer,
            dataset="ds",
            action="read_csv",
            action_input=state.get("action_input") or {},
            content={
                "columns": ["date", "amount"],
                "dtypes": {"date": "string", "amount": "float"},
                "row_count_estimate": 100,
            },
        )
        return store

    store = fake_tool_node_body(_state_for_read_csv())
    recs = store.list("dataset:ds")
    assert len(recs) == 1
    assert recs[0].payload["file_path"] == "transactions.csv"
    assert recs[0].payload["schema"] == {"date": "string", "amount": "float"}


def test_maybe_write_skips_non_dataset_actions(tmp_path: Path):
    from data_agent_langchain.agents import tool_node as tn

    cfg = MemoryConfig(mode="full", path=tmp_path)
    store = build_store(cfg)
    writer = build_writer(cfg, store=store)
    tn._maybe_write_dataset_knowledge(
        writer=writer,
        dataset="ds",
        action="execute_python",
        action_input={},
        content={"columns": []},
    )
    assert store.list("dataset:ds") == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_phase9_tool_node_memory_write.py -v`
Expected: FAIL（`AttributeError: _maybe_write_dataset_knowledge`）

- [ ] **Step 3: 修改 `tool_node.py`**

在 `tool_node.py` 顶部加 import：

```python
from data_agent_langchain.memory.records import DatasetKnowledgeRecord
from data_agent_langchain.memory.writers.store_backed import StoreBackedMemoryWriter
```

新增辅助函数（放在文件末尾、`__all__` 之前）：

```python
# v2 §5.1：dataset_knowledge 仅允许由这些 action 触发写入。
_DATASET_KNOWLEDGE_ACTIONS: dict[str, str] = {
    "read_csv": "csv",
    "read_json": "json",
    "read_doc": "doc",
    "inspect_sqlite_schema": "sqlite",
}


def _maybe_write_dataset_knowledge(
    *,
    writer: StoreBackedMemoryWriter,
    dataset: str,
    action: str,
    action_input: dict[str, Any],
    content: dict[str, Any],
) -> None:
    """工具成功后尝试写 DatasetKnowledgeRecord；任何字段缺失都安全跳过。"""
    file_kind = _DATASET_KNOWLEDGE_ACTIONS.get(action)
    if file_kind is None:
        return
    file_path = (
        action_input.get("file_path")
        or action_input.get("path")
        or content.get("file_path")
    )
    if not isinstance(file_path, str) or not file_path:
        return
    schema_src = content.get("dtypes") or content.get("schema") or {}
    if not isinstance(schema_src, dict):
        return
    schema = {str(k): str(v) for k, v in schema_src.items()}
    row_count = content.get("row_count_estimate")
    if not isinstance(row_count, int):
        row_count = None
    columns = content.get("columns") or []
    sample_columns = [str(c) for c in columns] if isinstance(columns, list) else []
    record = DatasetKnowledgeRecord(
        file_path=file_path,
        file_kind=file_kind,  # type: ignore[arg-type]
        schema=schema,
        row_count_estimate=row_count,
        sample_columns=sample_columns,
    )
    try:
        writer.write_dataset_knowledge(dataset, record)
    except Exception as exc:  # 写入失败不能影响主路径
        logger.warning("[tool_node] memory write skipped: %s", exc)
```

在 `tool_node()` 主体内、`if result.ok:` 分支末尾追加：

```python
        # v2 §5.1：dataset_knowledge 写入
        memory_cfg = getattr(app_config, "memory", None)
        if memory_cfg is not None and memory_cfg.mode != "disabled":
            try:
                from data_agent_langchain.memory.factory import (
                    build_store,
                    build_writer,
                )
                store = build_store(memory_cfg)
                writer = build_writer(memory_cfg, store=store)
                dataset_name = Path(state.get("dataset_root", "") or ".").name or "default"
                content = result.content if isinstance(result.content, dict) else {}
                _maybe_write_dataset_knowledge(
                    writer=writer,
                    dataset=dataset_name,
                    action=action,
                    action_input=action_input,
                    content=content,
                )
            except Exception as exc:
                logger.warning("[tool_node] memory subsystem error: %s", exc)
```

在顶部 import：

```python
from pathlib import Path
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_phase9_tool_node_memory_write.py tests/test_phase3_tool_node.py -v`
Expected: PASS（包括原 tool_node 测试不回归）

- [ ] **Step 5: 提交**

```bash
git add src/data_agent_langchain/agents/tool_node.py tests/test_phase9_tool_node_memory_write.py
git commit -m "feat(memory): write DatasetKnowledgeRecord from tool_node on success"
```

---

## Task M2.4：`planner_node` / `model_node` 召回 + `memory_recall` 事件

**Files:**
- Modify: `src/data_agent_langchain/agents/planner_node.py`
- Modify: `src/data_agent_langchain/agents/model_node.py`
- Test: `tests/test_phase9_memory_recall.py`

- [ ] **Step 1: 写失败测试**

```python
from pathlib import Path
from unittest.mock import MagicMock

from data_agent_langchain.config import MemoryConfig
from data_agent_langchain.memory.base import MemoryRecord
from data_agent_langchain.memory.factory import build_store
from data_agent_langchain.memory.types import MemoryHit


def test_recall_dataset_facts_returns_memory_hits(tmp_path: Path):
    from data_agent_langchain.agents.memory_recall import recall_dataset_facts

    cfg = MemoryConfig(mode="read_only_dataset", path=tmp_path)
    store = build_store(cfg)
    store.put(MemoryRecord(
        id="dk:ds:a.csv",
        namespace="dataset:ds",
        kind="dataset_knowledge",
        payload={
            "file_path": "a.csv",
            "file_kind": "csv",
            "schema": {"a": "string"},
            "row_count_estimate": 10,
            "encoding": None,
            "sample_columns": ["a"],
        },
    ))
    hits = recall_dataset_facts(
        memory_cfg=cfg, dataset="ds", node="planner_node", config=None
    )
    assert len(hits) == 1
    assert isinstance(hits[0], MemoryHit)
    assert "a.csv" in hits[0].summary


def test_recall_disabled_returns_empty(tmp_path: Path):
    from data_agent_langchain.agents.memory_recall import recall_dataset_facts

    cfg = MemoryConfig(mode="disabled", path=tmp_path)
    assert recall_dataset_facts(
        memory_cfg=cfg, dataset="ds", node="planner_node", config=None
    ) == []


def test_recall_dispatches_memory_recall_event(tmp_path: Path, monkeypatch):
    from data_agent_langchain.agents import memory_recall as mr

    captured: list[tuple[str, dict]] = []

    def fake_dispatch(name, data, config):
        captured.append((name, data))

    monkeypatch.setattr(mr, "dispatch_observability_event", fake_dispatch)

    cfg = MemoryConfig(mode="read_only_dataset", path=tmp_path)
    store = build_store(cfg)
    store.put(MemoryRecord(
        id="dk:ds:a.csv", namespace="dataset:ds",
        kind="dataset_knowledge",
        payload={"file_path": "a.csv", "file_kind": "csv",
                 "schema": {}, "row_count_estimate": None,
                 "encoding": None, "sample_columns": []},
    ))
    mr.recall_dataset_facts(
        memory_cfg=cfg, dataset="ds", node="planner_node", config=None
    )
    assert captured and captured[0][0] == "memory_recall"
    assert captured[0][1]["node"] == "planner_node"
    assert captured[0][1]["namespace"] == "dataset:ds"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_phase9_memory_recall.py -v`
Expected: FAIL（`No module named 'data_agent_langchain.agents.memory_recall'`）

- [ ] **Step 3: 新增 `agents/memory_recall.py`**

把召回逻辑集中到一个辅助模块，便于 planner / model 共用：

`src/data_agent_langchain/agents/memory_recall.py`：

```python
"""Memory 召回辅助：planner_node / model_node 复用同一路径。"""
from __future__ import annotations

from typing import Any

from data_agent_langchain.config import MemoryConfig
from data_agent_langchain.memory.factory import build_retriever, build_store
from data_agent_langchain.memory.types import MemoryHit
from data_agent_langchain.observability.events import dispatch_observability_event


def _summarize_dataset_knowledge(payload: dict[str, Any]) -> str:
    file_path = payload.get("file_path", "?")
    file_kind = payload.get("file_kind", "?")
    columns = payload.get("sample_columns") or list((payload.get("schema") or {}).keys())
    rows = payload.get("row_count_estimate")
    rows_str = f"Rows~: {rows}" if rows is not None else "Rows~: ?"
    return f"File: {file_path}  Kind: {file_kind}  Columns: {columns}  {rows_str}"


def recall_dataset_facts(
    *,
    memory_cfg: MemoryConfig,
    dataset: str,
    node: str,
    config: Any | None,
) -> list[MemoryHit]:
    if memory_cfg.mode == "disabled":
        return []
    store = build_store(memory_cfg)
    retriever = build_retriever(memory_cfg, store=store)
    namespace = f"dataset:{dataset}"
    results = retriever.retrieve(
        "", namespace=namespace, k=memory_cfg.retrieval_max_results
    )
    hits = [
        MemoryHit(
            record_id=r.record.id,
            namespace=r.record.namespace,
            score=r.score,
            summary=_summarize_dataset_knowledge(r.record.payload),
        )
        for r in results
    ]
    dispatch_observability_event(
        "memory_recall",
        {
            "node": node,
            "namespace": namespace,
            "k": memory_cfg.retrieval_max_results,
            "hit_ids": [h.record_id for h in hits],
            "reason": "exact_namespace",
        },
        config,
    )
    return hits


__all__ = ["recall_dataset_facts"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_phase9_memory_recall.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/data_agent_langchain/agents/memory_recall.py tests/test_phase9_memory_recall.py
git commit -m "feat(memory): add recall_dataset_facts helper with memory_recall event"
```

---

## Task M2.5：`planner_node` 接入召回

**Files:**
- Modify: `src/data_agent_langchain/agents/planner_node.py`
- Test: `tests/test_phase9_planner_memory_integration.py`

- [ ] **Step 1: 写失败测试**

```python
from pathlib import Path

from langchain_core.language_models import FakeListChatModel

from data_agent_langchain.agents.planner_node import planner_node
from data_agent_langchain.config import MemoryConfig
from data_agent_langchain.memory.base import MemoryRecord
from data_agent_langchain.memory.factory import build_store


def test_planner_node_writes_memory_hits_when_enabled(tmp_path: Path, monkeypatch):
    cfg = MemoryConfig(mode="read_only_dataset", path=tmp_path)
    store = build_store(cfg)
    store.put(MemoryRecord(
        id="dk:ds:a.csv", namespace="dataset:ds",
        kind="dataset_knowledge",
        payload={"file_path": "a.csv", "file_kind": "csv",
                 "schema": {"a": "string"}, "row_count_estimate": 1,
                 "encoding": None, "sample_columns": ["a"]},
    ))

    # 把 planner 用的 app_config 替换为 memory.mode=read_only_dataset 的版本
    from data_agent_langchain.agents import planner_node as pn
    from data_agent_langchain.config import default_app_config

    base_cfg = default_app_config()
    new_cfg = type(base_cfg)(
        **{**{f.name: getattr(base_cfg, f.name) for f in base_cfg.__dataclass_fields__.values()},
           "memory": cfg}
    )
    monkeypatch.setattr(pn, "_safe_get_app_config", lambda: new_cfg)

    fake_llm = FakeListChatModel(responses=['```json\n["List context", "Answer"]\n```'])
    state = {
        "task_id": "t1",
        "question": "q",
        "dataset_root": str(tmp_path / "ds"),
        "context_dir": str(tmp_path / "ds"),
        "task_dir": str(tmp_path / "ds"),
    }
    (tmp_path / "ds").mkdir(exist_ok=True)

    out = planner_node(state, config={"configurable": {"llm": fake_llm}})
    assert "memory_hits" in out
    assert out["memory_hits"][0].record_id == "dk:ds:a.csv"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_phase9_planner_memory_integration.py -v`
Expected: FAIL（`memory_hits` 未出现在 partial state）

- [ ] **Step 3: 修改 `planner_node.py`**

在 `planner_node()` 末尾构造 partial state 时追加召回：

```python
def planner_node(state: RunState, config: Any | None = None) -> dict[str, Any]:
    try:
        plan = _generate_plan_from_state(state, config=config)
        plan_ok = True
    except Exception:
        plan = list(FALLBACK_PLAN)
        plan_ok = False

    app_config = _safe_get_app_config()
    memory_cfg = getattr(app_config, "memory", None)
    memory_hits: list = []
    if memory_cfg is not None:
        from pathlib import Path

        from data_agent_langchain.agents.memory_recall import recall_dataset_facts

        dataset_name = Path(state.get("dataset_root", "") or ".").name or "default"
        memory_hits = recall_dataset_facts(
            memory_cfg=memory_cfg,
            dataset=dataset_name,
            node="planner_node",
            config=config,
        )

    out: dict[str, Any] = {
        "plan": plan,
        "plan_index": 0,
        "replan_used": int(state.get("replan_used", 0) or 0),
        "steps": [_planning_step(plan, ok=plan_ok)],
    }
    if memory_hits:
        out["memory_hits"] = memory_hits
    return out
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_phase9_planner_memory_integration.py tests/test_phase4_planner_node.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/data_agent_langchain/agents/planner_node.py tests/test_phase9_planner_memory_integration.py
git commit -m "feat(memory): recall dataset facts in planner_node"
```

---

## Task M2.6：`model_node` 在 prompt 后追加 facts

**Files:**
- Modify: `src/data_agent_langchain/agents/model_node.py`
- Test: `tests/test_phase9_model_node_memory.py`

- [ ] **Step 1: 写失败测试**

```python
from data_agent_langchain.agents.model_node import _build_messages_for_state
from data_agent_langchain.config import default_app_config
from data_agent_langchain.memory.types import MemoryHit


def test_messages_include_dataset_facts_when_state_has_hits(tmp_path):
    cfg = default_app_config()
    state = {
        "task_id": "t1",
        "question": "q",
        "dataset_root": str(tmp_path),
        "context_dir": str(tmp_path),
        "task_dir": str(tmp_path),
        "mode": "react",
        "action_mode": "json_action",
        "memory_hits": [
            MemoryHit(
                record_id="dk:ds:a.csv",
                namespace="dataset:ds",
                score=1.0,
                summary="File: a.csv  Kind: csv  Columns: ['a']  Rows~: 1",
            )
        ],
    }
    messages = _build_messages_for_state(state, cfg)
    joined = "\n".join(m.content for m in messages if isinstance(m.content, str))
    assert "Dataset facts" in joined
    assert "a.csv" in joined


def test_no_facts_when_memory_hits_empty(tmp_path):
    cfg = default_app_config()
    state = {
        "task_id": "t1", "question": "q",
        "dataset_root": str(tmp_path), "context_dir": str(tmp_path),
        "task_dir": str(tmp_path), "mode": "react", "action_mode": "json_action",
    }
    messages = _build_messages_for_state(state, cfg)
    joined = "\n".join(m.content for m in messages if isinstance(m.content, str))
    assert "Dataset facts" not in joined
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_phase9_model_node_memory.py -v`
Expected: FAIL

- [ ] **Step 3: 修改 `model_node.py`**

在 `_build_messages_for_state` 末尾追加：

```python
    # v2 §5.2：把 memory_hits 渲染成只读 facts 段
    hits = list(state.get("memory_hits") or [])
    if hits:
        from langchain_core.messages import HumanMessage

        from data_agent_langchain.agents.prompts import render_dataset_facts

        facts_text = render_dataset_facts(hits)
        if facts_text:
            messages = list(messages) + [HumanMessage(content=facts_text)]
    return messages
```

注意：函数内部已有 `return` 语句；把原 `return build_react_messages(...)` / `return build_plan_solve_execution_messages(...)` 改成赋值 `messages = ...` 后统一在末尾追加 facts 并返回。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_phase9_model_node_memory.py tests/test_phase3_model.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/data_agent_langchain/agents/model_node.py tests/test_phase9_model_node_memory.py
git commit -m "feat(memory): inject dataset facts after prompt in model_node"
```

---

# Milestone M3：评测协议 & CLI

## Task M3.1：CLI `--memory-mode` 覆盖

**Files:**
- Modify: `src/data_agent_langchain/cli.py`
- Test: `tests/test_phase9_cli_memory_mode.py`

- [ ] **Step 1: 写失败测试**

```python
from typer.testing import CliRunner

from data_agent_langchain.cli import app


def test_run_benchmark_accepts_memory_mode_flag():
    runner = CliRunner()
    result = runner.invoke(app, ["run-benchmark", "--help"])
    assert result.exit_code == 0
    assert "--memory-mode" in result.output


def test_run_task_accepts_memory_mode_flag():
    runner = CliRunner()
    result = runner.invoke(app, ["run-task", "--help"])
    assert result.exit_code == 0
    assert "--memory-mode" in result.output
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_phase9_cli_memory_mode.py -v`
Expected: FAIL

- [ ] **Step 3: 修改 `cli.py`**

在 `run-task` 与 `run-benchmark` 命令签名上追加：

```python
    memory_mode: Annotated[
        str | None,
        typer.Option(
            "--memory-mode",
            help="覆盖 memory.mode：disabled | read_only_dataset | full。",
        ),
    ] = None,
```

在命令体内 `load_app_config` 之后，若 `memory_mode is not None` 则用 `dataclasses.replace` 重建：

```python
    if memory_mode is not None:
        import dataclasses

        cfg = dataclasses.replace(
            cfg,
            memory=dataclasses.replace(cfg.memory, mode=memory_mode),
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_phase9_cli_memory_mode.py tests/test_phase5_runner_cli.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/data_agent_langchain/cli.py tests/test_phase9_cli_memory_mode.py
git commit -m "feat(memory): add --memory-mode flag to run-task and run-benchmark"
```

---

## Task M3.2：评测路径默认 `read_only_dataset`

**Files:**
- Modify: `src/data_agent_langchain/submission.py`
- Test: `tests/test_phase9_submission_memory_default.py`

- [ ] **Step 1: 写失败测试**

```python
from data_agent_langchain.submission import build_submission_config


def test_submission_default_memory_is_safe(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_API_URL", "http://example.com")
    monkeypatch.setenv("MODEL_NAME", "gpt-test")
    cfg = build_submission_config()
    # 评测默认必须是 disabled 或 read_only_dataset，不能是 full。
    assert cfg.memory.mode in {"disabled", "read_only_dataset"}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_phase9_submission_memory_default.py -v`
Expected: 取决于现有 `submission.py` 是否经过 `MemoryConfig` —— 至少不应是 `full`；若默认就是 `disabled`，则补一行显式断言 `read_only_dataset` 来锁定预期：

```python
    assert cfg.memory.mode == "read_only_dataset"
```

并使本测试 FAIL（因为 dataclass 默认是 `disabled`）。

- [ ] **Step 3: 修改 `submission.py`**

在 `build_submission_config()` 构造 `AppConfig` 时显式设置：

```python
    import dataclasses

    from data_agent_langchain.config import MemoryConfig

    cfg = dataclasses.replace(
        cfg,
        memory=dataclasses.replace(cfg.memory, mode="read_only_dataset"),
    )
```

（具体注入点取决于现有实现；如果用 `AppConfig(...)` 直接构造，则把 `memory=MemoryConfig(mode="read_only_dataset")` 加进参数。）

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_phase9_submission_memory_default.py tests/test_submission_config.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/data_agent_langchain/submission.py tests/test_phase9_submission_memory_default.py
git commit -m "feat(memory): default submission to memory.mode=read_only_dataset"
```

---

## Task M3.3：disabled 模式 parity 冒烟

**Files:**
- Test: `tests/test_phase9_memory_disabled_parity.py`

- [ ] **Step 1: 写测试**

```python
from pathlib import Path

from langchain_core.language_models import FakeListChatModel

from data_agent_langchain.agents.planner_node import planner_node
from data_agent_langchain.config import MemoryConfig, default_app_config


def test_disabled_mode_produces_no_memory_hits(tmp_path: Path, monkeypatch):
    from data_agent_langchain.agents import planner_node as pn

    base_cfg = default_app_config()
    cfg = type(base_cfg)(
        **{**{f.name: getattr(base_cfg, f.name) for f in base_cfg.__dataclass_fields__.values()},
           "memory": MemoryConfig(mode="disabled", path=tmp_path)}
    )
    monkeypatch.setattr(pn, "_safe_get_app_config", lambda: cfg)

    fake_llm = FakeListChatModel(responses=['```json\n["x"]\n```'])
    state = {
        "task_id": "t1", "question": "q",
        "dataset_root": str(tmp_path), "context_dir": str(tmp_path),
        "task_dir": str(tmp_path),
    }
    out = planner_node(state, config={"configurable": {"llm": fake_llm}})
    assert "memory_hits" not in out or out["memory_hits"] == []
```

- [ ] **Step 2: 运行测试确认通过**

Run: `pytest tests/test_phase9_memory_disabled_parity.py -v`
Expected: PASS（如不通过，说明召回路径未严格遵守 disabled；需回到 Task M2.5 修复）

- [ ] **Step 3: 提交**

```bash
git add tests/test_phase9_memory_disabled_parity.py
git commit -m "test(memory): verify disabled mode yields no memory_hits"
```

---

## Task M3.4：全量回归 + 文档对齐

**Files:**
- 无新代码，仅运行测试 + 更新 v1 文档脚注。

- [ ] **Step 1: 跑全部 Phase 9 测试**

Run: `pytest tests/test_phase9_*.py -v`
Expected: 全部 PASS

- [ ] **Step 2: 跑既有相关阶段回归**

Run: `pytest tests/test_phase1_runstate.py tests/test_phase3_tool_node.py tests/test_phase3_model.py tests/test_phase4_planner_node.py tests/test_phase5_config.py tests/test_submission_config.py -v`
Expected: 全部 PASS（无回归）

- [ ] **Step 3: 在 v1 设计文档末尾追加一行指向 v2**

修改 `src/计划/RAG记忆架构演示/2026-05-12-rag-memory-html-demo-design.md`，在最后追加：

```markdown
---

> **后续**：本文档为 HTML 演示页摘要；技术落地规格见 `2026-05-13-rag-memory-architecture-design-v2.md`，实施计划见 `2026-05-13-rag-memory-architecture-plan-v2.md`。
```

- [ ] **Step 4: 提交**

```bash
git add src/计划/RAG记忆架构演示/2026-05-12-rag-memory-html-demo-design.md
git commit -m "docs(memory): link v1 demo design to v2 architecture and plan"
```

---

## Self-Review

- **Spec coverage**：
  - v2 §3 原则 → 全部任务实现（dataclass 字段白名单、三态开关、store/retriever/writer 解耦、`RunState` 不存句柄、`memory_recall` 审计、RAG 边界）。
  - v2 §4 目录 → M0.1–M1.6 创建对应文件；`memory/__init__.py` 更新导出（M1.5）。
  - v2 §4.2 `MemoryConfig` → M1.1。
  - v2 §4.9 `memory_hits` → M2.1。
  - v2 §4.10 `render_dataset_facts` → M2.2 / M2.6。
  - v2 §4.11 RAG 占位 → M1.6。
  - v2 §5.1 写入路径 → M2.3。
  - v2 §5.2 召回路径 → M2.4 / M2.5 / M2.6。
  - v2 §6 配置示例 / §7 阶段 → M3.1 / M3.2 / M3.3。
  - v2 §8 测试矩阵 → 每个任务的 Step 1。
  - v2 §9 风险 → disabled parity 测试（M3.3）、写入路径异常吞噬（M2.3）、picklability（M2.1）。
- **Placeholder scan**：所有 step 都给出实际代码 / 命令 / 期望，未出现 TBD / TODO / 「适当处理」。
- **Type consistency**：`MemoryRecord.kind` Literal、`DatasetKnowledgeRecord.file_kind` Literal、`StoreBackedMemoryWriter(store, *, mode=...)` 签名、`build_writer(cfg, *, store=...)` 关键字参数、`recall_dataset_facts(memory_cfg=..., dataset=..., node=..., config=...)` 在任务间全部一致。
- **未覆盖项**：SQLite store、BM25 retriever、Corpus RAG 实现 → 已在 v2 §7 标记为独立提案（M4），本计划范围外。

---

## Execution Handoff

计划已保存至 `src/计划/RAG记忆架构演示/2026-05-13-rag-memory-architecture-plan-v2.md`。

两种执行方式可选：

1. **Subagent-Driven（推荐）**：每个 task 派发新 subagent，任务间审阅；快速迭代。
2. **Inline Execution**：在当前会话里按 `executing-plans` 流程批量执行 + checkpoint 审阅。

请告知选哪种，或先停留在计划评审阶段。
