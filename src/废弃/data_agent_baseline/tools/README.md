# `tools` 模块说明文档

## 一、模块概述

`tools` 模块是 KDD Cup 2026 Data Agent 的**工具系统**，为 ReAct Agent 提供了与外部环境交互的全部能力。Agent（大模型）在推理循环中通过调用本模块注册的工具来读取数据、执行代码、查询数据库，并最终提交答案。

整个模块的核心思想可以用一个餐厅类比来理解：

| 概念 | 类比 | 对应代码 |
|---|---|---|
| **ToolSpec** | 菜单（告诉顾客有什么菜） | 工具的元数据（名称、描述、参数结构），写入 System Prompt 供大模型参考 |
| **ToolHandler** | 后厨厨师（真正做菜的人） | 每个工具对应的 Python 执行函数 |
| **ToolRegistry** | 前台服务员（接单并派发给厨师） | 根据工具名称路由到对应的 Handler 并执行 |

---

## 二、文件结构

```
tools/
├── __init__.py        # 模块公开接口，导出核心类和工厂函数
├── registry.py        # 工具注册中心（核心文件）
├── filesystem.py      # 文件系统相关的底层工具函数
├── python_exec.py     # Python 代码执行引擎（子进程 + 超时控制）
└── sqlite.py          # SQLite 数据库只读查询工具
```

---

## 三、核心数据结构

### 3.1 `ToolSpec` — 工具说明书

```python
@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str                    # 工具名称，如 "read_csv"
    description: str             # 功能描述（自然语言）
    input_schema: dict[str, Any] # 参数结构示例
```

- **不可变**（`frozen=True`），保障线程安全。
- 被序列化为文本塞入 System Prompt，让大模型知道有哪些工具可用。

### 3.2 `ToolExecutionResult` — 工具执行结果

```python
@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    ok: bool                         # 执行是否成功
    content: dict[str, Any]          # 返回的数据或错误信息
    is_terminal: bool = False        # 是否为终止型工具（仅 answer 为 True）
    answer: AnswerTable | None = None # 终止时携带的最终答案
```

- 所有工具统一返回此格式，ReAct 主循环无需关心具体调用了哪个工具。
- `is_terminal=True` 是关键信号，ReAct 循环检测到后立即 `break`。

### 3.3 `ToolHandler` — 类型别名

```python
ToolHandler = Callable[[PublicTask, dict[str, Any]], ToolExecutionResult]
```

所有工具执行函数的统一签名：`(当前任务, 参数字典) -> 执行结果`。

### 3.4 `ToolRegistry` — 工具注册中心

```python
@dataclass(slots=True)
class ToolRegistry:
    specs: dict[str, ToolSpec]
    handlers: dict[str, ToolHandler]
```

核心方法：

| 方法 | 功能 |
|---|---|
| `describe_for_prompt()` | 将所有工具说明格式化为文本，用于写入 System Prompt |
| `execute(task, action, action_input)` | 根据工具名查找 Handler 并执行，返回 `ToolExecutionResult` |

---

## 四、内置工具清单（共 8 个）

通过 `create_default_tool_registry()` 工厂函数一键组装。

### 4.1 `list_context` — 列出目录结构

| 项目 | 内容 |
|---|---|
| **功能** | 查看任务文件夹的目录树（类似 `tree` 命令） |
| **参数** | `max_depth`（int，默认 4） |
| **典型场景** | Agent 拿到新任务后的第一步，先了解有哪些数据文件 |
| **底层实现** | `filesystem.list_context_tree()` — 递归遍历目录，返回每个条目的路径、类型、大小 |

### 4.2 `read_csv` — 预览 CSV 文件

| 项目 | 内容 |
|---|---|
| **功能** | 读取 CSV 文件的前 N 行 |
| **参数** | `path`（str，必填），`max_rows`（int，默认 20） |
| **注意** | 仅预览，不加载全部数据；完整处理应使用 `execute_python` 配合 pandas |
| **底层实现** | `filesystem.read_csv_preview()` — 使用标准库 `csv.reader` 读取 |

### 4.3 `read_json` — 预览 JSON 文件

| 项目 | 内容 |
|---|---|
| **功能** | 预览 JSON 文件的前 N 个字符 |
| **参数** | `path`（str，必填），`max_chars`（int，默认 4000） |
| **底层实现** | `filesystem.read_json_preview()` — 解析后重新格式化并截断 |

### 4.4 `read_doc` — 读取文本文档

| 项目 | 内容 |
|---|---|
| **功能** | 读取 `.md`、`.txt` 等文本文件 |
| **参数** | `path`（str，必填），`max_chars`（int，默认 4000） |
| **底层实现** | `filesystem.read_doc_preview()` — 直接读取文本并截断 |

### 4.5 `inspect_sqlite_schema` — 查看数据库表结构

| 项目 | 内容 |
|---|---|
| **功能** | 列出 SQLite 数据库中的所有表及其 `CREATE TABLE` 语句 |
| **参数** | `path`（str，必填） |
| **典型场景** | 面对 `.sqlite`/`.db` 文件时的侦查步骤，先看表结构再写 SQL |
| **底层实现** | `sqlite.inspect_sqlite_schema()` — 查询 `sqlite_master` 系统表 |

### 4.6 `execute_context_sql` — 执行只读 SQL

| 项目 | 内容 |
|---|---|
| **功能** | 在 SQLite 数据库上执行只读 SQL 查询 |
| **参数** | `path`（str，必填），`sql`（str，必填），`limit`（int，默认 200） |
| **安全限制** | 仅允许 `SELECT`/`WITH`/`PRAGMA` 开头的语句；以只读模式连接数据库 |
| **底层实现** | `sqlite.execute_read_only_sql()` — 白名单校验 + `fetchmany` 限量返回 |

### 4.7 `execute_python` — 执行 Python 代码

| 项目 | 内容 |
|---|---|
| **功能** | 执行任意 Python 代码，是 Agent 最强大的工具 |
| **参数** | `code`（str，必填） |
| **执行环境** | 工作目录设为任务的 `context_dir`；**30 秒超时**（强制终止） |
| **底层实现** | `python_exec.execute_python_code()` — 在独立子进程（`multiprocessing.Process`）中执行，捕获 stdout/stderr 并通过临时文件 + `Queue` 传回结果 |

### 4.8 `answer` — 提交最终答案 ⭐

| 项目 | 内容 |
|---|---|
| **功能** | 提交答案表格，**终止 ReAct 循环** |
| **参数** | `columns`（`list[str]`，必填），`rows`（`list[list]`，必填） |
| **特殊性** | 唯一一个 `is_terminal=True` 的工具 |
| **数据校验** | columns 必须为非空字符串列表；每行长度必须与列数一致；校验失败抛 `ValueError`，Agent 可在下一轮修正 |

---

## 五、安全机制

本模块在多个层面实施了安全防护：

1. **路径穿越防护** — `filesystem.resolve_context_path()` 检查解析后的绝对路径是否仍在 `context_dir` 内部，阻止 `../../etc/passwd` 等攻击路径。

2. **SQL 注入防护** — `sqlite.execute_read_only_sql()` 通过白名单校验只允许 `SELECT`/`WITH`/`PRAGMA` 语句，且以只读模式（`?mode=ro`）连接数据库。

3. **Python 执行沙箱** — `python_exec.execute_python_code()` 在**独立子进程**中执行代码，并施加 **30 秒超时**限制（`EXECUTE_PYTHON_TIMEOUT_SECONDS`）。超时后强制 `terminate()`。

4. **上下文窗口保护** — 所有文件读取工具都有 `max_rows` / `max_chars` / `limit` 参数，防止大文件或大查询结果撑爆 LLM 的上下文窗口。

5. **答案格式校验** — `_answer()` 对大模型提交的 `columns` 和 `rows` 进行严格的类型和维度校验，校验失败返回错误让大模型自行修正。

---

## 六、调用流程

```
大模型输出 JSON
  │  {"action": "read_csv", "action_input": {"path": "data.csv"}}
  ▼
react.py 解析 JSON
  │
  ▼
ToolRegistry.execute(task, "read_csv", {"path": "data.csv"})
  │
  ├─ 查找 handlers["read_csv"] → _read_csv()
  │
  ├─ _read_csv() 调用 filesystem.read_csv_preview()
  │
  └─ 返回 ToolExecutionResult(ok=True, content={...})
       │
       ▼
  react.py 将结果格式化为 Observation 追加到对话历史
       │
       ▼
  大模型在下一轮看到 Observation 并继续推理
```

---

## 七、扩展指南

如果需要新增一个工具，按以下步骤操作：

1. **编写底层函数** — 在 `filesystem.py`、`sqlite.py`、`python_exec.py` 中添加，或创建新文件。

2. **编写 Handler** — 在 `registry.py` 中添加 `_your_tool()` 函数，签名须符合 `ToolHandler` 类型：
   ```python
   def _your_tool(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
       ...
   ```

3. **注册 Spec** — 在 `create_default_tool_registry()` 的 `specs` 字典中添加 `ToolSpec` 条目。

4. **注册 Handler** — 在 `create_default_tool_registry()` 的 `handlers` 字典中添加映射，**键名必须与 specs 中一致**。

> **重要提醒**：如果新增了 spec 但忘记写 handler，Agent 调用时会抛出 `KeyError`。

---

## 八、依赖关系

```
tools/
  ├── 依赖 benchmark/schema.py  → PublicTask, AnswerTable
  ├── 被 react.py 调用          → ReAct 主循环通过 ToolRegistry.execute() 驱动工具
  └── 被 runner.py 初始化       → create_default_tool_registry() 创建工具箱
```

| 外部依赖 | 用途 |
|---|---|
| `csv` (stdlib) | CSV 文件解析 |
| `json` (stdlib) | JSON 文件解析 |
| `sqlite3` (stdlib) | SQLite 数据库查询 |
| `multiprocessing` (stdlib) | Python 代码子进程执行 |
| `pathlib` (stdlib) | 路径操作与安全校验 |

本模块**无第三方依赖**，全部基于 Python 标准库实现。
