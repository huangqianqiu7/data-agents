# `data_agent_refactored.tools` 模块文档

## 1. 模块概述

本模块是 KDD Cup 2026 Data Agent 的**工具层**，为上层 Agent 循环提供统一的工具注册、描述生成与调度入口。它将文件系统浏览、数据预览、SQL 查询、Python 代码执行及答案提交等能力封装为 8 个内置工具，并通过注册表模式（Registry Pattern）支持运行时插件式扩展。

## 2. 文件清单与职责

| 文件 | 职责 |
|---|---|
| `__init__.py` | 包入口，重新导出 `ToolSpec`、`ToolExecutionResult`、`ToolHandler`、`ToolRegistry`、`create_default_tool_registry` |
| `registry.py` | 定义核心数据结构（`ToolSpec`、`ToolExecutionResult`）、类型别名 `ToolHandler`，以及 `ToolRegistry` 注册表与默认工厂函数 |
| `handlers.py` | 8 个内置工具的 handler 实现，统一遵循 `ToolHandler` 签名 |
| `filesystem.py` | 文件系统操作工具函数：路径沙箱解析、目录树遍历、CSV / JSON / 文本文件预览 |
| `python_exec.py` | 沙箱化 Python 代码执行：子进程隔离、stdout/stderr 文件级捕获、超时终止 |
| `sqlite.py` | SQLite 只读操作：schema 检查与受限 SQL 查询执行 |

## 3. 核心数据结构

### 3.1 `ToolSpec`

不可变数据类（`frozen=True`），描述工具的元数据，序列化后注入系统提示词供 LLM 选择工具。

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | `str` | 工具唯一标识，同时用作调度键 |
| `description` | `str` | 工具功能描述，展示给 LLM |
| `input_schema` | `dict[str, Any]` | 输入参数示例字典，向 LLM 说明参数名与格式 |

### 3.2 `ToolExecutionResult`

不可变数据类，所有 handler 的统一返回值。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `ok` | `bool` | — | 执行是否成功 |
| `content` | `dict[str, Any]` | — | 返回给 Agent 的结构化结果 |
| `is_terminal` | `bool` | `False` | 是否为终止动作（仅 `answer` 工具为 `True`） |
| `answer` | `AnswerTable \| None` | `None` | 终止时携带的答案表 |

### 3.3 `ToolHandler`（类型别名）

```python
ToolHandler = Callable[[PublicTask, dict[str, Any]], ToolExecutionResult]
```

所有 handler 必须接收当前任务对象 `PublicTask` 与 LLM 提供的参数字典 `action_input`，返回 `ToolExecutionResult`。

## 4. `ToolRegistry` 详解

`ToolRegistry` 是一个可变数据类，内部维护两个同键字典：

- `specs: dict[str, ToolSpec]` — 工具元数据
- `handlers: dict[str, ToolHandler]` — 工具实现

### 4.1 `register(spec, handler, *, overwrite=False)`

注册一个工具。

| 参数 | 说明 |
|---|---|
| `spec` | 工具元数据 `ToolSpec` 实例 |
| `handler` | 符合 `ToolHandler` 签名的可调用对象 |
| `overwrite` | 为 `False`（默认）时，重复注册同名工具抛出 `ValueError`；为 `True` 时覆盖已有注册 |

### 4.2 `describe_for_prompt() -> str`

按工具名字母序生成纯文本描述，格式为：

```
- tool_name: description
  input_schema: {...}
```

用于拼接到系统提示词中，让 LLM 了解可用工具集。

### 4.3 `execute(task, action, action_input) -> ToolExecutionResult`

根据 `action`（工具名）查找对应 handler 并调用。若工具名未注册，抛出 `UnknownToolError`。

| 参数 | 说明 |
|---|---|
| `task` | 当前 `PublicTask` 实例 |
| `action` | 工具名字符串 |
| `action_input` | LLM 提供的参数字典 |

### 4.4 工厂函数 `create_default_tool_registry()`

创建一个预装 8 个内置工具的 `ToolRegistry` 实例，供 Agent 启动时一次性调用。handler 导入延迟到函数体内以避免循环依赖。

## 5. 内置工具一览

| 工具名 | 功能描述 | `input_schema` 关键参数 |
|---|---|---|
| `list_context` | 递归列出任务上下文目录的文件与子目录 | `max_depth`（默认 4） |
| `read_csv` | 预览 CSV 文件的前 N 行 | `path`、`max_rows`（默认 20） |
| `read_json` | 预览 JSON 文件内容（字符限制） | `path`、`max_chars`（默认 4000） |
| `read_doc` | 预览文本文档内容（字符限制） | `path`、`max_chars`（默认 4000） |
| `inspect_sqlite_schema` | 获取 SQLite 数据库的表名及建表 SQL | `path` |
| `execute_context_sql` | 对 SQLite 文件执行只读 SQL 查询 | `path`、`sql`、`limit`（默认 200） |
| `execute_python` | 以任务上下文目录为工作目录执行 Python 代码 | `code`（超时 30 秒） |
| `answer` | 提交最终答案表，唯一的终止动作 | `columns`（列名列表）、`rows`（行数据列表） |

## 6. 关键机制说明

### 6.1 路径沙箱（Path Sandboxing）

核心函数：`filesystem.resolve_context_path(task, relative_path)`。

1. 将 `relative_path` 拼接到 `task.context_dir` 后调用 `.resolve()` 得到绝对路径 `candidate`。
2. 对 `context_dir` 同样 `.resolve()` 得到 `context_root`。
3. 验证 `context_root` 是否在 `candidate.parents` 中，不满足则抛出 `ContextPathEscapeError`。
4. 验证文件存在性，不存在则抛出 `ContextAssetNotFoundError`。

这确保了即使输入包含 `..` 等路径遍历片段，也无法逃逸到上下文目录之外。所有文件读取类工具（`read_csv`、`read_json`、`read_doc`、`inspect_sqlite_schema`、`execute_context_sql`）均通过此函数解析路径。

### 6.2 Python 执行沙箱

核心函数：`python_exec.execute_python_code(context_root, code, *, timeout_seconds=30)`。

**子进程隔离**：使用 `multiprocessing.Process` 在独立进程中执行用户代码，主进程不受 `exec()` 副作用影响。

**超时控制**：`process.join(timeout_seconds)` 阻塞等待；若超时后进程仍存活，调用 `process.terminate()` 强制终止并返回超时错误。

**stdout/stderr 捕获**：
1. 在临时目录中创建 `stdout.txt` / `stderr.txt` 两个文件。
2. 子进程启动后通过 `os.dup2()` 将文件描述符级别的 stdout(fd 1) 和 stderr(fd 2) 重定向到这两个文件。
3. 同时替换 `sys.stdout` / `sys.stderr` 为指向相同文件描述符的 `TextIOWrapper`。
4. 执行结束后恢复原始描述符，主进程通过读取临时文件获取输出。
5. 子进程通过 `multiprocessing.Queue` 传回执行成功/失败状态及异常信息。

### 6.3 SQL 只读保护

核心函数：`sqlite.execute_read_only_sql(path, sql, *, limit=200)`。

**只读连接**：`_connect_read_only()` 使用 SQLite URI 模式 `file:<path>?mode=ro` 打开连接，数据库引擎层面即禁止写操作。

**语句白名单**：在执行前对 SQL 进行 `lstrip().lower()` 归一化，仅允许以 `select`、`with`、`pragma` 开头的语句，否则抛出 `ReadOnlySQLViolationError`。

**结果限制**：使用 `cursor.fetchmany(limit + 1)` 获取数据，多取一行用于判断是否截断，实际返回最多 `limit` 行。

## 7. 扩展指南

以新增一个名为 `my_tool` 的自定义工具为例：

**第 1 步 — 编写 handler 函数**

```python
# my_tools.py
from data_agent_refactored.benchmark.schema import PublicTask
from data_agent_refactored.tools.registry import ToolExecutionResult

def handle_my_tool(task: PublicTask, action_input: dict) -> ToolExecutionResult:
    result = action_input["param"]  # 从 action_input 取参数
    return ToolExecutionResult(ok=True, content={"result": result})
```

**第 2 步 — 定义 `ToolSpec`**

```python
from data_agent_refactored.tools.registry import ToolSpec

my_tool_spec = ToolSpec(
    name="my_tool",
    description="A short description shown to the LLM.",
    input_schema={"param": "example_value"},
)
```

**第 3 步 — 注册到 `ToolRegistry`**

```python
registry = create_default_tool_registry()  # 或已有的 registry 实例
registry.register(my_tool_spec, handle_my_tool)
```

如需覆盖同名内置工具，传入 `overwrite=True`。

## 8. 依赖与接口边界

### 外部模块依赖

| 依赖模块 | 使用位置 | 导入内容 |
|---|---|---|
| `data_agent_refactored.benchmark.schema` | `registry.py`、`handlers.py` | `PublicTask`、`AnswerTable` |
| `data_agent_refactored.exceptions` | `registry.py`、`filesystem.py`、`handlers.py`、`sqlite.py` | `UnknownToolError`、`ContextPathEscapeError`、`ContextAssetNotFoundError`、`ToolValidationError`、`ReadOnlySQLViolationError` |

### 标准库依赖

`csv`、`json`、`sqlite3`、`multiprocessing`、`tempfile`、`os`、`sys`、`io`、`contextlib`、`traceback`、`pathlib`、`logging`、`dataclasses`、`typing`

### 接口边界

- **对上游**：本模块不依赖 Agent 循环或 LLM 调用逻辑，仅通过 `PublicTask` 接收任务上下文。
- **对下游**：上层通过 `create_default_tool_registry()` 获取 `ToolRegistry` 实例，调用 `describe_for_prompt()` 生成提示词片段、调用 `execute()` 调度工具。
- **错误契约**：所有领域异常继承自 `DataAgentError`，上层可统一捕获。
