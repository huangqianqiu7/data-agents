# Refactoring Notes — `data_agent_refactored`

> Refactored from `data_agent_baseline` with **zero new dependencies** and **identical public API behaviour**.

---

## 变更总览

| # | 变更点 |
|---|--------|
| 1 | 新增 `exceptions.py`：自定义异常层次结构替代裸 `Exception` / `ValueError` / `KeyError` |
| 2 | `agents_v2/` → `agents/`：移除模块名中的版本后缀 |
| 3 | `gate.py` → `data_preview_gate.py`：模块名更具领域语义 |
| 4 | `registry.py` 拆分：工具 handler 函数提取至新文件 `tools/handlers.py`，registry 仅保留基础设施 |
| 5 | `ToolRegistry` 新增 `register()` 方法：支持插件式工具扩展 |
| 6 | 移除 `run/old_runner.py`：已废弃的冗余副本 |
| 7 | 修复 `run/runner.py` 的断裂导入：原代码从不存在的 `agents.model` 导入，改为正确的 `agents.model` |
| 8 | 所有公开函数和方法添加完整类型标注（参数 + 返回值） |
| 9 | `call_with_timeout` 添加泛型返回类型 `TypeVar("T")` 和 `Callable[..., T]` 签名 |
| 10 | 内部参数 `prog` 统一重命名为 `show_progress`（仅内部方法，不影响公开 API） |
| 11 | `_validate_and_execute_tool` 返回类型从 `object \| None` 改为 `ToolExecutionResult \| None` |
| 12 | `_handle_gate_block` 参数 `model_step` 添加类型标注 |
| 13 | `build_model_adapter` 添加返回类型 `OpenAIModelAdapter` |
| 14 | 关键路径添加 `logging` 模块结构化日志（模型重试、子进程生命周期、门控升级、重规划失败） |
| 15 | `load_app_config` 添加异常包装：YAML 解析失败抛出 `ConfigError` 而非裸异常 |
| 16 | `_answer` handler 中的 `ValueError` 替换为 `ToolValidationError` |
| 17 | `resolve_context_path` 中的 `ValueError` / `FileNotFoundError` 替换为 `ContextPathEscapeError` / `ContextAssetNotFoundError` |
| 18 | `execute_read_only_sql` 中的 `ValueError` 替换为 `ReadOnlySQLViolationError` |
| 19 | 数据集模块中的 `ValueError` / `FileNotFoundError` 替换为 `DatasetError` / `TaskNotFoundError` |
| 20 | `_SKIP_ACTIONS` → `_SANITIZE_ACTIONS`：变量名更清晰地表达"消毒"语义 |
| 21 | `ReActAgentConfig` 从 `agents/react_agent.py` 迁移至 `config.py`，`agents/__init__.py` 保持 re-export |
| 22 | `PlanAndSolveAgentConfig` 从 `agents/plan_solve_agent.py` 迁移至 `config.py`，`agents/__init__.py` 保持 re-export |
| 23 | 审计确认 `agents/` 下无工具逻辑需迁移至 `tools/`，所有辅助模块均为 Agent 内部逻辑 |

---

## 模块对照表

| 原文件 (`data_agent_baseline/`) | 新文件 (`data_agent_refactored/`) | 变更类型 |
|---|---|---|
| `__init__.py` | `__init__.py` | 保持 |
| `config.py` | `config.py` | 增强（类型标注 + `ConfigError`） |
| `cli.py` | `cli.py` | 更新导入路径 |
| *(无)* | `exceptions.py` | **新增** |
| `agents_v2/__init__.py` | `agents/__init__.py` | **重命名**（去版本后缀） |
| `agents_v2/base_agent.py` | `agents/base_agent.py` | 增强（类型标注 + 日志 + 参数重命名） |
| `agents_v2/context_manager.py` | `agents/context_manager.py` | 增强（类型标注 + 变量重命名） |
| `agents_v2/gate.py` | `agents/data_preview_gate.py` | **重命名** + 类型标注 |
| `agents_v2/json_parser.py` | `agents/json_parser.py` | 增强（`ModelResponseParseError`） |
| `agents_v2/model.py` | `agents/model.py` | 增强（`ModelCallError` + 类型标注） |
| `agents_v2/plan_solve_agent.py` | `agents/plan_solve_agent.py` | 增强（类型标注 + 日志 + 参数重命名） |
| `agents_v2/prompt.py` | `agents/prompt.py` | 增强（类型标注） |
| `agents_v2/react_agent.py` | `agents/react_agent.py` | 增强（类型标注 + 日志 + 参数重命名） |
| `agents_v2/runtime.py` | `agents/runtime.py` | 保持（类型已完备） |
| `agents_v2/text_helpers.py` | `agents/text_helpers.py` | 增强（类型标注） |
| `agents_v2/timeout.py` | `agents/timeout.py` | 增强（泛型类型标注） |
| `benchmark/__init__.py` | `benchmark/__init__.py` | 更新导入路径 |
| `benchmark/dataset.py` | `benchmark/dataset.py` | 增强（自定义异常 + 日志） |
| `benchmark/schema.py` | `benchmark/schema.py` | 保持 |
| `run/__init__.py` | `run/__init__.py` | 更新导入路径 |
| `run/runner.py` | `run/runner.py` | **重写**（修复断裂导入 + 类型标注 + 日志） |
| `run/old_runner.py` | *(移除)* | **删除**（冗余废弃代码） |
| `tools/__init__.py` | `tools/__init__.py` | 增强（导出 `ToolHandler`） |
| `tools/filesystem.py` | `tools/filesystem.py` | 增强（自定义异常 + 类型标注） |
| `tools/python_exec.py` | `tools/python_exec.py` | 增强（类型标注） |
| `tools/registry.py` | `tools/registry.py` | **拆分**（handler 提取至 `handlers.py`） |
| *(registry.py 内 handler 函数)* | `tools/handlers.py` | **新增**（从 registry 提取） |
| `tools/sqlite.py` | `tools/sqlite.py` | 增强（自定义异常 + 类型标注） |

---

## 设计决策说明

### 1. `agents_v2/` → `agents/`

**理由**：Python 包名中嵌入版本号 (`_v2`) 违反模块命名最佳实践。版本信息应由 `__version__` 或包管理元数据承载，而非包名。重命名后，导入路径更简洁：`from data_agent_refactored.agents import ReActAgent`。

### 2. `gate.py` → `data_preview_gate.py`

**理由**：`gate` 是一个过于泛化的名称，无法表达"数据预览门控"这一具体业务语义。新名称 `data_preview_gate` 自文档化，开发者无需打开文件即可理解其职责。

### 3. `registry.py` 拆分为 `registry.py` + `handlers.py`

**理由**：原 `registry.py` (482 行) 承担了三重职责——数据结构定义、8 个工具 handler 实现、注册表管理。拆分后：
- `registry.py` (~120 行)：仅包含 `ToolSpec`、`ToolExecutionResult`、`ToolRegistry` 和工厂函数
- `handlers.py` (~120 行)：8 个 handler 函数，每个函数职责单一

这使得新增工具只需在 `handlers.py` 中添加函数，并在工厂函数中注册，无需修改注册表基础设施。

### 4. `ToolRegistry.register()` 方法

**理由**：原版 `ToolRegistry` 仅在工厂函数中通过字典字面量初始化，外部无法动态注册新工具。新增的 `register(spec, handler)` 方法支持：
- 插件式工具扩展（第三方代码可在运行时注册工具）
- 重复注册检测（默认抛出 `ValueError`，可通过 `overwrite=True` 覆盖）
- 日志记录（`logger.debug` 跟踪注册事件）

### 5. 自定义异常层次结构 (`exceptions.py`)

**理由**：原代码使用裸 `ValueError` / `KeyError` / `FileNotFoundError`，调用方无法精确区分"工具名不存在"和"参数格式错误"等不同场景。新的异常层次：
```
DataAgentError
├── ConfigError
├── DatasetError → TaskNotFoundError
├── ToolError → UnknownToolError, ToolTimeoutError, ToolValidationError, ...
├── AgentError → ModelCallError, ModelResponseParseError
└── RunnerError → InvalidRunIdError
```
所有异常继承自 `DataAgentError`，调用方可按需捕获粗粒度或细粒度异常。

### 6. 移除 `old_runner.py`

**理由**：`old_runner.py` (281 行) 与 `runner.py` (354 行) 代码高度重复（>90% 相同），且存在已知缺陷（`process.join` 死锁风险）。`runner.py` 已修复所有已知问题并增加了进度打印支持，`old_runner.py` 不再有保留价值。

### 7. 修复 `runner.py` 断裂导入

**理由**：原 `runner.py` 执行 `from data_agent_baseline.agents.model import OpenAIModelAdapter`，但项目中实际只有 `agents_v2/` 目录，没有 `agents/` 目录。此导入在运行时必然抛出 `ModuleNotFoundError`。重构后统一指向正确的 `agents.model` 模块。

### 8. 内部参数 `prog` → `show_progress`

**理由**：`prog` 是非标准缩写，新读者需要查看上下文才能理解含义。`show_progress` 是自文档化的参数名，符合 PEP 8 "避免不必要的缩写"原则。此变更**仅影响内部方法**，公开 API（`ReActAgentConfig.progress`、`PlanAndSolveAgentConfig.progress`）保持不变。

### 9. Agent 配置类统一迁移至 `config.py`

**理由**：`ReActAgentConfig` 和 `PlanAndSolveAgentConfig` 原先分别定义在各自的 Agent 文件中，这导致配置参数分散在多个模块。将它们迁移至 `config.py` 带来以下好处：
- **集中管理可配置参数**：所有配置 dataclass（`DatasetConfig`、`AgentConfig`、`RunConfig`、`ReActAgentConfig`、`PlanAndSolveAgentConfig`）统一定义在同一文件中，便于发现和维护
- **便于 YAML 统一加载**：未来可在 `load_app_config` 中统一加载 Agent 特定参数，无需跨模块导入
- **减少 Agent 文件的职责范围**：`react_agent.py` 和 `plan_solve_agent.py` 仅关注推理循环逻辑，不再承担配置定义职责
- **公开 API 兼容**：`agents/__init__.py` 仍然 re-export 这两个配置类，现有的 `from data_agent_refactored.agents import ReActAgentConfig` 导入路径继续可用

---

## 迁移指南

### 1. 包名变更

所有 `data_agent_baseline` 替换为 `data_agent_refactored`：

```python
# Before
from data_agent_baseline.agents_v2 import ReActAgent, ReActAgentConfig
from data_agent_baseline.tools.registry import ToolRegistry, create_default_tool_registry
from data_agent_baseline.config import load_app_config

# After
from data_agent_refactored.agents import ReActAgent, ReActAgentConfig
from data_agent_refactored.tools import ToolRegistry, create_default_tool_registry
from data_agent_refactored.config import load_app_config
```

### 2. `agents_v2` → `agents`

```python
# Before
from data_agent_baseline.agents_v2.model import OpenAIModelAdapter
from data_agent_baseline.agents_v2.gate import has_data_preview

# After
from data_agent_refactored.agents.model import OpenAIModelAdapter
from data_agent_refactored.agents.data_preview_gate import has_data_preview
```

### 3. `gate` → `data_preview_gate`

```python
# Before
from data_agent_baseline.agents_v2.gate import GATED_ACTIONS, DATA_PREVIEW_ACTIONS

# After
from data_agent_refactored.agents.data_preview_gate import GATED_ACTIONS, DATA_PREVIEW_ACTIONS
```

### 4. 异常类型变更

如果你的代码捕获了工具相关的 `ValueError` / `KeyError`，需要更新：

```python
# Before
try:
    registry.execute(task, action, action_input)
except KeyError:
    ...

# After
from data_agent_refactored.exceptions import UnknownToolError
try:
    registry.execute(task, action, action_input)
except UnknownToolError:
    ...

# Or catch the entire family:
from data_agent_refactored.exceptions import ToolError
try:
    registry.execute(task, action, action_input)
except ToolError:
    ...
```

### 5. 插件式工具注册

```python
from data_agent_refactored.tools import ToolRegistry, ToolSpec, ToolExecutionResult, create_default_tool_registry

registry = create_default_tool_registry()

# Register a custom tool at runtime
registry.register(
    ToolSpec(name="my_tool", description="Does something", input_schema={"param": "value"}),
    my_handler_function,
)
```

### 6. CLI 入口点

如果 `pyproject.toml` 中配置了 CLI 入口点，更新为：

```toml
[project.scripts]
dabench = "data_agent_refactored.cli:main"
```
