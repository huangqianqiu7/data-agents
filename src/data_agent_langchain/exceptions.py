"""
LangGraph 后端的异常层次。

所有领域异常都继承自 :class:`DataAgentError`，调用方一条 ``except
DataAgentError`` 就能捕获全家族。

本模块是 langchain backend 异常体系的唯一来源；Phase 7 之后不再从
``data_agent_common`` 导入。
"""
from __future__ import annotations


class DataAgentError(Exception):
    """所有数据智能体异常的根类。"""


# ---------------------------------------------------------------------------
# 配置类异常
# ---------------------------------------------------------------------------
class ConfigError(DataAgentError):
    """应用配置缺失或非法时抛出。"""


class GatewayCapsMissingError(ConfigError):
    """启动时 ``gateway_caps.yaml`` 缺失（v4 M3）。

    Phase 0.5 smoke test 产物必须先存在再编译图，否则系统会在网关其实
    不支持 tool-calling 的情况下假装支持。
    """


class ReproducibilityViolationError(ConfigError):
    """``evaluation.reproducible=true`` 与破坏确定性的特性同时启用时抛出。

    例如开了 LangSmith / 缺少 seed 等。
    """


# ---------------------------------------------------------------------------
# benchmark / 数据集异常
# ---------------------------------------------------------------------------
class DatasetError(DataAgentError):
    """数据集结构层级错误（目录缺失、JSON 损坏等）。"""


class TaskNotFoundError(DatasetError):
    """指定 task_id 在数据集中不存在。"""


# ---------------------------------------------------------------------------
# 工具相关异常
# ---------------------------------------------------------------------------
class ToolError(DataAgentError):
    """工具相关错误的基类。"""


class UnknownToolError(ToolError):
    """智能体请求了未注册的工具名。"""

    def __init__(self, tool_name: str, available_tools: frozenset[str] | set[str]) -> None:
        self.tool_name = tool_name
        self.available_tools = available_tools
        super().__init__(
            f"Unknown tool '{tool_name}'. "
            f"Available tools: {', '.join(sorted(available_tools))}"
        )


class ToolTimeoutError(ToolError):
    """工具执行超出时间预算。"""

    def __init__(self, tool_name: str, timeout_seconds: float) -> None:
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Tool '{tool_name}' timed out after {timeout_seconds}s. "
            f"Try simplifying your code or query."
        )


class ToolValidationError(ToolError):
    """工具输入校验失败（如 ``answer`` 表格 shape 不合法）。"""


class ContextPathEscapeError(ToolError):
    """相对路径试图逃出任务的 ``context_dir``。"""

    def __init__(self, relative_path: str) -> None:
        self.relative_path = relative_path
        super().__init__(f"Path escapes context dir: {relative_path}")


class ContextAssetNotFoundError(ToolError):
    """引用的文件在 ``context_dir`` 中不存在。"""

    def __init__(self, relative_path: str) -> None:
        self.relative_path = relative_path
        super().__init__(
            f"Missing context asset: {relative_path}. "
            f"Call `list_context` first to discover the actual files available."
        )


class ReadOnlySQLViolationError(ToolError):
    """SQL 语句不是只读形式。"""


# ---------------------------------------------------------------------------
# 智能体 / 模型异常
# ---------------------------------------------------------------------------
class AgentError(DataAgentError):
    """智能体层错误的基类。"""


class ModelCallError(AgentError):
    """模型 API 调用在所有重试用尽后仍失败。"""

    def __init__(self, attempts: int, last_error: str) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Model call failed after {attempts} attempts: {last_error}")


class ModelResponseParseError(AgentError):
    """模型 JSON 响应无法解析。"""


# ---------------------------------------------------------------------------
# Runner 异常
# ---------------------------------------------------------------------------
class RunnerError(DataAgentError):
    """运行器层错误的基类。"""


class InvalidRunIdError(RunnerError):
    """run_id 字符串不合法。"""


__all__ = [
    "AgentError",
    "ConfigError",
    "ContextAssetNotFoundError",
    "ContextPathEscapeError",
    "DataAgentError",
    "DatasetError",
    "GatewayCapsMissingError",
    "InvalidRunIdError",
    "ModelCallError",
    "ModelResponseParseError",
    "ReadOnlySQLViolationError",
    "ReproducibilityViolationError",
    "RunnerError",
    "TaskNotFoundError",
    "ToolError",
    "ToolTimeoutError",
    "ToolValidationError",
    "UnknownToolError",
]