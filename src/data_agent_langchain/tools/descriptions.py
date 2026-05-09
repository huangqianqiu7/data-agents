"""
工具 description 注册表 —— 让新 ``BaseTool`` 子类的 description 块与 legacy
``ToolRegistry.describe_for_prompt()`` 输出 **字符级** 一致（v3 D8 / §6.5）。

为什么重要
----------
``json_action`` 模式下 description 块会拼接进系统提示词。任何措辞漂移都会
偏移 LLM 输出分布，破坏 parity 测试；description 块也是 trace 偏差最常见
的隐藏元凶，所以 parity 测试断言字符串严格相等。

结构
----
``_LEGACY_DESCRIPTIONS`` 以 8 个内置工具名为 key，每条记录两个字段：
``description``（LLM 看到的人读说明）和 ``input_schema``（冒号后的示例
载荷）。``EXECUTE_PYTHON_TIMEOUT_SECONDS`` 从共享常量导入（v4 E11），让
描述里的超时数与实际沙箱超时永不漂移。
"""
from __future__ import annotations

from typing import Any

from data_agent_langchain.agents.gate import (
    DATA_PREVIEW_ACTIONS,
    GATED_ACTIONS,
)
from data_agent_langchain.tools.python_exec import EXECUTE_PYTHON_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# 工具名分组 —— 从 ``agents.gate`` re-export，让 legacy backend、新 gate
# 节点和新 tools 层共享同一个 frozenset 对象（identity 相等而非仅 value
# 相等）。这能消除一类 "value 相等但 identity 测试失败" 的常见 bug。
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 字符级 description 注册表
# ---------------------------------------------------------------------------

# 真值来源：``data_agent_refactored/tools/registry.py:130-204``。
# 任何修改都必须同步到 legacy registry，否则 parity 测试
# ``test_phase2_descriptions_parity`` 会立即失败。
_LEGACY_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "answer": {
        "description": (
            "Submit the final answer table. "
            "This is the only valid terminating action."
        ),
        "input_schema": {
            "columns": ["column_name"],
            "rows": [["value_1"]],
        },
    },
    "execute_context_sql": {
        "description": (
            "Run a read-only SQL query against a sqlite/db file inside context."
        ),
        "input_schema": {
            "path": "relative/path/to/file.sqlite",
            "sql": "SELECT ...",
            "limit": 200,
        },
    },
    "execute_python": {
        # v4 E11：description 中的超时数必须来自共享常量，确保它跟踪实际
        # 沙箱设置。
        "description": (
            "Execute arbitrary Python code with the task context directory as the "
            "working directory. The tool returns the code's captured stdout as `output`. "
            f"The execution timeout is fixed at {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
        ),
        "input_schema": {
            "code": "import os\nprint(sorted(os.listdir('.')))",
        },
    },
    "inspect_sqlite_schema": {
        "description": (
            "Inspect tables and columns in a sqlite/db file inside context."
        ),
        "input_schema": {"path": "relative/path/to/file.sqlite"},
    },
    "list_context": {
        "description": "List files and directories available under context.",
        "input_schema": {"max_depth": 4},
    },
    "read_csv": {
        "description": "Read a preview of a CSV file inside context.",
        "input_schema": {
            "path": "relative/path/to/file.csv",
            "max_rows": 20,
        },
    },
    "read_doc": {
        "description": "Read a text-like document inside context.",
        "input_schema": {
            "path": "relative/path/to/file.md",
            "max_chars": 4000,
        },
    },
    "read_json": {
        "description": "Read a preview of a JSON file inside context.",
        "input_schema": {
            "path": "relative/path/to/file.json",
            "max_chars": 4000,
        },
    },
}


# v4 E12：工具名集合，被 parity 测试用来断言新 backend 没有删 / 改名 / 新
# 增任何工具。
ALL_TOOL_NAMES: frozenset[str] = frozenset(_LEGACY_DESCRIPTIONS.keys())


# ---------------------------------------------------------------------------
# 渲染辅助函数
# ---------------------------------------------------------------------------

def render_legacy_description(name: str) -> str:
    """返回 ``BaseTool.description`` 用的字符串。

    *name* 不是注册的内置工具时直接抛 ``KeyError``（不包装异常），让开发
    阶段的拼写错误立即在模块 import 时暴露。
    """
    return _LEGACY_DESCRIPTIONS[name]["description"]


def render_legacy_input_schema(name: str) -> dict[str, Any]:
    """返回某工具的 ``input_schema`` 示例字典。"""
    return dict(_LEGACY_DESCRIPTIONS[name]["input_schema"])


def render_legacy_prompt_block(tool_names: list[str] | None = None) -> str:
    """渲染 ``ToolRegistry.describe_for_prompt()`` 历史输出格式。

    legacy 实现遍历 ``sorted(self.specs)`` 并每个工具产出两行::

        - {name}: {description}
          input_schema: {input_schema}

    本函数字符级复刻该输出，parity 测试就是据此断言。

    Args:
        tool_names: 显式工具名列表；``None`` 时渲染所有已注册工具
            （字典序）。
    """
    names = sorted(tool_names if tool_names is not None else _LEGACY_DESCRIPTIONS.keys())
    lines: list[str] = []
    for name in names:
        spec = _LEGACY_DESCRIPTIONS[name]
        lines.append(f"- {name}: {spec['description']}")
        lines.append(f"  input_schema: {spec['input_schema']}")
    return "\n".join(lines)


__all__ = [
    "ALL_TOOL_NAMES",
    "DATA_PREVIEW_ACTIONS",
    "GATED_ACTIONS",
    "render_legacy_description",
    "render_legacy_input_schema",
    "render_legacy_prompt_block",
]