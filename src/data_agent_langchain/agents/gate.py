"""
``gate_node`` —— 数据预览门，作为图节点强制执行（§7）。

行为对齐 legacy ``PlanAndSolveAgent._handle_gate_block``，确保新后端
与 baseline 保持步骤级 parity：

  L1（连续阻塞次数 < ``max_gate_retries``）：
      - 追加一条 gate-block ``StepRecord``（action 沿用原工具名，
        observation = ``GATE_REMINDER``）。
      - ``gate_decision = "block"``，``skip_tool = True``，让条件边
        跳过 ``tool_node`` 直奔 ``advance_node``。

  L2（连续阻塞 == ``max_gate_retries``）：
      - 与 L1 同样的 block step，加上把当前 ``state["plan"][plan_index]``
        改写为 legacy 的 mandatory-inspect 文案。
      - 路由与 L1 一致（跳工具，下轮回到 model_node）。

  L3（连续阻塞 >= ``max_gate_retries`` + 2）：
      - 同样的 block step。
      - 直接调用 ``list_context``（短路，**不** 经过 ``model_node``），
        追加一条 forced-inject step。
      - **v4 M1**：``skip_tool = False``，让条件边继续走 ``tool_node``
        执行原 action。这样每次 L3 trip 在 trace 中正好留下 3 条 step
        （block + forced + 原 action），与 baseline fall-through
        路径一致（``plan_solve_agent.py:577-597``）。
      - 把 ``consecutive_gate_blocks = 0`` 重置，让计数干净循环。

本节点还处理一个 "透明" 情况：上一节点已经写了
``last_error_kind == "parse_error"``（``parse_action_node`` 没产出
有效 action），此时直接返回，不让空 ``action`` 触发任何 gate 规则。
（§9.2.1 中的软契约。）
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from data_agent_langchain.agents.runtime import StepRecord


# ---------------------------------------------------------------------------
# Gate 常量与判定函数
# ---------------------------------------------------------------------------
# 这几个名字也被 ``tools/descriptions.py`` / ``agents/tool_node.py`` /
# ``memory/working.py`` 引用；定义在文件最顶端，保证从那些模块迟到的
# 反向 import 不会与下面的 ``tools.factory`` 重型 import 形成竞态。

# 算作 "数据检视" 的工具名（成功后会解锁受门约束的工具）。
DATA_PREVIEW_ACTIONS: frozenset[str] = frozenset({
    "read_csv",
    "read_json",
    "read_doc",
    "inspect_sqlite_schema",
})

# 必须先做过数据预览才能用的工具名。
GATED_ACTIONS: frozenset[str] = frozenset({
    "execute_python",
    "execute_context_sql",
    "answer",
})

# gate block 时作为 observation 注入的提醒消息。
GATE_REMINDER: str = (
    "Blocked: inspect data first. Call read_csv / read_json / read_doc / "
    "inspect_sqlite_schema before using execute_python, execute_context_sql, or answer."
)


def has_data_preview(steps: Iterable[StepRecord]) -> bool:
    """*steps* 中是否存在至少一条成功的数据预览。

    形参是任何 ``Iterable[StepRecord]``，所以同时兼容 legacy step 列表
    与 LangGraph ``RunState["steps"]``。
    """
    return any(s.ok and s.action in DATA_PREVIEW_ACTIONS for s in steps)


# ---------------------------------------------------------------------------
# 重型 import 放在 gate 常量定义之后 —— 这样任何在 ``tools.factory``
# 初始化期间反向 import 本模块的消费者都能看到一个完整的命名空间。
# 会重新进入 ``tools.*`` 的 import 进一步推迟到函数局部
# （``_force_inject_list_context`` 内部），避免顶层 import 形成循环。
# ---------------------------------------------------------------------------
from data_agent_langchain.config import AppConfig  # noqa: E402
from data_agent_langchain.observability.events import dispatch_observability_event  # noqa: E402
from data_agent_langchain.runtime.context import get_current_app_config  # noqa: E402
from data_agent_langchain.runtime.rehydrate import build_runtime, rehydrate_task  # noqa: E402
from data_agent_langchain.runtime.state import RunState  # noqa: E402

logger = logging.getLogger(__name__)

# v2 C13 默认值：legacy backend 始终启用数据预览门，但 **不** 强制
# known-paths-only / discovery_done。Phase 3 沿用此默认；L3 escalation
# parity 测试基于此前提。


# ---------------------------------------------------------------------------
# 公共节点
# ---------------------------------------------------------------------------

def gate_node(state: RunState, config: Any | None = None) -> dict[str, Any]:
    """判定当前 action 是否通过数据预览门。

    L1/L2/L3 算法详见模块 docstring；返回 LangGraph reducer 用的
    partial state。
    """
    # parse_error 已经写过 step，这里直接放行（empty action 不该被 gate）。
    if state.get("last_error_kind") == "parse_error":
        return {
            "gate_decision": "pass",
            "skip_tool": True,  # parse_error 路径整体跳过 tool_node
        }

    app_config = _safe_get_app_config()

    if not _should_block(state, app_config):
        return {
            "consecutive_gate_blocks": 0,
            "gate_decision": "pass",
            "skip_tool": False,
            "last_error_kind": None,
        }

    blocks = int(state.get("consecutive_gate_blocks", 0) or 0) + 1
    dispatch_observability_event(
        "gate_block",
        {
            "step_index": int(state.get("step_index", 0) or 0),
            "level": _gate_level(blocks, int(getattr(app_config.agent, "max_gate_retries", 2))),
            "consecutive": blocks,
        },
        config,
    )
    block_step = _gate_block_record(state)
    update: dict[str, Any] = {
        "consecutive_gate_blocks": blocks,
        "steps": [block_step],
        "gate_decision": "block",
        "skip_tool": True,
        "last_tool_ok": False,
        "last_tool_is_terminal": False,
        "last_error_kind": "gate_block",
    }

    max_gate_retries = int(getattr(app_config.agent, "max_gate_retries", 2))

    if blocks >= max_gate_retries + 2:
        # ----- L3：短路 list_context，再 fall-through 到 tool_node -----
        forced_step = _force_inject_list_context(state, app_config)
        update["steps"] = [block_step, forced_step]
        update["consecutive_gate_blocks"] = 0
        update["gate_decision"] = "forced_inject"
        update["skip_tool"] = False  # v4 M1：原 action 同步执行
        update["last_tool_ok"] = forced_step.ok
        update["last_tool_is_terminal"] = False
        if forced_step.ok:
            update["discovery_done"] = True
        # 备注：known_paths 维护推迟到 Phase 5（plan §7.2 / C13），
        # parity baseline 从未设置过它。这里刻意不把 last_error_kind 改回
        # None —— 让它继续是 ``gate_block``，让 advance_node 走它原本对
        # gate fall-through 的规则。
        return update

    if blocks >= max_gate_retries:
        # ----- L2：改写当前 plan 步骤（仅 plan_solve 模式有意义） -----
        new_plan = _rewrite_current_step(state)
        if new_plan is not None:
            update["plan"] = new_plan

    return update


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _safe_get_app_config() -> AppConfig:
    """读取 contextvar 上的 AppConfig；测试场景下退化到默认配置。"""
    try:
        return get_current_app_config()
    except RuntimeError:
        # 不走 contextvar 直接驱动节点的测试也能拿到一份可用配置；
        # 生产 runner（D4）总是在调 ``compiled.invoke`` 之前
        # 调用 ``set_current_app_config`` 初始化 contextvar。
        from data_agent_langchain.config import default_app_config
        return default_app_config()


def _should_block(state: RunState, app_config: AppConfig) -> bool:
    """对应 legacy ``BaseAgent._check_gate``。"""
    if not bool(getattr(app_config.agent, "require_data_preview_before_compute", True)):
        return False
    action = state.get("action") or ""
    if action not in GATED_ACTIONS:
        return False
    return not has_data_preview(state.get("steps") or [])


def _gate_level(blocks: int, max_gate_retries: int) -> str:
    if blocks >= max_gate_retries + 2:
        return "L3"
    if blocks >= max_gate_retries:
        return "L2"
    return "L1"


def _gate_block_record(state: RunState) -> StepRecord:
    """构造 L1 block step，schema / 内容与 legacy 一致。"""
    action = state.get("action") or ""
    return StepRecord(
        step_index=int(state.get("step_index", 0) or 0),
        thought=state.get("thought", "") or "",
        action=action,
        action_input=dict(state.get("action_input") or {}),
        raw_response=state.get("raw_response", "") or "",
        observation={"ok": False, "tool": action, "error": GATE_REMINDER},
        ok=False,
        phase=state.get("phase", "") or "",
        plan_progress=state.get("plan_progress", "") or "",
        plan_step_description=state.get("plan_step_description", "") or "",
    )


def _force_inject_list_context(state: RunState, app_config: AppConfig) -> StepRecord:
    """直接执行 ``list_context``，对齐 legacy L3 行为。"""
    # 函数局部惰性 import：避免顶层与 ``tools.descriptions`` 形成循环
    # （后者会从本模块取 gate 常量）。
    from data_agent_langchain.tools.factory import create_all_tools
    from data_agent_langchain.tools.timeout import call_tool_with_timeout

    try:
        task = rehydrate_task(state)
        runtime = build_runtime(task, app_config)
        tools = {t.name: t for t in create_all_tools(task, runtime)}
        forced = call_tool_with_timeout(
            tools["list_context"],
            {"max_depth": 4},
            float(getattr(app_config.agent, "tool_timeout_s", 180.0)),
        )
        observation: dict[str, Any] = {
            "ok": forced.ok,
            "tool": "list_context",
            "content": forced.content,
        }
        ok = forced.ok
    except Exception as exc:
        # legacy 在这种情况下仅 log + 写一条失败 step；这里照搬。
        logger.warning("[gate] L3 force-inject list_context failed: %s", exc)
        observation = {"ok": False, "error": str(exc)}
        ok = False

    return StepRecord(
        step_index=int(state.get("step_index", 0) or 0),
        thought="[auto-injected by gate escalation]",
        action="list_context",
        action_input={"max_depth": 4},
        raw_response="",
        observation=observation,
        ok=ok,
        phase=state.get("phase", "execution") or "execution",
        plan_progress=state.get("plan_progress", "") or "",
        plan_step_description=state.get("plan_step_description", "") or "",
    )


def _rewrite_current_step(state: RunState) -> list[str] | None:
    """L2 plan 改写 —— 对齐 legacy 的 "MANDATORY: Inspect data files" 文案。"""
    plan = list(state.get("plan") or [])
    if not plan:
        return None
    plan_index = int(state.get("plan_index", 0) or 0)
    if plan_index >= len(plan):
        return None
    plan[plan_index] = (
        "MANDATORY: Inspect data files first by calling "
        "read_csv, read_json, read_doc, or inspect_sqlite_schema "
        "before any compute or answer action."
    )
    return plan


__all__ = ["gate_node"]