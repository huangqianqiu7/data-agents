"""
任务 & 工具运行时的单路径重建（v3 D17 + v4 E8）。

需要调工具的图节点入口都先从可 pickle 的 ``RunState`` 重建 task 与
``ToolRuntime``。把重建放在节点 **内部**（而不是把 live 对象塞进 state）
让 ``RunState`` 保持 picklable，``MemorySaver`` / ``SqliteSaver``
checkpoint 才能序列化（§5.3 / C4）。

两个辅助函数：

  - :func:`rehydrate_task` 从 ``state["dataset_root"] + state["task_id"]``
    重新加载 :class:`PublicTask`（单路径，D17）。
  - :func:`build_runtime` 用重建的 task + 当前 ``AppConfig`` 产出新的
    :class:`ToolRuntime`（E8 / E10）。

两者都是微秒级开销 —— 不开文件、不连数据库、不 fork 子进程，
所以每次节点入口调一次都安全。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from data_agent_langchain.benchmark.dataset import DABenchPublicDataset
from data_agent_langchain.benchmark.schema import PublicTask
from data_agent_langchain.config import AppConfig
from data_agent_langchain.tools.tool_runtime import ToolRuntime


def rehydrate_task(state: Mapping[str, Any]) -> PublicTask:
    """从 ``state["dataset_root"]`` + ``task_id`` 重新加载 :class:`PublicTask`。

    形参 *state* 接受任何支持 ``.get("dataset_root")`` 与 ``["task_id"]``
    的 ``Mapping``（不强求 ``RunState`` TypedDict），方便用 plain dict 写测试。

    ``state["dataset_root"]`` 缺失时抛 ``RuntimeError`` —— runner 的
    ``_build_initial_state`` 必须填好这个字段（v4 E8）。
    """
    dataset_root = state.get("dataset_root")
    if not dataset_root:
        raise RuntimeError(
            "rehydrate_task requires state['dataset_root']; "
            "runner must populate it in _build_initial_state (v4 E8)."
        )
    task_id = state.get("task_id")
    if not task_id:
        raise RuntimeError("rehydrate_task requires state['task_id'].")
    dataset = DABenchPublicDataset(Path(dataset_root))
    return dataset.get_task(str(task_id))


def build_runtime(task: PublicTask, app_config: AppConfig) -> ToolRuntime:
    """用 *task* + *app_config* 构造新的 :class:`ToolRuntime`。

    返回的是 frozen dataclass，字段全是基本类型（无 ``Path`` / 无闭包），
    可以安全地被新 ``BaseTool`` 子类与 ``multiprocessing`` 子进程消费。
    """
    return ToolRuntime(
        task_dir=str(task.assets.task_dir),
        context_dir=str(task.assets.context_dir),
        python_timeout_s=app_config.tools.python_timeout_s,
        sql_row_limit=app_config.tools.sql_row_limit,
        max_obs_chars=app_config.agent.max_obs_chars,
    )


__all__ = ["build_runtime", "rehydrate_task"]