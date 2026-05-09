"""
LangGraph 后端的子进程隔离 runner。

主要导出（详见 :mod:`run.runner`）：

  - :func:`run_single_task` —— 跑单个任务，写 trace.json / prediction.csv /
    metrics.json。
  - :func:`run_benchmark` —— 跑批量任务并写 summary.json，支持线程池并发。
  - :func:`create_run_output_dir` —— 创建本次 run 的输出目录。
  - :func:`resolve_run_id` —— 校验 / 生成 run_id。
  - :class:`TaskRunArtifacts` —— 单任务产物清单（路径 + 成功/失败状态）。

子进程内通过 ``set_current_app_config`` 把配置注入 contextvar
（D4 / §13.1），避免 ``RunnableConfig`` 序列化 ``AppConfig`` 失败。
``MetricsCollector`` 仅在 ``compiled.invoke`` 单点注入（D11 / §11.4.1）。
"""
from data_agent_langchain.run.runner import (
    TaskRunArtifacts,
    create_run_id,
    create_run_output_dir,
    resolve_run_id,
    run_benchmark,
    run_single_task,
)

__all__ = [
    "TaskRunArtifacts",
    "create_run_id",
    "create_run_output_dir",
    "resolve_run_id",
    "run_benchmark",
    "run_single_task",
]