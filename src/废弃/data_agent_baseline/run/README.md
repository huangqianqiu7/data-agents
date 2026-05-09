# `run` 模块说明文档

## 1. 模块概述

`run` 模块是 KDD Cup 2026 Data Agent 基线方案的**任务执行引擎**，负责：

1. 调度 ReAct Agent 对 DABench 数据集中的任务逐一"做题"
2. 管理超时、子进程隔离、并发控制等运行时安全机制
3. 将每次运行的预测结果和推理轨迹（trace）持久化到磁盘

## 2. 文件结构

| 文件 | 职责 |
|---|---|
| `__init__.py` | 对外暴露 `TaskRunArtifacts`、`create_run_id`、`run_single_task`、`run_benchmark` |
| `runner.py` | **当前主逻辑**——包含完整的单任务执行、超时控制、多线程批量跑分流水线 |
| `old_runner.py` | 旧版 runner 备份，与 `runner.py` 结构基本一致，但缺少部分改进（见第 7 节） |

## 3. 核心数据结构

### `TaskRunArtifacts`（`dataclass`）

表示单次任务运行后的产物清单：

| 字段 | 类型 | 含义 |
|---|---|---|
| `task_id` | `str` | 任务 ID |
| `task_output_dir` | `Path` | 该任务专属输出文件夹 |
| `prediction_csv_path` | `Path \| None` | 预测 CSV 路径（失败时为 `None`） |
| `trace_path` | `Path` | 推理轨迹 JSON 文件路径 |
| `succeeded` | `bool` | 是否成功完成 |
| `failure_reason` | `str \| None` | 失败原因 |

提供 `to_dict()` 方法，用于 JSON 序列化。

## 4. 关键函数一览

### 4.1 辅助 / 工具函数

| 函数 | 作用 |
|---|---|
| `create_run_id()` | 生成形如 `20260414T134153Z` 的 UTC 时间戳批次号 |
| `resolve_run_id(run_id)` | 校验/规范化用户传入的 `run_id`，为空时自动生成 |
| `create_run_output_dir(output_root, run_id)` | 在 `output_root` 下创建以 `run_id` 命名的目录，返回 `(run_id, Path)` |
| `build_model_adapter(config)` | 根据 `AppConfig` 中的 agent 配置实例化 `OpenAIModelAdapter` |
| `_write_json(path, payload)` | 以 UTF-8 编码写出格式化 JSON |
| `_write_csv(path, columns, rows)` | 以 UTF-8 编码写出 CSV |
| `_failure_run_result_payload(task_id, reason)` | 快速构造一个"失败"结果字典 |

### 4.2 核心执行链

执行链按调用深度由浅到深排列：

```
run_benchmark (批量入口)
  └─ run_single_task (单任务入口，带计时)
       ├─ _run_single_task_with_timeout (超时控制 + 子进程隔离)
       │    ├─ _run_single_task_in_subprocess (子进程内执行体)
       │    │    └─ _run_single_task_core (真正调用 Agent 做题)
       │    └─ _reap_process (清理僵尸子进程)
       └─ _run_single_task_core (无隔离直连模式)
```

#### `_run_single_task_core`

- 加载指定 `task_id` 对应的任务数据
- 创建 `ReActAgent`（可复用外部传入的 `model` / `tools`，否则自行构建）
- 执行 `agent.run(task)` 并返回结果字典

#### `_run_single_task_in_subprocess`

- 在独立子进程中调用 `_run_single_task_core`
- 通过 `multiprocessing.Queue` 将结果（成功 / 异常）传回主进程
- 任何 `BaseException` 均被捕获，保证子进程不会静默崩溃

#### `_run_single_task_with_timeout`

- 若 `task_timeout_seconds ≤ 0`，直接在主进程裸跑（无隔离）
- 否则启动子进程 + 用 `queue.get(timeout=...)` 等待结果
- 超时后执行 `terminate → kill` 三级终止策略
- **关键改进**：先 `queue.get()` 再 `join()`，避免大 payload 导致管道死锁

#### `_reap_process`

- 等待子进程退出（`join`）
- 若仍存活则依次 `terminate` → `kill`，防止 Windows 上残留僵尸进程

#### `run_single_task`

- 公开 API，自带 `perf_counter` 计时
- 根据是否传入自定义 `model`/`tools` 决定走隔离子进程路径还是直连路径
- 调用 `_write_task_outputs` 将结果写入磁盘，返回 `TaskRunArtifacts`

#### `run_benchmark`

- 批量跑分的顶层入口
- 加载数据集，可通过 `limit` 参数限制任务数量
- **单线程模式**（`max_workers == 1` 或传入自定义 model/tools）：顺序执行，共享同一个 model 和 tools 实例
- **多线程模式**（`max_workers > 1`）：`ThreadPoolExecutor` 分发任务，每个线程内部再通过子进程隔离
- 执行完毕后生成 `summary.json` 汇总报告

## 5. 输出目录结构

一次完整的 `run_benchmark` 调用会产出如下目录结构：

```
artifacts/runs/
└── 20260414T134153Z/          # run_id
    ├── task_001/
    │   ├── trace.json          # 推理轨迹（含所有 steps）
    │   └── prediction.csv      # 预测结果（仅成功时存在）
    ├── task_002/
    │   ├── trace.json
    │   └── prediction.csv
    └── summary.json            # 汇总：run_id, 任务数, 成功数, 各任务详情
```

### `summary.json` 结构示例

```json
{
  "run_id": "20260414T134153Z",
  "task_count": 10,
  "succeeded_task_count": 8,
  "max_workers": 4,
  "tasks": [
    {
      "task_id": "task_001",
      "task_output_dir": "...",
      "prediction_csv_path": "...",
      "trace_path": "...",
      "succeeded": true,
      "failure_reason": null
    }
  ]
}
```

## 6. 并发与隔离模型

```
主进程 (run_benchmark)
│
├─ ThreadPoolExecutor (max_workers 个线程)
│   ├─ Thread-1 → run_single_task → subprocess (Agent 做题)
│   ├─ Thread-2 → run_single_task → subprocess (Agent 做题)
│   └─ ...
```

- **线程层**：`ThreadPoolExecutor` 控制并发度（默认 `max_workers=4`）
- **进程层**：每个任务在独立 `multiprocessing.Process` 中运行，内存泄漏或崩溃不影响主进程
- **超时层**：`queue.get(timeout=...)` 提供精确的超时控制（默认 600 秒）

当传入自定义 `model` 或 `tools` 时，强制降为单线程 + 无子进程隔离模式（因为这些对象通常不可跨进程序列化）。

## 7. `runner.py` vs `old_runner.py` 主要差异

| 方面 | `old_runner.py` | `runner.py`（当前版本） |
|---|---|---|
| 超时等待方式 | 先 `process.join(timeout)` 再 `queue.get()` | 先 `queue.get(timeout)` 再 `_reap_process()` |
| 死锁风险 | 大 payload 时 `join` 与 `put` 互相阻塞 → 死锁 | 已修复：先读管道再 join |
| 僵尸进程清理 | 内联处理 | 抽取为独立函数 `_reap_process` |
| 文件编码 | `_write_json` / `_write_csv` 未显式指定 encoding | 全部显式 `encoding="utf-8"` |
| 进度日志 | 无 | `_run_single_task_core` 增加 `progress` 参数，可打印加载日志 |

## 8. 依赖关系

```
run/runner.py
  ├── data_agent_baseline.agents.model.OpenAIModelAdapter   # LLM 接口适配器
  ├── data_agent_baseline.agents.react.ReActAgent           # ReAct 推理 Agent
  ├── data_agent_baseline.benchmark.dataset.DABenchPublicDataset  # 数据集加载
  ├── data_agent_baseline.config.AppConfig                  # 全局配置
  └── data_agent_baseline.tools.registry.ToolRegistry       # 工具注册表
```

## 9. 典型使用方式

```python
from data_agent_baseline.config import load_app_config
from data_agent_baseline.run import run_benchmark, run_single_task, create_run_id

config = load_app_config(Path("config.yaml"))

# 跑整个数据集
run_output_dir, artifacts = run_benchmark(config=config, limit=5)

# 或者只跑单个任务
from data_agent_baseline.run.runner import create_run_output_dir
_, run_dir = create_run_output_dir(config.run.output_dir)
artifact = run_single_task(task_id="task_001", config=config, run_output_dir=run_dir)
```
