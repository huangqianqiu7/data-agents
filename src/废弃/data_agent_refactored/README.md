# Data Agent Refactored

> 基于 KDD Cup 2026 DataAgent-Bench 竞赛的重构版数据分析智能体，支持 ReAct 和 Plan-and-Solve 两种推理范式，自动完成数据探索、分析与答案提交。

## 📝 项目简介

本项目是对 `data_agent_baseline` 的深度重构版本，面向 **KDD Cup 2026 DataAgent-Bench** 数据智能体挑战赛。智能体读取结构化/半结构化数据任务，通过 LLM 驱动的推理循环自动完成数据分析并输出预测结果。

- **解决什么问题？** 自动化处理多种格式（CSV、JSON、SQLite、文本）的数据分析任务，无需人工编写分析代码，LLM 自主决策工具调用链路。
- **有什么特色功能？** 双范式支持（ReAct / Plan-and-Solve）、数据预览门控机制、插件式工具扩展、完整的自定义异常层次结构、全链路类型标注。
- **适用于什么场景？** DataAgent-Bench 竞赛评测、自动化数据探索与报表生成、LLM 工具调用范式的研究与实验。

## ✨ 核心功能

- [x] **ReAct 推理循环**：Thought → Action → Observation 循环，逐步推理并执行工具调用
- [x] **Plan-and-Solve 两阶段推理**：Phase 1 生成分步计划，Phase 2 逐步执行并支持动态重规划（Replan）
- [x] **8 种内置工具**：`list_context`、`read_csv`、`read_json`、`read_doc`、`inspect_sqlite_schema`、`execute_context_sql`、`execute_python`、`answer`
- [x] **数据预览门控**：强制 Agent 在执行代码/提交答案前先预览数据，三级升级策略（提醒 → 改写 → 强制注入）
- [x] **插件式工具注册**：`ToolRegistry.register()` 支持运行时动态注册自定义工具
- [x] **并行基准测试**：多线程并发执行任务，Rich 进度条实时展示状态
- [x] **自定义异常体系**：`DataAgentError` 为根的层次化异常，精确区分配置、工具、模型、数据集等错误
- [x] **YAML 配置驱动**：零代码修改即可切换模型、API、超时、并发等参数

## 🛠️ 技术栈

- **智能体范式**：ReAct（Reasoning + Acting）、Plan-and-Solve（规划 + 执行 + 重规划）
- **LLM 接口**：OpenAI 兼容 API（通过 `openai` SDK），支持任意兼容端点（如阿里云 DashScope）
- **数据处理**：`pandas`、`polars`、`numpy`、`pyarrow`、`openpyxl`、`duckdb`
- **数据库**：SQLite（只读 SQL 执行）
- **CLI 框架**：`typer` + `rich`（命令行界面与终端美化输出）
- **配置管理**：`pyyaml`（YAML 配置加载）
- **类型系统**：`pydantic`（数据校验）、全量 Python 类型标注
- **构建工具**：`hatchling`（PEP 517 构建后端）
- **包管理**：`uv`（推荐）

## 🚀 快速开始

### 环境要求

- Python 3.10+
- `uv` 包管理器（推荐）或 `pip`

### 安装依赖

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -e .
```

### 配置 API 密钥

```bash
# 复制示例配置
cp configs/react_baseline.example.yaml configs/react_baseline.local.yaml

# 编辑配置文件，填入你的模型名称、API 地址和密钥
```

配置文件示例：

```yaml
dataset:
  root_path: data/public/input

agent:
  model: YOUR_MODEL_NAME
  api_base: YOUR_API_BASE_URL
  api_key: YOUR_API_KEY
  max_steps: 16
  temperature: 0.0

run:
  output_dir: artifacts/runs
  run_id:
  max_workers: 4
  task_timeout_seconds: 600
```

### 运行项目

```bash
# 查看项目状态与数据集信息
uv run dabench status --config configs/react_baseline.local.yaml

# 查看单个任务详情
uv run dabench inspect-task task_1 --config configs/react_baseline.local.yaml

# 运行单个任务
uv run dabench run-task task_1 --config configs/react_baseline.local.yaml

# 运行完整基准测试
uv run dabench run-benchmark --config configs/react_baseline.local.yaml

# 限制任务数量
uv run dabench run-benchmark --config configs/react_baseline.local.yaml --limit 10
```

## 📖 使用示例

### 代码中直接调用 Agent

```python
from data_agent_refactored.agents import ReActAgent, ReActAgentConfig, OpenAIModelAdapter
from data_agent_refactored.tools import create_default_tool_registry
from data_agent_refactored.config import load_app_config
from pathlib import Path

# 加载配置
config = load_app_config(Path("configs/react_baseline.local.yaml"))

# 创建模型适配器与工具注册表
model = OpenAIModelAdapter(
    model=config.agent.model,
    api_base=config.agent.api_base,
    api_key=config.agent.api_key,
    temperature=config.agent.temperature,
)
tools = create_default_tool_registry()

# 创建并运行 ReAct Agent
agent = ReActAgent(model=model, tools=tools, config=ReActAgentConfig(max_steps=16))
```

### 注册自定义工具

```python
from data_agent_refactored.tools import ToolRegistry, ToolSpec, ToolExecutionResult, create_default_tool_registry

registry = create_default_tool_registry()

# 运行时注册自定义工具
registry.register(
    ToolSpec(name="my_tool", description="自定义工具描述", input_schema={"param": "value"}),
    my_handler_function,
)
```

## 🎯 项目亮点

- **双范式架构**：同时支持 ReAct 和 Plan-and-Solve 两种主流智能体范式，通过配置切换，无需改动代码
- **数据预览门控（Data Preview Gate）**：三级升级策略确保 Agent 在分析数据前先"看"数据，显著降低幻觉和错误答案
- **零新依赖重构**：从 `data_agent_baseline` 重构而来，保持完全相同的公开 API 行为，无需新增任何第三方依赖
- **完备的异常体系**：层次化自定义异常（`DataAgentError` → `ToolError` / `AgentError` / `ConfigError` 等），支持粗粒度或细粒度捕获
- **插件式工具扩展**：`ToolRegistry.register()` 方法支持运行时动态注册工具，适合二次开发
- **全量类型标注**：所有公开函数和方法均添加完整类型标注（参数 + 返回值），IDE 友好

## 📁 项目结构

```text
src/data_agent_refactored/
├── __init__.py                 # 包入口，版本定义
├── config.py                   # 全局配置（AppConfig / AgentConfig / RunConfig 等）
├── cli.py                      # CLI 命令行入口（status / inspect-task / run-task / run-benchmark）
├── exceptions.py               # 自定义异常层次结构
├── agents/                     # 智能体核心模块
│   ├── base_agent.py           # Agent 抽象基类（重试、工具执行、门控等共享逻辑）
│   ├── react_agent.py          # ReAct 范式 Agent
│   ├── plan_solve_agent.py     # Plan-and-Solve 范式 Agent
│   ├── model.py                # LLM 模型适配器（OpenAI 兼容）
│   ├── prompt.py               # 系统提示词与消息构建
│   ├── json_parser.py          # 模型 JSON 输出解析
│   ├── data_preview_gate.py    # 数据预览门控逻辑
│   ├── context_manager.py      # 上下文窗口管理（历史消息截断）
│   ├── runtime.py              # 运行时状态与步骤记录
│   ├── text_helpers.py         # 文本工具函数
│   └── timeout.py              # 超时控制（泛型实现）
├── tools/                      # 工具模块
│   ├── registry.py             # 工具注册表（ToolSpec / ToolRegistry / 工厂函数）
│   ├── handlers.py             # 8 个内置工具 handler 实现
│   ├── filesystem.py           # 文件系统工具（list_context / read_csv / read_json / read_doc）
│   ├── python_exec.py          # Python 代码执行工具
│   └── sqlite.py               # SQLite 工具（schema 检查 / 只读 SQL 执行）
├── benchmark/                  # 基准测试模块
│   ├── dataset.py              # 数据集加载器（DABenchPublicDataset）
│   └── schema.py               # 数据结构定义（PublicTask / AnswerTable）
└── run/                        # 运行器模块
    └── runner.py               # 单任务 / 批量基准测试执行器
```

## 📊 性能评估

运行基准测试后，结果输出至 `artifacts/runs/<run_id>/` 目录：

- **每个任务**：生成 `trace.json`（推理轨迹）和 `prediction.csv`（预测结果）
- **汇总报告**：生成 `summary.json`，包含成功率、失败原因等统计

```text
artifacts/runs/<run_id>/
├── summary.json
├── task_1/
│   ├── trace.json
│   └── prediction.csv
├── task_2/
│   ├── trace.json
│   └── prediction.csv
└── ...
```

## 🔮 未来计划

- [ ] 支持更多智能体范式（如 Reflexion、Tree-of-Thought）
- [ ] 集成更多数据处理工具（Excel 写入、图表生成等）
- [ ] 添加单元测试与集成测试覆盖
- [ ] 支持流式输出与中间结果可视化
- [ ] 优化上下文窗口管理策略，降低 token 消耗

## 🤝 贡献指南

欢迎提出 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m 'Add your feature'`
4. 推送分支：`git push origin feature/your-feature`
5. 创建 Pull Request

## 📄 许可证

MIT License

## 🔗 相关链接

- **竞赛官网**：[https://dataagent.top](https://dataagent.top)
- **GitHub Issues**：[https://github.com/HKUSTDial/kddcup2026-data-agents-starter-kit/issues](https://github.com/HKUSTDial/kddcup2026-data-agents-starter-kit/issues)
- **Discord 社区**：[https://discord.com/invite/7eFwJQN3Fx](https://discord.com/invite/7eFwJQN3Fx)
