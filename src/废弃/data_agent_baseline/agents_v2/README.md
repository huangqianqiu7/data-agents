# agents_v2 — 重构后的 Agent 模块

## 重构自
`data_agent_baseline.agents` (原 `plan_and_solve.py` 819 行单体 + `react.py` 427 行)

## 重构目标
1. **消除重复**：JSON 解析、门控逻辑、进度打印等在 react.py 和 plan_and_solve.py 中重复实现 → 提取为共享模块
2. **单一职责**：将 819 行的 plan_and_solve.py 拆分为可独立测试的小模块
3. **修复审计隐患**：线程池死锁（隐患1）、API 无重试（隐患4）、关键步骤丢失（隐患3）等
4. **继承复用**：通过 `BaseAgent` 基类封装共享逻辑，两个 Agent 只需实现各自的主循环

## 文件结构

```
agents_v2/
├── __init__.py              # 统一导出
├── model.py                 # 数据结构：ModelAdapter, ModelMessage, ModelStep, 适配器
├── runtime.py               # 运行时状态：StepRecord, AgentRuntimeState, AgentRunResult
├── prompt.py                # 所有提示词（ReAct + Plan-and-Solve）+ 构建函数
├── json_parser.py           # JSON 解析工具（三级降级：严格 → 修复括号 → json_repair）
├── timeout.py               # 线程超时包装器（独立 daemon 线程，修复线程池死锁）
├── text_helpers.py          # 文本工具：进度打印、JSON 预览、token 估算、observation 截断
├── gate.py                  # 数据预览门控：常量 + 检查逻辑
├── context_manager.py       # 历史上下文管理：钉住关键步骤 + 滑动窗口淘汰
├── base_agent.py            # 抽象基类：模型重试、工具校验执行、收尾封装
├── react_agent.py           # ReAct Agent（继承 BaseAgent）
├── plan_solve_agent.py      # Plan-and-Solve Agent（继承 BaseAgent）
└── README.md                # 本文件
```

## 模块依赖关系

```
base_agent.py
  ├── model.py
  ├── runtime.py
  ├── timeout.py
  ├── text_helpers.py
  └── gate.py

react_agent.py (extends BaseAgent)
  ├── base_agent.py
  ├── prompt.py
  └── json_parser.py

plan_solve_agent.py (extends BaseAgent)
  ├── base_agent.py
  ├── prompt.py
  ├── json_parser.py
  ├── gate.py
  ├── context_manager.py
  └── timeout.py

context_manager.py
  ├── model.py
  ├── runtime.py
  ├── prompt.py
  ├── gate.py
  └── text_helpers.py
```

## 使用示例

```python
from data_agent_baseline.agents_v2 import (
    PlanAndSolveAgent, PlanAndSolveAgentConfig,
    ReActAgent, ReActAgentConfig,
    OpenAIModelAdapter,
)

# Plan-and-Solve Agent
ps_agent = PlanAndSolveAgent(
    model=my_model,
    tools=my_tools,
    config=PlanAndSolveAgentConfig(max_steps=20, progress=True),
)
result = ps_agent.run(task)

# ReAct Agent
react_agent = ReActAgent(
    model=my_model,
    tools=my_tools,
    config=ReActAgentConfig(max_steps=16, progress=True),
)
result = react_agent.run(task)
```

## 审计隐患修复清单

| 隐患 | 等级 | 修复位置 |
|------|------|----------|
| 1. 线程池死锁 | Critical | `timeout.py` — 改用独立 daemon 线程 |
| 2. step_index 重复 | High | `plan_solve_agent.py` — 重规划失败用 step_index=-1 |
| 3. 早期步骤丢失 | High | `context_manager.py` — 钉住数据预览步骤 |
| 4. API 无重试 | High | `base_agent.py` — 指数退避重试 |
| 5. base_tokens 溢出 | Medium | `context_manager.py` — 超预算警告 + 最小余量 |
| 6. 门控死循环 | Medium | `plan_solve_agent.py` — 分级升级 + 强制注入 |
| 7. 工具名未校验 | Medium | `base_agent.py` — 预校验 + 清晰错误消息 |
| 9. 单体方法 | Low | 拆分为 BaseAgent + 子方法 |
