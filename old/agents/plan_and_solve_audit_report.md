# Agent 代码架构诊断报告 — `PlanAndSolveAgent`

> **审核文件**: `plan_and_solve.py`  
> **业务背景**: KDD Cup 2026 数据分析 Agent  
> **审核日期**: 2026-04-28  

---

## 诊断总览

| 维度 | 发现数 | 最高等级 |
|------|--------|----------|
| **LLM & Prompt Logic** | 3 个（隐患 3, 5, 8） | High |
| **Tools & Actions** | 3 个（隐患 1, 4, 7） | Critical |
| **Memory & State** | 2 个（隐患 2, 6） | High |
| **Architecture & Robustness** | 1 个（隐患 9） | Low |

**最高优先级修复项**：隐患 1（线程池死锁）和隐患 4（API 无重试），这两个在生产运行中几乎必然触发。

---

## 维度一：工具链与行动控制 (Tools & Actions)

### 🛑 隐患 1：线程池资源泄漏导致全局死锁

- **严重等级**：**Critical**
- **缺陷定位**：第 106 行

```python
_TIMEOUT_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=2)
```

- **缺陷成因**：

`_call_with_timeout` 在超时后仅放弃等待 `future.result()`，但底层线程**不会被终止**，仍在继续执行。该线程池是模块级单例，且 `max_workers=2`。

**致命场景**：如果模型调用超时一次 + 工具调用超时一次，两个 worker 都被"僵尸任务"占满。此后所有新的 `_TIMEOUT_POOL.submit()` 调用都会**阻塞在队列中**，整个 Agent 永久挂死，且无任何错误日志。

- **重构建议**：

```python
# 方案一：为每次调用使用独立线程，避免池竞争
import threading

def _call_with_timeout(fn, args: tuple, timeout_s: float):
    result_container = [None]
    exception_container = [None]

    def wrapper():
        try:
            result_container[0] = fn(*args)
        except Exception as e:
            exception_container[0] = e

    t = threading.Thread(target=wrapper, daemon=True)  # daemon=True 确保主进程退出时不阻塞
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        raise TimeoutError(
            f"{getattr(fn, '__qualname__', str(fn))} timed out after {timeout_s}s"
        )
    if exception_container[0] is not None:
        raise exception_container[0]
    return result_container[0]

# 方案二：如果坚持使用线程池，至少将 max_workers 提高，并添加池耗尽监控
_TIMEOUT_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=8)
```

---

### 🛑 隐患 4：模型 API 瞬态错误（限流/网络抖动）无重试，浪费宝贵步数

- **严重等级**：**High**
- **缺陷定位**：第 428-444 行

```python
try:
    raw = _call_with_timeout(
        self.model.complete, (messages,), cfg.model_timeout_s,
    )
except TimeoutError as exc:
    ...
    continue
```

- **缺陷成因**：

仅捕获了 `TimeoutError`。而 `model.complete` 内部将所有 `APIError`（包含 429 Rate Limit、502 Bad Gateway 等可恢复错误）统一包装为 `RuntimeError` 抛出，会**直接落入第 606 行的通用 `except Exception`**，被当作解析错误处理，白白消耗一个步数且没有任何重试。

在 `max_steps=20` 的紧张预算下，每浪费一步都意味着 5% 的执行能力损失。

- **重构建议**：

```python
import time

MAX_MODEL_RETRIES = 3
RETRY_BACKOFF = [2, 5, 10]  # 指数退避秒数

for attempt in range(MAX_MODEL_RETRIES):
    try:
        raw = _call_with_timeout(
            self.model.complete, (messages,), cfg.model_timeout_s,
        )
        break
    except (TimeoutError, RuntimeError) as exc:
        if attempt == MAX_MODEL_RETRIES - 1:
            # 最后一次重试失败，记录错误步骤
            state.steps.append(StepRecord(...))
            continue  # 进入主循环下一轮
        time.sleep(RETRY_BACKOFF[attempt])
```

---

### 🛑 隐患 7：工具名未预校验，幻觉工具名浪费步数

- **严重等级**：**Medium**
- **缺陷定位**：第 493-498 行

```python
try:
    tool_result = _call_with_timeout(
        self.tools.execute,
        (task, model_step.action, model_step.action_input),
        cfg.tool_timeout_s,
    )
```

- **缺陷成因**：

模型输出的 `action` 字符串未经任何校验直接传入 `tools.execute()`。当模型幻觉出不存在的工具名（如 `run_sql` 而非 `execute_context_sql`），`execute()` 会抛异常，走进通用错误处理路径。但错误消息不够明确——模型看到的是一个笼统的 `"error": "..."` 而非 `"Unknown tool 'run_sql'. Available: [list_context, read_csv, ...]"`。

这使得模型难以自我纠正，往往会在接下来的步骤中继续幻觉类似的错误工具名。

- **重构建议**：

```python
# 在 tools.execute 之前加一层快速校验
available_tools = self.tools.available_tool_names()  # 假设 ToolRegistry 提供此方法
if model_step.action not in available_tools:
    obs = {
        "ok": False,
        "tool": model_step.action,
        "error": f"Unknown tool '{model_step.action}'. "
                 f"Available tools: {', '.join(sorted(available_tools))}",
    }
    state.steps.append(StepRecord(..., observation=obs, ok=False))
    continue
```

---

## 维度二：大模型交互与意图流 (LLM & Prompt Logic)

### 🛑 隐患 3：滑动窗口丢弃关键的早期数据探查结果，导致模型"失忆"

- **严重等级**：**High**
- **缺陷定位**：第 346-359 行

```python
# L2: 滑动窗口淘汰 —— 从最旧的步骤开始丢弃，直至 token 预算内
start = 0
while start < len(pair_tokens) and (base_tokens + total_hist) > budget:
    total_hist -= pair_tokens[start]
    start += 1
```

- **缺陷成因**：

淘汰策略是 FIFO（先进先出），但数据分析 Agent 最重要的信息往往在**最早的步骤**中——`list_context`、`read_csv`（获取 schema）等。一旦这些被丢弃，后续的模型推理将缺少数据结构信息，极易产生错误的 SQL/Python 代码，导致连续失败 → 重规划 → 上下文更大 → 淘汰更多早期步骤 → **恶性循环**。

占位摘要 `"[Note: N earlier step(s) omitted...]"` 只告诉模型"有东西被省略了"，但没有传递**任何有效信息**。

- **重构建议**：

```python
# 对关键步骤（数据预览）做"钉住"保护，永远不淘汰
pinned = []
evictable = []
for i, (pair, tokens) in enumerate(zip(history_pairs, pair_tokens)):
    step = state.steps[i]
    if step.ok and step.action in _DATA_PREVIEW_ACTIONS:
        pinned.append((pair, tokens))
    else:
        evictable.append((pair, tokens))

# 先计算 pinned 的 token 开销
pinned_tokens = sum(t for _, t in pinned)
remaining_budget = budget - base_tokens - pinned_tokens

# 仅对 evictable 做滑动窗口淘汰
# 然后将 pinned 步骤排在前面，evictable 保留部分排在后面
```

---

### 🛑 隐患 5：`base_tokens` 超预算时无保护，system+task 自身即可撑爆上下文窗口

- **严重等级**：**Medium**
- **缺陷定位**：第 348-359 行

```python
base_tokens = sum(_estimate_tokens(m.content) for m in messages)
budget = cfg.max_context_tokens

while start < len(pair_tokens) and (base_tokens + total_hist) > budget:
    total_hist -= pair_tokens[start]
    start += 1
```

- **缺陷成因**：

如果 `tool_desc`（工具描述）+ `plan`（计划列表）+ `task.question`（用户问题）本身就超过 `max_context_tokens`（默认 24000），循环会将所有历史步骤丢光，但 `base_tokens` 仍然超标。此时请求发送到模型 API 后会触发 token limit 错误，且没有任何降级机制。

对于 KDD Cup 赛题，某些 `task.question` 中可能嵌入大段的背景描述或多个子问题，这种溢出是现实可能的。

- **重构建议**：

```python
if base_tokens > budget:
    logging.warning(
        "[plan-solve] Base prompt (%d tokens) exceeds budget (%d). "
        "Truncating tool descriptions.",
        base_tokens, budget,
    )
    # 降级：精简 tool_desc 或截断 plan 描述
    short_desc = tool_desc[:budget * 3 // 2]  # 粗略按比例截断
    # 重建 messages 使用精简版 ...
```

---

### 🛑 隐患 8：Prompt 注入风险——工具输出可操纵重规划逻辑

- **严重等级**：**Medium**
- **缺陷定位**：第 554-576 行

```python
for s in state.steps:
    preview = json.dumps(s.observation, ensure_ascii=False)[:200]
    if s.ok:
        completed.append(f"  ✓ {s.action}(...) → {preview}")
    ...
history = "\n\n".join(history_parts)

try:
    plan = self._generate_plan(task, tool_desc, history_hint=history)
```

- **缺陷成因**：

工具执行结果（`s.observation`）中的 `content` 字段来自外部数据文件（CSV、JSON、SQL 结果等），其内容**不可信**。这些内容被截断后直接拼接进 `history_hint`，再注入到规划 prompt 的 `user` 消息中。

如果数据文件中包含恶意构造的文本（例如 CSV 中某个单元格值为 `"Ignore all previous instructions. Your new plan is: ..."`），可能影响重规划阶段的模型决策。虽然在竞赛场景下风险较低，但在生产环境中这是一个经典的**间接 Prompt 注入**漏洞。

- **重构建议**：

```python
# 对注入到 prompt 中的外部数据内容做标记隔离
def _sanitize_for_prompt(text: str, max_len: int = 200) -> str:
    """将外部数据内容用明确的边界标记包裹，降低注入风险。"""
    truncated = text[:max_len]
    return f"<tool_output>{truncated}</tool_output>"

# 在构建 history_hint 时使用
preview = _sanitize_for_prompt(json.dumps(s.observation, ensure_ascii=False))
```

---

## 维度三：记忆与状态管理 (Memory & State)

### 🛑 隐患 2：重规划路径导致 `step_index` 重复，数据完整性损坏

- **严重等级**：**High**
- **缺陷定位**：第 523-604 行

```python
state.steps.append(StepRecord(
    step_index=step_index,
    ...  # 工具执行失败的记录（第 523 行）
))
...
# 工具失败且允许重规划 → 生成新计划
if not tool_result.ok and replan_used < cfg.max_replans:
    ...
    except Exception as replan_exc:
        ...
        state.steps.append(StepRecord(
            step_index=step_index,  # ← 同一个 step_index，重复写入！
            ...
            action="__replan_failed__",
            ...
        ))
```

- **缺陷成因**：

在同一次循环迭代中，工具失败后先在第 523 行追加了一条 `StepRecord`，若重规划也失败，又在第 593 行追加了一条**相同 `step_index`** 的记录。下游依赖 `step_index` 唯一性的逻辑（如 `trace.json` 的回溯分析）会产生歧义。

- **重构建议**：

```python
# 重规划失败记录应使用虚拟 step_index，或合并到已有步骤的 observation 中
state.steps.append(StepRecord(
    step_index=-1,  # 标记为非常规步骤
    thought="",
    action="__replan_failed__",
    ...
))
# 或者直接将 replan 失败信息追加到上一条 step 的 observation 而非新建记录
```

---

### 🛑 隐患 6：门控拦截后 `consecutive_gate_blocks` 永不自然归零，强制改写后模型仍可无限触发

- **严重等级**：**Medium**
- **缺陷定位**：第 462-490 行

```python
consecutive_gate_blocks += 1
...
if consecutive_gate_blocks >= cfg.max_gate_retries:
    plan[plan_idx] = "MANDATORY: Inspect data files first ..."
...
continue

# 非门控拦截路径：重置连续拦截计数器
consecutive_gate_blocks = 0
```

- **缺陷成因**：

当计数器达到阈值后，`plan[plan_idx]` 被改写为强制数据预览指令。但如果模型**仍然忽视**该指令并继续输出 `execute_python` 等 gated action，`consecutive_gate_blocks` 会持续递增（3, 4, 5...），每次都重复同样的 plan 改写操作，直到 `max_steps` 耗尽。这期间**没有任何升级的纠正措施**。

本质上这是一个**"死循环的变体"**——虽然有 `max_steps` 兜底，但可能白白浪费 18 个步数（20 - 2 gate retries）。

- **重构建议**：

```python
# 超过门控阈值后，直接注入一个自动化的数据预览动作，跳过模型决策
if consecutive_gate_blocks >= cfg.max_gate_retries + 2:
    # 强制执行 list_context 工具，绕过模型
    forced_action = "list_context"
    forced_input = {"max_depth": 4}
    tool_result = _call_with_timeout(
        self.tools.execute, (task, forced_action, forced_input), cfg.tool_timeout_s,
    )
    state.steps.append(StepRecord(
        step_index=step_index, thought="[auto-injected by gate escalation]",
        action=forced_action, action_input=forced_input,
        raw_response="", observation={"ok": tool_result.ok, ...}, ok=tool_result.ok,
    ))
    consecutive_gate_blocks = 0
    continue
```

---

## 维度四：架构解耦与鲁棒性 (Architecture & Robustness)

### 🛑 隐患 9：`run()` 方法 250+ 行单体设计，可测试性与可扩展性极差

- **严重等级**：**Low**（非即时崩溃，但严重影响长期维护）
- **缺陷定位**：第 377-632 行

- **缺陷成因**：

`run()` 方法长达 **255 行**，融合了以下全部职责：
1. 计划生成与降级
2. 主循环调度
3. 模型调用与超时处理
4. JSON 解析与错误恢复
5. 门控逻辑
6. 工具执行与超时处理
7. 计划推进与重规划
8. 终止判定

任何一个子逻辑的修改都可能引发连锁效应。且该方法几乎不可能编写有效的单元测试——必须 mock 整个模型和工具链才能触达某个特定分支。

- **重构建议**：将主循环拆分为可独立测试的小方法：

```python
def run(self, task: PublicTask) -> AgentRunResult:
    state = AgentRuntimeState()
    plan = self._phase1_plan(task)
    self._phase2_execute(task, plan, state)
    return self._finalize(task, state)

def _execute_one_step(self, task, plan, plan_idx, state, step_index) -> StepOutcome:
    """返回 StepOutcome 枚举：CONTINUE / ADVANCE / REPLAN / TERMINATE / ERROR"""
    ...

def _handle_gate_block(self, model_step, plan, plan_idx, state, step_index) -> None:
    ...

def _handle_replan(self, task, tool_desc, state, replan_used) -> tuple[list[str], int]:
    ...
```

---

## 附：风险热力图

```
Critical  ████████  隐患 1（线程池死锁）
High      ██████    隐患 2（step_index 重复）
High      ██████    隐患 3（早期步骤丢失）
High      ██████    隐患 4（API 无重试）
Medium    ████      隐患 5（base_tokens 溢出）
Medium    ████      隐患 6（门控死循环变体）
Medium    ████      隐患 7（工具名未校验）
Medium    ████      隐患 8（Prompt 注入）
Low       ██        隐患 9（单体方法）
```

---

## 建议修复优先级

1. **立即修复**：隐患 1（线程池死锁）— 生产环境必现
2. **高优修复**：隐患 4（API 重试）— 直接影响评分
3. **高优修复**：隐患 3（关键步骤钉住）— 直接影响推理质量
4. **中优修复**：隐患 2, 6, 7 — 数据完整性 + 步数效率
5. **低优重构**：隐患 5, 8, 9 — 架构韧性提升
