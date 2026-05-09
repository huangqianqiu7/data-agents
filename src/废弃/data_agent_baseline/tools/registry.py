"""
registry.py — 工具注册中心 (Tool Registry)

本文件是整个 Agent 工具系统的"总控中心"，承担三大职责：
1. 定义工具的"规格说明书" (ToolSpec) —— 告诉大模型有哪些工具可用、怎么用
2. 定义工具的"执行结果格式" (ToolExecutionResult) —— 统一所有工具返回值的数据结构
3. 实现工具的"注册表" (ToolRegistry) —— 把工具名称映射到真正的 Python 执行函数

打个比方：
- ToolSpec 就像餐厅的菜单（告诉顾客有什么菜、每道菜需要什么配料）
- ToolHandler 就像后厨的厨师（真正动手做菜的人）
- ToolRegistry 就像前台服务员（顾客点菜后，服务员把订单传给对应的厨师去执行）

文件结构：
├── 基础数据结构定义 (ToolSpec, ToolExecutionResult, ToolHandler 类型别名)
├── 具体工具的底层执行函数 (_list_context, _read_csv, _answer 等)
├── ToolRegistry 类（工具箱管理器）
└── create_default_tool_registry() 工厂函数（一键组装默认工具箱）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.filesystem import (
    list_context_tree,
    read_csv_preview,
    read_doc_preview,
    read_json_preview,
    resolve_context_path,
)
from data_agent_baseline.tools.python_exec import execute_python_code
from data_agent_baseline.tools.sqlite import execute_read_only_sql, inspect_sqlite_schema

# 核心安全机制：防止大模型写出死循环代码导致测评系统无限卡死
# 超过 30 秒的 Python 代码执行会被强制终止
EXECUTE_PYTHON_TIMEOUT_SECONDS = 30

# ==========================================
# 1. 基础数据结构定义
# ==========================================

@dataclass(frozen=True, slots=True)
class ToolSpec:
    """
    工具的元数据规范（发给大模型看的"使用说明书"）。
    
    这个类本身不包含任何执行逻辑，它只是一份"产品说明"，
    会被序列化成文字塞进 System Prompt 里发给大模型，
    让大模型知道"我有哪些工具可以用"以及"每个工具需要什么参数"。
    
    frozen=True: 说明书一旦印好就不能改（不可变对象），保障线程安全。
    slots=True: 使用 __slots__ 优化内存占用，不允许动态添加属性。
    
    属性:
        name: 工具名称（如 "execute_python"），大模型在 action 字段里填这个名字来调用工具
        description: 工具功能描述（人类可读的自然语言），帮助大模型理解什么场景该用这个工具
        input_schema: 工具的输入参数结构定义（以字典形式展示参数名和示例值），
                      告诉大模型 action_input 应该怎么填
    """
    name: str                       # 工具名称，如 "read_csv", "execute_python"
    description: str                # 工具描述，如 "Read a preview of a CSV file inside context."
    input_schema: dict[str, Any]    # 工具的输入参数结构定义（包含参数名和示例值）


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """
    工具执行完毕后的标准化返回结果。
    
    不管是执行 Python 代码、读 CSV 文件、还是提交最终答案，
    所有工具执行完后都必须返回这个统一格式的对象。
    这样 ReActAgent 的主循环就不需要关心"到底调了什么工具"，
    只需要统一处理这个结果对象即可（面向接口编程的思想）。
    
    属性:
        ok: 执行是否成功（True=成功, False=报错了）
        content: 工具返回的具体数据（成功时是结果数据，失败时是错误信息）
        is_terminal: 关键信号位！是否为"终止型工具"。
                     只有 answer 工具会把这个设为 True，表示 Agent 决定交卷了，
                     ReAct 主循环检测到这个信号后会立刻 break 退出循环。
        answer: 如果是终止工具，这里存放解析并验证过的最终答案表格（AnswerTable 对象）。
                非终止工具时为 None。
    """
    ok: bool                        # 执行是否成功
    content: dict[str, Any]         # 工具返回的具体数据或错误堆栈信息
    is_terminal: bool = False       # 关键信号：是否为终止工具（如果是，则结束 Agent 循环）
    answer: AnswerTable | None = None # 如果是终止工具，这里用来存放解析并验证过的最终答案


# ToolHandler 类型别名：所有工具的执行函数都必须符合这个签名
# 输入: (当前任务对象, 大模型传来的参数字典)
# 输出: ToolExecutionResult（标准化的执行结果）
# 这样做的好处是：ToolRegistry 不需要知道每个工具内部怎么实现，
# 它只要确认每个工具都"长得一样"（相同签名），就能统一调度。
ToolHandler = Callable[[PublicTask, dict[str, Any]], ToolExecutionResult]

# ==========================================
# 2. 具体工具的底层执行函数 (Handlers)
# ==========================================
# 以下每个函数都是某个具体工具的"真正干活"的代码。
# 它们的函数签名都遵循 ToolHandler 类型别名的约定：
#   输入 (task: PublicTask, action_input: dict) -> 输出 ToolExecutionResult
#
# 这些函数本身不直接暴露给大模型，而是通过 ToolRegistry 注册后，
# 由 ToolRegistry.execute() 方法间接调用。

def _list_context(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    """
    工具: list_context
    功能: 查看当前任务文件夹下的目录结构（类似 Linux 的 tree 命令）。
    
    这通常是 Agent 拿到一个新任务后的第一步操作——
    先看看这个任务给了什么文件，然后才知道该怎么分析。
    
    参数:
        max_depth: 最大递归深度（默认 4 层），防止目录过深导致输出太长撑爆上下文窗口
    """
    max_depth = int(action_input.get("max_depth", 4))
    return ToolExecutionResult(ok=True, content=list_context_tree(task, max_depth=max_depth))


def _read_csv(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    """
    工具: read_csv
    功能: 预览 CSV 文件的前 N 行。
    
    注意这里只是"预览"（preview），不是把整个文件读进来。
    因为有些 CSV 文件可能有几十万行，全部读进来会撑爆大模型的上下文窗口。
    Agent 看完预览后如果需要完整处理，应该使用 execute_python 工具用 pandas 来操作。
    
    参数:
        path: CSV 文件的相对路径（相对于任务的 context 目录）
        max_rows: 最多预览多少行（默认 20 行）
    """
    path = str(action_input["path"])
    max_rows = int(action_input.get("max_rows", 20))
    return ToolExecutionResult(ok=True, content=read_csv_preview(task, path, max_rows=max_rows))


def _read_json(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    """
    工具: read_json
    功能: 预览 JSON 文件的内容（截取前 N 个字符）。
    
    与 _read_csv 类似，也是只预览一部分，防止大文件撑爆上下文。
    
    参数:
        path: JSON 文件的相对路径
        max_chars: 最多预览多少个字符（默认 4000 字符）
    """
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 4000))
    return ToolExecutionResult(ok=True, content=read_json_preview(task, path, max_chars=max_chars))


def _read_doc(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    """
    工具: read_doc
    功能: 读取文本类文档的内容（如 .md, .txt 等）。
    
    参数:
        path: 文档文件的相对路径
        max_chars: 最多预览多少个字符（默认 4000 字符）
    """
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 4000))
    return ToolExecutionResult(ok=True, content=read_doc_preview(task, path, max_chars=max_chars))


def _inspect_sqlite_schema(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    """
    工具: inspect_sqlite_schema
    功能: 查看 SQLite 数据库文件的表结构（有哪些表、每张表有哪些列）。
    
    这是 Agent 在面对 .sqlite 或 .db 文件时的"侦查"步骤。
    先用这个工具看清楚数据库长什么样，然后再用 execute_context_sql 工具写 SQL 查询。
    
    参数:
        path: SQLite 文件的相对路径
    
    注意: 这里调用了 resolve_context_path() 来把相对路径转成绝对路径，
          同时做了安全校验（防止大模型传入 "../../etc/passwd" 之类的恶意路径）。
    """
    path = resolve_context_path(task, str(action_input["path"]))
    return ToolExecutionResult(ok=True, content=inspect_sqlite_schema(path))


def _execute_context_sql(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    """
    工具: execute_context_sql
    功能: 在 SQLite 数据库上执行只读 SQL 查询。
    
    特别注意"只读"二字（read-only）——这是安全限制，
    Agent 只能用 SELECT 查数据，不能用 INSERT/UPDATE/DELETE 改数据。
    
    参数:
        path: SQLite 文件的相对路径
        sql: 要执行的 SQL 查询语句（如 "SELECT * FROM users LIMIT 10"）
        limit: 最多返回多少行结果（默认 200 行），防止大查询结果撑爆上下文
    """
    path = resolve_context_path(task, str(action_input["path"]))
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", 200))
    return ToolExecutionResult(ok=True, content=execute_read_only_sql(path, sql, limit=limit))


def _execute_python(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    """
    工具: execute_python
    功能: 执行任意 Python 代码。这是 Agent 最强大也最危险的工具。
    
    Agent 可以通过这个工具写并执行任意的 Python 脚本来分析数据，
    比如用 pandas 读取 CSV、用 matplotlib 画图、用 numpy 做数学计算等。
    
    代码会在一个受限环境中执行：
    - 工作目录 (cwd) 会被设置为任务的 context_dir（数据文件所在目录）
    - 有超时限制（EXECUTE_PYTHON_TIMEOUT_SECONDS = 30 秒）
    - 代码的 stdout 输出会被捕获并返回给 Agent
    
    参数:
        code: 要执行的 Python 代码字符串
    
    返回:
        ok 字段取决于代码执行是否成功（content 中的 "success" 键）
    """
    code = str(action_input["code"])
    content = execute_python_code(
        context_root=task.context_dir,
        code=code,
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    # content 字典中包含 "success" 键，表示代码是否执行成功
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


def _answer(_: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    """
    工具: answer  ⭐ 唯一的终止型工具 ⭐
    功能: 提交最终答案表格，结束 Agent 的 ReAct 循环。
    
    这是整个工具系统中最特殊的一个工具：
    - 它是唯一一个 is_terminal=True 的工具
    - 当 Agent 调用这个工具时，ReAct 主循环会立即 break 退出
    - 它不需要 task 参数（所以用 _ 占位符忽略），因为它只处理答案数据
    
    参数:
        columns: 列名列表，如 ["name", "age", "city"]
        rows: 数据行列表，每行是一个列表，如 [["Alice", 25, "Beijing"], ["Bob", 30, "Shanghai"]]
    
    这个函数包含严格的数据校验（防御性编程）：
    1. columns 必须是非空的字符串列表
    2. rows 必须是列表
    3. 每一行的元素个数必须和 columns 的列数一致
    
    因为大模型经常会犯格式错误（比如把列名写成字典、行数和列数不匹配等），
    如果校验失败会抛出 ValueError，这个异常会被 react.py 的 try-except 捕获，
    然后把错误信息反馈给大模型，让它修正后重新提交。
    """
    # 从大模型的 action_input 中提取 columns 和 rows
    columns = action_input.get("columns")
    rows = action_input.get("rows")

    # ---- 校验 columns ----
    # isinstance(columns, list): columns 必须是列表
    # not columns: 列表不能为空
    # all(isinstance(item, str) for item in columns): 列表中的每个元素都必须是字符串
    if not isinstance(columns, list) or not columns or not all(isinstance(item, str) for item in columns):
        raise ValueError("answer.columns must be a non-empty list of strings.")

    # ---- 校验 rows ----
    if not isinstance(rows, list):
        raise ValueError("answer.rows must be a list.")

    # ---- 逐行校验并规范化 ----
    normalized_rows: list[list[Any]] = []
    for row in rows:
        # 每行必须是列表（不能是字典、元组等其他类型）
        if not isinstance(row, list):
            raise ValueError("Each answer row must be a list.")
        # 每行的元素个数必须和列数一致（防止列数不匹配导致下游解析出错）
        if len(row) != len(columns):
            raise ValueError("Each answer row must match the number of columns.")
        # list(row) 做一次浅拷贝，确保后续修改不会影响原始数据
        normalized_rows.append(list(row))

    # 构建标准的 AnswerTable 对象（定义在 benchmark/schema.py 中）
    answer = AnswerTable(columns=list(columns), rows=normalized_rows)

    # 返回结果，注意两个关键字段：
    # - is_terminal=True: 告诉 ReAct 主循环"Agent 决定交卷了，请停止循环"
    # - answer=answer: 把解析好的答案对象传出去，供 runner.py 保存为 prediction.csv
    return ToolExecutionResult(
        ok=True,
        content={
            "status": "submitted",           # 提交状态标记
            "column_count": len(columns),     # 列数（方便调试查看）
            "row_count": len(normalized_rows), # 行数（方便调试查看）
        },
        is_terminal=True,   # ⭐ 终止信号！ReAct 循环看到这个就 break
        answer=answer,       # ⭐ 最终答案
    )


# ==========================================
# 3. 工具注册中心 (ToolRegistry 类)
# ==========================================

@dataclass(slots=True)
class ToolRegistry:
    """
    工具箱管理器——Agent 调用工具的唯一入口。
    
    它维护两个核心字典：
    - specs: 工具名称 -> 工具说明书 (ToolSpec)，用于生成 System Prompt 里的工具列表
    - handlers: 工具名称 -> 执行函数 (ToolHandler)，用于根据工具名找到对应的 Python 函数
    
    工作流程（以大模型调用 "read_csv" 为例）：
    1. 大模型返回 JSON: {"action": "read_csv", "action_input": {"path": "data.csv"}}
    2. react.py 调用 registry.execute(task, "read_csv", {"path": "data.csv"})
    3. ToolRegistry 在 handlers 字典中查找 "read_csv" 对应的函数 (_read_csv)
    4. 调用 _read_csv(task, {"path": "data.csv"}) 并返回 ToolExecutionResult
    """
    specs: dict[str, ToolSpec]          # 工具说明书字典: {工具名: 说明书对象}
    handlers: dict[str, ToolHandler]    # 工具执行函数字典: {工具名: 执行函数}

    def describe_for_prompt(self) -> str:
        """
        将所有工具的说明书格式化为纯文本字符串，塞进 System Prompt。
        
        输出示例（会被塞到大模型的系统提示词中）：
        - answer: Submit the final answer table. This is the only valid terminating action.
          input_schema: {'columns': ['column_name'], 'rows': [['value_1']]}
        - execute_python: Execute arbitrary Python code...
          input_schema: {'code': "import os\\nprint(sorted(os.listdir('.')))"}
        
        sorted(self.specs) 对工具名按字母序排序，确保每次生成的 Prompt 是确定性的，
        方便调试和复现。
        """
        lines = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            lines.append(f"- {spec.name}: {spec.description}")
            lines.append(f"  input_schema: {spec.input_schema}")
        return "\n".join(lines)

    def execute(self, task: PublicTask, action: str, action_input: dict[str, Any]) -> ToolExecutionResult:
        """
        根据大模型指定的 action 名称，找到对应的执行函数并调用。
        
        这是 Agent 调用工具的唯一入口点。react.py 中的核心调用：
            tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
        
        参数:
            task: 当前任务对象（包含任务 ID、问题文本、数据文件路径等）
            action: 大模型想调用的工具名称（如 "execute_python"）
            action_input: 大模型传给工具的参数字典（如 {"code": "print('hello')"}）
        
        返回:
            ToolExecutionResult: 工具的标准化执行结果
        
        异常:
            KeyError: 如果大模型填了一个不存在的工具名（比如拼写错误 "exeucte_python"），
                      会抛出 KeyError，这个异常会被 react.py 的 try-except 捕获，
                      然后把 "Unknown tool: exeucte_python" 的错误信息反馈给大模型，
                      让它在下一轮循环中改正。
        """
        if action not in self.handlers:
            raise KeyError(f"Unknown tool: {action}")
        # 查找工具名对应的执行函数并调用，传入任务对象和参数字典
        return self.handlers[action](task, action_input)


# ==========================================
# 4. 工厂函数：一键组装默认工具箱
# ==========================================

def create_default_tool_registry() -> ToolRegistry:
    """
    创建并返回一个包含所有默认工具的 ToolRegistry 实例。
    
    这是一个"工厂函数"设计模式——外部代码不需要自己手动组装工具，
    只需要调用这个函数就能拿到一套开箱即用的完整工具箱。
    
    这个函数做了两件事：
    1. 定义 specs 字典：每个工具的"说明书"（名称、描述、参数结构），
       这些信息会被发给大模型，让它知道自己有哪些工具可用。
    2. 定义 handlers 字典：每个工具对应的真正的 Python 执行函数，
       当大模型选择某个工具时，ToolRegistry 就通过这个字典找到对应的函数来执行。
    
    当前默认工具箱包含 8 个工具：
    ┌──────────────────────┬────────────────────────────┬──────────┐
    │ 工具名                │ 功能                        │ 类型     │
    ├──────────────────────┼────────────────────────────┼──────────┤
    │ answer               │ 提交最终答案                  │ 终止型   │
    │ execute_context_sql  │ 在 SQLite 上执行只读 SQL      │ 普通型   │
    │ execute_python       │ 执行任意 Python 代码          │ 普通型   │
    │ inspect_sqlite_schema│ 查看 SQLite 数据库的表结构     │ 普通型   │
    │ list_context         │ 列出任务目录下的文件结构        │ 普通型   │
    │ read_csv             │ 预览 CSV 文件                │ 普通型   │
    │ read_doc             │ 读取文本文档                  │ 普通型   │
    │ read_json            │ 预览 JSON 文件               │ 普通型   │
    └──────────────────────┴────────────────────────────┴──────────┘
    
    使用示例（在 runner.py 中）：
        tools = create_default_tool_registry()
        agent = ReActAgent(model=model, tools=tools)
    """

    # ---- specs: 工具说明书（发给大模型看的菜单）----
    # input_schema 中的值是示例值，帮助大模型理解参数应该长什么样
    specs = {
        "answer": ToolSpec(
            name="answer",
            description="Submit the final answer table. This is the only valid terminating action.",
            input_schema={
                "columns": ["column_name"],          # 示例：列名是字符串列表
                "rows": [["value_1"]],               # 示例：每行是值列表的列表
            },
        ),
        "execute_context_sql": ToolSpec(
            name="execute_context_sql",
            description="Run a read-only SQL query against a sqlite/db file inside context.",
            input_schema={"path": "relative/path/to/file.sqlite", "sql": "SELECT ...", "limit": 200},
        ),
        "execute_python": ToolSpec(
            name="execute_python",
            description=(
                "Execute arbitrary Python code with the task context directory as the "
                "working directory. The tool returns the code's captured stdout as `output`. "
                f"The execution timeout is fixed at {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
            ),
            input_schema={
                "code": "import os\nprint(sorted(os.listdir('.')))",
            },
        ),
        "inspect_sqlite_schema": ToolSpec(
            name="inspect_sqlite_schema",
            description="Inspect tables and columns in a sqlite/db file inside context.",
            input_schema={"path": "relative/path/to/file.sqlite"},
        ),
        "list_context": ToolSpec(
            name="list_context",
            description="List files and directories available under context.",
            input_schema={"max_depth": 4},
        ),
        "read_csv": ToolSpec(
            name="read_csv",
            description="Read a preview of a CSV file inside context.",
            input_schema={"path": "relative/path/to/file.csv", "max_rows": 20},
        ),
        "read_doc": ToolSpec(
            name="read_doc",
            description="Read a text-like document inside context.",
            input_schema={"path": "relative/path/to/file.md", "max_chars": 4000},
        ),
        "read_json": ToolSpec(
            name="read_json",
            description="Read a preview of a JSON file inside context.",
            input_schema={"path": "relative/path/to/file.json", "max_chars": 4000},
        ),
    }

    # ---- handlers: 工具执行函数映射（后厨厨师名单）----
    # 每个键（工具名）必须和 specs 中的键一一对应！
    # 如果你新增了一个 spec 但忘了写 handler，Agent 调用时会抛 KeyError。
    handlers = {
        "answer": _answer,
        "execute_context_sql": _execute_context_sql,
        "execute_python": _execute_python,
        "inspect_sqlite_schema": _inspect_sqlite_schema,
        "list_context": _list_context,
        "read_csv": _read_csv,
        "read_doc": _read_doc,
        "read_json": _read_json,
    }

    # 将说明书和执行函数打包成 ToolRegistry 对象返回
    return ToolRegistry(specs=specs, handlers=handlers)
