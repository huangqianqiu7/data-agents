'''
记录Agent在运行工程中，它思考了什么、调用了什么工具、报错了没有、最终答案是什么。
trace.json文件就是有这里的代码生成
'''

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from data_agent_baseline.benchmark.schema import AnswerTable

# =====================================================================
# 1. 步骤记录簿 (StepRecord)
# 作用：记录 Agent 在 ReAct 循环中某“一轮”的完整经历。
# 对应 trace.json 里 "steps" 数组中的某一个对象。
# =====================================================================
@dataclass(frozen=True, slots=True)
class StepRecord:
    """
    不可变的数据类（frozen=True），一旦记录下这一步的动作，就不能再篡改历史。
    """
    step_index: int                 # 这是第几步
    thought: str                    # 大模型在这一步的思考过程
    action: str                     # 大模型决定调用的工具名称
    action_input: dict[str, Any]    # 传递个这个工具的具体参数
    raw_response: str               # 大模型返回的原始字符串
    observation: dict[str, Any]     # 环境/工具返回的观察结果（成功拿到的数据，或者抛出的 Error 报错）
    ok: bool                        # 一步的工具调用是否成功？（True 代表成功，False 代表报错了）

    def to_dict(self) -> dict[str, Any]:
        """
        把这个对象转换为普通的 Python 字典，方便后续用 json.dump 存到硬盘里。
        asdict 是 dataclasses 自带的神器，一键转换。
        """
        return asdict(self)

# =====================================================================
# 2. 运行时状态 (AgentRuntimeState)
# 作用：Agent 的“短期记忆”或“草稿本”。
# 在 react.py 的 for 循环里，每跑一轮，就会往这里面塞点东西。
# =====================================================================
@dataclass(slots=True)
class AgentRuntimeState:
    """
    这是一个可变的数据类。随着 Agent 循环的推进，它的状态会不断更新。
    """
    # 步骤记录列表：把上面的 StepRecord 一个个存进来，形成历史记忆。
    # field(default_factory=list) 确保每次实例化时，都会生成一个全新的空列表，防止数据串线。
    steps: list[StepRecord] = field(default_factory=list)
    
    # 最终的答案表格：Agent 没调用 answer 工具前，它是 None；调用后，就把答案填进来。
    answer: AnswerTable | None = None
    
    # 失败原因：如果超时了，或者发生严重崩溃没给出答案，就会在这里记下死因。
    failure_reason: str | None = None

# =====================================================================
# 3. 最终运行报告 (AgentRunResult)
# 作用：这是 Agent 跑完一个任务后，交出的“最终成绩单”。
# runner.py 就是拿着这个对象，决定是给你生成 prediction.csv 还是报 Timeout 错误的。
# =====================================================================
@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """
    不可变类。任务一旦结束，成绩单就盖棺定论了。
    """
    task_id: str                    # 任务编号（比如"task_11"）
    answer: AnswerTable | None      # 最终计算出的表格答案（如果失败了就是None）  
    steps: list[StepRecord]         # 从头到尾所有步骤录像（方便你复盘看 trace.json)
    failure_reason: str | None      # 失败的原因（没失败就是None)

    @property
    def succeeded(self) -> bool:
        """
        动态属性：判断这个任务是不是大获全胜了？
        条件很简单：必须要有答案（answer is not None）且 没有失败原因（failure_reason is None）。
        """
        return self.answer is not None and self.failure_reason is None

    def to_dict(self) -> dict[str, Any]:
        """
        将整份成绩单打包成字典。
        这个方法生成的字典，就是你最终在硬盘上看到的那个巨大的 trace.json 文件的全部内容！
        """
        return {
            "task_id": self.task_id,
            # 如果有答案，就把答案也转成字典（包含 columns 和 rows）；否则存 None
            "answer": self.answer.to_dict() if self.answer is not None else None,
            # 把步骤录像列表里的每一个动作，都转成字典
            "steps": [step.to_dict() for step in self.steps],
            "failure_reason": self.failure_reason,
            # 存一个布尔值，方便你在 summary.json 里统计它成功了几题
            "succeeded": self.succeeded,
        }
