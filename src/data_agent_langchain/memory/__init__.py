"""
LangGraph 后端的记忆层（memory layer）。

主方案只承载 ``working``（单 run 内的 scratchpad / pinned-preview /
sanitize 处理），跨任务的更高级记忆（dataset-knowledge / tool-playbook /
corpus 等）已抽到独立提案 ``MEMORY_MODULE_PROPOSAL.md``，这里有意不再
出现，避免 parity 测试受其影响。
"""
from data_agent_langchain.memory.working import (
    build_scratchpad_messages,
    render_step_messages,
    select_steps_for_context,
    truncate_observation,
)

__all__ = [
    "build_scratchpad_messages",
    "render_step_messages",
    "select_steps_for_context",
    "truncate_observation",
]