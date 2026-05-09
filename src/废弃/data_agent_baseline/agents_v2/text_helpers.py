"""
文本处理工具函数。

从 react.py 和 plan_and_solve.py 中提取的共享工具：
  - progress():             实时打印进度
  - preview_json():         对象 → 截断 JSON 预览
  - estimate_tokens():      粗估 token 数
  - truncate_observation(): 截断过大的 observation content
"""
from __future__ import annotations

import json


def progress(msg: str) -> None:
    """实时打印进度（flush 防止多进程延迟）。"""
    print(msg, flush=True)


def preview_json(obj: object, max_len: int = 140) -> str:
    """将对象转为截断的 JSON 预览字符串，用于日志。"""
    try:
        text = json.dumps(obj, ensure_ascii=False)
    except TypeError:
        text = str(obj)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def estimate_tokens(text: str) -> int:
    """粗估 token 数。中英混合场景下 1 token ≈ 3 字符，偏保守以留安全余量。"""
    return len(text) // 3 + 1


def truncate_observation(observation: dict, max_chars: int) -> dict:
    """对 observation 中的 content 字段做硬截断，防止单条 observation 过大。

    截断逻辑：
      - 仅截断 'content' 字段（CSV 预览、SQL 结果等大文本的载体）
      - 截断后追加提示标记，让大模型知道数据被裁剪了
      - ok / tool / error 等元字段保持原样
    """
    content = observation.get("content")
    if not isinstance(content, str) or len(content) <= max_chars:
        return observation
    truncated = dict(observation)  # 浅拷贝，避免污染原始 state
    truncated["content"] = (
        content[:max_chars]
        + f"\n... [truncated, showing first {max_chars} chars of {len(content)}]"
    )
    return truncated
