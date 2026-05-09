"""
``GatewayCaps`` —— 从 yaml 加载的网关能力 dataclass。

这份 yaml 由 Phase 0.5 ``dabench-lc gateway-smoke`` 命令产出，对评测
网关探测 ``tool_calling`` / ``parallel_tool_calls`` / ``seed`` /
``strict`` 是否支持。runner 启动时强校验 yaml 存在；缺失会让
``bind_tools_for_gateway`` 无法决定该传哪些字段，直到 HTTP 400 才暴露
（v4 M3 / §11.1 / §15）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from data_agent_langchain.exceptions import GatewayCapsMissingError


@dataclass(frozen=True, slots=True)
class GatewayCaps:
    """已配置网关在 OpenAI 兼容性面上的快照。"""

    tool_calling: bool                      # 是否支持 ``bind_tools``
    parallel_tool_calls: bool | None        # True/False = 已探测；None = 已跳过
    seed_param: bool                        # 是否真正接受 ``seed=...``
    strict_mode: bool                       # OpenAI strict mode 是否生效

    @classmethod
    def from_yaml(cls, path: Path) -> "GatewayCaps":
        """从 yaml 加载 caps；文件缺失抛 :class:`GatewayCapsMissingError`。"""
        if not path.exists():
            raise GatewayCapsMissingError(
                f"Gateway capabilities file missing: {path}. "
                "Run `dabench-lc gateway-smoke --config <yaml>` (Phase 0.5) before "
                "evaluation."
            )
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(**data["gateway_caps"])


__all__ = ["GatewayCaps"]