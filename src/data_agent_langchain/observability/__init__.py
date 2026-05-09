"""
LangGraph 后端的可观测性栈。

子模块职责：
  - ``events.dispatch_observability_event(name, data, config)``：业务节点
    发自定义事件的统一入口（v4 M4 / §11.5）；底层走 LangGraph 0.4 的
    ``dispatch_custom_event``。
  - ``tracer.build_callbacks(config, *, task_id, mode)``：返回 LangSmith
    回调列表（D1：仅在 ``compiled.invoke`` 单点注入）。
  - ``metrics.MetricsCollector``：每任务级 callback，写 ``metrics.json``。
  - ``gateway_caps.GatewayCaps``：从 ``artifacts/gateway_caps.yaml`` 加载
    的网关能力 dataclass；runner 启动时强校验该文件存在（v4 M3）。
  - ``gateway_smoke.run_gateway_smoke``：Phase 0.5 探针，写入
    gateway_caps.yaml。
  - ``reporter.aggregate_metrics``：把每任务 metrics 聚合成批量 summary。
"""
__all__: list[str] = []