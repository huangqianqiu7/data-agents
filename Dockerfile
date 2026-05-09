# KDD Cup 2026 DABench 正式提交镜像 —— v5 §二.1 / §四第 3 步骨架。
#
# 设计要点：
# - base image 与 dataagent dev 环境（Python 3.13.12）完全对齐，避免
#   typing / 语法行为漂移（v5 §四第 0 步实测）。
# - 显式 pin Debian Bookworm，确保比赛期间 Docker Hub 默认 tag 切换不
#   会被动改变镜像内容。
# - slim 比完整 python 镜像小 ~50%；目标镜像总大小 1-1.5 GB。
# - 不用 Alpine / musl，避免 numpy / pandas wheel 兼容风险。
# - ENTRYPOINT 通过 python -m 调用 data_agent_langchain.submission，所有
#   日志由 logging.FileHandler('/logs/runtime.log') 控制；禁止 shell tee。
FROM --platform=linux/amd64 python:3.13-slim-bookworm

WORKDIR /app

# 仅复制最小构建上下文：pyproject 与 langchain backend 源码。
# baseline 大依赖（datasets / duckdb / huggingface-hub / openpyxl /
# polars / pyarrow / rich）已下放到 [project.optional-dependencies].baseline，
# 主依赖只装 langchain-core / langchain-openai / langgraph / openai /
# pandas / pydantic / pyyaml / typer / numpy。
COPY pyproject.toml /app/
COPY README.md /app/
COPY src/data_agent_langchain /app/src/data_agent_langchain

# pip install 不带 extras（v5 §二.8.3）。
RUN pip install --no-cache-dir .

# 提交态网关能力快照：固定写入 /app/gateway_caps.yaml；提交入口直接
# 通过 ObservabilityConfig.gateway_caps_path = Path('/app/gateway_caps.yaml')
# 引用，不从开发态 artifacts/ 复制。
COPY docker/gateway_caps.yaml /app/gateway_caps.yaml

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 提交态严禁交互输入。所有配置走 MODEL_* / DABENCH_* 环境变量。
ENTRYPOINT ["python", "-m", "data_agent_langchain.submission"]
