# RAG 默认离线模式设计

## 背景

本地全量 RAG benchmark 在并发 worker 启动时反复访问 HuggingFace：

```text
https://huggingface.co/microsoft/harrier-oss-v1-270m/resolve/main/modules.json
```

由于网络超时，进度停留在 `0%`。单任务 `task_11` 曾成功运行，说明 Harrier 模型权重已存在本地缓存；问题不是模型不可用，而是 `sentence-transformers` / `huggingface_hub` 在每个子进程启动时尝试联网探测文件。

容器 RAG 镜像 `docker/Dockerfile.rag` 已经采用 build-time 下载、runtime 离线的方向：builder 阶段预烘 `microsoft/harrier-oss-v1-270m` 到 `/opt/models/hf`，runtime 设置 `HF_HOME=/opt/models/hf`、`TRANSFORMERS_OFFLINE=1`、`HF_HUB_OFFLINE=1`。

## 目标

1. 本地 CLI 在启用 corpus RAG 时默认使用 HuggingFace 本地缓存，不再主动联网请求 `huggingface.co`。
2. 容器 RAG 镜像继续保持 build-time 下载、runtime 离线运行。
3. 保留用户显式覆盖能力：如果用户已经设置 HF 相关环境变量，CLI 不强行覆盖。
4. 用测试固定 Dockerfile 的离线 runtime 不变量，避免后续回归。

## 非目标

1. 不把 Windows 宿主机 HuggingFace cache 直接 COPY 进 Docker build context。
2. 不改变 base `Dockerfile`；base 镜像仍是不带 RAG extras 的最小提交镜像。
3. 不改变 `CorpusRagConfig` 默认 `enabled=false`；RAG 仍需 `--memory-rag` 或 `DATA_AGENT_RAG=1` 显式启用。
4. 不在代码里硬编码用户本机缓存路径。

## 方案

### 本地 CLI 默认离线

在 `src/data_agent_langchain/cli.py` 新增一个小 helper，例如 `_apply_hf_offline_defaults_for_rag(cfg)`。

触发条件：

- `cfg.memory.rag.enabled is True`
- `cfg.memory.mode` 是 `read_only_dataset` 或 `full`

行为：

- 若 `HF_HUB_OFFLINE` 未设置，则设置为 `1`
- 若 `TRANSFORMERS_OFFLINE` 未设置，则设置为 `1`

不覆盖用户显式设置。例如用户手动设置 `HF_HUB_OFFLINE=0` 或 `HF_ENDPOINT=https://hf-mirror.com` 时，helper 不修改这些值。

接入点：

- `run_task_command`：加载 YAML 并应用 memory override 后调用 helper
- `run_benchmark_command`：加载 YAML 并应用 memory override 后调用 helper

这样 `dabench-lc run-task ... --memory-rag` 和 `dabench-lc run-benchmark ... --memory-rag` 默认都走本地缓存。

### 容器 RAG 镜像

保留当前 `docker/Dockerfile.rag` 结构：

- builder 阶段设置 `HF_HOME=/opt/models/hf`
- builder 阶段调用 `SentenceTransformer('microsoft/harrier-oss-v1-270m')` 触发模型下载
- runtime 阶段 copy `/opt/models/hf`
- runtime 阶段设置 `HF_HOME=/opt/models/hf`、`SENTENCE_TRANSFORMERS_HOME=/opt/models/hf/sentence_transformers`、`TRANSFORMERS_OFFLINE=1`、`HF_HUB_OFFLINE=1`

需要补测试强化这些不变量，尤其是 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`。

### 文档

更新 README：

- 本地 RAG 命令说明中补充：启用 RAG 后 CLI 默认设置 HF 离线模式，直接使用本地缓存。
- 容器 RAG 说明中补充：`docker/Dockerfile.rag` build 阶段需要网络下载模型，runtime 评测阶段离线运行。
- 如果本地缓存缺失，提供两个恢复方式：临时取消离线或设置 `HF_ENDPOINT=https://hf-mirror.com` 后先跑一次单任务预热缓存。

## 测试计划

1. CLI 单元测试：启用 `--memory-rag` 且 memory mode 为 `read_only_dataset` 时，未设置的 `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` 被设为 `1`。
2. CLI 单元测试：用户已设置 `HF_HUB_OFFLINE=0` 或 `TRANSFORMERS_OFFLINE=0` 时不覆盖。
3. CLI 单元测试：未启用 RAG 或 `memory.mode=disabled` 时不设置离线环境变量。
4. Dockerfile 静态测试：`docker/Dockerfile.rag` runtime ENV 包含 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`。
5. Dockerfile 静态测试：runtime copy `/opt/models/hf`，并保留 build-time `SentenceTransformer(...)` 预加载。

## 风险与处理

- 本地缓存缺失时，离线模式会让 Harrier 加载失败，RAG fail-closed。处理方式是在 README 中说明先预热缓存，或临时显式关闭离线变量。
- 并发 worker 会继承父进程环境变量，因此 CLI 在启动 runner 前设置即可覆盖本地 benchmark 场景。
- 容器 build 阶段仍需要网络；这是可接受的，因为目标是 runtime 离线评测。
