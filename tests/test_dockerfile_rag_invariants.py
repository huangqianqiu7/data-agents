"""``docker/Dockerfile.rag`` 静态不变量测试（G.2）。

不跑真正的 ``docker build``（评测节点 / CI 都太昂贵），只对 Dockerfile 文本
做关键不变量断言。任何会让评测镜像跑不起来的结构性变更（base image 改
切，extras 没装，ENTRYPOINT 漂移等）会立即被这里 catch。

设计依据：
  - ``A2 + B2 + C1`` 决策（详见 ``02-implementation-plan-v2.md`` 完成节 / G.2）。
  - A2：build 时预烘 Harrier-OSS-v1-270m 权重到 ``/opt/models/hf``，评测离线可用。
  - B2：multi-stage build（builder + runtime），减 layer 体积。
  - C1：``ENV DATA_AGENT_RAG=1`` 默认开启 RAG（提交即想 RAG）。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE_RAG = PROJECT_ROOT / "docker" / "Dockerfile.rag"
DOCKERFILE_BASE = PROJECT_ROOT / "Dockerfile"


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    """读取 ``docker/Dockerfile.rag`` 的全文。"""
    assert DOCKERFILE_RAG.is_file(), (
        f"docker/Dockerfile.rag must exist (G.2 deliverable). "
        f"Searched at: {DOCKERFILE_RAG}"
    )
    return DOCKERFILE_RAG.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def from_lines(dockerfile_text: str) -> list[str]:
    """提取所有 ``FROM`` 指令行。"""
    return [
        line.strip()
        for line in dockerfile_text.splitlines()
        if line.strip().upper().startswith("FROM ")
    ]


# ---------------------------------------------------------------------------
# 多阶段 build (B2 决策)
# ---------------------------------------------------------------------------


def test_dockerfile_rag_uses_multi_stage_build(from_lines: list[str]) -> None:
    """B2：必须是 multi-stage build —— 至少 2 条 ``FROM`` 指令。"""
    assert len(from_lines) >= 2, (
        f"expected multi-stage build (>=2 FROM lines), got {len(from_lines)}: {from_lines}"
    )


def test_dockerfile_rag_first_stage_is_named_builder(from_lines: list[str]) -> None:
    """B2：第一阶段必须命名（典型 ``AS builder``），便于第二阶段 ``COPY --from=``。"""
    first_stage = from_lines[0]
    assert re.search(r"\sAS\s+\w+", first_stage, re.IGNORECASE), (
        f"first FROM must be named (e.g. 'FROM ... AS builder'), got: {first_stage}"
    )


def test_dockerfile_rag_runtime_stage_uses_python_slim_bookworm(
    from_lines: list[str],
) -> None:
    """runtime 阶段（最后一个 FROM）必须用 ``python:3.13-slim-bookworm``，
    与 base ``Dockerfile`` 一致避免 ABI 漂移。"""
    runtime = from_lines[-1]
    assert "python:3.13-slim-bookworm" in runtime, (
        f"runtime stage must use python:3.13-slim-bookworm (parity with base Dockerfile), "
        f"got: {runtime}"
    )


def test_dockerfile_rag_pins_linux_amd64_platform(from_lines: list[str]) -> None:
    """提交镜像必须 pin ``linux/amd64`` 平台（评测节点 amd64）。"""
    runtime = from_lines[-1]
    assert "linux/amd64" in runtime, (
        f"runtime stage must pin --platform=linux/amd64 (KDD Cup eval is amd64), "
        f"got: {runtime}"
    )


def test_dockerfile_rag_runtime_copies_from_builder(dockerfile_text: str) -> None:
    """B2：runtime 阶段必须用 ``COPY --from=<builder>`` 把 builder 产物拿过来，
    否则 multi-stage 等于浪费一个阶段。"""
    assert re.search(r"COPY\s+--from=\w+", dockerfile_text), (
        "runtime stage must contain at least one 'COPY --from=<builder> ...' "
        "to actually benefit from multi-stage build"
    )


# ---------------------------------------------------------------------------
# RAG 依赖与 Harrier 权重 (A2 决策)
# ---------------------------------------------------------------------------


def test_dockerfile_rag_installs_rag_extras(dockerfile_text: str) -> None:
    """必须显式装 ``[rag]`` extras（chromadb + sentence-transformers + transformers + torch）。"""
    # 接受 ".[rag]" / "'.[rag]'" / '".[rag]"'，正则容忍引号差异。
    pattern = r'pip\s+install[^\n]*\.\[rag\]'
    assert re.search(pattern, dockerfile_text), (
        "Dockerfile.rag must run `pip install .[rag]` (or quoted variant) to "
        "install chromadb / sentence-transformers / transformers / torch"
    )


def test_dockerfile_rag_installs_cpu_only_torch(dockerfile_text: str) -> None:
    """2026-05-15 P1 §2 followup 实测发现：torch 默认 PyPI wheel 自带 CUDA libs
    （cudnn 366 MB + cublas 423 MB + nccl 206 MB + nvshmem 60 MB + triton 201 MB
    + cuda_bindings ...）≥1.5 GB，评测节点为 CPU-only 时全是死重量。

    必须显式走 PyTorch 官方 CPU wheel index（``download.pytorch.org/whl/cpu``）
    才能拿到 cpu-only 版本，整镜像回到 ~3.7 GB 预算。

    防御性测试：任何让该约束消失的修改（例如直接 ``pip install torch``）会让
    image bloat 回 5+ GB，本测试立即 fail。
    """
    # 接受 ``--index-url https://download.pytorch.org/whl/cpu`` 或
    # ``--extra-index-url`` 形式，只要 ``whl/cpu`` 这个 channel 出现且与 torch
    # 安装关联。
    cpu_channel_pattern = r"download\.pytorch\.org/whl/cpu"
    assert re.search(cpu_channel_pattern, dockerfile_text), (
        "Dockerfile.rag must install torch from the CPU-only wheel index "
        "(https://download.pytorch.org/whl/cpu); default PyPI wheel pulls in "
        "1.5+ GB of unused CUDA libs (P1 §2 followup finding)"
    )


def test_dockerfile_rag_preloads_harrier_weights(dockerfile_text: str) -> None:
    """A2：build 时必须预烘 ``microsoft/harrier-oss-v1-270m`` 权重到镜像，
    评测节点离线可用。"""
    assert "microsoft/harrier-oss-v1-270m" in dockerfile_text, (
        "Dockerfile.rag must preload microsoft/harrier-oss-v1-270m weights at build time "
        "(A2 decision: eval node is offline)"
    )
    # 同时必须真的调用 SentenceTransformer 触发权重下载。
    assert re.search(r"SentenceTransformer\s*\(", dockerfile_text), (
        "Dockerfile.rag must invoke SentenceTransformer(...) at build time to "
        "trigger HF weight download into the image layer"
    )


def test_dockerfile_rag_pins_hf_home(dockerfile_text: str) -> None:
    """``HF_HOME`` 必须固定到镜像内路径，让 sentence-transformers 默认走预烘缓存。"""
    # 接受 ENV HF_HOME=... 或 ENV ... HF_HOME=... 形式。
    assert re.search(r"\bHF_HOME\s*=\s*/", dockerfile_text), (
        "Dockerfile.rag must set ENV HF_HOME=/<absolute path> so runtime picks up preloaded weights"
    )


# ---------------------------------------------------------------------------
# 提交态行为 (C1 决策)
# ---------------------------------------------------------------------------


def test_dockerfile_rag_default_enables_rag_via_env(dockerfile_text: str) -> None:
    """C1：``ENV DATA_AGENT_RAG=1`` 默认打开 RAG（与提交即 RAG 主场景对齐）。"""
    # 接受 `ENV DATA_AGENT_RAG=1` 或 `ENV DATA_AGENT_RAG="1"`。
    pattern = r'ENV\s+(?:[A-Z_]+=\S+\s+)*DATA_AGENT_RAG\s*=\s*"?1"?'
    assert re.search(pattern, dockerfile_text), (
        'Dockerfile.rag must set ENV DATA_AGENT_RAG=1 (C1 decision: '
        'submitting the RAG image means we want RAG)'
    )


def test_dockerfile_rag_python_unbuffered_and_no_bytecode(dockerfile_text: str) -> None:
    """与 base Dockerfile 一致：``PYTHONUNBUFFERED=1`` + ``PYTHONDONTWRITEBYTECODE=1``。"""
    assert "PYTHONUNBUFFERED=1" in dockerfile_text, (
        "Dockerfile.rag must set PYTHONUNBUFFERED=1 (parity with base Dockerfile)"
    )
    assert "PYTHONDONTWRITEBYTECODE=1" in dockerfile_text, (
        "Dockerfile.rag must set PYTHONDONTWRITEBYTECODE=1 (parity with base Dockerfile)"
    )


def test_dockerfile_rag_entrypoint_matches_base(dockerfile_text: str) -> None:
    """ENTRYPOINT 必须与 base Dockerfile 完全一致（评测调用方式不变）。"""
    expected_pattern = (
        r'ENTRYPOINT\s+\[\s*"python"\s*,\s*"-m"\s*,\s*"data_agent_langchain\.submission"\s*\]'
    )
    assert re.search(expected_pattern, dockerfile_text), (
        'Dockerfile.rag ENTRYPOINT must be ["python", "-m", "data_agent_langchain.submission"] '
        "to match base Dockerfile (KDD Cup eval invokes the same way)"
    )


def test_dockerfile_rag_copies_gateway_caps(dockerfile_text: str) -> None:
    """提交态必须 copy ``docker/gateway_caps.yaml``，与 base Dockerfile 一致。"""
    assert "docker/gateway_caps.yaml" in dockerfile_text, (
        "Dockerfile.rag must COPY docker/gateway_caps.yaml /app/gateway_caps.yaml "
        "(parity with base Dockerfile; submission entrypoint reads it)"
    )


# ---------------------------------------------------------------------------
# 防御性约束
# ---------------------------------------------------------------------------


def test_dockerfile_rag_does_not_leak_dev_artifacts(dockerfile_text: str) -> None:
    """Dockerfile.rag 不应 copy 开发态 artifacts（``artifacts/`` / ``tests/``）—
    这些只用于本地，会膨胀镜像。"""
    forbidden = [
        re.compile(r"COPY\s+artifacts(?:/|\s)", re.IGNORECASE),
        re.compile(r"COPY\s+tests(?:/|\s)", re.IGNORECASE),
        re.compile(r"COPY\s+\.windsurf(?:/|\s)", re.IGNORECASE),
    ]
    for pat in forbidden:
        assert not pat.search(dockerfile_text), (
            f"Dockerfile.rag must not COPY dev-only directories; matched pattern: {pat.pattern}"
        )


def test_dockerfile_rag_workdir_is_app(dockerfile_text: str) -> None:
    """``WORKDIR /app`` 与 base 一致，让 ``submission.py`` 找到 ``/app/gateway_caps.yaml``。"""
    assert re.search(r"WORKDIR\s+/app\b", dockerfile_text), (
        "Dockerfile.rag must set WORKDIR /app (parity with base Dockerfile)"
    )
