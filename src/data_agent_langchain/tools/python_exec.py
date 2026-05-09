"""
``execute_python`` 工具的底层执行实现：子进程沙箱 + 硬超时。

把 LLM 给出的 Python 代码放到独立 ``multiprocessing.Process`` 里执行，
通过文件描述符级 dup2 重定向 stdout / stderr 到临时文件，超过 timeout
就 ``terminate``。这套机制保证：

  - 用户代码不会拖死整张图（每个工具调用都有独立进程，挂掉就 kill）。
  - stdout / stderr 被完整捕获并回传，对 LLM 提供有用的 observation。
  - ``EXECUTE_PYTHON_TIMEOUT_SECONDS`` 是 prompt 描述里写明的安全上界，
    必须与本模块常量保持一致（描述文本由 ``tools/descriptions.py`` 渲染）。
"""
from __future__ import annotations

import contextlib
import io
import multiprocessing
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Generator


# 安全上界：execute_python 单次最长执行时间（秒）。
EXECUTE_PYTHON_TIMEOUT_SECONDS: int = 30


@contextlib.contextmanager
def _capture_process_streams(
    stdout_path: Path, stderr_path: Path
) -> Generator[None, None, None]:
    """把进程级 fd 1/2 重定向到文件，离开 with 时还原。"""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)

    with stdout_path.open("w+b") as stdout_file, stderr_path.open("w+b") as stderr_file:
        try:
            if original_stdout is not None:
                original_stdout.flush()
            if original_stderr is not None:
                original_stderr.flush()

            os.dup2(stdout_file.fileno(), 1)
            os.dup2(stderr_file.fileno(), 2)

            stdout_encoding = getattr(original_stdout, "encoding", None) or "utf-8"
            stderr_encoding = getattr(original_stderr, "encoding", None) or "utf-8"

            sys.stdout = io.TextIOWrapper(
                os.fdopen(os.dup(1), "wb"),
                encoding=stdout_encoding,
                errors="replace",
                line_buffering=True,
                write_through=True,
            )
            sys.stderr = io.TextIOWrapper(
                os.fdopen(os.dup(2), "wb"),
                encoding=stderr_encoding,
                errors="replace",
                line_buffering=True,
                write_through=True,
            )
            yield
        finally:
            if sys.stdout is not None:
                sys.stdout.flush()
            if sys.stderr is not None:
                sys.stderr.flush()

            if sys.stdout is not original_stdout:
                sys.stdout.close()
            if sys.stderr is not original_stderr:
                sys.stderr.close()

            sys.stdout = original_stdout
            sys.stderr = original_stderr
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)


def _read_captured_stream(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _run_python_code(
    context_root: str,
    code: str,
    stdout_path: str,
    stderr_path: str,
    queue: multiprocessing.Queue[Any],
) -> None:
    """子进程目标函数：在独立进程里执行用户代码，捕获 stdout / stderr / 异常。"""
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__name__": "__main__",
        "context_root": context_root,
        "Path": Path,
    }
    resolved_stdout_path = Path(stdout_path)
    resolved_stderr_path = Path(stderr_path)

    try:
        os.chdir(context_root)
        with _capture_process_streams(resolved_stdout_path, resolved_stderr_path):
            exec(code, namespace, namespace)  # noqa: S102
        queue.put({"success": True})
    except BaseException as exc:  # noqa: BLE001
        queue.put(
            {
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def execute_python_code(
    context_root: Path, code: str, *, timeout_seconds: int = EXECUTE_PYTHON_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """在以 *context_root* 为工作目录的子进程中执行 *code*。

    返回结构：

      ``{"success": bool, "output": str, "stderr": str,
         "error"?: str, "traceback"?: str}``

    超时时返回 ``success=False`` + ``error`` 文案。
    """
    resolved_context_root = context_root.resolve()
    with tempfile.TemporaryDirectory() as temp_dir:
        stdout_path = Path(temp_dir) / "stdout.txt"
        stderr_path = Path(temp_dir) / "stderr.txt"
        stdout_path.write_text("")
        stderr_path.write_text("")

        queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
        os.environ.setdefault("PYTHONUTF8", "1")
        process = multiprocessing.Process(
            target=_run_python_code,
            args=(
                resolved_context_root.as_posix(),
                code,
                stdout_path.as_posix(),
                stderr_path.as_posix(),
                queue,
            ),
        )
        process.start()
        process.join(timeout_seconds)

        if process.is_alive():
            process.terminate()
            process.join()
            return {
                "success": False,
                "output": _read_captured_stream(stdout_path),
                "stderr": _read_captured_stream(stderr_path),
                "error": f"Python execution timed out after {timeout_seconds} seconds.",
            }

        if queue.empty():
            return {
                "success": False,
                "output": _read_captured_stream(stdout_path),
                "stderr": _read_captured_stream(stderr_path),
                "error": "Python execution exited without returning a result.",
            }

        result: dict[str, Any] = queue.get()
        result["output"] = _read_captured_stream(stdout_path)
        result["stderr"] = _read_captured_stream(stderr_path)
        return result


__all__ = ["EXECUTE_PYTHON_TIMEOUT_SECONDS", "execute_python_code"]