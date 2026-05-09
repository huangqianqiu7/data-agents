import json
import os

os.environ.setdefault("LLM_API_KEY", "test-key")

import bench_comparison


class FakeMessage:
    content = json.dumps({"status": "一致", "reason": "Pass"})


class FakeChoice:
    message = FakeMessage()


class FakeResponse:
    choices = [FakeChoice()]


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeResponse()


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self):
        self.chat = FakeChat()


def test_evaluate_with_llm_sets_request_timeout():
    client = FakeClient()

    result = bench_comparison.evaluate_with_llm(client, "[]", "[]")

    assert result == {"status": "一致", "reason": "Pass"}
    assert client.chat.completions.kwargs["timeout"] == bench_comparison.LLM_REQUEST_TIMEOUT_SECONDS
