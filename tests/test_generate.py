import pytest

from api import backend


class FakeResp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def reply(text, finish="STOP"):
    return {"candidates": [{"finishReason": finish, "content": {"parts": [{"text": text}]}}]}


HITS = [{"content": "Spacing is 3 diameters.", "source": "AC.pdf", "page": 1}]


def test_complete_answer_is_returned(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(backend.requests, "post", lambda *a, **k: FakeResp(reply("Use 3 diameters [1].")))
    assert backend.generate("q?", HITS) == "Use 3 diameters [1]."


def test_truncated_answer_is_rejected(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(backend.requests, "post", lambda *a, **k: FakeResp(reply("Use 3", finish="MAX_TOKENS")))
    with pytest.raises(ValueError):
        backend.generate("q?", HITS)


def test_thinking_is_disabled_only_for_25_models():
    assert backend._generation_config("gemini-2.5-flash")["thinkingConfig"] == {"thinkingBudget": 0}
    assert "thinkingConfig" not in backend._generation_config("gemini-3.5-flash-lite")
    assert backend._generation_config("gemini-2.5-flash")["maxOutputTokens"] == 800
