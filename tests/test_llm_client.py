"""LLM 客户端单元测试：JSON 提取 + 结构化输出校验与重试"""
import pytest

from agent.llm_client import LLMClient, LLMCallError, _extract_json
from agent.schemas import Intent, VideoSummary


def test_extract_json_object():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_array():
    assert _extract_json('[{"bvid": "BV1"}]') == [{"bvid": "BV1"}]


def test_extract_json_code_fence():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_with_noise():
    text = '好的，结果是：\n{"action": "execute", "topic": "Python"}\n以上'
    assert _extract_json(text) == {"action": "execute", "topic": "Python"}


def test_extract_json_invalid():
    assert _extract_json("not json at all") is None
    assert _extract_json("") is None


def test_complete_structured_object(monkeypatch):
    client = LLMClient(api_key="sk-test")

    def fake_complete(*a, **k):
        return '{"action": "execute", "topic": "Python", "level": "入门", "daily_hours": 2.0, "confidence": 0.9}'

    monkeypatch.setattr(client, "complete", fake_complete)
    result = client.complete_structured("sys", "user", Intent)
    assert isinstance(result, Intent)
    assert result.topic == "Python"
    assert result.action == "execute"


def test_complete_structured_list(monkeypatch):
    client = LLMClient(api_key="sk-test")

    def fake_complete(*a, **k):
        return '[{"bvid": "BV1", "match_score": 8}]'

    monkeypatch.setattr(client, "complete", fake_complete)
    result = client.complete_structured("sys", "user", list[VideoSummary])
    assert isinstance(result, list) and len(result) == 1
    assert result[0].bvid == "BV1"
    assert result[0].match_score == 8


def test_complete_structured_retry_then_success(monkeypatch):
    client = LLMClient(api_key="sk-test")
    calls = {"n": 0}

    def fake_complete(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json at all"
        return '{"action": "execute", "topic": "Python"}'

    monkeypatch.setattr(client, "complete", fake_complete)
    result = client.complete_structured("sys", "user", Intent)
    assert result.topic == "Python"
    assert calls["n"] == 2


def test_missing_api_key_raises():
    with pytest.raises(LLMCallError):
        LLMClient(api_key="")
