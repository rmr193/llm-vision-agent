"""
Unit tests for src/llm/agent.py — focused on AgentAction parsing/validation,
which does not require a live vLLM server.
"""
import pytest

from src.llm.agent import AgentAction


def test_from_json_valid_click():
    payload = {
        "reasoning": "The play button is visible.",
        "action": "click",
        "target_element_id": 3,
        "coordinates": None,
        "text": None,
        "key_name": None,
        "scroll_amount": None,
        "done_summary": None,
    }
    action = AgentAction.from_json(payload)
    assert action.action == "click"
    assert action.target_element_id == 3


def test_from_json_invalid_action_raises():
    with pytest.raises(ValueError):
        AgentAction.from_json({"action": "delete_system32", "reasoning": ""})


def test_from_json_missing_action_raises():
    with pytest.raises(ValueError):
        AgentAction.from_json({"reasoning": "no action field"})


def test_to_history_line_includes_key_fields():
    action = AgentAction(
        reasoning="typing search query",
        action="type_text",
        text="hello world",
    )
    line = action.to_history_line()
    assert "action=type_text" in line
    assert "text='hello world'" in line


def test_done_action_roundtrip():
    payload = {"action": "done", "reasoning": "goal complete", "done_summary": "Calculator shows 714"}
    action = AgentAction.from_json(payload)
    assert action.action == "done"
    assert action.done_summary == "Calculator shows 714"
