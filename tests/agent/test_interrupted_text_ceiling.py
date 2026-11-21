"""Empty interrupted streams must not be diagnosed as output-budget exhaustion.

Exercise the real truncation recovery/continuation helpers without making API calls.
The terminal response identifies the final attempt, not the cause of every attempt.
"""
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.agent_runtime_helpers import strip_think_blocks
from agent.transports.chat_completions import ChatCompletionsTransport
from agent.turn_retry_state import TurnRetryState
from agent.turn_truncation import recover_from_truncation
from hermes_constants import FINISH_REASON_LENGTH, PARTIAL_STREAM_STUB_ID


@pytest.fixture
def recovery_agent():
    transport = ChatCompletionsTransport()
    strip = partial(strip_think_blocks, None)
    return SimpleNamespace(
        provider="custom",
        api_mode="chat_completions",
        context_compressor=SimpleNamespace(context_length=65536),
        log_prefix="",
        _fallback_chain=[],
        _fallback_index=0,
        _ephemeral_reasoning_off=False,
        _get_transport=lambda: transport,
        _strip_think_blocks=strip,
        _has_content_after_think_block=lambda text: bool(strip(text).strip()),
        _build_assistant_message=lambda msg, reason: {
            "role": "assistant", "content": msg.content, "finish_reason": reason,
        },
        _vprint=Mock(),
        _cleanup_task_resources=Mock(),
        _persist_session=Mock(),
    )


def _response(*, interrupted, clean_eof=False, content=None, prompt_tokens=1000, dropped_tools=None):
    return SimpleNamespace(
        id=PARTIAL_STREAM_STUB_ID if interrupted else "real-length-response",
        _clean_eof=clean_eof,
        _dropped_tool_names=dropped_tools,
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=None),
            # Hermes maps incomplete transport stubs to length for recovery.
            finish_reason=FINISH_REASON_LENGTH,
        )],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=524, total_tokens=prompt_tokens + 524),
    )


def _exhaust(agent, responses):
    messages = [{"role": "user", "content": "answer the question"}]
    retries, parts = 0, []
    for count, response in enumerate(responses, 1):
        retry = TurnRetryState()
        verdict = recover_from_truncation(
            agent, response, FINISH_REASON_LENGTH, retry,
            messages=messages, conversation_history=None, api_kwargs={},
            api_call_count=count, effective_task_id="test-task", current_turn_user_idx=0,
            length_continue_retries=retries, truncated_response_parts=parts,
            truncated_tool_call_retries=0, retry_count=0, compression_attempts=0,
        )
        retries, parts = verdict.length_continue_retries, verdict.truncated_response_parts
        if count < len(responses):
            assert verdict.action == "break"
            assert retry.restart_with_length_continuation is True
        else:
            assert verdict.action == "return"
    assert verdict.result["completed"] is False
    assert verdict.result["partial"] is True
    assert verdict.result["api_calls"] == len(responses)
    assert (verdict.result["failure_reason"], verdict.result["failure_retryable"]) == ("truncated", True)
    assert agent._ephemeral_reasoning_off is False
    assert not any(
        message.get("_length_continuation_fragment") or message.get("_length_continuation_nudge")
        for message in verdict.result["messages"]
    )
    agent._cleanup_task_resources.assert_called_once_with("test-task")
    agent._persist_session.assert_called_once_with(messages, None)
    return verdict.result


@pytest.mark.parametrize("clean_eof", [True, False], ids=["clean-eof", "transport-drop"])
@pytest.mark.parametrize("content", [None, ""], ids=["none", "empty"])
def test_empty_interrupted_stream_ceiling_reports_interruption(recovery_agent, clean_eof, content):
    result = _exhaust(recovery_agent, [
        _response(interrupted=True, clean_eof=clean_eof, content=content) for _ in range(4)
    ])
    final = result["final_response"]
    assert result["error"] == "Response stream interrupted at the continuation limit"
    assert not any(word in final for word in ("output-token limit", "/reasoning", "max_tokens"))
    assert "no answer was produced" in final and "/retry" in final
    if clean_eof:
        assert "ended the stream" in final and "dropped" not in final
    else:
        assert "dropped" in final and "ended the stream" not in final


def test_true_length_ceiling_keeps_output_budget_diagnosis(recovery_agent):
    result = _exhaust(recovery_agent, [_response(interrupted=False) for _ in range(4)])
    assert result["final_response"] == (
        "⚠️ **No visible answer was produced.** The model hit its output-token limit on every "
        "continuation attempt — its reasoning consumed the entire budget each time.\n\nTo fix this:\n"
        "→ Lower reasoning effort: `/reasoning low` or `/reasoning none`\n→ Or raise max_tokens for this model"
    )
    assert result["error"] == "Response remained truncated after 4 continuation attempts"


@pytest.mark.parametrize("final_interrupted", [True, False], ids=["ends-interrupted", "ends-length"])
def test_mixed_attempts_classify_final_response_only(recovery_agent, final_interrupted):
    result = _exhaust(recovery_agent, [
        _response(interrupted=not final_interrupted),
        *[_response(interrupted=final_interrupted) for _ in range(3)],
    ])
    if final_interrupted:
        assert "dropped before any answer text" in result["final_response"]
        assert "stream interrupted" in result["error"]
        text = result["final_response"].lower()
        assert "every continuation attempt" not in text  # earlier attempts ended differently
        assert "output-token limit" not in text and "/reasoning" not in text
    else:
        # Preserve the existing true-length path even if an earlier attempt was a stub.
        assert "output-token limit" in result["final_response"]
        assert result["error"] == "Response remained truncated after 4 continuation attempts"


@pytest.mark.parametrize("interrupted", [True, False], ids=["interrupted", "length"])
def test_context_full_notice_takes_precedence(recovery_agent, interrupted):
    result = _exhaust(recovery_agent, [
        _response(interrupted=interrupted, prompt_tokens=65500),
    ])
    assert "**Context window full.**" in result["final_response"]
    assert result["error"] == "Prompt used 65500 of 65536 context tokens; no room to answer"
    assert "stream interrupted" not in result["error"]


@pytest.mark.parametrize("interrupted", [True, False], ids=["interrupted", "length"])
def test_visible_partial_is_preserved_at_ceiling(recovery_agent, interrupted):
    fragments = ["first fragment ", "second fragment ", "third fragment ", "fourth fragment."]
    result = _exhaust(recovery_agent, [
        _response(interrupted=interrupted, content=fragment) for fragment in fragments
    ])
    assert result["final_response"] == "".join(fragments)
    assert result["messages"][-1]["content"] == result["final_response"]
    if interrupted:
        assert result["error"] == "Response stream interrupted at the continuation limit"
    else:
        assert result["error"] == "Response remained truncated after 4 continuation attempts"


def test_stub_with_dropped_tool_calls_still_gets_stream_copy(recovery_agent):
    result = _exhaust(recovery_agent, [
        _response(interrupted=True, dropped_tools=["write_file"]) for _ in range(4)
    ])
    assert "dropped before any answer text" in result["final_response"]
    assert "output-token limit" not in result["final_response"]


def _diagnostic(agent):
    return [c.args[0] for c in agent._vprint.call_args_list if c.kwargs.get("diagnostic")][-1]


@pytest.mark.parametrize("content", ["", "partial answer"], ids=["no-text", "with-text"])
def test_stub_diagnostic_is_a_complete_sentence(recovery_agent, content):
    _exhaust(recovery_agent, [_response(interrupted=True, clean_eof=True, content=content) for _ in range(4)])
    line = _diagnostic(recovery_agent)
    assert "ended without reporting an output limit" in line
    assert line.endswith(("keeping the partial response received so far.", "no visible text was produced."))


def test_length_diagnostic_unchanged(recovery_agent):
    _exhaust(recovery_agent, [_response(interrupted=False, content="") for _ in range(4)])
    line = _diagnostic(recovery_agent)
    assert "Response still truncated after" in line and line.endswith("no visible text was produced.")
