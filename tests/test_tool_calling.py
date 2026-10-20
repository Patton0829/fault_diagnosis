"""Offline native-tool protocol tests; HTTP is always an in-memory transport."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import socket
from unittest.mock import Mock

import dotenv
import httpx
from openai import OpenAI
from pydantic import ValidationError
import pytest

from qa_platform.providers import tool_calling as tool


SECRET = "SECRET_MUST_NOT_APPEAR"


@pytest.fixture(autouse=True)
def forbid_external_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tool protocol tests must not use network or .env")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(dotenv, "load_dotenv", forbidden)
    monkeypatch.setattr(dotenv, "dotenv_values", forbidden)


def first_response():
    return {
        "id": "demo-completion", "object": "chat.completion", "created": 1,
        "model": "synthetic-model",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "call-demo-1", "type": "function", "function": {
                "name": tool.TOOL_NAME, "arguments": json.dumps({"code": tool.DEMO_CODE}),
            }}],
        }}],
    }


def final_response(content):
    result = first_response()
    result["choices"][0] = {"index": 0, "finish_reason": "stop", "message": {
        "role": "assistant", "content": content, "tool_calls": None,
    }}
    return result


def marker_in_request(request):
    return json.loads(json.loads(request.content)["messages"][-1]["content"])["verification_marker"]


def sdk_client(handler):
    return OpenAI(
        api_key=SECRET, base_url="https://synthetic.invalid/v1", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler), trust_env=False),
    )


def test_native_two_round_trip_uses_unknown_local_marker_and_real_call_id():
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, json=first_response())
        return httpx.Response(200, json=final_response("  " + marker_in_request(request) + "\n"))

    with sdk_client(handler) as client:
        result = tool.probe(client)

    first, second = [json.loads(request.content) for request in requests]
    assert first["tool_choice"] == "auto"
    assert second["tool_choice"] == "none"
    assert first["tools"] == second["tools"] == tool.TOOLS
    assert first["stream"] is second["stream"] is False
    assert first["enable_thinking"] is second["enable_thinking"] is False
    assert first["max_tokens"] == second["max_tokens"] == tool.MAX_TOKENS
    assert second["messages"][:-2] == first["messages"]
    assert second["messages"][-2]["tool_calls"] == [result["tool_call"]]
    assert second["messages"][-1]["role"] == "tool"
    assert second["messages"][-1]["tool_call_id"] == "call-demo-1"
    assert result["tool_result"]["code"] == tool.DEMO_CODE
    assert result["tool_result"]["is_synthetic"] is True
    marker = result["tool_result"]["verification_marker"]
    assert marker.startswith("TC-RESULT-")
    assert marker not in requests[0].content.decode()
    assert result["final_answer"] == marker
    assert result["schema_version"] == "chat-tool-probe/v1"
    assert result["model"] == tool.DEFAULT_MODEL
    assert result["native_tool_call_observed"] is True
    assert result["finish_reasons"] == ["tool_calls", "stop"]
    assert result["request_counts"] == {"http": 2, "chat": 2}
    assert result["elapsed_seconds"] >= 0
    assert tool.MODEL_CONFIGS[tool.DEFAULT_MODEL]["extra_body"] == {"enable_thinking": True}


@pytest.mark.parametrize("failure", [
    "plain_text", "truncated", "multiple_calls", "unknown_tool", "invalid_json",
    "extra_argument", "wrong_type", "wrong_code", "empty_id", "control_id", "multiple_choices",
])
def test_first_round_invalid_protocol_never_executes_tool_or_sends_second(monkeypatch, failure):
    response = first_response()
    choice = response["choices"][0]
    call = choice["message"]["tool_calls"][0]
    if failure == "plain_text":
        choice["finish_reason"] = "stop"
        choice["message"].update(content='{"tool_calls": "' + SECRET + '"}', tool_calls=None)
    elif failure == "truncated":
        choice["finish_reason"] = "length"
    elif failure == "multiple_calls":
        choice["message"]["tool_calls"].append(deepcopy(call))
    elif failure == "unknown_tool":
        call["function"]["name"] = "execute_shell_" + SECRET
    elif failure == "invalid_json":
        call["function"]["arguments"] = SECRET
    elif failure == "extra_argument":
        call["function"]["arguments"] = json.dumps({"code": tool.DEMO_CODE, "path": SECRET})
    elif failure == "wrong_type":
        call["function"]["arguments"] = '{"code": 7319}'
    elif failure == "wrong_code":
        call["function"]["arguments"] = json.dumps({"code": SECRET})
    elif failure == "empty_id":
        call["id"] = " "
    elif failure == "control_id":
        call["id"] = "call\n" + SECRET
    elif failure == "multiple_choices":
        response["choices"].append(deepcopy(choice))
    executed = Mock(side_effect=AssertionError("Tool must not execute"))
    monkeypatch.setattr(tool, "lookup_demo_record", executed)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=response)

    with sdk_client(handler) as client, pytest.raises(tool.ProbeError) as caught:
        tool.probe(client)
    error = caught.value
    assert error.stage == "validate_tool_call"
    assert error.attempted_requests == len(requests) == 1
    assert error.native_tool_call_observed is (failure not in {"plain_text", "multiple_choices"})
    assert SECRET not in str(error)
    assert "不代表服务一定不支持" in str(error)
    executed.assert_not_called()


@pytest.mark.parametrize("failure", ["wrong_marker", "extra_text", "truncated", "new_tool_call"])
def test_second_round_must_finish_and_exactly_echo_locally_generated_result(failure):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, json=first_response())
        marker = marker_in_request(request)
        response = final_response(marker)
        if failure == "wrong_marker":
            response["choices"][0]["message"]["content"] = SECRET
        elif failure == "extra_text":
            response["choices"][0]["message"]["content"] = "Result: " + marker
        elif failure == "truncated":
            response["choices"][0]["finish_reason"] = "length"
        else:
            response = first_response()
        return httpx.Response(200, json=response)

    with sdk_client(handler) as client, pytest.raises(tool.ProbeError) as caught:
        tool.probe(client)
    assert caught.value.stage == "validate_final_answer"
    assert caught.value.attempted_requests == len(requests) == 2
    assert caught.value.native_tool_call_observed is True
    assert SECRET not in str(caught.value)


@pytest.mark.parametrize("fail_at,stage", [(1, "request_tool_call"), (2, "request_final_answer")])
def test_http_error_has_no_implicit_retry_or_response_leak(fail_at, stage):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == fail_at:
            return httpx.Response(500, json={"error": {"message": SECRET}})
        return httpx.Response(200, json=first_response())

    with sdk_client(handler) as client, pytest.raises(tool.ProbeError) as caught:
        tool.probe(client)
    assert caught.value.stage == stage
    assert caught.value.attempted_requests == len(requests) == fail_at
    assert caught.value.native_tool_call_observed is (fail_at == 2)
    assert SECRET not in str(caught.value)
    assert caught.value.__cause__ is None


def test_local_tool_is_whitelisted_random_and_has_no_external_side_effects():
    first = tool.lookup_demo_record(tool.DEMO_CODE)
    second = tool.lookup_demo_record(tool.DEMO_CODE)
    assert first["verification_marker"] != second["verification_marker"]
    with pytest.raises(ValueError):
        tool.lookup_demo_record("/tmp/run-command")
    for data in ({"code": 1}, {"code": "ok", "shell": "ls"}, {"code": ""}):
        with pytest.raises(ValidationError):
            tool.LookupArgs.model_validate(data)


def test_unknown_model_is_rejected_before_any_client_call():
    client = Mock()
    with pytest.raises(ValueError):
        tool.probe(client, "unconfigured-model")
    client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("model", list(tool.MODEL_CONFIGS))
def test_cli_plan_does_not_read_key_create_client_or_probe(monkeypatch, capsys, model):
    forbidden = Mock(side_effect=AssertionError("Plan must not perform I/O"))
    monkeypatch.setattr(tool, "load_chat_key", forbidden)
    monkeypatch.setattr(tool, "OpenAI", forbidden)
    monkeypatch.setattr(tool.httpx, "Client", forbidden)
    monkeypatch.setattr(tool, "probe", forbidden)
    assert tool.main(["--model", model]) == 0
    output = capsys.readouterr().out
    assert "HTTP=0" in output
    assert "不是模型能力已通过" in output
    forbidden.assert_not_called()


def test_cli_glm_never_sends_cloud_key_and_disables_proxy_redirect_retry(monkeypatch, capsys):
    monkeypatch.setenv("DASHSCOPE_API_KEY", SECRET)
    monkeypatch.setenv("OPENAI_ORG_ID", "SYNTHETIC_ORG_MUST_NOT_SEND")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "SYNTHETIC_PROJECT_MUST_NOT_SEND")
    original_http_client, original_sdk = httpx.Client, OpenAI
    requests, http_options, sdk_options = [], [], []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, json=first_response())
        return httpx.Response(200, json=final_response(marker_in_request(request)))

    class ProbeHTTPClient(original_http_client):
        def __init__(self, **kwargs):
            http_options.append(kwargs)
            super().__init__(**kwargs, transport=httpx.MockTransport(handler))

    def make_sdk(**kwargs):
        sdk_options.append(kwargs)
        return original_sdk(**kwargs)

    monkeypatch.setattr(tool.httpx, "Client", ProbeHTTPClient)
    monkeypatch.setattr(tool, "OpenAI", make_sdk)
    assert tool.main(["--run", "--model", "glm"]) == 0
    assert http_options == [{"trust_env": False, "follow_redirects": False, "timeout": 60.0}]
    assert sdk_options[0]["api_key"] == "not-needed"
    assert sdk_options[0]["max_retries"] == 0
    assert sdk_options[0]["organization"] == ""
    assert sdk_options[0]["project"] == ""
    assert sdk_options[0]["base_url"] == tool.MODEL_CONFIGS["glm"]["base_url"]
    assert len(requests) == 2
    for request in requests:
        assert SECRET not in str(request.headers)
        assert SECRET not in request.content.decode()
        assert "SYNTHETIC_ORG_MUST_NOT_SEND" not in str(request.headers)
        assert "SYNTHETIC_PROJECT_MUST_NOT_SEND" not in str(request.headers)
        assert "enable_thinking" not in json.loads(request.content)
    output = capsys.readouterr()
    assert SECRET not in output.out + output.err
    assert "[成功]" in output.out


def test_cli_failure_prints_safe_stage_and_count_only(monkeypatch, capsys):
    monkeypatch.setattr(tool, "load_chat_key", lambda model: SECRET)
    original_http_client = httpx.Client
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(400, json={"error": {"message": SECRET}})

    class ProbeHTTPClient(original_http_client):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(tool.httpx, "Client", ProbeHTTPClient)
    assert tool.main(["--run"]) == 1
    output = capsys.readouterr()
    assert len(requests) == 1
    assert "阶段=request_tool_call" in output.err
    assert "HTTP尝试=1" in output.err
    assert SECRET not in output.out + output.err


def test_script_entry_point_delegates_default_plan(monkeypatch):
    main = Mock(return_value=0)
    monkeypatch.setattr(tool, "main", main)
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_tool_calling.py"
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(script), run_name="__main__")
    assert caught.value.code == 0
    main.assert_called_once_with()
