from tooling import parse_tool_call


def test_parse_tool_call_valid():
    raw = '{"tool": "fetch_url", "args": {"url": "https://example.com"}}'
    call = parse_tool_call(raw)
    assert call is not None
    assert call.name == "fetch_url"
    assert call.args["url"] == "https://example.com"


def test_parse_tool_call_invalid():
    assert parse_tool_call("hello") is None
    assert parse_tool_call("{\"tool\": \"nope\"}") is None
