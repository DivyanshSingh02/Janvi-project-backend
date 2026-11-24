import json
import logging

from app.core.logging import JsonFormatter


def test_json_formatter_outputs_valid_json_with_extra_fields():
    record = logging.LogRecord("app.test", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    record.request_id = "abc123"
    data = json.loads(JsonFormatter().format(record))
    assert data["message"] == "hello world"
    assert data["level"] == "INFO"
    assert data["request_id"] == "abc123"
    assert data["timestamp"].endswith("+00:00")
