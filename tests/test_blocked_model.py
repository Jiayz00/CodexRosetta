from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from codex_rosetta.main import app
from codex_rosetta.state.conversation_store import InMemoryConversationStore


def test_blocked_model_returns_clear_400_without_upstream():
    upstream = AsyncMock()
    blocked_settings = SimpleNamespace(BLOCKED_MODEL_IDS="gpt-5.6-luna")
    with patch("codex_rosetta.api.router.get_settings", return_value=blocked_settings), \
         patch("codex_rosetta.api.router.get_upstream_client", return_value=upstream), \
         patch("codex_rosetta.api.router.get_conversation_store", return_value=InMemoryConversationStore()):
        with TestClient(app) as client:
            resp = client.post("/v1/responses", json={
                "model": "gpt-5.6-luna",
                "input": "Hi",
                "stream": True,
            })

    assert resp.status_code == 400
    assert "gpt-5.6-luna" in resp.json()["error"]["message"]
    upstream.chat_completions.assert_not_called()
    upstream.chat_completions_stream.assert_not_called()
