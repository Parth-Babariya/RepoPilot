import pytest
from pydantic import BaseModel

from repopilot.services import llm_service


@pytest.fixture
def calls(monkeypatch):
    made = []
    monkeypatch.setattr(llm_service, "init_chat_model",
                        lambda model, model_provider, **kw: made.append((model_provider, model, kw)) or "model")
    monkeypatch.setitem(llm_service.config, "llm", {"provider": "groq", "model": "openai/gpt-oss-120b"})
    return made


def test_groq_is_a_provider_with_its_own_key(calls):
    assert llm_service.chat_provider() == "groq"
    assert llm_service.PROVIDER_API_KEYS["groq"] == "GROQ_API_KEY"
    llm_service.chat_model()
    assert calls[0][:2] == ("groq", "openai/gpt-oss-120b")
    assert calls[0][2]["timeout"] == 120 and calls[0][2]["max_retries"] == 6


def test_provider_prefix_picks_another_provider_for_one_model(calls):
    llm_service.chat_model("gemini:gemini-3.5-flash")
    assert calls[0][:2] == ("google_genai", "gemini-3.5-flash")
    llm_service.chat_model("meta-llama/llama-x:scout")          # not a provider prefix → unchanged
    assert calls[1][:2] == ("groq", "meta-llama/llama-x:scout")


def test_fallback_is_skipped_when_unset_or_same_as_main(calls):
    assert llm_service.get_fallback_llm() is None
    llm_service.config["llm"]["fallback_model"] = "openai/gpt-oss-120b"
    assert llm_service.get_fallback_llm() is None


def test_startup_asks_for_every_provider_the_models_use(monkeypatch):
    monkeypatch.setitem(llm_service.config, "llm", {
        "provider": "gemini", "model": "gemini-3.5-flash-lite",
        "judge_model": "groq:openai/gpt-oss-20b", "fallback_model": "gemini-3.5-flash",
    })
    assert set(llm_service.required_api_keys()) == {"GOOGLE_API_KEY", "GROQ_API_KEY"}


def test_groq_uses_json_schema_for_structured_output():
    class ChatGroq:
        def with_structured_output(self, schema, **kw):
            return kw.get("method")

    class OtherModel(ChatGroq):
        pass

    assert llm_service.with_schema(ChatGroq(), Verdict) == "json_schema"
    assert llm_service.with_schema(OtherModel(), Verdict) is None         # provider default
    assert type(llm_service.response_format(ChatGroq(), Verdict)).__name__ == "ProviderStrategy"
    assert llm_service.response_format(OtherModel(), Verdict) is Verdict


class Verdict(BaseModel):
    passed: bool
