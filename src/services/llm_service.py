from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from repopilot.config import config
from repopilot.observability.logger import get_logger
from repopilot.observability.usage import TrackedEmbeddings, usage_callbacks

logger = get_logger(__name__)

# config.yaml name → LangChain provider id
_CHAT_PROVIDERS = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "google_genai",
    "google_genai": "google_genai",
    "groq": "groq",
}

# Environment variable each provider needs (checked at startup in main.py)
PROVIDER_API_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
}


def chat_provider() -> str:
    name = config["llm"]["provider"]
    if name not in _CHAT_PROVIDERS:
        raise ValueError(f"Unknown llm.provider '{name}' in config.yaml — use one of: openai, anthropic, gemini, groq")
    return _CHAT_PROVIDERS[name]


def chat_model(model: str | None = None, feature: str | None = None, **kwargs):
    """
    Create a chat model for the configured provider. Every model RepoPilot uses
    (/ask agent, planner, task agents, judges, summariser) comes from here, so
    switching provider is one line in config.yaml. Calls are recorded for /stats;
    feature overrides the caller's context label (e.g. "summarize").

    max_retries makes the SDK back off and retry on rate limits — important on
    free tiers, which allow only a handful of requests per minute. timeout stops
    a hung request from freezing a /plan run; it then counts as a failed attempt.
    """
    model = model or config["llm"]["model"]
    provider = chat_provider()
    # "provider:model" picks another provider for this one model — e.g. a Groq setup
    # with fallback_model "gemini:gemini-3.5-flash".
    prefix, _, rest = model.partition(":")
    if rest and prefix in _CHAT_PROVIDERS:
        provider, model = _CHAT_PROVIDERS[prefix], rest
    kwargs.setdefault("max_retries", config["llm"].get("max_retries", 6))
    kwargs.setdefault("timeout", config["llm"].get("timeout_seconds", 120))
    return init_chat_model(
        model,
        model_provider=provider,
        callbacks=usage_callbacks(model, feature),
        **kwargs,
    )


def get_llm(feature: str | None = None):
    """The main chat model from config (llm.model)."""
    logger.info(f"Using LLM provider: {config['llm']['provider']}, model: {config['llm']['model']}")
    return chat_model(feature=feature)


def get_judge_llm(feature: str, **kwargs):
    """The cheaper model used for LLM-as-judge (llm.judge_model, falling back to llm.model)."""
    return chat_model(config["llm"].get("judge_model") or config["llm"]["model"], feature=feature, **kwargs)


def get_fallback_llm(feature: str | None = None, **kwargs):
    """
    The model to switch to when the main one keeps failing (llm.fallback_model),
    or None if none is configured. It is used only after the main model's own
    retries are used up — e.g. a free-tier model returning 503 "overloaded" for minutes.
    """
    name = config["llm"].get("fallback_model")
    if not name or name == config["llm"]["model"]:
        return None
    return chat_model(name, feature=feature, **kwargs)


def required_api_keys() -> dict[str, str]:
    """Env var → why it's needed, for every provider the configured models use (checked at startup)."""
    needed = {PROVIDER_API_KEYS[chat_provider()]: f"llm.provider is '{config['llm']['provider']}'"}
    for setting in ("model", "judge_model", "fallback_model"):
        prefix, _, rest = (config["llm"].get(setting) or "").partition(":")
        if rest and prefix in _CHAT_PROVIDERS:
            needed.setdefault(PROVIDER_API_KEYS[_CHAT_PROVIDERS[prefix]], f"llm.{setting} uses {prefix}")
    return needed


# ------------------------------------------------------------------
# Structured output (judges, planner)
# ------------------------------------------------------------------
# Groq validates tool calls server-side and rejects a malformed one with a 400:
# with the default (tool-calling) method gpt-oss-20b failed 3 of 4 judge calls.
# Its json_schema mode constrains the output to the schema and passed 4 of 4,
# so Groq models use it; other providers keep LangChain's default.


def _native_schema(llm) -> bool:
    return type(llm).__name__ == "ChatGroq"


def with_schema(llm, schema):
    """llm.with_structured_output(schema), using the method that is reliable for the provider."""
    if _native_schema(llm):
        return llm.with_structured_output(schema, method="json_schema")
    return llm.with_structured_output(schema)


def response_format(llm, schema):
    """create_agent's response_format for this model (the same choice as with_schema)."""
    if _native_schema(llm):
        from langchain.agents.structured_output import ProviderStrategy

        return ProviderStrategy(schema)
    return schema


def fallback_middleware(**kwargs) -> list:
    """Agent middleware that retries a failed model call on llm.fallback_model ([] if not configured)."""
    from langchain.agents.middleware import ModelFallbackMiddleware

    fallback = get_fallback_llm(**kwargs)
    return [ModelFallbackMiddleware(fallback)] if fallback is not None else []


# ------------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------------


class LocalEmbeddings(Embeddings):
    """
    Free, offline embeddings via fastembed (ONNX — no PyTorch, no API key).
    The model is downloaded once on first use. Documents and queries use
    passage_embed / query_embed, which add the prefixes models like BGE expect.
    """

    def __init__(self, model: str) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


@lru_cache(maxsize=2)
def _local_embeddings(model: str) -> LocalEmbeddings:
    # Loading an ONNX model takes seconds — load once per process, not per call
    logger.info(f"Loading local embedding model {model} (downloaded on first use)")
    return LocalEmbeddings(model)


def get_embedder():
    """Return the right LangChain embedder based on config (wrapped for /stats tracking)."""
    provider = config["embeddings"]["provider"]
    model = config["embeddings"]["model"]
    logger.info(f"Using embeddings provider: {provider}, model: {model}")
    if provider == "fastembed":
        return TrackedEmbeddings(_local_embeddings(model), model)
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return TrackedEmbeddings(OpenAIEmbeddings(model=model), model)
    raise ValueError(f"Unknown embeddings.provider '{provider}' in config.yaml (use fastembed or openai)")


def embedding_slug() -> str:
    """
    Short, collection-safe name for the configured embedding model. Vector store
    collections include it, so switching embedding model builds a fresh index
    instead of failing on a vector-size mismatch with the old one.
    """
    model = config["embeddings"]["model"].split("/")[-1].lower()
    return "".join(ch if ch.isalnum() else "_" for ch in model)[:40]
