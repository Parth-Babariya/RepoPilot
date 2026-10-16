from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings

from repopilot.config import config
from repopilot.observability.logger import get_logger
from repopilot.observability.usage import TrackedEmbeddings, usage_callbacks

logger = get_logger(__name__)

# config.yaml name → LangChain provider id
_CHAT_PROVIDERS = {
  "openai":       "openai",
  "anthropic":    "anthropic",
  "gemini":       "google_genai",
  "google_genai": "google_genai",
}

# Environment variable each provider needs (checked at startup in main.py)
PROVIDER_API_KEYS = {
  "openai":       "OPENAI_API_KEY",
  "anthropic":    "ANTHROPIC_API_KEY",
  "google_genai": "GOOGLE_API_KEY",
}


def chat_provider() -> str:
  name = config["llm"]["provider"]
  if name not in _CHAT_PROVIDERS:
      raise ValueError(f"Unknown llm.provider '{name}' in config.yaml — use one of: openai, anthropic, gemini")
  return _CHAT_PROVIDERS[name]


def chat_model(model: str | None = None, feature: str | None = None, **kwargs):
  """
  Create a chat model for the configured provider. Every model RepoPilot uses
  (/ask agent, planner, task agents, judges, summariser) comes from here, so
  switching provider is one line in config.yaml. Calls are recorded for /stats;
  feature overrides the caller's context label (e.g. "summarize").

  max_retries makes the SDK back off and retry on rate limits — important on
  free tiers, which allow only a handful of requests per minute.
  """
  model = model or config["llm"]["model"]
  kwargs.setdefault("max_retries", config["llm"].get("max_retries", 6))
  return init_chat_model(
      model,
      model_provider=chat_provider(),
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
  if provider == "huggingface":
      from langchain_huggingface import HuggingFaceEmbeddings
      return TrackedEmbeddings(HuggingFaceEmbeddings(model_name=model), model)
  from langchain_openai import OpenAIEmbeddings
  return TrackedEmbeddings(OpenAIEmbeddings(model=model), model)


def embedding_slug() -> str:
  """
  Short, collection-safe name for the configured embedding model. Vector store
  collections include it, so switching embedding model builds a fresh index
  instead of failing on a vector-size mismatch with the old one.
  """
  model = config["embeddings"]["model"].split("/")[-1].lower()
  return "".join(ch if ch.isalnum() else "_" for ch in model)[:40]
