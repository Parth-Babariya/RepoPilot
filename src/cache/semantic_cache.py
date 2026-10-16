from __future__ import annotations

import asyncio
import hashlib
import platform
import socket
import subprocess
import time
from urllib.parse import urlparse

from redis.asyncio import Redis
from redisvl.index import AsyncSearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag
from redisvl.redis.utils import array_to_buffer

from repopilot.config import config
from repopilot.llm.factory import get_embedder
from repopilot.observability.logger import get_logger

logger = get_logger(__name__)

VECTOR_DTYPE = "float32"


def _build_index_schema(dims: int) -> dict:
    return {
        # One index per vector size, so switching embedding model never queries an
        # index built for a different size.
        "index": {"name": f"semantic_cache_{dims}", "prefix": f"cache:{dims}:"},
        "fields": [
            {
                "name": "query_vector",
                "type": "vector",
                "attrs": {
                    "dims": dims,
                    "algorithm": "HNSW",
                    "distance_metric": "cosine",
                    "datatype": VECTOR_DTYPE,
                },
            },
            {"name": "response", "type": "text"},
            {"name": "query_text", "type": "text"},
            {"name": "domain", "type": "tag"},
            {"name": "model", "type": "tag"},
            {"name": "created_at", "type": "numeric"},
        ],
    }


class SemanticCache:
    """Async Redis-backed semantic cache for /ask responses.

    Embeddings come from the app's own get_embedder() (so the cache follows
    whichever provider config.yaml selects) and lookups filter on both
    domain and model, since this app can switch LLM providers at runtime
    and a cached answer from one model shouldn't be served under another.
    """

    def __init__(self, redis_url: str, threshold: float, dims: int):
        self.client = Redis.from_url(redis_url)
        self.embedder = get_embedder()
        self.threshold = threshold
        self.index = AsyncSearchIndex.from_dict(
            _build_index_schema(dims), redis_client=self.client
        )

    async def setup(self) -> None:
        # overwrite=False: reuse the index if it already exists from a
        # previous run instead of wiping cached entries on every startup.
        await self.index.create(overwrite=False)

    async def _embed(self, text: str) -> list[float]:
        return await self.embedder.aembed_query(text)

    async def _top_match(self, query: str, domain: str, model: str) -> dict | None:
        vector = await self._embed(query)
        q = VectorQuery(
            vector=vector,
            vector_field_name="query_vector",
            return_fields=["response", "query_text", "vector_distance"],
            num_results=1,
            dtype=VECTOR_DTYPE,
        )
        q.set_filter((Tag("domain") == domain) & (Tag("model") == model))
        results = await self.index.query(q)
        return results[0] if results else None

    async def get(self, query: str, domain: str, model: str) -> str | None:
        """Look up a cached response. Returns None on miss."""
        hit = await self._top_match(query, domain=domain, model=model)
        if hit is None:
            return None
        similarity = 1 - float(hit["vector_distance"])
        if similarity >= self.threshold:
            return hit["response"]
        # A candidate exists but isn't close enough - treat as a miss rather
        # than returning a possibly-wrong cached answer.
        return None

    async def put(self, query: str, response: str, domain: str, model: str, ttl: int) -> None:
        """Store a (query, response) pair."""
        vector = await self._embed(query)
        doc_id = hashlib.sha256(query.encode()).hexdigest()[:16]
        redis_key = self.index.key(doc_id)
        entry = {
            "query_vector": array_to_buffer(vector, VECTOR_DTYPE),
            "response": response,
            "query_text": query,
            "domain": domain,
            "model": model,
            "created_at": time.time(),
        }
        await self.index.load([entry], keys=[redis_key], ttl=ttl)

    async def invalidate_domain(self, domain: str) -> None:
        """Delete all cache entries for a domain (e.g. after a codebase change)."""
        async for key in self.client.scan_iter("cache:*"):
            entry = await self.client.hget(key, "domain")
            if entry and entry.decode() == domain:
                await self.client.delete(key)


def get_repo_domain(repo_path: str) -> str:
    """Stable cache domain derived from the repo path, so cached entries
    persist across restarts for the same repo without colliding with
    entries from a different repo indexed by the same tool."""
    return hashlib.sha256(repo_path.encode()).hexdigest()[:16]


def _redis_reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _ensure_wsl_redis(distro: str, redis_url: str, wait_seconds: float = 20) -> bool:
    """
    Windows only: make sure Redis running inside a WSL distro is up.

    WSL stops a distro (and its Redis) a few seconds after the last command
    attached to it exits. So we start the distro's redis-server service and
    keep the distro alive with a detached `sleep infinity`. Redis then stays
    up until Windows restarts or `wsl --shutdown` — the cache survives across
    RepoPilot runs. Returns True once the port accepts connections.
    """
    parsed = urlparse(redis_url)
    host, port = parsed.hostname or "localhost", parsed.port or 6379
    if _redis_reachable(host, port):
        return True
    if platform.system() != "Windows":
        return False

    logger.info(f"Starting Redis in WSL distro '{distro}'")
    try:
        subprocess.Popen(
            ["wsl", "-d", distro, "-u", "root", "--", "bash", "-c",
             "systemctl start redis-server; exec sleep infinity"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS,
        )
    except OSError as e:
        logger.warning(f"Could not launch WSL: {e}")
        return False

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if _redis_reachable(host, port):
            logger.info("Redis in WSL is up")
            return True
        time.sleep(0.5)
    return False


async def build_semantic_cache() -> SemanticCache | None:
    """Build and initialize the semantic cache per config.yaml.

    Returns None (and logs why) if caching is disabled or Redis is
    unreachable, so the caller can run without caching instead of crashing.
    """
    cache_config = config.get("semantic_cache", {})
    if not cache_config.get("enabled", False):
        logger.info("Semantic cache: disabled (semantic_cache.enabled is false in config.yaml)")
        return None

    redis_url = cache_config.get("redis_url", "redis://localhost:6379")
    wsl_distro = cache_config.get("wsl_distro")
    if wsl_distro and not await asyncio.to_thread(_ensure_wsl_redis, wsl_distro, redis_url):
        logger.warning(f"Semantic cache: disabled (could not start Redis in WSL distro '{wsl_distro}')")
        return None

    try:
        # Ask the embedder for its vector size instead of trusting config,
        # so changing embeddings.model can't silently break the cache.
        dims = len(await get_embedder().aembed_query("dimension probe"))
        cache = SemanticCache(
            redis_url=redis_url,
            threshold=cache_config.get("threshold", 0.85),
            dims=dims,
        )
        await cache.setup()
        logger.info(f"Semantic cache: enabled (threshold={cache.threshold}, dims={dims})")
        return cache
    except Exception as error:
        logger.warning(f"Semantic cache: disabled (init failed: {error})")
        return None