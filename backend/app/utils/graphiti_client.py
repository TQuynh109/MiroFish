"""Graphiti instance factory.

Mỗi worker thread có Graphiti instance + dedicated event loop riêng
để tránh lỗi "Future attached to a different loop" khi Neo4j async driver
bị dùng across multiple event loops.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Optional

from graphiti_core import Graphiti
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient

from ..config import Config
from .logger import get_logger

logger = get_logger("mirofish.graphiti")


# Thread-local storage: mỗi thread có Graphiti instance + event loop riêng
_local = threading.local()

# Cross-encoder reranker (BGE, chạy local) được share toàn process:
# model nặng ~2GB nên chỉ load một lần; .predict() chạy trong executor nên thread-safe.
_cross_encoder = None
_cross_encoder_lock = threading.Lock()


def _get_cross_encoder() -> BGERerankerClient:
    """Lazy-load & share một BGERerankerClient cho cả process."""
    global _cross_encoder
    if _cross_encoder is None:
        with _cross_encoder_lock:
            if _cross_encoder is None:
                logger.info("Loading BGE reranker (BAAI/bge-reranker-v2-m3)...")
                _cross_encoder = BGERerankerClient()
                logger.info("BGE reranker loaded")
    return _cross_encoder


def _create_graphiti() -> Graphiti:
    """Tạo Graphiti instance mới."""
    llm_config = LLMConfig(
        api_key=Config.LLM_API_KEY,
        model=Config.LLM_MODEL_NAME,
        base_url=Config.LLM_BASE_URL,
    )
    llm_client = OpenAIGenericClient(config=llm_config)

    embedder_config = OpenAIEmbedderConfig(
        api_key=Config.EMBEDDING_API_KEY,
        base_url=Config.EMBEDDING_BASE_URL,
        embedding_model=Config.EMBEDDING_MODEL,
    )
    embedder = OpenAIEmbedder(config=embedder_config)

    # Cross-encoder reranker (BGE local) dùng cho EDGE_HYBRID_SEARCH_CROSS_ENCODER.
    # Share một model cho cả process để khỏi load lại trên từng thread.
    cross_encoder = _get_cross_encoder()

    instance = Graphiti(
        uri=Config.NEO4J_URI,
        user=Config.NEO4J_USER,
        password=Config.NEO4J_PASSWORD,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )
    logger.info(
        f"Graphiti instance created for thread {threading.current_thread().name} "
        f"(neo4j={Config.NEO4J_URI}, llm={Config.LLM_MODEL_NAME}, embed={Config.EMBEDDING_MODEL})"
    )
    return instance


def get_graphiti() -> Graphiti:
    """Trả về Graphiti instance cho thread hiện tại.

    Mỗi thread có instance riêng để Neo4j async driver
    luôn chạy trên cùng một event loop.
    """
    instance = getattr(_local, 'graphiti', None)
    if instance is not None:
        return instance

    instance = _create_graphiti()
    _local.graphiti = instance

    # Tạo indices/constraints trên instance mới
    try:
        run_async(instance.build_indices_and_constraints())
        logger.info("Neo4j indices and constraints created successfully")
    except Exception as e:
        logger.warning(f"Failed to build indices (may already exist): {e}")

    return instance


def run_async(coro):
    """Chạy async coroutine trong sync context.

    Dùng dedicated event loop per-thread để đảm bảo
    Neo4j async driver luôn chạy trên cùng một loop.
    """
    loop = getattr(_local, 'loop', None)

    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _local.loop = loop

    return loop.run_until_complete(coro)
