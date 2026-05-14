"""Corpus RAG boundary reserved for Phase M4."""
from __future__ import annotations


def build_corpus_retriever(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise NotImplementedError("Corpus RAG is reserved for Phase M4")


__all__ = ["build_corpus_retriever"]
