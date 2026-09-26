"""Fishora RAG evaluation harness and test suite.

Everything here runs offline against the real candidate corpus and the real
local E5 model; nothing needs Postgres or an LLM key. ``python -m evals.run``
writes the evaluation artifact; ``python -m scripts.quality`` runs the test
suite and the evaluation together and feeds the dashboard at ``/quality``.
"""
