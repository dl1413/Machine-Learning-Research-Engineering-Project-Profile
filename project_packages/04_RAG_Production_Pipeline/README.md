# Project 04 — RAG: Production Pipeline with Vector Embeddings & LLM Ensemble

**Author:** Derek Lankeaux, MS Applied Statistics
**Date:** 2026
**Compliance:** IEEE 2830-2025 · ISO/IEC 23894:2025 · EU AI Act 2025

## Publication

| Document | File |
|----------|------|
| Technical Report (PDF) | [`RAG_Project_Publication.pdf`](./RAG_Project_Publication.pdf) |

## Summary

Production-grade Retrieval-Augmented Generation system combining hybrid vector embeddings, multi-model LLM orchestration, and statistical hallucination detection for enterprise knowledge retrieval.

**Key Results:**
- 94.2% citation precision (vs. 71.8% LLM-only baseline: +22.4 pp)
- 2.4% hallucination rate — below the 5% operational threshold
- <200ms end-to-end latency (p50) at 1,240 req/sec throughput
- 96.3% Recall@10 (hybrid embedding ensemble vs. 94.1% single-model)
- 99.97% uptime over 30-day production window
- Expected Calibration Error (ECE): 0.011 (1.1%)

**Tech Stack:** `OpenAI` `Anthropic` `Llama-3.2` `Qdrant` `ColBERT` `BM25` `Kafka` `Kubernetes` `Prometheus` `MLflow`
