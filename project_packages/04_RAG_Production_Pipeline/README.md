# RAG Production Pipeline

**Type:** Independent systems-design case study
**Focus:** Hybrid retrieval, grounding, confidence calibration, and observability
**Report version:** 3.0.0 · April 2026

| Read | Link |
|---|---|
| Full technical report | [Markdown](../../RAG_Project_Report.md) |
| Publication-formatted report | [PDF](./RAG_Project_Publication.pdf) |
| Portfolio overview | [README](../../README.md) |

## Project at a glance

| Problem | Approach | Reported evidence |
|---|---|---|
| Improve retrieval quality and grounding in a RAG system design. | Dense and lexical retrieval, re-ranking, citation checks, Bayesian confidence calibration, and monitoring design. | 96.3% Recall@10; 94.2% citation precision; 2.4% reported hallucination rate. |

## What this demonstrates

- An end-to-end view of RAG quality: retrieval, answer grounding, latency,
  confidence, and failure modes.
- How hybrid retrieval and re-ranking can be evaluated separately from
  generation.
- The deployment controls—monitoring, privacy safeguards, rollback, and drift
  detection—that need to accompany a production implementation.

## Scope

This is a systems-design and evaluation case study. Its metrics are reported
from the documented evaluation, not from a live production service. A real
deployment would require security review, workload testing, data governance,
and ongoing operational validation.

## Methods

`Qdrant` · `BM25` · `ColBERT` · `OpenAI` · `Kafka` · `Kubernetes` · `Prometheus` · `MLflow`
