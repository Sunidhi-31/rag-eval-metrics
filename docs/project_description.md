> **Historical document.** Written at the planning stage (September 2026), before the experiments were run. It describes the original Kaggle-based plan and an earlier wording of the contribution. It is kept for the record only; the completed study is described in the repository `README.md`.

# Evaluation Metrics for Knowledge-Grounded Text Generation using RAG

**Researcher:** Sunidhi | MTech (Artificial Intelligence), Amity University
**Type:** MTech Minor Research Project

## 1. Background & Motivation

Retrieval-Augmented Generation (RAG) has become the dominant architecture for building factual, up-to-date AI systems: instead of relying only on what a language model memorized during training, RAG retrieves relevant external documents at query time and conditions the generated answer on that retrieved evidence. This makes RAG central to modern chatbots, enterprise search, customer support, and domain-specific QA (medical, legal, educational) systems.

However, **evaluating whether a RAG system's output is actually good remains an unsolved problem**:

- Traditional NLG metrics (BLEU, ROUGE, METEOR) only measure surface-level word overlap with a reference answer. They cannot tell whether an answer is *faithful* to the retrieved documents or whether it *hallucinates* facts not present in them.
- RAG systems fail in two distinct places — bad **retrieval** (wrong/irrelevant documents fetched) and bad **generation** (model ignores good documents and fabricates anyway) — and most simple metrics cannot distinguish between the two.
- Newer LLM-as-judge evaluation frameworks (e.g., RAGAS, ARES, TruLens, G-Eval) attempt to score faithfulness, answer relevance, and context precision/recall using another LLM as evaluator. These are more capable but are computationally expensive, can be inconsistent across runs, carry the judge model's own biases, and are largely validated only on English, generic-domain data.
- There is no community consensus on which metric(s) to use, making RAG systems reported in different papers difficult to compare fairly.

This project addresses that gap: **it is not about building another RAG pipeline, but about rigorously studying, comparing, and improving how we measure the quality of RAG-generated, knowledge-grounded text.**

## 2. Problem Statement

Existing evaluation approaches for knowledge-grounded, RAG-based text generation are either too shallow (lexical-overlap metrics that ignore grounding/faithfulness) or too costly and under-validated (LLM-as-judge metrics not benchmarked against human judgment, especially outside generic English domains). There is a need for an evaluation methodology that is (a) empirically validated against human judgment, (b) able to separately diagnose retrieval quality vs. generation faithfulness, and (c) lightweight enough to be practical.

## 3. Goals & Objectives

1. **Survey and categorize** existing evaluation metrics for RAG/knowledge-grounded generation — lexical (BLEU, ROUGE), semantic (BERTScore, embedding similarity), and RAG-specific (RAGAS: faithfulness, answer relevance, context precision, context recall; LLM-as-judge scoring).
2. **Build/select an evaluation dataset**: a set of questions, retrieved contexts, generated answers, and human-annotated quality judgments (faithfulness, relevance, correctness), potentially with a domain or language focus (e.g., Hindi/Indian-context QA) that existing English-generic benchmarks (RAGAS, ARES, KILT) underserve.
3. **Empirically benchmark** each existing metric against human judgments — measure correlation (e.g., Pearson/Spearman) to determine which metrics actually track human-perceived quality, and where each one fails.
4. **Propose a lightweight hybrid/composite metric** that combines cheap, interpretable signals (e.g., retrieval-context overlap + NLI-based entailment between the generated answer and retrieved passages) to approximate the accuracy of expensive LLM-judge metrics at lower computational cost.
5. **Validate the proposed metric** against the human-annotated ground truth and existing metrics, reporting where it matches, exceeds, or falls short.

## 4. Novelty / Contribution

- A **systematic empirical comparison** of lexical, semantic, and RAG-specific/LLM-judge metrics against human judgment, going beyond simply citing existing metrics.
- A **domain/language-focused evaluation angle** (e.g., Hindi or Indian-context knowledge-grounded QA) that most existing RAG evaluation benchmarks (RAGAS, ARES) do not cover, since they are built and validated primarily on English, generic-domain data.
- A **proposed lightweight composite metric** as a practical, lower-cost alternative to full LLM-as-judge evaluation, with empirical evidence of how well it approximates more expensive methods.

*(This scope/direction is a starting recommendation — final novelty angle to be confirmed before deep implementation begins.)*

## 5. Expected Outcomes / Deliverables

- A literature review chapter positioning this work against RAGAS, ARES, TruLens, KILT, and related RAG-evaluation research.
- An annotated evaluation dataset (questions, retrieved contexts, generated answers, human quality labels).
- Quantitative results: correlation tables/plots showing how each existing metric aligns (or fails to align) with human judgment.
- A working implementation of the proposed hybrid metric, with code (Python) and reproducible experiments.
- A comparative analysis/discussion of strengths, weaknesses, and recommended use-cases for each metric.
- Final dissertation/report and (if targeted) a paper draft suitable for a conference/workshop submission.

## 6. Scope Notes

- This is a minor MTech project (single-semester scale) — scope is intentionally an empirical study + one focused metric contribution, not a full new RAG system or a large-scale multi-domain benchmark.
- Dataset size, domain, and language focus are to be finalized based on available resources (annotation effort, compute, existing public datasets).

## 7. Finalized Plan (2026-09-15)

Novelty, environment, dataset, literature review, and base paper have been finalized under zero-budget / Colab-Kaggle-VS Code-only / ~2-week constraints. Full details in [research_plan.md](research_plan.md):

- **Novelty:** Benchmark free/open-source RAG faithfulness metrics (ROUGE, BERTScore, NLI-entailment, a free local LLM-judge) against RAGTruth's human hallucination labels; propose a lightweight hybrid metric.
- **Dataset:** RAGTruth (human-annotated RAG hallucination corpus, ~17,790 responses, MIT license) — no new annotation needed.
- **Environment:** Kaggle Notebooks (primary compute, GPU T4/P100, 30 GPU-hrs/week) + VS Code (local dev/writing) + Colab (overflow only).
- **Base paper:** RAGTruth (arXiv:2401.00396) — supplies both dataset and directly comparable baseline results.
- **Timeline:** 14-day day-by-day plan in research_plan.md.

## 8. Remaining Open Decisions

- [ ] Confirm Kaggle account is set up and RAGTruth uploaded as a Kaggle Dataset.
- [ ] Confirm subsample size (1,500 vs. 2,000 vs. full 17,790 responses).
- [ ] Decide target venue: internal thesis only, or also a conference/workshop paper submission.
