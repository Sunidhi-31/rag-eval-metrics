> **Historical document.** Written at the planning stage (September 2026), before the experiments were run. It describes the original Kaggle-based plan and an earlier wording of the contribution. It is kept for the record only; the completed study is described in the repository `README.md`.

# Research Plan — Evaluation Metrics for Knowledge-Grounded Text Generation using RAG
*(Zero-budget, Colab/Kaggle/VS Code only, ~2-week timeline)*

---

## 1. Finalized Novelty (budget- and time-adjusted)

> **Using RAGTruth — a publicly available, human-annotated hallucination corpus for RAG outputs — as ground truth, empirically benchmark free/open-source automatic evaluation metrics (lexical overlap, BERTScore, NLI-based entailment/faithfulness, and a free locally-run open-source LLM-as-judge) for their ability to detect hallucinated/unfaithful content in RAG-generated text, and propose a lightweight hybrid metric that approaches LLM-judge accuracy at near-zero compute cost.**

Why this version is feasible in 2 weeks at zero cost:

- **No annotation needed.** RAGTruth already has human-labeled hallucination spans/case labels across 3 task types (summarization, QA, data-to-text) — this is the single biggest time-saver versus building your own annotated dataset.
- **No paid APIs anywhere.** Every metric candidate (ROUGE, BERTScore, NLI entailment, open-source LLM-judge) runs on free, open-source models — nothing needs an OpenAI/Anthropic/paid key.
- **Modest GPU need.** A single free-tier T4 (Colab or Kaggle) is enough — models involved are ≤8B parameters, used with 4-bit quantization for the LLM-judge.
- **Clean, bounded evaluation.** Ground truth is binary/categorical (hallucinated vs. not, per response or span), so scoring is just precision/recall/F1/AUROC against RAGTruth's labels — no subjective new human study required.
- **Directly comparable to prior work.** The RAGTruth paper itself reports baseline detector performance (prompted GPT-4, fine-tuned small LM, SelfCheckGPT-style methods) — you get a ready-made comparison table to slot your numbers into.

---

## 2. Environment Comparison — Colab vs Kaggle vs VS Code

| Criterion | Google Colab (free) | Kaggle Notebooks (free) | VS Code (local) |
|---|---|---|---|
| GPU | T4 (16GB), availability varies day to day, session/idle limits are unpredictable on the free tier | T4 x2 or P100, **30 GPU-hours/week quota**, sessions up to ~9–12h, more predictable | None (unless your own machine has one) |
| Session persistence | Ephemeral; must remount Google Drive each session | Ephemeral runtime, but easy to attach persistent Kaggle Datasets/Outputs across sessions | Fully persistent (it's your machine) |
| Best for | Quick one-off GPU experiments | **Longer, repeatable GPU experiment runs + hosting the dataset itself** | Writing/organizing code, git, report writing, no-GPU steps (ROUGE, data wrangling, plotting) |
| Cost | Free, $0 | Free, $0 | Free, $0 (your hardware) |
| Dataset hosting | Needs Drive upload each time | **Can upload RAGTruth once as a Kaggle Dataset and reuse across notebooks** | Local disk |

### Recommendation
~~**Primary compute: Kaggle Notebooks.**~~ **Superseded 2026-09-29: Kaggle account banned, project moved to Google Colab as primary compute.** Kaggle's 30 GPU-hrs/week quota was preferable while available, but is no longer an option.

**Primary compute: Google Colab.** Since there's no persistent dataset hosting equivalent to Kaggle Datasets, mount Google Drive each session and treat a Drive folder as the persistence layer for the sample/eval CSVs (Colab's local disk is wiped on disconnect). Free-tier session/idle limits are less predictable than Kaggle's were — checkpoint outputs to Drive frequently and disconnect the runtime when not actively computing.

**Secondary/writing environment: VS Code**, used locally for:
- Editing code before pasting into the Colab notebook (better editor experience, git integration)
- CPU-only steps (data cleaning, ROUGE scoring, plotting, report/thesis writing)
- Version-controlling the project via the GitHub repo (free) — pull CSVs down from Drive and commit them here for the reproducibility artifact

**Reproducibility note for the paper:** the original Kaggle-hosted sample CSV is unrecoverable due to the account ban. The stratified sample is rebuilt on Colab using the identical procedure (proportional stratification by `task_type`, `random_state=42`) against the same RAGTruth source data, yielding the same composition (~627/600/572 across Data2txt/QA/Summary). Document this infrastructure change candidly in the reproducibility/limitations section rather than presenting it as a single continuous run.

**GPU requirement:** Yes, but modest. A single T4 (16GB VRAM) is sufficient for:
- BERTScore / NLI model (roberta-large-mnli or a DeBERTa-v3 NLI model, ~350M–430M params) — fast even on T4
- A 4-bit-quantized open-source instruct LLM (Llama-3.1-8B-Instruct, Mistral-7B-Instruct, or Phi-3-mini) as the free "LLM-as-judge" baseline
- ROUGE and basic lexical metrics need no GPU at all

---

## 3. Environment Setup

### Repo structure (created in this scratch workspace as a starting point)
```
rag-eval-project/
├── README.md
├── requirements.txt
└── notebooks/
    └── 00_setup_and_data_exploration.ipynb
```

### Libraries needed (all free/open-source)
- `transformers`, `datasets`, `accelerate`, `bitsandbytes` (for 4-bit LLM loading)
- `sentence-transformers` (for embedding-based context-overlap signal)
- `bert-score`, `rouge-score`, `nltk`
- `evaluate` (Hugging Face's metrics wrapper)
- `torch`
- `scikit-learn`, `pandas`, `numpy`
- `matplotlib`, `seaborn` (for result plots)

### Setup steps
1. Create a free GitHub repo (e.g., `rag-eval-metrics`) — push code from VS Code, pull/clone into Kaggle notebooks so both environments stay in sync.
2. On Kaggle: **New Notebook → Settings → Accelerator → GPU T4 x2** (or P100). Add RAGTruth as a Kaggle Dataset (Add Data → Upload, using the JSONL files from the [RAGTruth GitHub repo](https://github.com/ParticleMedia/RAGTruth)) so it persists across sessions without re-downloading.
3. `pip install -r requirements.txt` in the first notebook cell.
4. In VS Code: clone the same GitHub repo locally, do CPU-only development/testing there, push changes, pull them into Kaggle when you need GPU runs.
5. Track GPU-hour usage on Kaggle (visible in your account) so you don't blow the 30h/week quota mid-week — batch your GPU experiments rather than running them interactively all week.

A starter `requirements.txt` and a setup/data-exploration notebook skeleton are included in this folder — see [requirements.txt](requirements.txt) and [notebooks/00_setup_and_data_exploration.ipynb](notebooks/00_setup_and_data_exploration.ipynb).

---

## 4. Dataset Comparison

| Dataset | Contents | Size | Free? | Fits constraints? | Fit for our novelty |
|---|---|---|---|---|---|
| **RAGTruth** | RAG responses (GPT-3.5, GPT-4, Llama-2, Mistral-7B) across summarization, QA, data-to-text, with **word/case-level human hallucination annotations** | ~2,965 source docs → 17,790 responses | Yes, MIT license, GitHub + also mirrored on Hugging Face (`wandb/RAGTruth-processed`) | Yes — JSONL, small enough to fully load in memory, no scraping/API needed | **Best fit** — ready-made human ground truth for faithfulness/hallucination, exactly what we need to benchmark metrics against |
| **HaluEval** | 35,000 ChatGPT QA/dialogue/summarization samples labeled hallucinated vs. not (ChatGPT-generated + human-checked) | 35K samples | Yes, free on GitHub/arXiv/HF | Yes | Good secondary/validation set; less "RAG-native" (many samples are general QA, not all retrieval-grounded) |
| **FaithDial** | Edited Wizard-of-Wikipedia dialogues rewritten to be faithful, with faithfulness labels | ~50,761 turns | Yes, free (TACL 2022, HF datasets) | Yes | Good for dialogue-specific grounding, but dialogue-only — narrower than RAGTruth's 3 task types |
| **RGB Benchmark** (Chen et al.) | Constructed RAG QA benchmark testing noise robustness, negative rejection, counterfactual robustness (English + Chinese) | Moderate, constructed from news | Yes, free on GitHub | Yes | Interesting but built for testing *LLM robustness*, not primarily for *metric* benchmarking — less direct fit |

### Recommendation
**Use RAGTruth as the primary dataset.** It's the only one purpose-built for exactly this problem (human-annotated hallucination in RAG outputs, across 3 realistic task types), it's small enough to fit comfortably in Colab/Kaggle memory and free GPU-hour budgets, and it comes with baseline detector results in its paper for direct comparison. To control scope for the 2-week window, **use a stratified random subsample (~1,500–2,000 responses across the 3 task types)** rather than the full 17,790 — this keeps GPU/runtime cost low without sacrificing evaluation validity.

---

## 5. Literature Review — 10 Verified, Non-Survey Papers

All verified live via arXiv/ACL Anthology search (not from memory alone) on 2026-09-15.

### 1. RAGAS: Automated Evaluation of Retrieval Augmented Generation
- **Authors:** Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert
- **Year / Venue:** 2023 (arXiv), published EACL 2024 (demo track)
- **Link:** [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)
- **Main idea:** Reference-free evaluation framework for RAG scoring three dimensions — faithfulness, answer relevance, context relevance — without needing human-labeled ground truth.
- **Methodology:** Uses an LLM to decompose the generated answer into statements, checks whether each is inferable from retrieved context (faithfulness); generates synthetic questions from the answer and measures similarity to the original question (answer relevance); scores overlap between question and retrieved context (context relevance).
- **Dataset:** WikiEval (self-constructed from Wikipedia pages)
- **Results:** Correlates reasonably with human judgment on their constructed eval set; became the most widely-adopted open-source RAG eval toolkit.
- **Limitations:** Originally designed around a paid LLM (GPT) as the judge; reference-free scores can drift with judge-model choice; validated on a small, English, general-domain set.
- **Relation to our project:** Direct predecessor — we're testing whether RAGAS-style faithfulness scoring (adapted to a free LLM) actually tracks *human* labels, using a larger, human-annotated corpus (RAGTruth) rather than RAGAS's own small constructed set.
- **How we differ:** We swap the paid-LLM judge for a free open-source model, validate against real human annotations (not another LLM's judgment), and add a lexical/NLI-based hybrid metric RAGAS doesn't propose.

### 2. ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems
- **Authors:** Jon Saad-Falcon, Omar Khattab, Christopher Potts, Matei Zaharia
- **Year / Venue:** 2023 (arXiv), NAACL 2024
- **Link:** [arXiv:2311.09476](https://arxiv.org/abs/2311.09476)
- **Main idea:** Fine-tunes lightweight LM judges (not a giant LLM) on synthetic training data to score context relevance, faithfulness, and answer relevance, using a small number of human annotations for prediction-powered inference (PPI) confidence correction.
- **Methodology:** Synthetic query/answer generation → fine-tune small judge models → PPI to combine model predictions with a small human-labeled set for statistically calibrated scores.
- **Dataset:** KILT, SuperGLUE, AIS (8 knowledge-intensive tasks)
- **Results:** Accurate evaluation using only a few hundred human annotations; robust across domain shifts.
- **Limitations:** Still requires fine-tuning a judge model per new domain; synthetic data generation step adds complexity/compute.
- **Relation to our project:** Shows that *lightweight, non-LLM-judge* models can approximate expensive evaluation — directly supports our hybrid-metric approach.
- **How we differ:** We avoid the fine-tuning step entirely (zero training budget/time) and instead combine off-the-shelf free NLI + lexical signals; ARES needs synthetic data generation infrastructure we don't have time for.

### 3. HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models
- **Authors:** Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, Ji-Rong Wen
- **Year / Venue:** EMNLP 2023
- **Link:** [arXiv:2305.11747](https://arxiv.org/abs/2305.11747)
- **Main idea:** Constructs 35K hallucinated/normal samples (QA, knowledge-grounded dialogue, summarization) via a ChatGPT sampling-then-filtering pipeline plus human annotation, then tests how well LLMs can recognize hallucination in text.
- **Methodology:** Two-step generation (sample plausible hallucinations, filter with ChatGPT) + human verification; evaluates LLMs' own hallucination-recognition accuracy.
- **Dataset:** Self-constructed (HaluEval)
- **Results:** ChatGPT hallucinates in ~19.5% of general queries; LLMs struggle to self-recognize hallucination without external knowledge/reasoning aids.
- **Limitations:** Not all samples are RAG-grounded (many are general/parametric-knowledge QA); relies on ChatGPT for construction (though usable for free evaluation afterward).
- **Relation to our project:** A candidate secondary validation dataset; also evidence that hallucination-recognition is hard even for LLM judges — motivates checking whether our hybrid metric does better than a raw LLM-judge prompt.
- **How we differ:** We focus specifically on retrieval-grounded (RAG) outputs (RAGTruth), not general LLM hallucination, and we benchmark multiple metric types, not just LLM self-recognition.

### 4. RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models
- **Authors:** (Illinois/ParticleMedia team; corpus + detection benchmark paper)
- **Year / Venue:** ACL 2024 (arXiv Jan 2024)
- **Link:** [arXiv:2401.00396](https://arxiv.org/abs/2401.00396)
- **Main idea:** Introduces a ~18K-response corpus with **word-level human-annotated hallucination spans** across summarization, QA, and data-to-text RAG tasks from multiple LLMs (GPT-3.5/4, Llama-2, Mistral-7B), and benchmarks existing hallucination-detection methods (prompted GPT-4, fine-tuned small LM, SelfCheckGPT-style) against these labels.
- **Methodology:** Human annotators label hallucination spans with type (evident/subtle conflict, evident/subtle baseless info) and intensity; baseline detectors are then scored against these labels.
- **Dataset:** RAGTruth (self-constructed — this is our primary dataset)
- **Results:** Fine-tuned small LMs on RAGTruth can rival GPT-4-prompting for hallucination detection; existing detection methods still leave substantial room for improvement.
- **Limitations:** Fine-tuning a detector (their best method) requires labeled training data and compute; prompted-GPT-4 baseline needs a paid API (not usable under our constraints).
- **Relation to our project:** **This is our dataset AND our natural base paper** — same data, same evaluation goal (does an automatic metric predict human hallucination labels), directly comparable baseline numbers.
- **How we differ:** We restrict ourselves to zero-cost detectors (no GPT-4 API), we test a broader mix of cheap classical + NLI + open LLM-judge metrics rather than fine-tuning a new detector model, and we propose a composite metric rather than a single detector.

### 5. FaithDial: A Faithful Benchmark for Information-Seeking Dialogue
- **Authors:** Nouha Dziri, Ehsan Kamalloo, Sivan Milton, Osmar Zaiane, Mo Yu, Edoardo M. Ponti, Siva Reddy
- **Year / Venue:** TACL 2022
- **Link:** [arXiv:2204.10757](https://arxiv.org/abs/2204.10757)
- **Main idea:** Builds a faithfulness-focused dialogue dataset by editing hallucinated turns in Wizard-of-Wikipedia, and trains a "hallucination critic" classifier to detect unfaithful utterances.
- **Methodology:** Human editing of hallucinated dialogue turns to be grounded; trains/evaluates a faithfulness critic model; tests zero-shot transfer to other dialogue datasets.
- **Dataset:** FaithDial (derived from Wizard of Wikipedia)
- **Results:** Faithfulness critic trained on FaithDial improves F1 by 12.8 points on the BEGIN faithfulness benchmark vs. prior data; generalizes to CMU-Dog, TopicalChat.
- **Limitations:** Dialogue-only domain; critic model needs training data (some compute/time cost).
- **Relation to our project:** Validates that a trained faithfulness classifier can be a strong grounding metric — supports considering an NLI/classifier-based signal in our hybrid metric.
- **How we differ:** We stay in the RAG QA/summarization/data-to-text space (not open dialogue) and use off-the-shelf pretrained NLI rather than training a new critic from scratch (time constraint).

### 6. SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models
- **Authors:** Potsawee Manakul, Adian Liusie, Mark Gales
- **Year / Venue:** EMNLP 2023
- **Link:** [arXiv:2303.08896](https://arxiv.org/abs/2303.08896)
- **Main idea:** Detects hallucination without any external knowledge base by sampling multiple stochastic generations from the same LLM and checking consistency — consistent facts are likely true, divergent ones are likely hallucinated.
- **Methodology:** Sample N responses to the same prompt, measure inter-sample consistency (via BERTScore, NLI, QA-based, or n-gram methods) as a proxy for factuality.
- **Dataset:** WikiBio-based benchmark (biography generation)
- **Results:** Outperforms grey-box (probability-based) methods without needing access to token probabilities; works with black-box APIs.
- **Limitations:** Requires multiple generations per query (more compute per evaluated item); designed for context-free hallucination, not explicitly retrieval-grounded faithfulness.
- **Relation to our project:** One of RAGTruth's own baseline comparison methods; relevant as a *zero-external-knowledge* alternative to our context-grounded NLI approach.
- **How we differ:** We exploit the fact that RAG *does* have retrieved context available, so we check entailment against that context directly rather than relying on sampling-consistency (cheaper — one generation, not N).

### 7. FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation
- **Authors:** Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettlemoyer, Hannaneh Hajishirzi
- **Year / Venue:** EMNLP 2023
- **Link:** [arXiv:2305.14251](https://arxiv.org/abs/2305.14251)
- **Main idea:** Breaks long-form generations into "atomic facts" and scores the percentage supported by a reliable knowledge source, giving fine-grained (not binary) factual precision.
- **Methodology:** Decompose text → atomic facts → verify each against a knowledge source (e.g., Wikipedia) → compute % supported.
- **Dataset:** Biography generation from various LLMs
- **Results:** Differentiates model quality much better than binary correct/incorrect judgments (e.g., 66.7% vs 10.0% instead of both scoring 0).
- **Limitations:** Atomic fact decomposition + verification is often done via an LLM (cost scales with text length); built for long-form generation, not short RAG answers specifically.
- **Relation to our project:** A strong conceptual precedent for fine-grained (not just binary) faithfulness scoring, relevant if we want to move past simple hallucinated/not labels.
- **How we differ:** RAGTruth already gives us span-level ground truth, so we don't need our own atomic-fact decomposition pipeline — we instead test whether cheaper aggregate metrics can approximate this fine-grained signal.

### 8. TRUE: Re-evaluating Factual Consistency Evaluation
- **Authors:** Or Honovich, Roee Aharoni, et al.
- **Year / Venue:** NAACL 2022
- **Link:** [arXiv:2204.04991](https://arxiv.org/abs/2204.04991)
- **Main idea:** Standardizes many existing factual-consistency datasets into one meta-evaluation benchmark, and systematically compares automatic factual-consistency metrics (NLI-based, QA-based, etc.) against human-annotated ground truth using a unified, example-level, ROC/AUC-style protocol.
- **Methodology:** Collects diverse human-annotated factual-consistency datasets, standardizes labels, computes example-level correlation/AUC for each candidate metric.
- **Dataset:** Aggregated existing factual-consistency datasets (not RAG-specific)
- **Results:** Large-scale NLI-based and QA-generation-based metrics perform best and are complementary.
- **Limitations:** Predates the RAG/LLM era (summarization-focused, pre-ChatGPT); not retrieval-augmented generation specifically.
- **Relation to our project:** **This is our key methodological template** — we are essentially running a "TRUE-style" meta-evaluation, but for RAG-specific data (RAGTruth) instead of generic summarization datasets, and restricted to free/open-source metrics.
- **How we differ:** New domain (RAG, not classic summarization), newer free NLI/LLM models, and an added proposed hybrid metric which TRUE's original protocol doesn't include (TRUE is purely a comparison study, not a new-metric proposal).

### 9. Benchmarking Large Language Models in Retrieval-Augmented Generation (RGB)
- **Authors:** Jiawei Chen, Hongyu Lin, Xianpei Han, Le Sun
- **Year / Venue:** AAAI 2024 (arXiv Sept 2023)
- **Link:** [arXiv:2309.01431](https://arxiv.org/abs/2309.01431)
- **Main idea:** Constructs the RGB benchmark to test 4 core RAG abilities in LLMs: noise robustness, negative rejection, information integration, and counterfactual robustness, in English and Chinese.
- **Methodology:** Builds test cases from recent news with injected noise/counterfactual documents, evaluates 6 LLMs' RAG behavior.
- **Dataset:** RGB (self-constructed from news)
- **Results:** LLMs handle noisy retrieval reasonably but struggle badly with negative rejection (saying "I don't know") and counterfactual/conflicting documents.
- **Limitations:** Evaluates *LLM robustness*, not metric quality per se — it's a system benchmark, not a metrics-comparison study.
- **Relation to our project:** Useful context on *why* faithfulness evaluation matters (LLMs do fail in these ways), and a reminder that our metric should ideally be robust to noisy/conflicting retrieved context too.
- **How we differ:** Our focus is evaluating *metrics*, not LLM robustness directly — RGB is background motivation, not something we replicate.

### 10. Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models
- **Authors:** Bernd Bohnet et al. (22 authors, Google Research)
- **Year / Venue:** 2022 (arXiv), EMNLP 2023 (related attribution track)
- **Link:** [arXiv:2212.08037](https://arxiv.org/abs/2212.08037)
- **Main idea:** Studies "attribution" — whether an LLM's answer is properly supported by a cited source passage — proposing a reproducible human-evaluation framework and testing whether automatic metrics correlate with human attribution judgments.
- **Methodology:** Human-annotated gold standard for attribution (AIS-style judgments) on Natural Questions passages; benchmarks automatic correlated metrics against these human labels.
- **Dataset:** Attributed QA (built on Natural Questions + Wikipedia passages)
- **Results:** A correlated automatic metric is shown to be usable for model development/iteration, though not a full replacement for human evaluation.
- **Limitations:** Focused on short-answer QA with explicit citations, not summarization/data-to-text; human annotation pipeline is resource-intensive to reproduce.
- **Relation to our project:** Directly parallel goal — "does an automatic metric correlate with human grounding judgment" — but for attribution/citation rather than hallucination-span detection.
- **How we differ:** We use an existing human-annotated corpus (RAGTruth) rather than running our own annotation study (time/budget constraint), and we cover 3 task types instead of QA-only.

---

## 6. Base Paper Shortlist & Final Recommendation

| Candidate | Why it could be the base paper | What we'd take | How we'd differ | Comparability |
|---|---|---|---|---|
| **RAGTruth** (#4) | Supplies both the dataset *and* baseline detector results for the exact same problem (predict hallucination from human labels) | Dataset, task framing, evaluation protocol (per-response/span hallucination detection) | We skip their paid-API and fine-tuning baselines, use only free/open metrics, and add a hybrid metric | Direct — same data, same labels, their baseline numbers become our comparison table |
| **TRUE** (#8) | Cleanest methodological template for "meta-evaluate metrics against human labels" | Example-level meta-evaluation protocol (AUC/correlation-based metric comparison) | Applied to RAG data (RAGTruth) instead of classic summarization datasets, free-only metrics | Methodological, not data-level comparability |
| **RAGAS** (#1) | Most recognizable RAG-eval framework; faithfulness metric definition is a natural free baseline to reimplement | Faithfulness metric formulation | Reimplemented with a free LLM instead of GPT, validated against real human labels instead of RAGAS's own small eval set | Conceptual baseline within our own experiments, not a dataset match |

### Final recommendation: **RAGTruth (arXiv:2401.00396)** as the base paper.

Reasons:
- It is the only candidate that gives us **both the dataset and directly comparable baseline results** — critical for a 2-week project with zero time to build new labeled data.
- It's realistic under our constraints: the dataset is small, free, and already downloaded/verified above; their non-paid-API baselines (fine-tuned small LM, SelfCheckGPT-style) are things we can also run for free, giving an apples-to-apples comparison.
- **RAGAS** and **TRUE** remain first-class *related work / methodological references* — RAGAS's faithfulness formulation is worth reimplementing as one of our free baseline metrics, and TRUE's meta-evaluation protocol (AUC against human labels) is the exact statistical approach we'll use to score every candidate metric, including our own hybrid one.

---

## 7. Two-Week Implementation Plan

Assumes ~2–3 focused hours/day; adjust as needed. Built around a subsampled RAGTruth set (~1,500–2,000 responses, stratified across summarization/QA/data-to-text) to stay within free GPU-hour budgets.

| Day | Task |
|---|---|
| 1 | Set up GitHub repo, Kaggle notebook (GPU on), VS Code clone. Download RAGTruth, upload as Kaggle Dataset. Install libraries. |
| 2 | Explore RAGTruth structure (`response.jsonl`, `source_info.jsonl`); build the stratified ~1,500–2,000 sample subset; write literature review section (papers #1–5) into the report. |
| 3 | Write literature review section (papers #6–10) + finalize base-paper comparison section. |
| 4 | Implement lexical/overlap baseline metrics (ROUGE, source-overlap score) on the subset. |
| 5 | Implement BERTScore baseline. |
| 6 | Implement NLI-based faithfulness metric (roberta-large-mnli or DeBERTa-v3-NLI): sentence-split responses, check entailment vs. source passages, aggregate a score. |
| 7 | Set up and run the free open-source LLM-judge baseline (4-bit quantized Llama-3.1-8B-Instruct or Mistral-7B-Instruct on Kaggle T4) prompted to flag hallucinated content. |
| 8 | Run all four metrics across the full subsample; save all scores + human ground-truth labels to a results table. |
| 9 | Compute evaluation stats (precision/recall/F1/AUROC per metric vs. RAGTruth human labels); compare to RAGTruth paper's own reported baseline numbers. |
| 10 | Design + implement the hybrid metric (e.g., weighted/learned combination of NLI-entailment score + context-overlap score); tune on a small held-out split. |
| 11 | Evaluate hybrid metric the same way; compute cost/accuracy tradeoff (runtime, model size) vs. the LLM-judge baseline. |
| 12 | Error analysis: look at cases where each metric fails, categorize failure types, plot results (bar chart of F1/AUROC per metric, cost-vs-accuracy scatter). |
| 13 | Write results + discussion + limitations sections of the report. |
| 14 | Buffer day: polish code repo, finalize report, prepare literature review/base-paper writeup for submission. |

---

## Open items to confirm with you
- [ ] Confirm you're happy locking in **RAGTruth + this hybrid-metric novelty** (vs. any of the other angles).
- [ ] Confirm **Kaggle as primary compute** works for you (you'll need a free Kaggle account if you don't have one).
- [ ] Decide subsample size (1,500 vs. 2,000 vs. full 17,790 — full is possible but riskier on the 2-week/GPU-quota budget).
