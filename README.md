# Evaluation Metrics for Knowledge-Grounded Text Generation using RAG

A controlled, zero-training comparison of four families of faithfulness signals for
retrieval-augmented generation (RAG), evaluated against the human hallucination
annotations of the [RAGTruth](https://github.com/ParticleMedia/RAGTruth) corpus.

Systems compared (all used off the shelf, nothing is fine-tuned):

| System | Score used (higher = more likely hallucinated) | Model / tool | Hardware |
|---|---|---|---|
| ROUGE | `1 - ROUGE-1 precision` of the response against the source context | `rouge-score`, stemming on | CPU |
| BERTScore | `1 - BERTScore precision` of the response against the source context | `roberta-large` | T4 GPU |
| NLI | `1 - fraction of response sentences predicted "entailment"` | `roberta-large-mnli`, premise truncated to 512 tokens | T4 GPU |
| LLM judge | probability of answering "No" to "is every claim supported by the source?" | `microsoft/Phi-3-mini-4k-instruct`, zero-shot, fp16 | T4 GPU |
| Hybrid-4 | equal-weight mean of the four scores' percentile ranks | no model | - |
| Hybrid-3 | the same without NLI | no model | - |

ROUGE and BERTScore are computed against the **source context**, not against a
reference answer (RAGTruth has none). This is an adapted use of both metrics.

## Data and experimental design

- **Dataset:** RAGTruth (`response.jsonl`, `source_info.jsonl`), 17,790 responses over three
  task types (Data2txt, QA, Summary), with span-level human hallucination annotations.
- **Sample:** proportional stratified sample by task type, `n = round(share * 1800)` per task,
  `random_state=42`, giving 1,799 responses. 18 are excluded for generation-quality flags
  (16 `incorrect_refusal`, 2 `truncated`), leaving **1,781** (Data2txt 627, QA 583, Summary 571).
- **Label:** a response is hallucinated if it has at least one annotated span (764 of 1,781, 42.9%).
- **Canonical split:** train 1,068 / validation 356 / test 357, stratified jointly on task type and
  label, `random_state=42`. Stored in `results/scores/ragtruth_split_assignment.csv` and reused by
  every notebook by `id`. It is never re-created.
- **Thresholds:** chosen on the validation split by maximising F1, then frozen for the test split.

  | System | Rule for "hallucinated" | Validation-selected threshold |
  |---|---|---|
  | ROUGE | raw ROUGE-1 precision <= t | 0.8118 (as saved, 4 decimals) |
  | BERTScore | raw BERTScore precision <= t | 0.902768 |
  | NLI | score >= t | 0.600000 |
  | LLM judge | P(No) >= t | 0.314042 |
  | Hybrid-4 | score >= t | 0.355454 |
  | Hybrid-3 | score >= t | 0.347066 |

  The ROUGE and BERTScore thresholds are on the raw precision values, as saved in
  `metric_run_metadata.csv`. On the oriented scores used elsewhere (`1 - precision`)
  the same rules read `score >= 0.1882` and `score >= 0.097232`.

- **Hybrids:** each oriented score is converted to a percentile rank against the *training split
  only* (no labels), then averaged with equal weights. No weights are learned. The design was fixed
  before any hybrid test result was computed.
- **Uncertainty and tests:** 95% percentile bootstrap intervals (2,000 resamples, seed 42);
  DeLong's test for paired ROC-AUC differences; paired bootstrap for PR-AUC differences;
  Holm correction within each family of comparisons; McNemar's exact test as a secondary
  analysis of thresholded predictions.
- **Primary vs supplementary:** the held-out **test set (357 responses)** is the primary result.
  Results on **all 1,781 responses** are supplementary and are *not* held-out performance; they
  are reported because per-task and per-subtype test groups are small.

## Main results (held-out test set, n = 357, 153 hallucinated)

| System | ROC-AUC [95% CI] | PR-AUC [95% CI] | Accuracy | F1 | Compute time, 1,781 responses |
|---|---|---|---|---|---|
| ROUGE-1 precision | 0.730 [0.678, 0.781] | 0.606 [0.530, 0.702] | 0.653 | 0.679 | 25 s (CPU) |
| BERTScore precision | 0.722 [0.668, 0.776] | 0.610 [0.534, 0.705] | 0.577 | 0.646 | 169 s (T4 GPU) |
| NLI entailment | 0.511 [0.463, 0.558] | 0.431 [0.374, 0.490] | 0.476 | 0.606 | 1,305 s (T4 GPU) |
| LLM judge (Phi-3-mini) | 0.678 [0.621, 0.733] | 0.539 [0.466, 0.625] | 0.616 | 0.658 | 914 s (T4 GPU) |
| Hybrid-4 (all four) | 0.739 [0.689, 0.791] | 0.600 [0.529, 0.691] | 0.602 | 0.668 | sum of components |
| Hybrid-3 (no NLI) | 0.745 [0.693, 0.797] | 0.610 [0.537, 0.702] | 0.633 | 0.678 | sum of components |

Majority-class accuracy on the test set is 0.571.

What the Holm-corrected paired tests support on the test set:

- NLI is significantly worse than ROUGE, BERTScore and the LLM judge (ROC-AUC and PR-AUC).
- ROUGE, BERTScore and the LLM judge are **not** statistically distinguishable from each other.
- Neither hybrid is significantly better than ROUGE or BERTScore; both are significantly better
  than the LLM judge alone and than NLI on ROC-AUC.

Further findings (see `notebooks/04_comparative_analysis.ipynb`):

- Pooled scores are confounded by task type; within a task type the overlap metrics score lower.
- On all 1,781 responses (supplementary) Hybrid-3 is significantly better than every individual
  metric on ROC-AUC, by a small margin (+0.023 over ROUGE).
- Subtype analysis covers three of RAGTruth's four subtypes. *Subtle Conflict* has only 16
  responses in the sample (3 in the test set) and is not analysed. With task-adjusted ROC-AUC,
  *Evident Conflict* is the hardest subtype for the overlap metrics. Subtype results are descriptive.

## Repository layout

```
notebooks/
  00_setup_and_data_exploration.ipynb   load RAGTruth, sample, build the evaluation dataframe
  01_baseline_metrics.ipynb             ROUGE and BERTScore; creates the canonical split
  02_nli_metric.ipynb                   sentence-level NLI
  03_llm_judge.ipynb                    zero-shot LLM judge
  04_comparative_analysis.ipynb         alignment, per-task results, paired tests, hybrids,
                                        final tables, subtype analysis, figures
results/
  tables/    final tables, all comparative_*.csv results, per-metric bootstrap intervals,
             metric_run_metadata.csv (thresholds, timings, test metrics)
  scores/    canonical split, per-response scores of each metric, aligned score tables
  figures/   fig1 ROC curves (test), fig2 per-task ROC-AUC (test), fig2s the same on all data
             (supplementary), fig3 ROC-AUC against compute time; PNG (300 dpi) and PDF
docs/
  environment_versions.txt              package versions of the analysis runtime
  research_plan.md, project_description.md   historical planning documents
```

The notebooks are the executed Google Colab notebooks, with outputs. Notebook 01 also contains
earlier exploratory cells (full-data and non-canonical-split ROUGE results); the results that count
are the ones computed on the canonical split, which are the ones saved in `results/`.

## Reproducing

The notebooks were run on Google Colab and read and write one flat Google Drive folder:
`/content/drive/MyDrive/rag-eval-metrics/`.

1. Run notebook 00 to clone RAGTruth and build `ragtruth_eval_dataframe.csv` in that folder.
   The file is not versioned here because it copies dataset text; the `id` column of
   `results/scores/ragtruth_split_assignment.csv` lists exactly which 1,781 responses were used.
2. To reuse the saved scores instead of recomputing them, copy the contents of `results/scores/`
   and `results/tables/` into the Drive folder (flat, without the sub-folders).
3. Run notebooks 01 to 03 on a GPU runtime to recompute the metric scores, or skip them and run
   notebook 04 on a CPU runtime; it uses only saved scores and re-runs no model.

Not versioned (too large or intermediate): the sentence-level NLI score file (about 33 MB),
the LLM-judge checkpoint file, and one superseded exploratory bootstrap file.

## Environment

- Package versions of the analysis runtime (notebook 04, Colab CPU) are in
  `docs/environment_versions.txt`.
- The GPU runtimes used for notebooks 01 to 03 were not recorded and may have differed.
- The `bert-score` package version was not recorded and could not be recovered from the notebooks.
- Timings are single runs on shared Colab hardware. They exclude package installation, model
  download and model loading, and were not measured identically for every metric.

## Limitations

- One dataset and a 1,781-response sample; the test set has 357 responses, and per-task and
  per-subtype groups are small.
- BERTScore and NLI truncate long contexts at about 512 tokens; the NLI premise is a flattened
  version of structured source data.
- One small judge model and one prompt; larger judges may behave differently.
- Bootstrap intervals reflect sampling of the test set only, not annotation uncertainty.
- The original sample was first built on Kaggle; after losing access to that account it was rebuilt
  on Colab with the same procedure and seed. All reported results come from the Colab run.

## Dataset licence

RAGTruth is distributed by its authors under the MIT licence. This repository does not
redistribute the dataset files; notebook outputs show a few example records.
