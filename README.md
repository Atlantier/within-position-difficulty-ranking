# Ranking Programming Problems Within Contest Positions Under a Limited Review Budget

This repository provides the replication package, verified item-level score tables, analysis scripts, publication figures, and English manuscript artifacts for the study *"Ranking Programming Problems Within Contest Positions Under a Limited Review Budget"*.

All evaluations operate strictly on release-visible problem statements without post-contest submission logs, tags, or contest metadata at inference time. To respect platform terms of service and copyright, this repository contains zero raw problem statement prose, zero raw HTML, and zero API credentials.

## Overview and Principal Empirical Findings

Offline benchmarks that pool comparisons across unequal contest slots allow a zero-training alphabetic contest-position rule to reach 91.01%--93.43% pairwise accuracy across Codeforces and AtCoder while yielding exact ties on every same-position pair (50.00%). When problem-bank maintainers curate adjacent-contest problems sharing the same contest position under a fixed chronological manual review budget, the operational problem splits into two separately testable decisions: (i) whose predictions order the automated pairs, and (ii) whose score margins select which pairs are referred for manual inspection.

We formulate this workflow as a retrospective decision simulation over historical adjacent-contest blocks across three primary cohorts: Codeforces Div. 2 (`n = 400` raw requests; 274 directional, 126 official ties), AtCoder Beginner Contest (`n = 646` directional requests, 0 ties), and Codeforces Other Divisions (`n = 341` raw requests; 227 directional, 114 official ties).

### Summary at the Fixed 30% Chronological Review Budget

AtCoder ABC 2x2 counterfactual matrix (`646` requests: `193` referred, `453` automated; verifiable automatic errors out of 453):

| Who Refers (193 of 646 Pairs) | Automated Ranker: `Joint Ridge` | Automated Ranker: `DeepSeek-Flash` |
| --- | :---: | :---: |
| `Joint Ridge` (margin) | 213 (47.02% auto error) | 141 (31.13% auto error) |
| `DeepSeek-Flash` (margin) | 213 (47.02% auto error) | 119 (26.27% auto error) |

Same-ranker margin referral versus exact expected intra-block uniform random referral (`Random - Margin`, saved verifiable errors per 100 raw requests, 95% contest-pair block-bootstrap CI, `B = 2,000`):

| Fixed Automated Ranker | AtCoder ABC (`n = 646`) | Codeforces Div. 2 (`n = 400`) | Codeforces Other Div. (`n = 341`) |
| --- | :---: | :---: | :---: |
| `DeepSeek-Flash` | **+3.67 [+2.06, +5.38]** (119.0 vs. 142.7) | -0.93 [-2.86, +1.22] (87.0 vs. 83.3) | +1.02 [-0.97, +3.16] (65.0 vs. 68.5) |
| `Qwen3-A0` (LoRA, seed 42) | +0.78 [-1.13, +2.61] (168.0 vs. 173.1) | -1.16 [-3.18, +1.05] (90.0 vs. 85.4) | -1.68 [-3.63, +0.38] (75.0 vs. 69.3) |
| `Joint Ridge` | -0.14 [-2.14, +1.86] (213.0 vs. 212.1) | 0.00 [-1.82, +1.98] (82.0 vs. 82.0) | -0.52 [-2.42, +1.48] (73.0 vs. 71.2) |

Key takeaways:

- **Whole-policy versus fixed-ranker referral separation on AtCoder ABC:** Switching the complete policy from `Joint Ridge` to `DeepSeek-Flash` reduced automated misorderings by 14.55 per 100 requests (213 to 119; 26.27% residual automated error rate). Holding `DeepSeek-Flash` fixed as the ranker, margin-based referral still saved +3.67 errors per 100 requests over uniform random referral (119.0 vs. 142.7). Local `Qwen3-A0` reduced whole-policy errors across all three training seeds (168, 170, and 180 vs. 213) at zero API cost, whereas its fixed-ranker margin gain over random referral (+0.78, -1.13 to +2.61) was uncertain.
- **Platform boundaries on Codeforces:** On Codeforces Div. 2 (400 requests: 120 referred, 280 automated), `Joint Ridge` and `DeepSeek-Flash` left 82 and 87 verifiable errors alongside 80 and 93 official rating ties, with uncertain margin-referral advantage on both Codeforces cohorts. Official ratings and in-contest solve rates agreed directionally on 262 of 274 Div. 2 pairs (95.62%, 95% CI 93.01%--97.93%), reproducing the same null separation under behavioral solve-rate evaluation.
- **Controlled `Qwen3-Embedding-0.6B` 2x2 factorial (`F0, F1, A0, A1`):** Adding a *training-only position offset* `b_k` (discarded at inference) yielded no stable within-position gain on Codeforces Div. 2 (`A1 - A0 = -1.09` percentage points, 95% CI -6.09 to +4.03; three seeds: -1.09, +1.09, +1.46) and reversed sign on one of three seeds on AtCoder ABC (-4.80, -2.48, +0.93). Un-offset LoRA (`A0`) consistently improved ABC accuracy over frozen probing (`F0`, 55.88%) across all three seeds (61.61%, 59.13%, 59.13%; primary-seed gain +5.73 points, 95% CI +1.85 to +9.66).

## Repository Structure

```text
within-position-difficulty-ranking/
├── README.md                              # Documentation and reproduction guide
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies
├── SHA256SUMS.txt                         # SHA-256 checksums for all tracked files
├── reproduce_v4_tables.py                 # Standalone verification and table recomputation script
├── statement_feature_parser_v2.py         # Inequality-preserving DOM parser and feature extractor
├── train_qwen3_factorial_reference.py     # Reference training script for Qwen3-Embedding-0.6B factorial
├── data_scores/                           # Zero-prose item-level score tables and frozen JSON specs
│   ├── v2_all_model_scores_with_dev.csv
│   ├── qwen3_item_scores.csv
│   ├── llm_single_item_scores.csv
│   ├── qwen3_cost_and_throughput.csv
│   ├── frozen_experiment_spec.json
│   └── prospective_abc_protocol.json
├── reference_tables/                      # Archived Step 2/3 reference CSVs for parity verification
│   ├── step2_main_comparison_table.csv
│   ├── step2_ablation_4cell_table.csv
│   ├── step2_multiseed_A0_A1.csv
│   ├── step3_budget_curves.csv
│   ├── step3_budget30_disposition_table.csv
│   ├── step3_referral_vs_ranking_2x2.csv
│   ├── step3_cf_behavioral_budget30.csv
│   ├── step3_tie_salt_sensitivity.csv
│   └── step3_cost_and_breakeven.csv
├── figures/                               # High-resolution publication figures (PNG 600 dpi and PDF)
│   ├── fig1_workflow_real_pair_en.{png,pdf}
│   ├── fig2_within_position_effects_en.{png,pdf}
│   └── fig3_review_budget_curves_en.{png,pdf}
└── paper/                                 # Compiled English manuscript and supplementary materials
    ├── manuscript_en_v4.{pdf,md,tex}
    └── supplement_en_v4.{pdf,md,tex}
```

## Three-Tier Reproducibility Scope

| Tier | Scope | Entry Point / Artifacts |
| --- | --- | --- |
| **1. Bundled directly** | Zero-prose item-level scores, derived surface features, frozen experiment and prospective protocol specifications, reference tables, publication figures, and manuscript PDFs. | Verified via `SHA256SUMS.txt` |
| **2. Recomputed from bundled scores** | Table 2 within-position and pooled accuracies with `B = 2,000` contest-pair block CIs (`A1 - A0`, `A0 - F0`, `A0 - Joint`, `DeepSeek - Joint`); three-seed stability (`42 / 101 / 202`) for accuracy and ABC 30% errors (`168 / 170 / 180`); Table 3 and Table 4a 30%-budget counts and ABC 2x2 swap (`213 / 141 / 119`); Table 4b same-ranker margin-versus-random referral gains (`+3.67 [+2.06, +5.38]`); and the unified 1,292-request-item AtCoder ABC token and GPU-inference cost audit. | `python reproduce_v4_tables.py` |
| **3. Retraining from public statements** | Retrieve public Codeforces and AtCoder statements matching the archived `problem_key` identifiers, preprocess with `statement_feature_parser_v2.py` (inequality-preserving DOM repair and word unigram/bigram TF-IDF), and retrain the controlled 2x2 factorial with pinned open weights. | `python statement_feature_parser_v2.py --self-test` and `python train_qwen3_factorial_reference.py --help` |

## Quick Verification

Minimal reproduction of all headline tables and statistical checks requires only Python 3.10+, `numpy`, and `pandas`:

```bash
pip install -r requirements.txt
python reproduce_v4_tables.py
python statement_feature_parser_v2.py --self-test
python train_qwen3_factorial_reference.py --help
```

To verify file integrity against `SHA256SUMS.txt` on Linux or macOS:

```bash
sha256sum -c SHA256SUMS.txt
```

## Pinned Model and Feature Specifications

- **Neural 2x2 Factorial Backbone:** `Qwen/Qwen3-Embedding-0.6B` (HuggingFace revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`), BF16 precision, left padding, head-plus-tail truncation (`max_len = 768`, retaining the final 256 constraint tokens), L2-normalized last-token pooling, rank-8 LoRA (`alpha = 16`, `q_proj` and `v_proj`), and a training-only position offset `b_k` (`b_A = 0`, `lambda_b = 1e-3`, discarded at inference).
- **Early Frozen Reference Encoders:** `google-bert/bert-base-uncased` (HuggingFace revision `86b5e0934494bd15c9632b12f734a8a67f723594`) and `BAAI/bge-base-en-v1.5` (HuggingFace revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`).
- **Lexical-and-Surface Baselines (`Joint Ridge`, `PW-Joint`):** 11 release-time surface features combined with word unigram and bigram TF-IDF features (`ngram_range = (1, 2)`, `min_df = 3`, `max_features = 20,000`, `sublinear_tf = True`).
- **External Generative Judge (`DeepSeek-Flash`):** Requested model `deepseek-flash`, observed server identifier `deepseek-v4.1-flash-1017`, `temperature = 0`, `max_tokens = 128`, `thinking = "disabled"`, `response_format = "json_object"`, `tools = "disabled"` (`3,062` target evaluation items plus `20` development anchor items = `3,082` total; `70.0%` exact-match repeatability and `2.5` mean absolute difference on 10 repeated development items, indicating non-strict determinism at temperature 0). The verbatim prompt and `prompt_sha256` are documented in `paper/supplement_en_v4.md` (Supplement S4).

## Scope Limitations and Ethics

- **Retrospective simulation:** All review-budget metrics reflect offline retrospective decision simulation on historical contests rather than live human maintainer trials; editorial time savings and post-review human accuracy are not claimed.
- **Pretraining exposure caveat:** Because historical contest archives are publicly available, model release dates alone cannot rule out exposure of past statements during foundation-model pretraining.
- **Data redistribution:** Raw contest statements are owned by their respective platforms (Codeforces and AtCoder) and are therefore excluded; only derived features, item identifiers, and model score tables are distributed here.

## Release Archives

The GitHub Release attaches two pre-packaged archives:

1. `within-position-difficulty-ranking-v1.0.0.zip`: Complete English academic repository archive including the manuscript, supplementary materials, figures, scripts, and score data.
2. `anonymous_replication_v4.zip`: Minimal standalone zero-prose replication archive matching the journal submission supplement.

## Citation

If you use this codebase or dataset artifacts in academic work, please cite the manuscript:

```bibtex
@misc{within_position_difficulty_ranking_2026,
  title        = {Ranking Programming Problems Within Contest Positions Under a Limited Review Budget},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/Atlantier/within-position-difficulty-ranking}
}
```

## License

Code, scripts, and derived benchmark artifacts in this repository are released under the MIT License (see `LICENSE`).
