# Supplementary Materials: Ranking Programming Problems Within Contest Positions Under a Limited Review Budget

These supplementary materials provide detailed literature positioning, preprocessing audits, encoder and four-cell adaptation diagnostics, rolling development-selection robustness, verbatim external generative judge specifications, prospective protocol precision estimates, behavioral solve-rate checks, complete cohort timelines, three-denominator cost accounting, and full fixed-budget replay tables supporting the main manuscript.

---

## Supplement S1. Related-Work Matrix, Statement Preprocessing Repair, and Version Sensitivity

### S1.1 Positioning Against Prior Studies

**Table S1a. Comparison of related programming-difficulty, ranking-with-abstention, and learning-to-defer studies.**

| Study | Release-time inputs | Candidate formation | Evaluation unit | Separates ranking vs. budget referral? |
| --- | --- | --- | --- | --- |
| PSG (Kim et al., 2024) [1] | Release-time English statements | Independent single problems | Single problem (tags & difficulty tier) | No |
| Tabib & Deedar (2025) [2] | Problem statements (tree adds acceptance rate) | Independent single problems (LeetCode) | Single problem (difficulty tier) | No |
| Park et al. (2024) [9] | Zero-shot LLM simulating multi-level students | Independent single problems | Single-item difficulty estimate | No |
| Online-judge & IRT studies [3–8, 22] | Post-contest submission logs or learner histories | Historical submission traces | Problem, sequence, or learner | No |
| Krichene & Rendle (2020) [10] | User–item interaction logs | Sampled negative candidates | User-level sampled ranking metrics | Metric rank flips only |
| Geifman & El-Yaniv (2017) [23] | Classifier confidence scores | Selective abstention over test items | Coverage–risk curve | Abstention (classification) |
| Cheng et al. (2010) [26] | Feature vectors for label ranking | Partial-order abstention on uncertain pairs | Pairwise partial-order loss | Abstention (no per-block budget swap) |
| Mozannar & Sontag (2020) [27] | Features + expert oracle labels | Joint predictor and rejector | Surrogate deferral loss | Classification deferral (not pairwise) |
| **This study** | **Release-visible statements and constraints only** | **Adjacent same-series within-position pairs** | **Contest-pair blocks & chronological requests** | **Yes (2×2 swap + same-ranker random baseline)** |

### S1.2 Inequality-Preserving Statement Repair and Sensitivity Audit

Earlier statement extraction applied `<[^>]+>` after HTML entity decoding, inadvertently stripping literal mathematical inequalities (`<`, `>`). Confining HTML tag removal to the raw DOM layer prior to `html.unescape` repaired **79 of 8,502** archived items (`CF_train` 35, `ABC_test` 18, `ARC_test` 10, `CF_test` 6, `ARC_train` 4, `ARC_valid` 4, `ABC_train` 2, `E1` 0, `E2` 0) with 0 duplicate-flag changes and `0.00e+00` max embedding drift on unchanged anchor items. Refitting all models on repaired inputs shifted headline displayed contrasts by at most **0.65 percentage points** (`ARC` `BoW − Length` $D$: −3.81 to −3.16 pp) and shifted the full 105-contrast parity audit by at most **1.87 percentage points** (`ARC` `PW-Joint − Joint Ridge` $D$: −1.36 to +0.52 pp), while leaving the primary Codeforces Div. 2 `Joint − Surface` within-position gain unchanged at **+2.19 pp** (`95% CI [−3.08, +7.64]`).

---

## Supplement S2. Encoder Configurations, Four-Cell Qwen3 Diagnostics, Multi-Seed Stability, and Reference Shift

### S2.1 Model Configurations and Internal Engineering Sanity Check

- **Position Diagnostic Baseline:** Reads the contest slot letter directly (`A < B < C < ...`) solely as a zero-training structural diagnostic (not a statement-only ranker), tying all same-position pairs (`50.00%`).
- **Lexical-and-Surface Linear Rankers (`Joint Ridge`, `PW-Joint`):** Combine 11 release-time surface features (statement length, digit/formula density, constraint upper-bound indicators) with **word unigram and bigram** TF-IDF features (`ngram_range=(1, 2)`, `min_df=3`, `max_features=20,000`, `sublinear_tf=True`, `token_pattern=(?u)\b\w+\b`).
- **Early Reference Frozen Encoders:** `google-bert/bert-base-uncased` (pinned HuggingFace revision `86b5e0934494bd15c9632b12f734a8a67f723594`, mean-pooled Ridge) and **`BAAI/bge-base-en-v1.5`** (pinned HuggingFace revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, mean-pooled and CLS-pooled Ridge) trained on identical repaired Codeforces Div. 2 Train statements.
- **Controlled Qwen3 Backbone:** `Qwen/Qwen3-Embedding-0.6B` (pinned HuggingFace revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`), BF16 precision, left padding, head+tail truncation (`max_len = 768`, retaining final `256` constraint tokens; truncation rate `0.86%` on CF Train, `1.73%` on CF Div. 2 Test, `0.07%` on ABC Test, `2.00%` on CF E1), L2-normalized last non-padding token pooling into `Linear(1024, 1)`.
- **LoRA & Training-Only Position Offset Hyperparameters:** Rank `r = 8`, `lora_alpha = 16`, `lora_dropout = 0.0`, target modules `q_proj, v_proj`, `AdamW` with cosine 10% warmup, gradient clip `1.0`, `weight_decay = 0.01`, `epochs = 5`, 2-block groups (~37.5 pairs/step). Cells `F1` and `A1` add a *training-only position offset* $(b_{k_i} - b_{k_j})$ to $f(x_i) - f(x_j)$ with $b_A \equiv 0$ and L2 penalty $\lambda_b = 10^{-3}$, discarding $b_k$ at inference. Learning rates (`lr_head = 1e-2`, `lr_lora = 1e-4`, `lr_bias = 1e-2`) were sanity-checked strictly on an internal training split (`<=2020 -> 2021`, never touching 2022 dev or test cohorts): internal 2021 accuracy rose from `51.69%` (frozen) to `55.06%` (LoRA A1) with zero non-finite gradients.

**Table S2a. Complete four-cell Qwen3 factorial, reference encoders, and auxiliary boundary cohorts (within-position accuracy %, 95% contest-pair block CI).**

| Cohort (`n` pairs) | F0 (Frozen) | F1 (Frozen+Offset) | A0 (`s42`) | A1 (`s42`) | `F1 − F0` | `A0 − F0` | `A1 − A0` | BERT-Base | BGE-Base-Mean | BGE-Base-CLS |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CF Div. 2 Dev (`100`) | 59.00 [50.00, 68.63] | 60.00 [50.52, 70.21] | 58.00 [49.53, 67.42] | 56.00 [45.63, 66.36] | 1.00 [-3.96, 6.67] | -1.00 [-10.31, 8.70] | -2.00 [-9.00, 4.94] | 57.00 [46.90, 67.23] | 56.00 [46.30, 65.56] | 57.00 [47.42, 66.25] |
| CF Div. 2 Test (`274`) | 59.85 [54.29, 65.23] | 58.03 [52.88, 63.51] | 55.11 [50.00, 59.92] | 54.01 [48.77, 59.74] | -1.82 [-4.69, 0.70] | -4.74 [-11.11, 1.54] | -1.09 [-6.09, 4.03] | 55.47 [49.63, 61.68] | 54.01 [47.97, 59.86] | 53.28 [47.17, 59.07] |
| AtCoder ABC Test (`646`) | 55.88 [52.11, 59.47] | 56.97 [53.26, 60.65] | 61.61 [57.89, 65.27] | 56.81 [52.52, 60.78] | 1.08 [-0.31, 2.62] | 5.73 [1.85, 9.66] | -4.80 [-7.86, -1.84] | 55.73 [52.48, 59.04] | 51.86 [48.13, 55.64] | 51.86 [47.99, 55.81] |
| CF Other Div. E1 (`227`) | 51.98 [46.09, 57.75] | 53.30 [47.35, 59.29] | 56.39 [51.57, 61.44] | 54.63 [48.51, 60.46] | 1.32 [-0.87, 3.69] | 4.41 [-2.34, 11.38] | -1.76 [-7.63, 3.65] | 53.74 [47.90, 59.59] | 51.10 [45.05, 57.51] | 50.22 [44.49, 56.31] |
| AtCoder ARC Aux. (`170`) | 48.24 [40.00, 55.87] | 45.88 [37.71, 53.49] | 52.35 [42.69, 61.68] | 50.59 [40.22, 60.67] | -2.35 [-5.75, 0.59] | 4.12 [-4.05, 12.12] | -1.76 [-7.27, 3.47] | 49.41 [42.01, 56.90] | 53.53 [44.97, 62.13] | 52.94 [44.31, 61.59] |
| ABC Post-Cutoff E2 (`21`) | 52.38 [28.57, 71.43] | 61.90 [42.86, 71.43] | 66.67 [57.14, 71.43] | 57.14 [42.86, 71.43] | 9.52 [0.00, 14.29] | 14.29 [0.00, 42.86] | -9.52 [-28.57, 0.00] | 42.86 [42.86, 42.86] | 42.86 [14.29, 71.43] | 42.86 [14.29, 71.43] |

**Table S2b. Multi-seed replication across seeds 42, 101, and 202 for adapted Qwen3 models (within-position accuracy %).**

| Cohort | A0 (`42 / 101 / 202`) | A0 Mean ± SD | A1 (`42 / 101 / 202`) | A1 Mean ± SD | `A1 − A0` (`42 / 101 / 202`) | `A1 − A0` Mean ± SD |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| `CF_dev` | 58.00 / 56.00 / 62.00 | 58.67 ± 3.06 | 56.00 / 54.00 / 62.00 | 57.33 ± 4.16 | -2.00 / -2.00 / +0.00 | -1.33 ± 1.15 |
| `CF_test` | 55.11 / 56.57 / 55.47 | 55.72 ± 0.76 | 54.01 / 57.66 / 56.93 | 56.20 ± 1.93 | -1.09 / +1.09 / +1.46 | +0.49 ± 1.38 |
| `ABC_test` | 61.61 / 59.13 / 59.13 | 59.96 ± 1.43 | 56.81 / 56.66 / 60.06 | 57.84 ± 1.92 | -4.80 / -2.48 / +0.93 | -2.12 ± 2.88 |
| `E1_CF_External` | 56.39 / 54.63 / 53.74 | 54.92 ± 1.35 | 54.63 / 54.63 / 55.95 | 55.07 ± 0.76 | -1.76 / +0.00 / +2.20 | +0.15 ± 1.99 |
| `ARC_test` | 52.35 / 48.82 / 50.59 | 50.59 ± 1.76 | 50.59 / 50.59 / 47.65 | 49.61 ± 1.70 | -1.76 / +1.76 / -2.94 | -0.98 ± 2.45 |
| `E2_ABC_PostCutoff` | 66.67 / 47.62 / 57.14 | 57.14 ± 9.52 | 57.14 / 61.90 / 52.38 | 57.14 ± 4.76 | -9.52 / +14.29 / -4.76 | +0.00 ± 12.60 |

**Table S2c. Best-epoch training dynamics, learned training-only position offsets $b_k$ ($b_A \equiv 0$), and local RTX 5070 Laptop GPU cost.**

| Run | Best Epoch | Dev Same (%) | Train Wall (s) | Infer (`items/s`) | Peak VRAM (MiB) | `b_B` | `b_C` | `b_D` | `b_E` | `b_F` | `b_G` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen_F0` | 3 | 59.00 | 3.3 | 19.9 | 62.3 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| `qwen_F1` | 3 | 60.00 | 3.6 | 19.9 | 62.3 | +0.003 | +0.771 | +1.241 | +1.613 | +1.592 | +1.140 |
| `qwen_A0_s42` | 4 | 58.00 | 3060.4 | 20.8 | 1452.5 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| `qwen_A1_s42` | 1 | 56.00 | 3780.2 | 22.1 | 1452.8 | -0.324 | +0.188 | +0.431 | +0.530 | +0.520 | +0.313 |
| `qwen_A0_s101` | 2 | 56.00 | 3340.2 | 13.5 | 1619.3 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| `qwen_A1_s101` | 1 | 54.00 | 3391.3 | 22.3 | 1620.8 | -0.333 | +0.220 | +0.441 | +0.550 | +0.443 | +0.348 |
| `qwen_A0_s202` | 2 | 62.00 | 2687.9 | 22.3 | 1619.8 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| `qwen_A1_s202` | 2 | 62.00 | 3391.1 | 19.1 | 1621.7 | -0.359 | +0.238 | +0.574 | +0.830 | +0.907 | +0.571 |

### S2.2 Reference-Shift Identity

Because replacing a weak length reference with strong `Joint Ridge` changes only the baseline term in $D = \Delta_{\mathrm{pool}} - \Delta_{\mathrm{same}}$, it subtracts a cohort-fixed constant (`11.69`, `9.90`, and `13.78` pp across CF Div. 2, AtCoder ABC, and CF E1, respectively) from every model's $D$ identically.

---

## Supplement S3. Rolling Out-of-Time Development Selection & Robustness

Across four successive annual test windows (2023–2026) and **12 candidate settings**—composed per platform (Codeforces Div. 2 and AtCoder ABC) of **4 fixed candidate pools** (`fam1_bow_vs_len` [2 models], `fam2_joint_vs_surf` [2 models], `fam3_all4_classic` [4 models], `fam4_all8_modern` [8 models]) plus **2 hyperparameter tuning grids** (`e2e_linear_10grid` [10 configurations], `e2e_full_58grid` [58 configurations])—we compared two development selection rules (pooled vs. within-position dev accuracy) inside every setting (`4 windows × 12 settings = 48` comparisons). The two rules selected different winners in **27 of 48** window comparisons (mean bootstrap flip rates `34.1%–88.3%`). When reselecting models inside every block-bootstrap replicate, all **12 of 12** unconditional 95% confidence intervals for the four-year pooled downstream gain $G$ included zero. Equal-weight (`50/50`) versus empirical-weight pairwise RankNet sensitivity across the 58-configuration grid showed no consistent out-of-time improvement.

---

## Supplement S4. External Generative Judge Configuration and Pairwise Diagnostic

- **Model & API Parameters:** Requested model `deepseek-flash`; observed server model identifier **`deepseek-v4.1-flash-1017`**; `temperature = 0`, `max_tokens = 128`, `thinking = "disabled"`, `response_format = "json_object"`, `tools = "disabled"`.
- **Coverage & Parse Rate:** Scored **3,062 target evaluation items** (`CF_test`, `ABC_test`, `E1_CF_External`, `E2_ABC_PostCutoff`) plus **20 development anchor items** (`3,082` unique items total) using anonymized `statement_v2_readable` inputs (`0.00%` parse failure).
- **Verbatim Prompt (`prompt_sha256: da1ee43c2c4b39ae4972e084f020e962d997b031f8fc308544c1dae8149d19dd`):** *"Estimate the difficulty for a competitive programmer to derive and implement a correct solution from the statement and constraints. Use an integer scale from 0 to 100. Use 0 for direct implementation from the statement, 50 for a standard algorithm requiring a nontrivial observation, and 100 for several dependent insights with demanding implementation. Return a JSON object with a single integer field named score."*
- **API Repeatability Audit:** Re-querying 10 development items under identical `temperature = 0` parameters yielded **`70.0%` exact match** and mean absolute score difference **`2.5`** (0–100 scale), confirming that remote API inference is **not strictly deterministic** even at temperature 0.
- **Pairwise-Prompting (`AB/BA`) Diagnostic:** On a separate 30-block, 153-pair subset, direct `AB/BA` pairwise prompting achieved `0.00%` format failure, `82.4%` order-swap consistency (`CF 89.4%`, `ABC 79.2%`), and `61.11%` pairwise accuracy vs. `58.17%` single-item accuracy on the identical pairs.

---

## Supplement S5. Prospective Contest Protocol & Precision Simulation

On **21 November 2026 (07:30:20 UTC)** we froze a separate prospective protocol (`protocol_sha256: 4057463a99c640e848ac4fcecaca64d0722858231af3c2ad21c16fff75c7ec36`) for future AtCoder Beginner Contest rounds (`abc485+`, zero post-freeze rounds included in this manuscript). The protocol specifies prediction sealing within 24 hours of contest end and independent rating retrieval on day 14. Block-clustered precision simulation shows that estimating `DeepSeek-Flash − Joint` within-position differences on future ABC rounds yields expected 95% CI half-widths of `±10.68 pp` at `K=10` blocks (~4.6 months), `±5.68 pp` at `K=40` (~1.5 years), and `±3.82 pp` at `K=91` (~3.5 years).

---

## Supplement S6. Behavioral In-Contest Solve-Rate Dual Criterion

Joining official Codeforces Div. 2 Test pairs (`274` directional same-slot pairs) with archived in-contest solver counts (`solved_in_contest / n_participants`) confirmed **262/274** directional agreement (**95.62%**, 95% CI `93.01%–97.93%`), rising to **189/190 (99.47%)** above the independently chosen `CF_dev` threshold $\delta_{\min} = 0.03$. Table S6 reports the 30%-budget decision simulation evaluated directly under both official rating order and behavioral solve-rate order.

**Table S6. Codeforces Div. 2 30%-budget dual-criterion comparison (400 requests: 120 referred, 280 automated).**

| Model | Official Verif. (`n`) | Official Ties (`n`) | Official Err / 100 | Official $\Delta$ vs. Joint [95% CI] | Behavioral Err / 100 (`n=280`) | Behavioral $\Delta$ vs. Joint [95% CI] | $|\Delta p| \ge 0.03$ Beh. Err / Verif. |
| --- | ---: | ---: | ---: | :---: | ---: | :---: | :---: |
| `joint` | 200 | 80 | 20.50 | 0.00 [0.00, 0.00] | 28.75 | 0.00 [0.00, 0.00] | 68 / 172 |
| `pw_joint` | 199 | 81 | 20.25 | -0.25 [-2.68, 2.10] | 28.00 | -0.75 [-3.58, 2.01] | 67 / 178 |
| `llm` | 187 | 93 | 21.75 | 1.25 [-2.70, 5.01] | 32.50 | 3.75 [-1.58, 8.55] | 76 / 183 |
| `qwen_F0` | 198 | 82 | 19.50 | -1.00 [-4.30, 2.40] | 28.75 | 0.00 [-3.88, 3.88] | 68 / 179 |
| `qwen_F1` | 200 | 80 | 19.25 | -1.25 [-4.94, 2.26] | 27.50 | -1.25 [-5.33, 2.79] | 67 / 182 |
| `qwen_A0_s42` | 200 | 80 | 22.50 | 2.00 [-1.47, 5.78] | 29.00 | 0.25 [-3.78, 4.37] | 72 / 178 |
| `qwen_A1_s42` | 203 | 77 | 21.25 | 0.75 [-2.29, 3.89] | 29.00 | 0.25 [-3.56, 4.22] | 74 / 176 |

---

## Supplement S7. Complete Cohort Timelines, All-Model Budget Tables, Three-Denominator Cost Audit, and Field Mapping

### S7.1 Complete Cohort Timelines and Study Audit Milestones

**Study Milestones (UTC):** (1) Historical contest archive cutoff: `2026-10-06`; (2) `DeepSeek-Flash` scoring pilot & audit: `2026-11-21T07:12:10Z` (`step6_dev_pilot.json`); (3) Prospective AtCoder ABC protocol freeze (`abc485+`, 0 post-freeze rounds included): `2026-11-21T07:30:20Z`; (4) Formal V4 factorial and replay specification freeze: `2026-11-21T11:01:50Z` (`spec_sha256: ee08471d870b021513024dd909918b319687503bf060e85891734ca9fb2338ef`). Baseline 30%-budget ABC counts (`213 / 141 / 119`) were observed during historical exploration prior to milestone (4), whereas all Qwen3 factorial checkpoints (`F0/F1/A0/A1`, 3 seeds) were trained and evaluated afterward.

**Table S7a. Exact UTC date spans and verified denominators across all nine historical cohorts.**

| Cohort ID | Contest Date Range (UTC) | Total Items | Paired Unique Items | Contest-Pair Blocks (Non-Empty Same) | Same-Slot Dir. / Cand. (Ties) | Pooled Dir. |
| --- | :---: | ---: | ---: | ---: | :---: | ---: |
| `CF_train` | N/A | 2568 | 2431 | 277 (273) | 800 / 974 (174) | 5193 |
| `CF_dev` | 2022-01-12 to 2022-12-16 | 350 | 331 | 33 (30) | 100 / 144 (44) | 778 |
| `CF_test` | 2023-01-05 to 2026-09-26 | 1042 | 983 | 101 (94) | 274 / 400 (126) | 2224 |
| `ABC_train` | N/A | 923 | 865 | 90 (89) | 410 / 411 (1) | 2393 |
| `ABC_dev` | 2022-01-08 to 2022-12-24 | 400 | 384 | 24 (24) | 192 / 192 (0) | 1536 |
| `ABC_test` | 2023-01-07 to 2026-10-03 | 1382 | 1298 | 91 (91) | 646 / 646 (0) | 4640 |
| `E1_CF_External` | 2023-01-27 to 2026-09-21 | 902 | 739 | 52 (51) | 227 / 341 (114) | 2399 |
| `ARC_test` | 2023-01-14 to 2026-10-04 | 422 | 352 | 33 (33) | 170 / 170 (0) | 948 |
| `E2_ABC_PostCutoff` | 2026-10-11 to 2026-11-14 | 42 | 42 | 3 (3) | 21 / 21 (0) | 147 |

### S7.2 Complete All-Model 30%-Budget Disposition and Three-Seed ABC Counts

**Table S7b. Full all-model 30%-budget disposition (extending compact Table 3 with `PW-Joint`, `Qwen3-F0`, `Qwen3-F1`, `Qwen3-A1`).**

| Cohort | Model | Auto Ties | Auto Verif. | Auto Errors | Errors / 100 [95% CI] | $\Delta$ vs. Joint [95% CI] |
| --- | --- | ---: | ---: | ---: | :---: | :---: |
| `CF_test` | `joint` | 80 | 200 | 82.0 | 20.50 [16.71, 23.84] | 0.00 [0.00, 0.00] |
| `CF_test` | `pw_joint` | 81 | 199 | 81.0 | 20.25 [16.14, 23.94] | -0.25 [-2.68, 2.10] |
| `CF_test` | `llm` | 93 | 187 | 87.0 | 21.75 [17.86, 25.27] | 1.25 [-2.70, 5.01] |
| `CF_test` | `qwen_F0` | 82 | 198 | 78.0 | 19.50 [16.03, 22.86] | -1.00 [-4.30, 2.40] |
| `CF_test` | `qwen_F1` | 80 | 200 | 77.0 | 19.25 [15.69, 22.57] | -1.25 [-4.94, 2.26] |
| `CF_test` | `qwen_A0_s42` | 80 | 200 | 90.0 | 22.50 [18.87, 26.09] | 2.00 [-1.47, 5.78] |
| `CF_test` | `qwen_A1_s42` | 77 | 203 | 85.0 | 21.25 [17.45, 24.88] | 0.75 [-2.29, 3.89] |
| `CF_test` | `qwen_A0_s101` | 76 | 204 | 82.0 | 20.50 [16.85, 23.86] | 0.00 [-3.11, 3.18] |
| `CF_test` | `qwen_A1_s101` | 86 | 194 | 73.0 | 18.25 [14.65, 21.56] | -2.25 [-5.59, 1.05] |
| `CF_test` | `qwen_A0_s202` | 84 | 196 | 77.0 | 19.25 [15.97, 22.32] | -1.25 [-4.08, 1.81] |
| `CF_test` | `qwen_A1_s202` | 94 | 186 | 72.0 | 18.00 [14.65, 21.19] | -2.50 [-5.67, 0.85] |
| `CF_test` | `rand_refer_joint_rank` | 89 | 191 | 82.0 | 20.50 [17.33, 23.47] | 0.00 [-1.82, 1.98] |
| `CF_test` | `rand_refer_llm_rank` | 89 | 191 | 83.3 | 20.82 [17.62, 23.67] | 0.32 [-2.99, 3.62] |
| `ABC_test` | `joint` | 0 | 453 | 213.0 | 32.97 [29.75, 36.19] | 0.00 [0.00, 0.00] |
| `ABC_test` | `pw_joint` | 0 | 453 | 211.0 | 32.66 [29.48, 35.87] | -0.31 [-2.32, 1.71] |
| `ABC_test` | `llm` | 0 | 453 | 119.0 | 18.42 [15.48, 21.48] | -14.55 [-18.57, -10.46] |
| `ABC_test` | `qwen_F0` | 0 | 453 | 188.0 | 29.10 [26.03, 32.14] | -3.87 [-7.28, -0.46] |
| `ABC_test` | `qwen_F1` | 0 | 453 | 183.0 | 28.33 [25.35, 31.27] | -4.64 [-8.04, -1.25] |
| `ABC_test` | `qwen_A0_s42` | 0 | 453 | 168.0 | 26.01 [23.03, 29.11] | -6.97 [-10.82, -3.27] |
| `ABC_test` | `qwen_A1_s42` | 0 | 453 | 196.0 | 30.34 [27.13, 33.75] | -2.63 [-6.35, 0.93] |
| `ABC_test` | `qwen_A0_s101` | 0 | 453 | 170.0 | 26.32 [23.18, 29.41] | -6.66 [-10.49, -3.07] |
| `ABC_test` | `qwen_A1_s101` | 0 | 453 | 184.0 | 28.48 [25.38, 31.59] | -4.49 [-7.64, -1.23] |
| `ABC_test` | `qwen_A0_s202` | 0 | 453 | 180.0 | 27.86 [24.81, 31.07] | -5.11 [-8.59, -1.71] |
| `ABC_test` | `qwen_A1_s202` | 0 | 453 | 175.0 | 27.09 [24.19, 30.19] | -5.88 [-9.33, -2.64] |
| `ABC_test` | `rand_refer_joint_rank` | 0 | 453 | 212.1 | 32.83 [30.48, 35.39] | -0.14 [-2.14, 1.86] |
| `ABC_test` | `rand_refer_llm_rank` | 0 | 453 | 142.7 | 22.09 [19.46, 24.89] | -10.88 [-14.43, -7.20] |
| `E1_CF_External` | `joint` | 68 | 171 | 73.0 | 21.41 [17.59, 25.14] | 0.00 [0.00, 0.00] |
| `E1_CF_External` | `pw_joint` | 75 | 164 | 72.0 | 21.11 [17.01, 25.15] | -0.29 [-2.69, 2.05] |
| `E1_CF_External` | `llm` | 77 | 162 | 65.0 | 19.06 [15.71, 22.38] | -2.35 [-5.85, 1.20] |
| `E1_CF_External` | `qwen_F0` | 76 | 163 | 73.0 | 21.41 [17.29, 25.31] | 0.00 [-4.34, 4.52] |
| `E1_CF_External` | `qwen_F1` | 78 | 161 | 71.0 | 20.82 [16.82, 24.71] | -0.59 [-4.79, 3.80] |
| `E1_CF_External` | `qwen_A0_s42` | 60 | 179 | 75.0 | 21.99 [18.34, 25.83] | 0.59 [-2.88, 4.14] |
| `E1_CF_External` | `qwen_A1_s42` | 70 | 169 | 72.0 | 21.11 [17.32, 25.00] | -0.29 [-3.38, 2.99] |
| `E1_CF_External` | `qwen_A0_s101` | 59 | 180 | 80.0 | 23.46 [19.76, 27.35] | 2.05 [-1.18, 5.37] |
| `E1_CF_External` | `qwen_A1_s101` | 65 | 174 | 73.0 | 21.41 [17.45, 25.73] | 0.00 [-3.75, 3.86] |
| `E1_CF_External` | `qwen_A0_s202` | 70 | 169 | 71.0 | 20.82 [17.26, 24.56] | -0.59 [-3.79, 2.62] |
| `E1_CF_External` | `qwen_A1_s202` | 76 | 163 | 66.0 | 19.35 [16.31, 22.62] | -2.05 [-5.00, 1.16] |
| `E1_CF_External` | `rand_refer_joint_rank` | 79 | 160 | 71.2 | 20.89 [17.53, 24.25] | -0.52 [-2.42, 1.48] |
| `E1_CF_External` | `rand_refer_llm_rank` | 79 | 160 | 68.5 | 20.08 [16.89, 23.20] | -1.33 [-4.71, 1.98] |

**Table S7c. Three-seed 30%-budget AtCoder ABC automated errors (`646` requests: `193` referred, `453` automated) vs. same-ranker random expectation.**

| Seed Run | Margin Auto Errors | Margin Errors / 100 | Same-Ranker Random Expected Errors |
| --- | ---: | ---: | ---: |
| `qwen_A0_s42` | 168.0 | 26.01 | 173.07 |
| `qwen_A0_s101` | 170.0 | 26.32 | 184.16 |
| `qwen_A0_s202` | 180.0 | 27.86 | 184.58 |
| `qwen_A1_s42` | 196.0 | 30.34 | 195.14 |
| `qwen_A1_s101` | 184.0 | 28.48 | 195.50 |
| `qwen_A1_s202` | 175.0 | 27.09 | 180.06 |

### S7.3 Three-Denominator AtCoder ABC Cost Accounting

**Table S7d. Transparent cost accounting across all three AtCoder ABC item denominators (primary manuscript basis: `1,292` request-involved problem keys).**

| Policy / Model | Denominator Scope | Item Count | Compute / Token Usage | Saved Errors (vs. Joint 213) | Incremental Cost per Saved Error |
| --- | --- | ---: | --- | ---: | --- |
| `Joint Ridge` (CPU) | Request-involved unique keys | 1,292 | Measured **`0.26 s`** CPU inference (`1.29 s` fit) | 0 (Baseline) | Baseline |
| `Qwen3-A0` (Local GPU) | Request-involved unique keys (Primary) | 1,292 | Est. **`62.1 s`** GPU infer (`20.8 items/s`) + `3060.4 s` one-time LoRA train | 45 (`213 -> 168`) | **`1.38 s`** infer / error (zero API fee) |
| `Qwen3-A0` (Local GPU) | Full ABC test cohort | 1,382 | Est. `66.4 s` GPU infer (`20.8 items/s`) + `3060.4 s` train | 45 (`213 -> 168`) | `1.48 s` infer / error (zero API fee) |
| `DeepSeek-Flash` (API) | Request-involved unique keys (Primary) | 1,292 | **`409,426`** prompt + **`9,836`** comp. tokens (`2020.1 s` latency) | 94 (`213 -> 119`) | **`4,356`** prompt + **`105`** comp. / error |
| `DeepSeek-Flash` (API) | Paired-contest scored items | 1,298 | `410,885` prompt + `9,881` comp. tokens (`2032.2 s` latency) | 94 (`213 -> 119`) | `4,371` prompt + `105` comp. / error |

*Note.* The 1,382-item ABC test cohort includes 84 items from unmatched single contests excluded by shift-0 adjacent pairing, leaving 1,298 paired-contest items, of which 1,292 distinct problem keys participate in the 646 same-position requests (6 items occupy unmatched slots without a same-slot partner). `Qwen3-A0` inference times are projected from measured `CF_test` throughput (`20.813 items/s`); CPU `Joint Ridge` timings (`0.26 s` inference, `1.29 s` fit) were measured directly on the 1,292 ABC request keys.

### S7.4 Complete 0%–50% Review-Budget Curves (Verifiable Automatic Errors per 100 Raw Requests)

**Table S7e. Verifiable automatic errors per 100 requests across review budgets (operational tie fallback; 0% academic 0.5-tie reference also shown).**

| Cohort | Model | 0% (Academic) | 0% (Op.) | 10% | 20% | 30% | 50% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `CF_test` | `joint` | 29.50 | 29.50 | 26.25 | 22.75 | 20.50 | 14.50 |
| `CF_test` | `pw_joint` | 30.00 | 30.00 | 25.75 | 23.75 | 20.25 | 14.50 |
| `CF_test` | `qwen_F0` | 27.50 | 27.50 | 23.75 | 21.25 | 19.50 | 13.00 |
| `CF_test` | `qwen_F1` | 28.75 | 28.75 | 24.25 | 21.75 | 19.25 | 14.00 |
| `CF_test` | `qwen_A0_s42` | 30.75 | 30.75 | 28.00 | 25.25 | 22.50 | 15.75 |
| `CF_test` | `qwen_A1_s42` | 31.50 | 31.50 | 28.25 | 24.25 | 21.25 | 14.50 |
| `CF_test` | `llm` | 31.25 | 29.75 | 27.00 | 24.50 | 21.75 | 14.00 |
| `CF_test` | `rand_refer_joint_rank` | — | 29.50 | 26.45 | 23.48 | 20.50 | 14.89 |
| `ABC_test` | `joint` | 46.90 | 46.90 | 41.80 | 37.31 | 32.97 | 23.84 |
| `ABC_test` | `pw_joint` | 45.98 | 45.98 | 41.80 | 37.46 | 32.66 | 22.76 |
| `ABC_test` | `qwen_F0` | 44.12 | 44.12 | 38.70 | 33.75 | 29.10 | 20.43 |
| `ABC_test` | `qwen_F1` | 43.03 | 43.03 | 37.77 | 33.28 | 28.33 | 19.50 |
| `ABC_test` | `qwen_A0_s42` | 38.39 | 38.39 | 35.14 | 30.96 | 26.01 | 17.34 |
| `ABC_test` | `qwen_A1_s42` | 43.19 | 43.19 | 39.16 | 34.67 | 30.34 | 19.97 |
| `ABC_test` | `llm` | 31.27 | 31.58 | 25.85 | 21.83 | 18.42 | 12.23 |
| `ABC_test` | `rand_refer_joint_rank` | — | 46.90 | 42.19 | 37.59 | 32.83 | 23.44 |
| `E1_CF_External` | `joint` | 29.91 | 29.91 | 28.15 | 24.63 | 21.41 | 16.13 |
| `E1_CF_External` | `pw_joint` | 32.55 | 32.55 | 28.45 | 25.51 | 21.11 | 16.13 |
| `E1_CF_External` | `qwen_F0` | 31.96 | 31.96 | 28.15 | 23.75 | 21.41 | 14.96 |
| `E1_CF_External` | `qwen_F1` | 31.09 | 31.09 | 28.15 | 23.75 | 20.82 | 14.37 |
| `E1_CF_External` | `qwen_A0_s42` | 29.03 | 29.03 | 27.86 | 24.93 | 21.99 | 14.08 |
| `E1_CF_External` | `qwen_A1_s42` | 30.21 | 30.21 | 28.15 | 25.22 | 21.11 | 14.96 |
| `E1_CF_External` | `llm` | 29.18 | 28.74 | 26.10 | 22.58 | 19.06 | 13.49 |
| `E1_CF_External` | `rand_refer_joint_rank` | — | 29.91 | 27.08 | 23.94 | 20.89 | 15.12 |

### S7.5 Five-Salt SHA-256 Margin Tie-Break Sensitivity (30% Budget)

**Table S7f. Sensitivity of verifiable automatic errors across 5 independent SHA-256 tie-break salts.**

| Cohort | Model | Min Errors | Max Errors | Range (Errors / 100 Requests) |
| --- | --- | ---: | ---: | :---: |
| `CF_test` | `joint` | 82.0 | 82.0 | 20.50 – 20.50 |
| `CF_test` | `llm` | 81.0 | 88.0 | 20.25 – 22.00 |
| `CF_test` | `qwen_A0_s42` | 90.0 | 90.0 | 22.50 – 22.50 |
| `CF_test` | `qwen_A1_s42` | 85.0 | 85.0 | 21.25 – 21.25 |
| `ABC_test` | `joint` | 213.0 | 213.0 | 32.97 – 32.97 |
| `ABC_test` | `llm` | 115.0 | 120.0 | 17.80 – 18.58 |
| `ABC_test` | `qwen_A0_s42` | 168.0 | 168.0 | 26.01 – 26.01 |
| `ABC_test` | `qwen_A1_s42` | 196.0 | 196.0 | 30.34 – 30.34 |
| `E1_CF_External` | `joint` | 73.0 | 73.0 | 21.41 – 21.41 |
| `E1_CF_External` | `llm` | 65.0 | 68.0 | 19.06 – 19.94 |
| `E1_CF_External` | `qwen_A0_s42` | 75.0 | 75.0 | 21.99 – 21.99 |
| `E1_CF_External` | `qwen_A1_s42` | 72.0 | 72.0 | 21.11 – 21.11 |

### S7.6 Manuscript Terminology to Data-Field Mapping

| Manuscript Term | Replication Package Identifier |
| --- | --- |
| Codeforces Div. 2 Test / Dev / Train | `CF_test` / `CF_dev` / `CF_train` |
| AtCoder ABC Test / Dev / Train | `ABC_test` / `ABC_dev` / `ABC_train` |
| Codeforces Other Divisions (E1) | `E1_CF_External` |
| AtCoder ABC Post-Cutoff Pilot (E2) | `E2_ABC_PostCutoff` |
| Within-position / Pooled pairs | `same_slot` / `pooled` |
| Repaired statement preprocessing | `p3-proto-v2` (`statement_v2_readable`) |
| Training-only position offset models (`F1, A1`) | `qwen_F1`, `qwen_A1_s42/101/202` |
| Un-offset LoRA / Frozen models (`A0, F0`) | `qwen_A0_s42/101/202`, `qwen_F0` |
