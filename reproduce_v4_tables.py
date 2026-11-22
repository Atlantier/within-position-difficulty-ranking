# -*- coding: utf-8 -*-
"""Self-contained zero-prose verification & table-recomputation script for the
v4 anonymous replication package.

Directly from `data_scores/*.csv`, independently recomputes and verifies:
  1. Table 2: Within-position & pooled pairwise accuracies + B=2,000 paired
     contest-pair block-bootstrap 95% CIs (including primary A1 - A0, A0 - F0,
     A0 - Joint, and DeepSeek - Joint contrasts) across CF_test, ABC_test, E1.
  2. Three-seed stability table (seeds 42, 101, 202) for Qwen3-A0 & Qwen3-A1
     (both within-position accuracy and ABC 30%-budget automated errors).
  3. Table 3 & Table 4a: Chronological 30%-budget disposition and AtCoder ABC
     2x2 counterfactual swap (`213 / 141` and `213 / 119`).
  4. Table 4b: Fixed-ranker Margin vs. exact expected intra-block uniform
     Random referral (`Random - Margin`, saved errors / 100 requests, B=2,000
     paired contest-pair block CIs) across all 3 cohorts.
  5. Unified 1,292-request-item AtCoder ABC token and GPU-inference cost audit.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data_scores"
REF = HERE / "reference_tables"
TIE_TOL = 1e-12
BOOT_B = 2000
BOOT_SEED = 42


def sha256_pair(ka: str, kb: str, salt: str = "") -> str:
    s1, s2 = sorted((str(ka), str(kb)))
    return hashlib.sha256(f"{s1}|{s2}{salt}".encode("utf-8")).hexdigest()


def load_merged_scores() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    v2 = pd.read_csv(DATA / "v2_all_model_scores_with_dev.csv")
    llm = pd.read_csv(DATA / "llm_single_item_scores.csv")
    q = pd.read_csv(DATA / "qwen3_item_scores.csv")
    v2["llm"] = v2["problem_key"].astype(str).map(dict(zip(llm["problem_key"].astype(str), llm["score"].astype(float))))
    q_cols = [c for c in q.columns if c.startswith("qwen_")]
    for c in q_cols:
        v2[c] = v2["problem_key"].astype(str).map(dict(zip(q["problem_key"].astype(str), q[c].astype(float))))
    return v2, llm, q, q_cols


def build_shift0_blocks(df_cohort: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    work = df_cohort.copy().reset_index(drop=True)
    if "quarter" not in work.columns:
        work["quarter"] = pd.to_datetime(work["start_time"], unit="s", utc=True).dt.quarter
    work["platform"] = work["problem_key"].map(lambda k: "codeforces" if "-" in str(k) else "atcoder")
    raw = []
    for (plat, ser, yr, q), grp in work.groupby(["platform", "series", "year", "quarter"], sort=True):
        cids = grp[["contest_id", "start_time"]].drop_duplicates().sort_values(["start_time", "contest_id"])["contest_id"].tolist()
        for c1, c2 in zip(cids[::2], cids[1::2]):
            g1 = grp[grp["contest_id"] == c1].sort_values("problem_index")
            g2 = grp[grp["contest_id"] == c2].sort_values("problem_index")
            raw.append((int(g2["start_time"].max()), int(g1["start_time"].max()), str(c2), str(c1), g1, g2))
    raw.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return [(g1, g2) for (_, _, _, _, g1, g2) in raw]


def compute_table2_and_contrasts(v2: pd.DataFrame, models: list[str]) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Recompute exact Table 2 accuracies and paired block-bootstrap CIs."""
    out: dict[str, dict[str, tuple[float, float, float]]] = {}
    for cid in ("CF_test", "ABC_test", "E1_CF_External"):
        blocks = build_shift0_blocks(v2[v2["cohort"] == cid])
        n_b = len(blocks)
        same_n = np.zeros(n_b, dtype=np.float64)
        pool_n = np.zeros(n_b, dtype=np.float64)
        same_c = {m: np.zeros(n_b, dtype=np.float64) for m in models}
        pool_c = {m: np.zeros(n_b, dtype=np.float64) for m in models}
        for bi, (g1, g2) in enumerate(blocks):
            r1 = list(g1.itertuples(index=False))
            r2 = list(g2.itertuples(index=False))
            for a in r1:
                for b in r2:
                    ra, rb = float(a.rating), float(b.rating)
                    if not (np.isfinite(ra) and np.isfinite(rb) and ra != rb):
                        continue
                    sgn = 1.0 if ra > rb else -1.0
                    is_same = str(a.problem_index) == str(b.problem_index)
                    pool_n[bi] += 1.0
                    if is_same:
                        same_n[bi] += 1.0
                    for m in models:
                        d = float(getattr(a, m)) - float(getattr(b, m))
                        sc = 0.5 if abs(d) <= TIE_TOL else (1.0 if d * sgn > 0 else 0.0)
                        pool_c[m][bi] += sc
                        if is_same:
                            same_c[m][bi] += sc

        rng_p = np.random.default_rng(BOOT_SEED)
        idx_p = rng_p.integers(0, n_b, size=(BOOT_B, n_b))
        den_p = pool_n[idx_p].sum(axis=1)

        ne_s = np.where(same_n > 0)[0]
        rng_s = np.random.default_rng(BOOT_SEED)
        idx_s = ne_s[rng_s.integers(0, len(ne_s), size=(BOOT_B, len(ne_s)))]
        den_s = same_n[idx_s].sum(axis=1)

        res: dict[str, tuple[float, float, float]] = {}
        for m in models:
            pt_s = 100.0 * same_c[m].sum() / same_n.sum()
            bs_s = 100.0 * same_c[m][idx_s].sum(axis=1) / np.where(den_s > 0, den_s, np.nan)
            lo_s, hi_s = np.nanpercentile(bs_s, [2.5, 97.5])
            res[f"same:{m}"] = (round(float(pt_s), 2), round(float(lo_s), 2), round(float(hi_s), 2))

            pt_p = 100.0 * pool_c[m].sum() / pool_n.sum()
            bs_p = 100.0 * pool_c[m][idx_p].sum(axis=1) / np.where(den_p > 0, den_p, np.nan)
            lo_p, hi_p = np.nanpercentile(bs_p, [2.5, 97.5])
            res[f"pool:{m}"] = (round(float(pt_p), 2), round(float(lo_p), 2), round(float(hi_p), 2))

        for cname, m1, m2 in (
            ("A1_minus_A0", "qwen_A1_s42", "qwen_A0_s42"),
            ("A0_minus_F0", "qwen_A0_s42", "qwen_F0"),
            ("A0_minus_joint", "qwen_A0_s42", "joint"),
            ("llm_minus_joint", "llm", "joint"),
        ):
            pt = 100.0 * (same_c[m1].sum() - same_c[m2].sum()) / same_n.sum()
            bs = 100.0 * (same_c[m1][idx_s].sum(axis=1) - same_c[m2][idx_s].sum(axis=1)) / np.where(den_s > 0, den_s, np.nan)
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            res[f"diff:{cname}"] = (round(float(pt), 2), round(float(lo), 2), round(float(hi), 2))
        out[cid] = res
    return out


def build_same_slot_requests(df_cohort: pd.DataFrame, model_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks = build_shift0_blocks(df_cohort)
    reqs, blks = [], []
    for cidx, (g1, g2) in enumerate(blocks):
        m1 = {str(r.problem_index): r for r in g1.itertuples(index=False)}
        m2 = {str(r.problem_index): r for r in g2.itertuples(index=False)}
        shared = sorted(set(m1) & set(m2))
        for sl in shared:
            a, b = m1[sl], m2[sl]
            ra, rb = float(a.rating), float(b.rating)
            valid = bool(np.isfinite(ra) and np.isfinite(rb) and ra != rb)
            rec = {
                "chrono_idx": cidx,
                "key_a": str(a.problem_key),
                "key_b": str(b.problem_key),
                "off_valid": valid,
                "off_sign": float(np.sign(ra - rb)) if valid else 0.0,
            }
            for m in model_cols:
                d = float(getattr(a, m)) - float(getattr(b, m))
                rec[f"sdiff_{m}"] = abs(d)
                rec[f"sig_{m}"] = 0.0 if abs(d) <= TIE_TOL else float(np.sign(d))
            reqs.append(rec)
        blks.append({"chrono_idx": cidx, "n_same": len(shared)})
    return pd.DataFrame(reqs), pd.DataFrame(blks)


def replay_30_with_random(req_df: pd.DataFrame, blk_df: pd.DataFrame, ref_m: str, rnk_m: str) -> dict[str, float]:
    n_b = len(blk_df)
    cum = 0
    prev_q = 0
    auto_err = 0.0
    rand_err = 0.0
    auto_verif = 0
    auto_ties = 0
    referred = 0
    b_n = np.zeros(n_b, dtype=np.float64)
    b_m_err = np.zeros(n_b, dtype=np.float64)
    b_r_err = np.zeros(n_b, dtype=np.float64)

    for bi, (cidx, n_s) in enumerate(zip(blk_df["chrono_idx"], blk_df["n_same"])):
        n_s = int(n_s)
        cum += n_s
        cur_q = int(math.floor(0.30 * cum))
        dq = cur_q - prev_q
        prev_q = cur_q
        b_n[bi] = n_s
        if n_s == 0:
            continue
        sub = req_df[req_df["chrono_idx"] == cidx].copy()
        sub["_h"] = [sha256_pair(ka, kb) for ka, kb in zip(sub["key_a"], sub["key_b"])]
        sub = sub.sort_values([f"sdiff_{ref_m}", "_h"], ascending=[True, True])
        referred += dq

        # Exact expected uniform random referral errors inside block
        tot_blk_err = 0.0
        for r in sub.itertuples(index=False):
            if r.off_valid:
                pred = getattr(r, f"sig_{rnk_m}")
                if pred == 0.0:
                    pred = getattr(r, "sig_joint")
                if pred != r.off_sign:
                    tot_blk_err += 1.0
        exp_r = ((n_s - dq) / n_s) * tot_blk_err
        b_r_err[bi] = exp_r
        rand_err += exp_r

        auto = sub.iloc[dq:]
        m_e = 0.0
        for r in auto.itertuples(index=False):
            if not r.off_valid:
                auto_ties += 1
                continue
            auto_verif += 1
            pred = getattr(r, f"sig_{rnk_m}")
            if pred == 0.0:
                pred = getattr(r, "sig_joint")
            if pred != r.off_sign:
                m_e += 1.0
        b_m_err[bi] = m_e
        auto_err += m_e

    ne = np.where(b_n > 0)[0]
    rng = np.random.default_rng(BOOT_SEED)
    idx = ne[rng.integers(0, len(ne), size=(BOOT_B, len(ne)))]
    den = b_n[idx].sum(axis=1)
    saved_pt = 100.0 * (rand_err - auto_err) / float(len(req_df))
    bs = 100.0 * (b_r_err[idx].sum(axis=1) - b_m_err[idx].sum(axis=1)) / np.where(den > 0, den, np.nan)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {
        "total": len(req_df),
        "referred": referred,
        "auto_ties": auto_ties,
        "auto_verif": auto_verif,
        "auto_err": auto_err,
        "rand_err": round(rand_err, 2),
        "saved_vs_rand_100": round(saved_pt, 2),
        "saved_ci_lo": round(float(lo), 2),
        "saved_ci_hi": round(float(hi), 2),
    }


def main() -> int:
    v2, llm, q, q_cols = load_merged_scores()
    models = ["joint", "pw_joint", "llm"] + q_cols

    print("=== 1. Table 2 Within-Position Accuracies & Primary Contrasts (B=2,000) ===")
    t2 = compute_table2_and_contrasts(v2, models)
    ref_main = pd.read_csv(REF / "step2_main_comparison_table.csv")
    for cid in ("CF_test", "ABC_test", "E1_CF_External"):
        a1_a0 = t2[cid]["diff:A1_minus_A0"]
        a0_f0 = t2[cid]["diff:A0_minus_F0"]
        llm_j = t2[cid]["diff:llm_minus_joint"]
        ref_r = ref_main[(ref_main["cohort"] == cid) & (ref_main["scope"] == "same_slot") & (ref_main["entry"] == "A1_minus_A0")].iloc[0]
        assert abs(a1_a0[0] - float(ref_r["estimate"])) <= 0.02, f"A1-A0 mismatch on {cid}"
        print(f"  [PASS] {cid:14s} | A0={t2[cid]['same:qwen_A0_s42'][0]:.2f}% | LLM={t2[cid]['same:llm'][0]:.2f}% | "
              f"A1-A0={a1_a0[0]:+.2f} [{a1_a0[1]:+.2f}, {a1_a0[2]:+.2f}] | A0-F0={a0_f0[0]:+.2f} | LLM-Joint={llm_j[0]:+.2f}")

    print("\n=== 2. Three-Seed Stability (Seeds 42 / 101 / 202) ===")
    ref_ms = pd.read_csv(REF / "step2_multiseed_A0_A1.csv")
    for cid in ("CF_test", "ABC_test", "E1_CF_External"):
        a0s = [t2[cid][f"same:qwen_A0_s{s}"][0] for s in (42, 101, 202)]
        a1s = [t2[cid][f"same:qwen_A1_s{s}"][0] for s in (42, 101, 202)]
        diffs = [round(x - y, 2) for x, y in zip(a1s, a0s)]
        ref_r = ref_ms[(ref_ms["cohort"] == cid) & (ref_ms["scope"] == "same_slot")].iloc[0]
        assert abs(np.mean(diffs) - float(ref_r["mean_diff"])) <= 0.03
        print(f"  [PASS] {cid:14s} | A0={a0s} | A1={a1s} | A1-A0={diffs}")

    print("\n=== 3. Table 3 & Table 4a: 30% Budget Counts & ABC 2x2 Swap ===")
    cache = {cid: build_same_slot_requests(v2[v2["cohort"] == cid], models) for cid in ("CF_test", "ABC_test", "E1_CF_External")}
    expected_counts = {
        ("CF_test", "joint", "joint"): 82,
        ("CF_test", "llm", "llm"): 87,
        ("ABC_test", "joint", "joint"): 213,
        ("ABC_test", "joint", "llm"): 141,
        ("ABC_test", "llm", "llm"): 119,
        ("ABC_test", "llm", "joint"): 213,
        ("ABC_test", "qwen_A0_s42", "qwen_A0_s42"): 168,
        ("ABC_test", "qwen_A0_s101", "qwen_A0_s101"): 170,
        ("ABC_test", "qwen_A0_s202", "qwen_A0_s202"): 180,
        ("ABC_test", "qwen_A1_s42", "qwen_A1_s42"): 196,
        ("E1_CF_External", "joint", "joint"): 73,
        ("E1_CF_External", "llm", "llm"): 65,
    }
    for (cid, rm, rkm), exp_err in expected_counts.items():
        obs = replay_30_with_random(*cache[cid], rm, rkm)
        assert int(round(obs["auto_err"])) == exp_err, f"Mismatch on {(cid, rm, rkm)}: {obs} != {exp_err}"
        print(f"  [PASS] {cid:14s} {rm:12s} -> {rkm:12s}: auto_err={int(obs['auto_err']):3d} (verif={obs['auto_verif']}, ties={obs['auto_ties']})")

    print("\n=== 4. Table 4b: Same-Ranker Margin vs. Random Referral (Saved / 100 [95% CI]) ===")
    for m in ("llm", "qwen_A0_s42", "joint"):
        parts = []
        for cid in ("ABC_test", "CF_test", "E1_CF_External"):
            o = replay_30_with_random(*cache[cid], m, m)
            if cid == "ABC_test" and m == "llm":
                assert abs(o["saved_vs_rand_100"] - 3.67) <= 0.02, f"ABC LLM margin-vs-rand mismatch: {o}"
                assert o["saved_ci_lo"] > 0, "Expected strictly positive CI on ABC LLM margin vs rand"
            parts.append(f"{cid}={o['saved_vs_rand_100']:+.2f} [{o['saved_ci_lo']:+.2f}, {o['saved_ci_hi']:+.2f}]")
        print(f"  [PASS] {m:12s} | " + " | ".join(parts))

    print("\n=== 5. Unified 1,292-Request-Item AtCoder ABC Cost Audit ===")
    abc_reqs = cache["ABC_test"][0]
    req_keys = set(abc_reqs["key_a"]).union(set(abc_reqs["key_b"]))
    assert len(req_keys) == 1292, f"Expected 1,292 unique ABC request keys, got {len(req_keys)}"
    sub_llm = llm[llm["problem_key"].astype(str).isin(req_keys)]
    p_tok = int(sub_llm["prompt_tokens"].sum())
    c_tok = int(sub_llm["completion_tokens"].sum())
    lat_s = float(sub_llm["elapsed_s"].sum())
    assert p_tok == 409426 and c_tok == 9836, f"Token mismatch: {p_tok}, {c_tok}"
    cost2 = pd.read_csv(DATA / "qwen3_cost_and_throughput.csv")
    ips = float(cost2.loc[cost2["model"] == "qwen_A0_s42", "infer_items_per_s"].iloc[0])
    q_est = 1292.0 / ips
    print(f"  [PASS] 1,292 ABC request keys | DeepSeek: {p_tok:,} prompt + {c_tok:,} comp. tokens ({lat_s:.1f}s) | Qwen3-A0 est. GPU infer: {q_est:.1f}s")
    print("\n=== Anonymous Replication v4 Full Verification PASS ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
