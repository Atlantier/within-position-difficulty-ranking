# -*- coding: utf-8 -*-
"""Reference training script for the controlled Qwen3-Embedding-0.6B 2x2 factorial
(`F0, F1, A0, A1`) evaluated in the paper.

Input CSV contract (`--statements-csv`):
  Required columns:
    - problem_key (str): e.g. '1900-B' (Codeforces) or 'abc319_b' (AtCoder)
    - cohort (str): 'CF_train', 'CF_dev', 'CF_test', 'ABC_test', 'E1_CF_External', etc.
    - contest_id (str/int), start_time (int UTC epoch), series (str), year (int)
    - problem_index (str): single slot letter 'A'..'I'
    - rating (float): within-platform published difficulty
    - statement_v2_readable (str): release-visible statement preprocessed via
      `statement_feature_parser_v2.py` (no tags, editorials, or post-contest stats).

Usage example:
  python train_qwen3_factorial_reference.py \
    --statements-csv /path/to/rebuilt_statements_v2.csv \
    --output-dir ./retrained_qwen3_outputs \
    --seeds 42 101 202
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REV = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
SLOTS = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
SLOT_TO_ID = {s: i for i, s in enumerate(SLOTS)}
TARGET_MAX_LEN = 768
TAIL_LEN = 256
LAMBDA_B = 1e-3
EPOCHS = 5
LR_HEAD = 1e-2
LR_LORA = 1e-4
LR_BIAS = 1e-2
WEIGHT_DECAY = 0.01


def tokenize_head_tail(texts: list[str], tokenizer, max_len: int = TARGET_MAX_LEN, tail_len: int = TAIL_LEN):
    """Preserve opening problem context + final `tail_len` constraint tokens with left padding."""
    import torch

    head_len = max_len - tail_len
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    seqs = []
    for txt in texts:
        ids = tokenizer.encode(str(txt), add_special_tokens=True, truncation=False)
        if len(ids) > max_len:
            ids = ids[:head_len] + ids[-tail_len:]
        seqs.append(ids)
    longest = max(len(s) for s in seqs)
    out_ids = torch.full((len(seqs), longest), pad_id, dtype=torch.long)
    out_msk = torch.zeros((len(seqs), longest), dtype=torch.long)
    for i, s in enumerate(seqs):
        out_ids[i, longest - len(s) :] = torch.tensor(s, dtype=torch.long)
        out_msk[i, longest - len(s) :] = 1
    return out_ids, out_msk


def build_training_blocks(df_train: pd.DataFrame) -> list[dict[str, Any]]:
    work = df_train.copy().reset_index(drop=True)
    work["quarter"] = pd.to_datetime(work["start_time"], unit="s", utc=True).dt.quarter
    blocks = []
    for (_, _, _), grp in work.groupby(["series", "year", "quarter"], sort=True):
        cids = grp[["contest_id", "start_time"]].drop_duplicates().sort_values(["start_time", "contest_id"])["contest_id"].tolist()
        for c1, c2 in zip(cids[::2], cids[1::2]):
            g1 = grp[grp["contest_id"] == c1].sort_values("problem_index")
            g2 = grp[grp["contest_id"] == c2].sort_values("problem_index")
            items = pd.concat([g1, g2], ignore_index=True)
            k2i = {str(r.problem_key): idx for idx, r in enumerate(items.itertuples(index=False))}
            pi, pj, py = [], [], []
            for a in g1.itertuples(index=False):
                for b in g2.itertuples(index=False):
                    ra, rb = float(a.rating), float(b.rating)
                    if np.isfinite(ra) and np.isfinite(rb) and ra != rb:
                        pi.append(k2i[str(a.problem_key)])
                        pj.append(k2i[str(b.problem_key)])
                        py.append(1.0 if ra > rb else -1.0)
            if pi:
                blocks.append({
                    "keys": items["problem_key"].astype(str).tolist(),
                    "slots": [SLOT_TO_ID[str(s)] for s in items["problem_index"].astype(str).tolist()],
                    "pi": pi, "pj": pj, "py": py,
                })
    return blocks


def train_cell(
    df_all: pd.DataFrame,
    backbone_mode: str,
    use_position_offset: bool,
    seed: int,
    model_id: str,
    revision: str,
) -> dict[str, float]:
    """Train one cell (`frozen` or `lora`, with/without training-only position offset b_k).
    At inference time b_k is discarded and predictions use strictly f(x)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, padding_side="left")
    base = AutoModel.from_pretrained(model_id, revision=revision, torch_dtype=torch.bfloat16).to(device)
    if backbone_mode == "lora":
        cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, r=8, lora_alpha=16, lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        )
        base = get_peft_model(base, cfg)
    else:
        for p in base.parameters():
            p.requires_grad = False

    hidden = int(base.config.hidden_size)
    head = nn.Linear(hidden, 1, dtype=torch.float32).to(device)
    slot_bias = nn.Parameter(torch.zeros(len(SLOTS) - 1, dtype=torch.float32, device=device))

    param_groups = [{"params": list(head.parameters()), "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY}]
    if backbone_mode == "lora":
        param_groups.append({"params": [p for p in base.parameters() if p.requires_grad], "lr": LR_LORA, "weight_decay": WEIGHT_DECAY})
    if use_position_offset:
        param_groups.append({"params": [slot_bias], "lr": LR_BIAS, "weight_decay": 0.0})
    opt = torch.optim.AdamW(param_groups)

    text_map = dict(zip(df_all["problem_key"].astype(str), df_all["statement_v2_readable"].astype(str)))
    train_blocks = build_training_blocks(df_all[df_all["cohort"] == "CF_train"])

    def encode_scores(keys: list[str], grad: bool) -> torch.Tensor:
        ids, msk = tokenize_head_tail([text_map[k] for k in keys], tok)
        ids, msk = ids.to(device), msk.to(device)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            hs = base(input_ids=ids, attention_mask=msk).last_hidden_state
            emb = F.normalize(hs[:, -1, :].float(), p=2, dim=-1)
            return head(emb).squeeze(-1)

    for _epoch in range(1, EPOCHS + 1):
        rng = np.random.default_rng(seed + _epoch)
        order = rng.permutation(len(train_blocks))
        base.train() if backbone_mode == "lora" else base.eval()
        head.train()
        for start in range(0, len(order), 2):
            opt.zero_grad()
            losses = []
            for bi in order[start : start + 2]:
                blk = train_blocks[int(bi)]
                scores = encode_scores(blk["keys"], grad=True)
                pi = torch.tensor(blk["pi"], dtype=torch.long, device=device)
                pj = torch.tensor(blk["pj"], dtype=torch.long, device=device)
                py = torch.tensor(blk["py"], dtype=torch.float32, device=device)
                diff = scores[pi] - scores[pj]
                if use_position_offset:
                    full_b = torch.cat([torch.zeros(1, device=device), slot_bias])
                    sl = torch.tensor(blk["slots"], dtype=torch.long, device=device)
                    diff = diff + (full_b[sl[pi]] - full_b[sl[pj]])
                losses.append(F.softplus(-py * diff).mean())
            loss = torch.stack(losses).mean()
            if use_position_offset:
                loss = loss + LAMBDA_B * torch.sum(slot_bias ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in param_groups for p in g["params"] if p.requires_grad], 1.0
            )
            opt.step()

    # Inference strictly via f(x); training-only position offset b_k is discarded.
    base.eval()
    head.eval()
    preds: dict[str, float] = {}
    all_keys = df_all["problem_key"].astype(str).drop_duplicates().tolist()
    for s in range(0, len(all_keys), 16):
        batch = all_keys[s : s + 16]
        vals = encode_scores(batch, grad=False).detach().cpu().tolist()
        for k, v in zip(batch, vals):
            preds[k] = float(v)
    return preds


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Reference Qwen3-Embedding-0.6B 2x2 factorial trainer.")
    ap.add_argument("--statements-csv", type=Path, required=True, help="CSV with statement_v2_readable & metadata.")
    ap.add_argument("--output-dir", type=Path, required=True, help="Directory to save predicted item scores.")
    ap.add_argument("--model-id", type=str, default=MODEL_ID)
    ap.add_argument("--revision", type=str, default=MODEL_REV)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 202])
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.statements_csv)
    req_cols = {"problem_key", "cohort", "contest_id", "start_time", "series", "year", "problem_index", "rating", "statement_v2_readable"}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {args.statements_csv}: {sorted(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"problem_key": df["problem_key"].astype(str).drop_duplicates().tolist()})
    specs = [("qwen_F0", "frozen", False, args.seeds[0]), ("qwen_F1", "frozen", True, args.seeds[0])]
    for sd in args.seeds:
        specs.append((f"qwen_A0_s{sd}", "lora", False, sd))
        specs.append((f"qwen_A1_s{sd}", "lora", True, sd))
    for name, mode, use_off, sd in specs:
        preds = train_cell(df, mode, use_off, sd, args.model_id, args.revision)
        out[name] = out["problem_key"].map(preds)
    out.to_csv(args.output_dir / "qwen3_item_scores_retrained.csv", index=False)
    print(f"Saved retrained scores to {args.output_dir / 'qwen3_item_scores_retrained.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
