# -*- coding: utf-8 -*-
"""P3 协议模块：题面清洗、统一表层结构特征、跨场配对、计分与块重采样。

协议版本：p3-proto-v1（供 P3 开发段定稿及 P4/P5/P6 只读冻结复用）。
严格约束：
  1. 冷启动特征仅取题面正文、显式约束与输入格式（屏蔽题头、比赛号、题位、分值、标签、题解、赛后统计）。
  2. 所有词表（TF-IDF）、缺失值中位数与连续变量标准化统计量只在 Codeforces 训练段（<=2021）拟合。
  3. 评价均在「同平台、同系列、同一年同季度、相邻不重叠两场」构造的跨场题对及比赛对块上进行。
"""

from __future__ import annotations

import html as _html
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

# Standalone compatibility shim replacing package-relative fetcher import
class _F_Compat:
    _NUM = r"(?:\d+(?:\.\d+)?(?:\s*[eE]\s*[+-]?\d+)?)"
    _POW10 = r"10\s*\^\s*\{?\s*(\d+)\s*\}?"
    _MULT = r"(?:(\d+(?:\.\d+)?)\s*(?:\\cdot|\\times|\*|·)?\s*)?"
    _MAX_EXPONENT = 64
    _MAX_ABS_VALUE = 1e30

    @staticmethod
    def _to_number(token: str) -> float | None:
        t = token.strip()
        if not t:
            return None
        m = re.match(rf"^{_F_Compat._MULT}{_F_Compat._POW10}$", t)
        if m:
            exp = int(m.group(2))
            if exp > _F_Compat._MAX_EXPONENT:
                return None
            mult = float(m.group(1)) if m.group(1) else 1.0
            return mult * (10.0 ** exp)
        m = re.match(rf"^{_F_Compat._POW10}$", t)
        if m:
            exp = int(m.group(1))
            if exp > _F_Compat._MAX_EXPONENT:
                return None
            return 10.0 ** exp
        t2 = t.replace(" ", "")
        m = re.match(rf"^{_F_Compat._NUM}$", t2)
        if m:
            try:
                v = float(m.group(0))
            except (ValueError, OverflowError):
                return None
            if not math.isfinite(v) or abs(v) > _F_Compat._MAX_ABS_VALUE:
                return None
            return v
        return None


F = _F_Compat()

# ================================================================ P3 冻结常量

PROTOCOL_VERSION = "p3-proto-v1"
PROTOCOL_VERSION_V2 = "p3-proto-v2"
P2_AUDIT_VERSION = "p2-audit-v1"

CF_TARGET_SERIES: tuple[str, ...] = ("Regular Div.2", "Educational Div.2")
ATCODER_PRIMARY_SERIES = "ABC"
ATCODER_SUPPLEMENT_SERIES: tuple[str, ...] = (
    "ARC Standard",
    "ARC Div.1",
    "ARC Div.2",
    "ARC++",
    "ARC--",
)
ATCODER_ALL_SERIES: tuple[str, ...] = (ATCODER_PRIMARY_SERIES,) + ATCODER_SUPPLEMENT_SERIES

CF_GAP_EDGES: tuple[float, ...] = (0.0, 200.0, 400.0, 800.0, float("inf"))
CF_GAP_LABELS: tuple[str, ...] = ("(0, 200]", "(200, 400]", "(400, 800]", "(800, +inf)")

# 依据 P2 审计后 AtCoder train+valid (<=2022) 同题位分差分布（ABC/ARC 约四分位数）锁定
ATCODER_GAP_EDGES: tuple[float, ...] = (0.0, 150.0, 300.0, 600.0, float("inf"))
ATCODER_GAP_LABELS: tuple[str, ...] = ("(0, 150]", "(150, 300]", "(300, 600]", "(600, +inf)")

BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42
TIE_TOL = 1e-12
DELTA_PRACTICAL_PP = 5.0  # 每 20 个有效有方向题对多排对 1 个（5 个百分点）

RIDGE_ALPHAS_CANDIDATES: tuple[float, ...] = (1.0, 10.0, 100.0)
DEFAULT_RIDGE_ALPHA = 10.0
TFIDF_PARAMS: dict[str, Any] = {
    "ngram_range": (1, 2),
    "min_df": 3,
    "max_features": 20000,
    "sublinear_tf": True,
    "strip_accents": "unicode",
    "token_pattern": r"(?u)\b\w+\b",
}

SURFACE_FEATURE_NAMES: tuple[str, ...] = (
    "log_word_count",
    "digit_density",
    "math_op_density",
    "log_array_len_upper",
    "array_len_missing",
    "log_value_range_upper",
    "value_range_missing",
    "has_multi_test",
    "log_multi_test_count",
    "has_mod",
    "is_interactive",
)

# ================================================================ 1. 统一题面清洗

LATEX_MAP_P3: dict[str, str] = {
    r"\leftrightarrow": "<->",
    r"\Leftrightarrow": "<=>",
    r"\leftarrow": "<-",
    r"\Leftarrow": "<=",
    r"\rightarrow": "->",
    r"\Rightarrow": "=>",
    r"\to": "->",
    r"\leqslant": "<=",
    r"\leqq": "<=",
    r"\leq": "<=",
    r"\le": "<=",
    r"\geqslant": ">=",
    r"\geqq": ">=",
    r"\geq": ">=",
    r"\ge": ">=",
    r"\neq": "!=",
    r"\ne": "!=",
    r"\times": "*",
    r"\cdot": "*",
    r"\div": "/",
    r"\lt": "<",
    r"\gt": ">",
    r"\pm": "+-",
    r"\sum": "sum",
    r"\prod": "prod",
    r"\infty": "infinity",
    r"\cdots": "...",
    r"\ldots": "...",
    r"\vdots": "...",
    r"\ddots": "...",
    r"\dots": "...",
    r"\pmod": "mod",
    r"\bmod": "mod",
    r"\mod": "mod",
    r"\lfloor": "floor(",
    r"\rfloor": ")",
    r"\lceil": "ceil(",
    r"\rceil": ")",
    r"\sqrt": "sqrt",
    r"\log": "log",
    r"\ln": "ln",
    r"\left": "",
    r"\right": "",
    r"\displaystyle": "",
    r"\alpha": "alpha",
    r"\beta": "beta",
    r"\gamma": "gamma",
    r"\theta": "theta",
    r"\pi": "pi",
    r"\epsilon": "epsilon",
    r"\phi": "phi",
    r"\lambda": "lambda",
    r"\mu": "mu",
    r"\sigma": "sigma",
    r"\Delta": "Delta",
    r"\Omega": "Omega",
    r"\{": "{",
    r"\}": "}",
    r"\,": " ",
    r"\;": " ",
    r"\!": " ",
    r"\:": " ",
    r"\ ": " ",
}
_LATEX_KEYS_SORTED = sorted(LATEX_MAP_P3.keys(), key=len, reverse=True)
_RE_LATEX_P3 = re.compile(
    "|".join(
        re.escape(k) + (r"(?![a-zA-Z])" if k[1:].isalpha() else "")
        for k in _LATEX_KEYS_SORTED
    )
)
_RE_LATEX_CMD = re.compile(r"\\[a-zA-Z]+\*?")
_RE_BRACES = re.compile(r"[{}]")
_RE_DOLLAR = re.compile(r"\$+")
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_WS = re.compile(r"\s+")
_RE_MATH_OPS = re.compile(r"<=|>=|!=|->|=>|<->|[+\-*/=<>^]")
_RE_SAMPLE_FB = re.compile(r"(?is)(example\s*\d*\s*)?(input|output)\s*(copy)?\s*:.*?(?=(example\s*\d|note\s*:|$))")
_RE_NOTE_FB = re.compile(r"(?is)\bnote\s*:.*$")


_RE_THOUSANDS = re.compile(r"(?<=\d)(?:\\,|\\;|\\:|\\ |,)(?=\d{3}(?:\D|$))")


def clean_statement_p3(raw: str, drop_samples: bool = False) -> str:
    """统一冷启动题面清洗（v1 历史归档版本）：保留供 v1 结果回放使用。"""
    if not isinstance(raw, str) or not raw:
        return ""
    s = _html.unescape(raw)
    s = _RE_HTML_TAG.sub(" ", s)
    s = _RE_THOUSANDS.sub("", s)
    for old, new in (("≤", "<="), ("≦", "<="), ("≥", ">="), ("≧", ">="), ("×", "*"), ("·", "*"), ("−", "-"), ("–", "-"), ("—", "-")):
        s = s.replace(old, new)
    s = _RE_LATEX_P3.sub(lambda m: " " + LATEX_MAP_P3[m.group(0)] + " ", s)
    s = _RE_LATEX_CMD.sub(" ", s)
    s = _RE_BRACES.sub(" ", s)
    s = _RE_DOLLAR.sub(" ", s)
    s = s.replace("\\", " ")
    if drop_samples:
        s = _RE_SAMPLE_FB.sub(" ", s)
        s = _RE_NOTE_FB.sub(" ", s)
    return _RE_WS.sub(" ", s).strip().lower()


def clean_statement_p3_v2(
    raw: str,
    drop_samples: bool = False,
    preserve_case: bool = False,
) -> str:
    """统一冷启动题面清洗（v2 修复版本，p3-proto-v2）。

    前置约定：HTML 标签剥离已在原始 HTML/DOM 层（如 `extract_cf_statement_parts`
    或 BeautifulSoup DOM 提取）完成，并早于 HTML 实体解码执行。本函数面向已提取的
    纯文本正文与约束，不再运行 `<[^>]+>` 正则，从而完整保留字面 `<`、`>`、
    成对尖括号（如 `<<>>`、`<lvalue>`）及跨句数学不等式（如 `0 <= k < n and x_k > k`）。
    """
    if not isinstance(raw, str) or not raw:
        return ""
    s = _html.unescape(raw)
    s = _RE_THOUSANDS.sub("", s)
    for old, new in (
        ("≤", "<="), ("≦", "<="), ("≥", ">="), ("≧", ">="),
        ("×", "*"), ("·", "*"), ("−", "-"), ("–", "-"), ("—", "-"),
    ):
        s = s.replace(old, new)
    s = _RE_LATEX_P3.sub(lambda m: " " + LATEX_MAP_P3[m.group(0)] + " ", s)
    s = _RE_LATEX_CMD.sub(" ", s)
    if not preserve_case:
        s = _RE_BRACES.sub(" ", s)
    s = _RE_DOLLAR.sub(" ", s)
    s = s.replace("\\", " ")
    if drop_samples:
        s = _RE_SAMPLE_FB.sub(" ", s)
        s = _RE_NOTE_FB.sub(" ", s)
    s = _RE_WS.sub(" ", s).strip()
    return s if preserve_case else s.lower()


def norm_canon(text: object) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return " ".join(str(text).lower().split())


# ================================================================ 2. 统一表层结构与约束解析

_NUM = r"(?:\d+(?:\.\d+)?(?:\s*[eE]\s*[+-]?\d+)?)"
_POW10 = r"10\s*\^\s*\{?\s*(\d+)\s*\}?"
_MULT = r"(?:(\d+(?:\.\d+)?)\s*(?:\\cdot|\\times|\*|·)?\s*)?"
_BOUND = rf"(?:{_MULT}{_POW10}|{_NUM})"
_VAR_ABS_OR_PLAIN = r"(?:\|\s*[a-zA-Z](?:\s*_\s*\{?\s*\w+\s*\}?)?\s*\|)|(?:[a-zA-Z](?:\s*_\s*\{?\s*\w+\s*\}?)?)"
_VARLIST = rf"{_VAR_ABS_OR_PLAIN}(?:\s*,\s*{_VAR_ABS_OR_PLAIN})*"
_SUP = r"(?:\\le|\\leq|<=|≤|\\lt|<)"
_RE_CHAIN = re.compile(
    rf"(?P<lo>{_BOUND})\s*{_SUP}\s*(?P<vars>{_VARLIST})(?:\s*{_SUP}\s*(?P<vars2>{_VARLIST}))?\s*{_SUP}\s*(?P<hi>{_BOUND})"
)
_RE_UPPER_ONLY = re.compile(rf"(?P<vars>{_VARLIST})\s*{_SUP}\s*(?P<hi>{_BOUND})")

_PARSE_FIXUPS: tuple[tuple[str, str], ...] = (
    (r"\leqslant", r"\le"),
    (r"\leqq", r"\le"),
    (r"\geqslant", r"\ge"),
    (r"\geqq", r"\ge"),
    (r"\left", ""),
    (r"\right", ""),
    (r"\displaystyle", ""),
    (r"\,", " "),
    (r"\;", " "),
    (r"\!", " "),
    (r"\:", " "),
    (r"\ ", " "),
    (r"\quad", " "),
    (r"\qquad", " "),
    ("≤", r"\le"),
    ("≦", r"\le"),
    ("≥", r"\ge"),
    ("≧", r"\ge"),
    ("×", "*"),
    ("·", "*"),
    ("−", "-"),
    ("–", "-"),
    ("—", "-"),
)
_LEN_SCALAR_VARS = {"n", "m", "q", "k", "h", "w", "len", "length"}
_MOD_RE = re.compile(
    r"(?:\bmod(?:ulo)?\b|remainder\s+when\s+divided\s+by|998\s*,?\s*244\s*,?\s*353|"
    r"10\s*\^\s*\{?\s*9\s*\}?\s*\+\s*[79]|100000000[79]|10\s*\^\s*9\s*\+\s*[79])",
    re.I,
)
_TC_PHRASE_RE = re.compile(r"\b(?:test\s*cases?|testcases?|multiple\s+tests|number\s+of\s+tests)\b", re.I)


def _normalize_for_parse(raw: str) -> str:
    if not raw:
        return ""
    s = _html.unescape(str(raw)).replace("\\\\", "\\")
    s = _RE_THOUSANDS.sub("", s)
    for old, new in _PARSE_FIXUPS:
        s = s.replace(old, new)
    s = re.sub(r"\\[a-zA-Z]+", lambda m: m.group(0).lower(), s)
    return s.lower()


def _classify_var(v_raw: str) -> str:
    v = v_raw.strip().lower()
    if "|" in v:
        return "len"
    v_clean = re.sub(r"\s+", "", re.sub(r"_\s*\{?\s*(\w+)\s*\}?", r"_\1", v))
    if v_clean == "t":
        return "t"
    if "_" in v_clean:
        base, sub = v_clean.split("_", 1)
        if base in ("n", "m", "k", "q", "h", "w") and sub in ("1", "2", "3"):
            return "len"
        return "val"
    if v_clean in _LEN_SCALAR_VARS:
        return "len"
    return "val"


def extract_surface_features(
    raw_text: str,
    cleaned_text: str,
    is_interactive_hint: int = 0,
) -> dict[str, float]:
    """统一抽取 CF 与 AtCoder 共享的 11 维冷启动表层结构特征。"""
    s_parse = _normalize_for_parse(raw_text)
    len_cands: list[float] = []
    val_cands: list[float] = []
    t_cands: list[float] = []

    for seg in re.split(r"[\n\r]+|(?<=\.)\s+", s_parse):
        seg_s = seg.strip()
        if not seg_s:
            continue
        is_sum_clause = bool(re.search(r"\b(?:sum|total)\s+of\b", seg_s))
        hits_seg = list(_RE_CHAIN.finditer(seg_s)) or list(_RE_UPPER_ONLY.finditer(seg_s))
        for m in hits_seg:
            hi = F._to_number(m.group("hi"))
            if hi is None:
                continue
            raw_vars = list(m.group("vars").split(","))
            if "vars2" in m.groupdict() and m.group("vars2"):
                raw_vars.extend(m.group("vars2").split(","))
            for vr in raw_vars:
                cls = _classify_var(vr)
                if cls == "t":
                    t_cands.append(hi)
                elif cls == "len":
                    if not is_sum_clause or not len_cands:
                        len_cands.append(hi)
                else:
                    val_cands.append(hi)

        if not hits_seg and not is_sum_clause:
            mb = re.search(rf"\bbetween\s+{_BOUND}\s+and\s+(?P<hi>{_BOUND})", seg_s)
            ma = re.search(rf"\b(?:at\s+most|doesn'?t\s+exceed|does\s+not\s+exceed|exceeding)\s+(?P<hi>{_BOUND})", seg_s)
            mx = mb or ma
            if mx:
                hi = F._to_number(mx.group("hi"))
                if hi is not None:
                    if re.search(r"\b(?:length|size|number\s+of)\b", seg_s):
                        len_cands.append(hi)
                    elif _TC_PHRASE_RE.search(seg_s):
                        t_cands.append(hi)
                    else:
                        head_vars = re.findall(r"\b([a-z](?:_\w+)?)\b", re.split(r"\b(?:is|are)\b", seg_s)[0])
                        if any(_classify_var(hv) == "len" for hv in head_vars):
                            len_cands.append(hi)
                        elif head_vars:
                            val_cands.append(hi)

    has_tc = bool(_TC_PHRASE_RE.search(s_parse)) and bool(t_cands)
    t_max = float(max(t_cands)) if has_tc else 1.0
    len_up = float(max(len_cands)) if len_cands else np.nan
    val_up = float(max(val_cands)) if val_cands else np.nan

    n_words = len(cleaned_text.split()) if cleaned_text else 0
    n_chars = len(cleaned_text)
    n_digits = len(re.findall(r"\d", cleaned_text)) if cleaned_text else 0
    n_ops = len(_RE_MATH_OPS.findall(cleaned_text)) if cleaned_text else 0
    has_mod = 1.0 if _MOD_RE.search(raw_text or "") else 0.0
    is_inter = 1.0 if (is_interactive_hint or re.search(r"\binteractive\b", raw_text or "", re.I)) else 0.0

    return {
        "word_count": float(n_words),
        "log_word_count": float(np.log1p(n_words)),
        "digit_density": float(n_digits / max(n_chars, 1)),
        "math_op_density": float(n_ops / max(n_chars, 1)),
        "array_len_upper": len_up,
        "log_array_len_upper": float(np.log1p(len_up)) if np.isfinite(len_up) else np.nan,
        "array_len_missing": 0.0 if np.isfinite(len_up) else 1.0,
        "value_range_upper": val_up,
        "log_value_range_upper": float(np.log1p(val_up)) if np.isfinite(val_up) else np.nan,
        "value_range_missing": 0.0 if np.isfinite(val_up) else 1.0,
        "has_multi_test": 1.0 if has_tc else 0.0,
        "multi_test_count": t_max,
        "log_multi_test_count": float(np.log1p(t_max)),
        "has_mod": has_mod,
        "is_interactive": is_inter,
    }


# ================================================================ 3. 训练集拟合的冷启动管线

@dataclass
class FrozenFeaturePipeline:
    """严格只在 CF 训练段 fit，然后对验证/测试/外部平台只读 transform。"""

    imputer: SimpleImputer
    scaler: StandardScaler
    tfidf: TfidfVectorizer
    train_struct_medians: dict[str, float]
    vocab_size: int

    @classmethod
    def fit(cls, train_df: pd.DataFrame) -> "FrozenFeaturePipeline":
        X_raw = train_df[list(SURFACE_FEATURE_NAMES)].to_numpy(dtype=np.float64)
        imputer = SimpleImputer(strategy="median")
        X_imp = imputer.fit_transform(X_raw)
        scaler = StandardScaler()
        scaler.fit(X_imp)

        tfidf = TfidfVectorizer(**TFIDF_PARAMS)
        tfidf.fit(train_df["statement_p3"].fillna("").tolist())

        medians = {
            col: float(imputer.statistics_[i])
            for i, col in enumerate(SURFACE_FEATURE_NAMES)
        }
        return cls(
            imputer=imputer,
            scaler=scaler,
            tfidf=tfidf,
            train_struct_medians=medians,
            vocab_size=len(tfidf.vocabulary_),
        )

    def transform_structure(self, df: pd.DataFrame) -> np.ndarray:
        X_raw = df[list(SURFACE_FEATURE_NAMES)].to_numpy(dtype=np.float64)
        return self.scaler.transform(self.imputer.transform(X_raw))

    def transform_text(self, df: pd.DataFrame) -> sparse.csr_matrix:
        return self.tfidf.transform(df["statement_p3"].fillna("").tolist())

    def transform_joint(self, df: pd.DataFrame) -> sparse.csr_matrix:
        xs = sparse.csr_matrix(self.transform_structure(df))
        xt = self.transform_text(df)
        return sparse.hstack([xs, xt], format="csr")


def fit_trainable_models(
    train_df: pd.DataFrame,
    pipe: FrozenFeaturePipeline,
    alpha: float = DEFAULT_RIDGE_ALPHA,
    n_threads: int = 2,
) -> dict[str, Ridge]:
    """在固定训练段上拟合表层结构、全文与联合三套 Ridge 模型。"""
    y = train_df["rating"].to_numpy(dtype=np.float64)
    Xs = pipe.transform_structure(train_df)
    Xt = pipe.transform_text(train_df)
    Xj = sparse.hstack([sparse.csr_matrix(Xs), Xt], format="csr")

    with threadpool_limits(limits=n_threads):
        m_struct = Ridge(alpha=alpha, solver="auto").fit(Xs, y)
        m_text = Ridge(alpha=alpha, solver="lsqr").fit(Xt, y)
        m_joint = Ridge(alpha=alpha, solver="lsqr").fit(Xj, y)
    return {"structure": m_struct, "text": m_text, "joint": m_joint}


def predict_all_methods(
    df: pd.DataFrame,
    pipe: FrozenFeaturePipeline,
    models: dict[str, Ridge],
    n_threads: int = 2,
) -> pd.DataFrame:
    """一次性生成 5 种协议方法的逐题分数（slot / length / structure / text / joint）。"""
    out = pd.DataFrame({"problem_key": df["problem_key"].to_numpy()})
    out["score_slot"] = np.array([float(ord(str(c)[0].upper()) - ord("A")) for c in df["problem_index"]], dtype=np.float64)
    out["score_length"] = df["word_count"].to_numpy(dtype=np.float64)

    Xs = pipe.transform_structure(df)
    Xt = pipe.transform_text(df)
    Xj = sparse.hstack([sparse.csr_matrix(Xs), Xt], format="csr")
    with threadpool_limits(limits=n_threads):
        out["score_structure"] = models["structure"].predict(Xs)
        out["score_text"] = models["text"].predict(Xt)
        out["score_joint"] = models["joint"].predict(Xj)
    return out


# ================================================================ 4. 配对构建（先赛程，后关联标签）

def build_contest_pairs(
    eval_df: pd.DataFrame,
    train_canon_shas: set[str] | None = None,
    allowed_series: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """按「同平台×同系列×同一年同一季度、相邻不重叠两场」生成跨场题对与块表。

    步骤严格遵从协议：先剔除训练段同文题、单字母筛选并生成日程配对，最后关联标签与真值分差。
    """
    work = eval_df.copy()
    audit: dict[str, Any] = {
        "n_input_rows": int(len(work)),
        "n_dropped_series": 0,
        "n_dropped_non_single_letter": 0,
        "n_dropped_train_dup": 0,
        "n_dropped_within_split_dup": 0,
        "odd_contests": [],
    }
    if allowed_series is not None:
        m_ser = work["series"].isin(list(allowed_series))
        audit["n_dropped_series"] = int((~m_ser).sum())
        work = work[m_ser].copy()

    m_single = work["problem_index"].astype(str).str.fullmatch(r"[A-Z]")
    audit["n_dropped_non_single_letter"] = int((~m_single).sum())
    work = work[m_single].copy()

    if "canon_sha256" not in work.columns:
        import hashlib
        work["canon_sha256"] = work["statement"].map(lambda s: hashlib.sha256(norm_canon(s).encode("utf-8")).hexdigest())

    if train_canon_shas:
        m_tr = work["canon_sha256"].isin(train_canon_shas)
        audit["n_dropped_train_dup"] = int(m_tr.sum())
        work = work[~m_tr].copy()

    work = work.sort_values(["start_time", "contest_id", "problem_key"]).reset_index(drop=True)
    m_wd = work.duplicated("canon_sha256", keep="first")
    audit["n_dropped_within_split_dup"] = int(m_wd.sum())
    work = work[~m_wd].copy()

    if "quarter" not in work.columns:
        work["quarter"] = pd.to_datetime(work["start_time"], unit="s", utc=True).dt.quarter

    # 先按日程生成相邻场次配对（不含标签计算）
    blocks_meta: list[dict[str, Any]] = []
    pair_skeletons: list[dict[str, Any]] = []
    block_id = 0

    for (plat, ser, yr, q), grp in work.groupby(["platform", "series", "year", "quarter"], sort=True):
        contests = (
            grp[["contest_id", "start_time"]]
            .drop_duplicates()
            .sort_values(["start_time", "contest_id"])["contest_id"]
            .tolist()
        )
        if len(contests) % 2 == 1:
            audit["odd_contests"].append({
                "platform": plat, "series": ser, "year": int(yr), "quarter": int(q),
                "contest_id": contests[-1],
                "n_problems": int((grp["contest_id"] == contests[-1]).sum()),
            })
        for c1, c2 in zip(contests[::2], contests[1::2]):
            block_id += 1
            bkey = f"{plat}|{ser}|{int(yr)}Q{int(q)}|{c1}_{c2}"
            g1 = grp[grp["contest_id"] == c1].sort_values("problem_index")
            g2 = grp[grp["contest_id"] == c2].sort_values("problem_index")
            blocks_meta.append({
                "block_id": block_id,
                "block_key": bkey,
                "platform": plat,
                "series": ser,
                "year": int(yr),
                "quarter": int(q),
                "contest_a": c1,
                "contest_b": c2,
                "n_problems_a": int(len(g1)),
                "n_problems_b": int(len(g2)),
            })
            for a in g1.itertuples(index=False):
                for b in g2.itertuples(index=False):
                    pair_skeletons.append({
                        "block_id": block_id,
                        "block_key": bkey,
                        "platform": plat,
                        "series": ser,
                        "year": int(yr),
                        "quarter": int(q),
                        "contest_a": c1,
                        "contest_b": c2,
                        "key_a": a.problem_key,
                        "key_b": b.problem_key,
                        "slot_a": a.problem_index,
                        "slot_b": b.problem_index,
                        "same_slot": a.problem_index == b.problem_index,
                    })

    pairs_df = pd.DataFrame(pair_skeletons)
    blocks_df = pd.DataFrame(blocks_meta)
    if pairs_df.empty:
        return pairs_df, blocks_df, audit

    # 生成候选清单之后再关联两题标签
    label_map = work.set_index("problem_key")["rating"].to_dict()
    pairs_df["rating_a"] = pairs_df["key_a"].map(label_map).astype(float)
    pairs_df["rating_b"] = pairs_df["key_b"].map(label_map).astype(float)
    pairs_df["truth_diff"] = pairs_df["rating_a"] - pairs_df["rating_b"]
    pairs_df["truth_sign"] = np.sign(pairs_df["truth_diff"])
    pairs_df["label_gap"] = np.abs(pairs_df["truth_diff"])
    pairs_df["is_valid"] = pairs_df["label_gap"] > 0.0

    audit["n_unique_problems_paired"] = int(pd.unique(pd.concat([pairs_df["key_a"], pairs_df["key_b"]])).size)
    audit["n_blocks"] = int(len(blocks_df))
    audit["n_odd_contests_dropped"] = len(audit["odd_contests"])
    audit["n_candidate_pairs"] = int(len(pairs_df))
    audit["n_valid_pairs"] = int(pairs_df["is_valid"].sum())
    audit["n_tied_pairs"] = int((~pairs_df["is_valid"]).sum())
    return pairs_df, blocks_df, audit


# ================================================================ 5. 计分、块重采样与分差标准化

def score_pairs_with_predictions(
    pairs_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    method_cols: Sequence[str] = ("score_slot", "score_length", "score_structure", "score_text", "score_joint"),
    tie_tol: float = TIE_TOL,
) -> pd.DataFrame:
    """把各方法逐题分数映射到候选题对并计算对错（1 / 0.5 / 0）与预测平分标记。"""
    scored = pairs_df.copy()
    score_lookup = scores_df.set_index("problem_key")
    truth = scored["truth_sign"].to_numpy(dtype=np.float64)
    for mcol in method_cols:
        short = mcol.replace("score_", "")
        va = scored["key_a"].map(score_lookup[mcol]).to_numpy(dtype=np.float64)
        vb = scored["key_b"].map(score_lookup[mcol]).to_numpy(dtype=np.float64)
        diff = va - vb
        pred_sign = np.where(np.abs(diff) <= tie_tol, 0.0, np.sign(diff))
        # 对有效题对（truth != 0）计算 (1 + pred_sign * truth)/2；并列题对保持 NaN
        c = np.where(truth != 0.0, 0.5 * (1.0 + pred_sign * truth), np.nan)
        scored[f"c_{short}"] = c
        scored[f"tie_pred_{short}"] = np.where(truth != 0.0, (pred_sign == 0.0).astype(float), np.nan)
    return scored


def summarize_subset_with_bootstrap(
    scored_subset: pd.DataFrame,
    all_block_ids: Sequence[int],
    methods: Sequence[str] = ("slot", "length", "structure", "text", "joint"),
    delta_pair: tuple[str, str] = ("joint", "structure"),
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """按固定比赛对块列表进行整组有放回重采样（2000 次，种子 42），报告百分位 95% CI 与块等权敏感性。"""
    n_cand = int(len(scored_subset))
    valid = scored_subset[scored_subset["is_valid"]].copy()
    n_valid = int(len(valid))
    n_tied = n_cand - n_valid

    res: dict[str, Any] = {
        "n_candidate_pairs": n_cand,
        "n_valid_pairs": n_valid,
        "n_tied_pairs": n_tied,
        "valid_pair_ratio": float(n_valid / n_cand) if n_cand else 0.0,
        "n_total_blocks": int(len(all_block_ids)),
    }
    if n_valid == 0:
        res["n_nonempty_blocks"] = 0
        res["n_zero_valid_blocks"] = int(len(all_block_ids))
        res["n_inestimable_draws"] = n_boot
        return res

    blk_index = pd.Index(sorted(set(all_block_ids)), name="block_id")
    c_cols = [f"c_{m}" for m in methods]
    tie_cols = [f"tie_pred_{m}" for m in methods]

    blk_sums = valid.groupby("block_id")[c_cols].sum().reindex(blk_index, fill_value=0.0)
    blk_counts = valid.groupby("block_id").size().reindex(blk_index, fill_value=0).to_numpy(dtype=np.float64)
    n_zero_blks = int((blk_counts == 0).sum())
    n_nonempty = int((blk_counts > 0).sum())
    res["n_nonempty_blocks"] = n_nonempty
    res["n_zero_valid_blocks"] = n_zero_blks

    # 点估计（题对加权 / 100 pp）及每个有效块等权均值
    sum_mat = blk_sums.to_numpy(dtype=np.float64)  # shape (B, M)
    tot_n = float(blk_counts.sum())
    nonempty_mask = blk_counts > 0

    for j, m in enumerate(methods):
        acc_pp = 100.0 * float(sum_mat[:, j].sum() / tot_n)
        eq_blk_pp = 100.0 * float(np.mean(sum_mat[nonempty_mask, j] / blk_counts[nonempty_mask]))
        res[f"acc_{m}_pp"] = acc_pp
        res[f"acc_{m}_block_equal_pp"] = eq_blk_pp
        res[f"pred_tie_rate_{m}"] = float(valid[tie_cols[j]].mean())

    m_hi, m_lo = delta_pair
    i_hi, i_lo = methods.index(m_hi), methods.index(m_lo)
    delta_pt = res[f"acc_{m_hi}_pp"] - res[f"acc_{m_lo}_pp"]
    delta_eq = res[f"acc_{m_hi}_block_equal_pp"] - res[f"acc_{m_lo}_block_equal_pp"]
    res["delta_pair_name"] = f"{m_hi}_minus_{m_lo}"
    res["delta_pp"] = float(delta_pt)
    res["delta_block_equal_pp"] = float(delta_eq)

    # 块重采样（按块矩阵索引一次抽完，保证所有方法在同一批块上做配对差值）
    rng = np.random.default_rng(seed)
    n_blks = len(blk_index)
    draw_idx = rng.integers(0, n_blks, size=(n_boot, n_blks))
    boot_n = blk_counts[draw_idx].sum(axis=1)  # (B_boot,)
    est_mask = boot_n > 0
    n_inest = int((~est_mask).sum())
    res["n_inestimable_draws"] = n_inest

    boot_sums = sum_mat[draw_idx].sum(axis=1)  # (B_boot, M)
    for j, m in enumerate(methods):
        vals = 100.0 * boot_sums[est_mask, j] / boot_n[est_mask]
        q025, q975 = np.quantile(vals, [0.025, 0.975])
        res[f"ci95_{m}_lo"] = float(q025)
        res[f"ci95_{m}_hi"] = float(q975)

    delta_draws = 100.0 * (boot_sums[est_mask, i_hi] - boot_sums[est_mask, i_lo]) / boot_n[est_mask]
    dq025, dq975 = np.quantile(delta_draws, [0.025, 0.975])
    res["delta_ci95_lo"] = float(dq025)
    res["delta_ci95_hi"] = float(dq975)

    # 同时给出 text - length 参考对比（用于与先导衔接）
    if "text" in methods and "length" in methods:
        it, il = methods.index("text"), methods.index("length")
        dtl = 100.0 * (boot_sums[est_mask, it] - boot_sums[est_mask, il]) / boot_n[est_mask]
        res["delta_text_minus_length_pp"] = float(res["acc_text_pp"] - res["acc_length_pp"])
        res["delta_text_minus_length_ci95_lo"] = float(np.quantile(dtl, 0.025))
        res["delta_text_minus_length_ci95_hi"] = float(np.quantile(dtl, 0.975))

    return res


def compute_gap_standardized_comparison(
    scored_df: pd.DataFrame,
    all_block_ids: Sequence[int],
    gap_edges: Sequence[float],
    gap_labels: Sequence[str],
    method_col: str = "c_joint",
    compare_col: str | None = "c_structure",
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """按同题位组真值分差分布对不同题位组分档正确率做标准化，支持空档重归一化与块重采样稳定度检测。"""
    valid = scored_df[scored_df["is_valid"]].copy()
    valid["band"] = pd.cut(valid["label_gap"], bins=list(gap_edges), labels=list(gap_labels), right=True)

    same_v = valid[valid["same_slot"]]
    diff_v = valid[~valid["same_slot"]]

    def _band_table(sub: pd.DataFrame) -> pd.DataFrame:
        g = sub.groupby("band", observed=False)[method_col].agg(["count", "sum", "mean"]).reindex(gap_labels)
        g["count"] = g["count"].fillna(0).astype(int)
        g["sum"] = g["sum"].fillna(0.0)
        return g

    tb_same = _band_table(same_v)
    tb_diff = _band_table(diff_v)
    common_bands = [b for b in gap_labels if tb_same.loc[b, "count"] > 0 and tb_diff.loc[b, "count"] > 0]

    per_band_rows = []
    for b in gap_labels:
        sc, ss, sm = int(tb_same.loc[b, "count"]), float(tb_same.loc[b, "sum"]), tb_same.loc[b, "mean"]
        dc, ds, dm = int(tb_diff.loc[b, "count"]), float(tb_diff.loc[b, "sum"]), tb_diff.loc[b, "mean"]
        row = {
            "band": b,
            "in_common_support": b in common_bands,
            "same_count": sc,
            "same_acc_pp": 100.0 * float(sm) if sc > 0 else np.nan,
            "diff_count": dc,
            "diff_acc_pp": 100.0 * float(dm) if dc > 0 else np.nan,
        }
        if compare_col is not None:
            row["same_comp_pp"] = 100.0 * float(same_v[same_v["band"] == b][compare_col].mean()) if sc > 0 else np.nan
            row["diff_comp_pp"] = 100.0 * float(diff_v[diff_v["band"] == b][compare_col].mean()) if dc > 0 else np.nan
        per_band_rows.append(row)

    out: dict[str, Any] = {
        "method": method_col.replace("c_", ""),
        "gap_edges": [float(x) if math.isfinite(x) else "inf" for x in gap_edges],
        "common_support_bands": common_bands,
        "all_bands_supported": len(common_bands) == len(gap_labels),
        "common_support_same_coverage": float(tb_same.loc[common_bands, "count"].sum() / max(len(same_v), 1)) if common_bands else 0.0,
        "common_support_diff_coverage": float(tb_diff.loc[common_bands, "count"].sum() / max(len(diff_v), 1)) if common_bands else 0.0,
        "per_band": per_band_rows,
    }

    if not common_bands:
        out["status"] = "inestimable_no_common_support"
        out["raw_diff_minus_same_pp"] = np.nan
        out["common_raw_diff_minus_same_pp"] = np.nan
        out["standardized_diff_acc_pp"] = np.nan
        out["standardized_diff_minus_same_pp"] = np.nan
        return out

    raw_same_all = 100.0 * float(same_v[method_col].mean())
    raw_diff_all = 100.0 * float(diff_v[method_col].mean())

    same_comm = same_v[same_v["band"].isin(common_bands)]
    diff_comm = diff_v[diff_v["band"].isin(common_bands)]
    raw_same_comm = 100.0 * float(same_comm[method_col].mean())
    raw_diff_comm = 100.0 * float(diff_comm[method_col].mean())

    w_comm = tb_same.loc[common_bands, "count"] / tb_same.loc[common_bands, "count"].sum()
    std_diff_comm = 100.0 * float((tb_diff.loc[common_bands, "mean"] * w_comm).sum())

    out["status"] = "ok"
    out["raw_same_all_pp"] = raw_same_all
    out["raw_diff_all_pp"] = raw_diff_all
    out["raw_diff_minus_same_pp"] = raw_diff_all - raw_same_all
    out["common_same_acc_pp"] = raw_same_comm
    out["common_diff_raw_acc_pp"] = raw_diff_comm
    out["common_raw_diff_minus_same_pp"] = raw_diff_comm - raw_same_comm
    out["standardized_diff_acc_pp"] = std_diff_comm
    out["standardized_diff_minus_same_pp"] = std_diff_comm - raw_same_comm
    gap_explains_ratio = (
        (raw_diff_all - std_diff_comm) / (raw_diff_all - raw_same_all)
        if abs(raw_diff_all - raw_same_all) > 1e-12
        else np.nan
    )
    out["gap_constitution_explained_fraction"] = float(gap_explains_ratio) if np.isfinite(gap_explains_ratio) else np.nan

    # 在比赛对块重采样中同步重算同题位权重、共同支持稳定度与标准化差值 95% 区间
    blk_index = pd.Index(sorted(set(all_block_ids)), name="block_id")
    k_bands = len(gap_labels)
    b_cnt = len(blk_index)
    blk_pos = {bid: i for i, bid in enumerate(blk_index)}
    band_pos = {b: j for j, b in enumerate(gap_labels)}

    same_c_arr = np.zeros((b_cnt, k_bands), dtype=np.float64)
    same_n_arr = np.zeros((b_cnt, k_bands), dtype=np.float64)
    diff_c_arr = np.zeros((b_cnt, k_bands), dtype=np.float64)
    diff_n_arr = np.zeros((b_cnt, k_bands), dtype=np.float64)

    for r in valid.itertuples(index=False):
        bi = blk_pos[r.block_id]
        bj = band_pos[str(r.band)]
        val = float(getattr(r, method_col))
        if r.same_slot:
            same_c_arr[bi, bj] += val
            same_n_arr[bi, bj] += 1.0
        else:
            diff_c_arr[bi, bj] += val
            diff_n_arr[bi, bj] += 1.0

    rng = np.random.default_rng(seed)
    draw_idx = rng.integers(0, b_cnt, size=(n_boot, b_cnt))
    sc_b = same_c_arr[draw_idx].sum(axis=1)
    sn_b = same_n_arr[draw_idx].sum(axis=1)
    dc_b = diff_c_arr[draw_idx].sum(axis=1)
    dn_b = diff_n_arr[draw_idx].sum(axis=1)

    std_gaps_draws = []
    std_diff_draws = []
    same_support_mask_cnt = 0
    inest_std = 0
    ref_mask = np.array([b in common_bands for b in gap_labels], dtype=bool)
    for d in range(n_boot):
        act = (sn_b[d] > 0) & (dn_b[d] > 0)
        if not act.any():
            inest_std += 1
            continue
        if np.array_equal(act, ref_mask):
            same_support_mask_cnt += 1
        w = sn_b[d, act] / sn_b[d, act].sum()
        std_diff_d = 100.0 * float(((dc_b[d, act] / dn_b[d, act]) * w).sum())
        std_same_d = 100.0 * float(sc_b[d, act].sum() / sn_b[d, act].sum())
        std_diff_draws.append(std_diff_d)
        std_gaps_draws.append(std_diff_d - std_same_d)

    out["n_inestimable_std_draws"] = inest_std
    out["common_support_exact_stability"] = float(same_support_mask_cnt / max(len(std_gaps_draws), 1))
    if std_gaps_draws:
        out["standardized_diff_ci95_lo"] = float(np.quantile(std_diff_draws, 0.025))
        out["standardized_diff_ci95_hi"] = float(np.quantile(std_diff_draws, 0.975))
        out["standardized_diff_minus_same_ci95_lo"] = float(np.quantile(std_gaps_draws, 0.025))
        out["standardized_diff_minus_same_ci95_hi"] = float(np.quantile(std_gaps_draws, 0.975))
    return out


# ================================================================ 6. 开发期边界手算单元测试

def verify_protocol_unit_tests() -> dict[str, Any]:
    """按 P3 验收要求，以手算样例逐条验证六类边界条件与计分/配对逻辑。"""
    # 边界 1 & 2 & 3：真实评分同分、预测平分、相反排序
    syn_df = pd.DataFrame([
        # 2022 Q1 同系列两场（可配 1 块，Q2 另 1 场独立且不跨季配对）
        {"platform": "cf", "series": "Regular Div.2", "year": 2022, "quarter": 1, "start_time": 100, "contest_id": 10, "problem_key": "10-A", "problem_index": "A", "rating": 800.0, "statement": "alpha array problem one", "canon_sha256": "h1"},
        {"platform": "cf", "series": "Regular Div.2", "year": 2022, "quarter": 1, "start_time": 100, "contest_id": 10, "problem_key": "10-B", "problem_index": "B", "rating": 1100.0, "statement": "beta graph problem two", "canon_sha256": "h2"},
        {"platform": "cf", "series": "Regular Div.2", "year": 2022, "quarter": 1, "start_time": 200, "contest_id": 11, "problem_key": "11-A", "problem_index": "A", "rating": 800.0, "statement": "gamma string problem three", "canon_sha256": "h3"},
        {"platform": "cf", "series": "Regular Div.2", "year": 2022, "quarter": 1, "start_time": 200, "contest_id": 11, "problem_key": "11-B", "problem_index": "B", "rating": 1300.0, "statement": "delta tree problem four", "canon_sha256": "h4"},
        # 边界 4：跨季孤立场（2022 Q2，与 2022 Q1 不得配对）+ 跨年场（2023 Q1）
        {"platform": "cf", "series": "Regular Div.2", "year": 2022, "quarter": 2, "start_time": 300, "contest_id": 12, "problem_key": "12-A", "problem_index": "A", "rating": 900.0, "statement": "q2 lonely contest", "canon_sha256": "h5"},
        {"platform": "cf", "series": "Regular Div.2", "year": 2023, "quarter": 1, "start_time": 400, "contest_id": 13, "problem_key": "13-A", "problem_index": "A", "rating": 950.0, "statement": "next year contest", "canon_sha256": "h6"},
        # 边界 5：与训练集文本规范化重复的题目必须从评估段剔除，以及同场/同评价内部重复
        {"platform": "cf", "series": "Regular Div.2", "year": 2022, "quarter": 1, "start_time": 100, "contest_id": 10, "problem_key": "10-C", "problem_index": "C", "rating": 1500.0, "statement": "leaked train dup", "canon_sha256": "train_dup_sha"},
        # 多字符题位必须剔除
        {"platform": "cf", "series": "Regular Div.2", "year": 2022, "quarter": 1, "start_time": 100, "contest_id": 10, "problem_key": "10-D1", "problem_index": "D1", "rating": 1900.0, "statement": "subtask multi", "canon_sha256": "h7"},
    ])
    pairs, blocks, pa = build_contest_pairs(syn_df, train_canon_shas={"train_dup_sha"})
    assert len(blocks) == 1, f"Expected 1 block, got {len(blocks)}"
    assert len(pairs) == 4, f"Expected 2x2=4 pairs, got {len(pairs)}"
    assert pa["n_dropped_train_dup"] == 1
    assert pa["n_dropped_non_single_letter"] == 1
    assert pa["n_odd_contests_dropped"] == 2  # 2022 Q2 1场, 2023 Q1 1场均未跨季/跨年错配
    assert pa["n_tied_pairs"] == 1 and pa["n_valid_pairs"] == 3  # (10-A=800, 11-A=800) tied

    # 手算验证得分：(10-A=800, 11-B=1300) truth=-1; (10-B=1100, 11-A=800) truth=+1; (10-B=1100, 11-B=1300) truth=-1
    scores = pd.DataFrame([
        {"problem_key": "10-A", "score_m": 10.0, "score_tie": 5.0},
        {"problem_key": "10-B", "score_m": 12.0, "score_tie": 5.0},
        {"problem_key": "11-A", "score_m": 20.0, "score_tie": 5.0},  # 错误排序 (pred 20>12 vs y 800<1100 -> 0)
        {"problem_key": "11-B", "score_m": 15.0, "score_tie": 5.0},  # 10-A vs 11-B (10<15 -> 1); 10-B vs 11-B (12<15 -> 1)
    ])
    scored = score_pairs_with_predictions(pairs, scores, method_cols=("score_m", "score_tie"))
    valid_s = scored[scored["is_valid"]]
    assert valid_s["c_m"].tolist() == [1.0, 0.0, 1.0] or math.isclose(valid_s["c_m"].mean(), 2.0 / 3.0)
    assert np.allclose(valid_s["c_tie"].to_numpy(), 0.5), "预测平分应全部计 0.5"
    assert scored.loc[~scored["is_valid"], "c_m"].isna().all(), "真实同分题对不得计分"

    # 边界 6：部分分档缺失与完全无共同支持分档
    gap_df = pd.DataFrame([
        {"block_id": 1, "same_slot": True, "label_gap": 100.0, "is_valid": True, "c_j": 1.0, "c_s": 0.0},
        {"block_id": 1, "same_slot": False, "label_gap": 100.0, "is_valid": True, "c_j": 0.0, "c_s": 0.0},
        {"block_id": 1, "same_slot": False, "label_gap": 900.0, "is_valid": True, "c_j": 1.0, "c_s": 1.0},  # same 无 >800 档
    ])
    gs_partial = compute_gap_standardized_comparison(
        gap_df, [1], CF_GAP_EDGES, CF_GAP_LABELS, method_col="c_j", compare_col="c_s", n_boot=50, seed=42
    )
    assert gs_partial["status"] == "ok"
    assert gs_partial["common_support_bands"] == ["(0, 200]"]
    assert math.isclose(gs_partial["common_support_same_coverage"], 1.0)
    assert math.isclose(gs_partial["common_support_diff_coverage"], 0.5)
    assert math.isclose(gs_partial["standardized_diff_acc_pp"], 0.0)

    gap_none = pd.DataFrame([
        {"block_id": 1, "same_slot": True, "label_gap": 100.0, "is_valid": True, "c_j": 1.0},
        {"block_id": 1, "same_slot": False, "label_gap": 500.0, "is_valid": True, "c_j": 1.0},
    ])
    gs_empty = compute_gap_standardized_comparison(
        gap_none, [1], CF_GAP_EDGES, CF_GAP_LABELS, method_col="c_j", compare_col=None, n_boot=50, seed=42
    )
    assert gs_empty["status"] == "inestimable_no_common_support"

    # 文本清洗与千分位约束解析核验：\left / \cdots / \{ / 1\,000 / 500\,000 必须准确还原
    cl = clean_statement_p3(r"We have $1 \le n \le 2 \cdot 10^5$, $\left|S\right| \le 10$ and $a_1, \cdots, a_n \in \{0, 1\}$.")
    assert "<=ft" not in cl and "*s" not in cl and "<=" in cl and "..." in cl and "\\" not in cl, cl

    feats_thou = extract_surface_features(
        r"The first line contains $n, k$ ($1 \le n, k \le 1\,000$). Next $m \le 500\,000$.",
        clean_statement_p3(r"The first line contains $n, k$ ($1 \le n, k \le 1\,000$). Next $m \le 500\,000$."),
    )
    assert math.isclose(feats_thou["array_len_upper"], 500000.0), feats_thou

    # 边界：非连续索引预测不发生 NaN 错位，且无有效题对块重采样可统计 n_zero_valid_blocks / n_inestimable_draws
    sub_syn = syn_df.iloc[[0, 1, 2, 3]].copy()
    sub_syn.index = [100, 200, 300, 400]
    sub_syn["statement_p3"] = sub_syn["statement"]
    for c in SURFACE_FEATURE_NAMES:
        sub_syn[c] = 1.0
    sub_syn["word_count"] = [5.0, 10.0, 8.0, 12.0]
    p_noncontig = FrozenFeaturePipeline.fit(sub_syn)
    m_noncontig = fit_trainable_models(sub_syn, p_noncontig, alpha=1.0)
    pred_noncontig = predict_all_methods(sub_syn.iloc[[1, 3]], p_noncontig, m_noncontig)
    assert pred_noncontig["score_slot"].notna().all() and pred_noncontig["score_slot"].tolist() == [1.0, 1.0]

    # 块重采样存在 0 有效题对块时的统计
    zero_blk_df = pd.DataFrame([
        {"block_id": 1, "is_valid": True, "c_slot": 1.0, "c_length": 1.0, "c_structure": 1.0, "c_text": 1.0, "c_joint": 1.0,
         "tie_pred_slot": 0.0, "tie_pred_length": 0.0, "tie_pred_structure": 0.0, "tie_pred_text": 0.0, "tie_pred_joint": 0.0},
        {"block_id": 2, "is_valid": False, "c_slot": np.nan, "c_length": np.nan, "c_structure": np.nan, "c_text": np.nan, "c_joint": np.nan,
         "tie_pred_slot": np.nan, "tie_pred_length": np.nan, "tie_pred_structure": np.nan, "tie_pred_text": np.nan, "tie_pred_joint": np.nan},
    ])
    zb_sum = summarize_subset_with_bootstrap(zero_blk_df, [1, 2], n_boot=200, seed=42)
    assert zb_sum["n_zero_valid_blocks"] == 1 and zb_sum["n_inestimable_draws"] > 0

    return {
        "unit_tests_passed": 8,
        "cleaned_sample_check": cl,
        "thousand_sep_array_len": feats_thou["array_len_upper"],
        "zero_valid_blk_inestimable_draws": zb_sum["n_inestimable_draws"],
        "partial_gap_common_bands": gs_partial["common_support_bands"],
        "empty_gap_status": gs_empty["status"],
    }


# ================================================================ 7. Standalone HTML / Section Extractors


_RE_PROBLEM_STATEMENT_HTML = re.compile(r'<div\s+class="problem-statement"[^>]*>(.*)', re.S | re.I)
_RE_SCRIPT_STYLE_HTML = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.S | re.I)
_RE_TAG_ANY_HTML = re.compile(r"<[^>]+>")
_RE_WS_HTML = re.compile(r"[ \t\r\f\v]+")
_RE_NL_HTML = re.compile(r"\n{3,}")
_RE_SUP_HTML = re.compile(r"(?is)<sup[^>]*>(.*?)</sup>")
_RE_SUB_HTML = re.compile(r"(?is)<sub[^>]*>(.*?)</sub>")


def _strip_balanced_div(html_str: str, class_name: str) -> str:
    """Remove balanced `<div class="...class_name...">...</div>` blocks."""
    out: list[str] = []
    i, n = 0, len(html_str)
    open_pat = re.compile(
        rf'<div\s+class="[^"]*(?<![\w-]){re.escape(class_name)}(?![\w-])[^"]*"[^>]*>',
        re.I,
    )
    div_pat = re.compile(r"<(/?)div\b[^>]*>", re.I)
    while i < n:
        m = open_pat.search(html_str, i)
        if not m:
            out.append(html_str[i:])
            break
        out.append(html_str[i:m.start()])
        depth, j = 1, m.end()
        while depth > 0:
            dm = div_pat.search(html_str, j)
            if not dm:
                j = n
                break
            depth += -1 if dm.group(1) else 1
            j = dm.end()
        i = j
    return "".join(out)


def _inner_of_div(html_str: str, start: int) -> str:
    m = re.compile(r"<div\b[^>]*>", re.I).match(html_str, start)
    if not m:
        return ""
    div_pat = re.compile(r"<(/?)div\b[^>]*>", re.I)
    depth, j = 1, m.end()
    while depth > 0:
        dm = div_pat.search(html_str, j)
        if not dm:
            return html_str[m.end():]
        depth += -1 if dm.group(1) else 1
        j = dm.end()
    return html_str[m.end():dm.start()]


def _extract_div_by_class(html_str: str, class_name: str) -> str:
    open_pat = re.compile(
        rf'<div\s+class="[^"]*(?<![\w-]){re.escape(class_name)}(?![\w-])[^"]*"[^>]*>',
        re.I,
    )
    m = open_pat.search(html_str)
    if not m:
        return ""
    div_pat = re.compile(r"<(/?)div\b[^>]*>", re.I)
    depth, j = 1, m.end()
    while depth > 0:
        dm = div_pat.search(html_str, j)
        if not dm:
            return html_str[m.end():]
        depth += -1 if dm.group(1) else 1
        j = dm.end()
    return html_str[m.end():dm.start()]


def _html_to_text(html_str: str) -> str:
    """Convert statement HTML fragment to plain text BEFORE entity unescaping."""
    s = _RE_SCRIPT_STYLE_HTML.sub(" ", html_str)
    s = _RE_SUP_HTML.sub(lambda m: "^{" + _RE_TAG_ANY_HTML.sub("", m.group(1)).strip() + "}", s)
    s = _RE_SUB_HTML.sub(lambda m: "_{" + _RE_TAG_ANY_HTML.sub("", m.group(1)).strip() + "}", s)
    s = re.sub(r"(?i)</(p|div|li|h[1-6]|tr|pre|section|article)>", "\n", s)
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = _RE_TAG_ANY_HTML.sub(" ", s)
    s = _html.unescape(s)
    s = s.replace("$$$", " ")
    s = _RE_WS_HTML.sub(" ", s)
    s = _RE_NL_HTML.sub("\n\n", s)
    return s.strip()


def extract_cf_statement_parts(html_str: str) -> dict[str, str]:
    """Extract Codeforces statement parts via DOM class-balanced slicing."""
    cleaned_html = _RE_SCRIPT_STYLE_HTML.sub(" ", html_str or "")
    m = _RE_PROBLEM_STATEMENT_HTML.search(cleaned_html)
    if not m:
        return {"preamble": "", "input_spec": "", "output_spec": "", "interaction_spec": "", "is_interactive": "0"}
    body = _inner_of_div(cleaned_html, m.start())
    input_spec = _extract_div_by_class(body, "input-specification")
    output_spec = _extract_div_by_class(body, "output-specification")
    interaction = _extract_div_by_class(body, "interaction-specification") or _extract_div_by_class(body, "interaction")
    prose = body
    for cls in (
        "header",
        "input-specification",
        "output-specification",
        "interaction-specification",
        "interaction",
        "sample-tests",
        "sample-test",
        "note",
    ):
        prose = _strip_balanced_div(prose, cls)
    prose_text = _html_to_text(prose)

    def _drop_title(seg: str) -> str:
        seg = re.sub(r'(?is)<div\s+class="section-title"[^>]*>.*?</div>', " ", seg)
        txt = _html_to_text(seg)
        first, _, rest = txt.partition("\n")
        if first.strip().lower().strip(" .:：") in ("input", "output", "interaction", "note", "example", "examples"):
            return rest.strip()
        return txt.strip()

    low_all = cleaned_html.lower()
    interactive = bool(
        interaction.strip()
        or re.search(r"hunk-title[^>]*>\s*interaction", low_all)
        or re.search(r"interactive\s+problem", low_all)
        or re.search(r"this\s+is\s+an?\s+interactive", low_all)
    )
    return {
        "preamble": prose_text,
        "input_spec": _drop_title(input_spec),
        "output_spec": _drop_title(output_spec),
        "interaction_spec": _drop_title(interaction),
        "is_interactive": "1" if interactive else "0",
    }


def _norm_key(s: object) -> str:
    return re.sub(r"[\s_\-]+", " ", str(s)).strip().lower()


def _sections_map(sections: object) -> dict[str, str]:
    if not isinstance(sections, dict):
        return {}
    return {_norm_key(k): v.strip() for k, v in sections.items() if isinstance(v, str) and v.strip()}


def _section_text(sec: dict[str, str], *names: str) -> str:
    for name in names:
        key = _norm_key(name)
        if key in sec:
            return sec[key]
    for k, v in sec.items():
        for name in names:
            if _norm_key(name) in k:
                return v
    return ""


def assemble_atcoder_statement(rec: dict[str, Any]) -> tuple[str, str]:
    """Assemble non-sample sections from an AtCoder statement record."""
    sec = _sections_map(rec.get("sections"))
    if sec:
        body = _section_text(sec, "problem statement", "statement")
        cons = _section_text(sec, "constraints", "constraint")
        inp = _section_text(sec, "input", "input format")
        parts = [p for p in (body, cons, inp) if p]
        if parts:
            return "\n\n".join(parts), "sections"
    for col in ("statement_en", "statement"):
        txt = rec.get(col)
        if isinstance(txt, str) and txt.strip():
            return txt.strip(), "statement"
    return "", "empty"


def _run_standalone_self_test() -> int:
    """Run 7 synthetic and adversarial self-tests covering p3-proto-v2 cleaning, DOM stripping, and P3 pairing."""
    # Test 1: Literal < ... > inequality must be preserved in v2 (while v1 regex swallows it)
    t1 = "If x < y and z > 0, print x."
    assert clean_statement_p3_v2(t1) == "if x < y and z > 0, print x.", clean_statement_p3_v2(t1)
    assert clean_statement_p3(t1) == "if x 0, print x.", clean_statement_p3(t1)

    # Test 2: Chained multi-line / subscript inequalities must survive intact
    t2 = "Find integers satisfying 0 <= k < n and x_k > k."
    assert clean_statement_p3_v2(t2) == "find integers satisfying 0 <= k < n and x_k > k.", clean_statement_p3_v2(t2)
    assert clean_statement_p3(t2) == "find integers satisfying 0 k.", clean_statement_p3(t2)

    # Test 3: Literal angle-bracket tokens (< or > and <<>>) must not vanish
    t3 = "Each character is either < or > and string <<>>."
    assert clean_statement_p3_v2(t3) == "each character is either < or > and string <<>>.", clean_statement_p3_v2(t3)
    assert clean_statement_p3(t3) == "each character is either and string >.", clean_statement_p3(t3)

    # Test 4: Grammar/BNF angle-bracket tokens (<lvalue>, <rvalue>) preserved
    t4 = "Form <lvalue> = <rvalue> or <lvalue> = <arg1><op><arg2>."
    assert clean_statement_p3_v2(t4) == "form <lvalue> = <rvalue> or <lvalue> = <arg1><op><arg2>.", clean_statement_p3_v2(t4)

    # Test 5: Adversarial HTML DOM stripping BEFORE html.unescape + entity recovery + interactive + surface extraction
    syn_html = (
        '<html><head><style>.hidden{display:none;}</style><script>var secret_js = 1;</script></head><body>'
        '<div class="problem-statement">'
        '<div class="header"><div class="title">A. Synthetic</div><div class="time-limit">1 second</div>'
        '<div class="input-file input-standard">standard input</div></div>'
        '<div><script>window.leak=1;</script><style>.x{color:red;}</style>'
        '<p>This is an interactive problem. Compute answers modulo $998244353$ whenever '
        '0 &lt;= k &lt; n and x &gt; k, or string &lt;&lt;&gt;&gt;.</p></div>'
        '<div class="input-specification"><div class="section-title">Input</div>'
        r'<p>Each test contains multiple test cases. First line contains $t$ ($1 \le t \le 10<sup class="upper-index">4</sup>$). '
        r'Next line contains $n$ ($1 \le n \le 2 \cdot 10<sup class="upper-index">5</sup>$) and '
        r'$1 \le a<sub class="lower-index">i</sub> \le 10<sup class="upper-index">9</sup>$.</p></div>'
        '<div class="sample-tests"><div class="sample-test"><div class="input">ExampleSampleInput 999</div></div></div>'
        '<div class="note"><p>Sample note must vanish.</p></div>'
        '</div><div id="footer">Copyright footer</div></body></html>'
    )
    parts = extract_cf_statement_parts(syn_html)
    assert parts["is_interactive"] == "1", parts
    combined_cf = "\n\n".join(x for x in (parts["preamble"], parts["input_spec"]) if x).strip()
    low_comb = combined_cf.lower()
    for banned in (
        "standard input",
        "time limit",
        "secret_js",
        "window.leak",
        "display:none",
        "color:red",
        "examplesampleinput",
        "sample note",
        "copyright footer",
    ):
        assert banned not in low_comb, f"Banned HTML residue leaked: {banned} in {combined_cf}"
    cl_cf = clean_statement_p3_v2(combined_cf, drop_samples=False)
    assert "0 <= k < n and x > k" in cl_cf and "<<>>" in cl_cf and "998244353" in cl_cf, cl_cf
    feats_cf = extract_surface_features(combined_cf, cl_cf, is_interactive_hint=1)
    assert feats_cf["has_mod"] == 1.0 and feats_cf["is_interactive"] == 1.0 and feats_cf["has_multi_test"] == 1.0
    assert math.isclose(feats_cf["array_len_upper"], 200000.0), feats_cf
    assert math.isclose(feats_cf["value_range_upper"], 1e9), feats_cf

    # Test 6: Synthetic AtCoder assembly + thousands separator + exponent overflow guard + drop_samples=True
    syn_ac = {
        "sections": {
            "Problem Statement": r"Given $N$ integers $A_1, \dots, A_N$, count pairs with $A_i < A_j$.",
            "Constraints": r"$1 \le N \le 500\,000$, $0 \le A_i \le 10^9$, bound $x \le 10^{1000}$.",
            "Input": "N then A_1 ... A_N",
            "Sample Input 1": "3\n1 2 3",
        }
    }
    raw_ac, src_ac = assemble_atcoder_statement(syn_ac)
    assert src_ac == "sections" and "sample input" not in raw_ac.lower()
    cl_ac = clean_statement_p3_v2(raw_ac + "\nExample\nInput:\n1 2\nOutput:\n3", drop_samples=True)
    assert "example" not in cl_ac and "a_i < a_j" in cl_ac, cl_ac
    feats_ac = extract_surface_features(raw_ac, cl_ac, is_interactive_hint=0)
    assert math.isclose(feats_ac["array_len_upper"], 500000.0), feats_ac
    assert math.isclose(feats_ac["value_range_upper"], 1e9), feats_ac
    assert feats_ac["is_interactive"] == 0.0

    # Test 7: Full P3 pairing, scoring, zero-valid-block bootstrap, and 4-band common-support verification
    unit_res = verify_protocol_unit_tests()
    assert unit_res["unit_tests_passed"] == 8, unit_res

    print("statement_feature_parser_v2.py --self-test PASS (7/7 synthetic & protocol test groups passed).")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Standalone p3-proto-v2 (and legacy v1) HTML/statement cleaner, 11-d surface extractor, and P3 pairing verifier."
    )
    parser.add_argument("--self-test", action="store_true", help="Run 7 synthetic and adversarial self-test groups.")
    args = parser.parse_args()
    if args.self_test:
        raise SystemExit(_run_standalone_self_test())
    parser.print_help()
