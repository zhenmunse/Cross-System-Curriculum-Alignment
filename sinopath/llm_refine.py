# -*- coding: utf-8 -*-
"""
LLM/规则微调模块：可选地对初始 confidence 进行规则或 LLM 的后处理。
- mode='none': 不做处理，直接返回。
- mode='rule': 简单规则：若源/目标模块名高度相似，提高少量置信度；若差异很大且包含负向词，降低置信度。
- mode='openai': 预留接口；如未安装 openai 或无 API key，则回退为 rule 或 none。

注意：为保证离线可运行，默认 mode='none'，不会产生网络调用。
"""
from typing import Literal
import os
import re
import pandas as pd

LLMMode = Literal['none', 'rule', 'openai']

_NEGATIVE_PATTERNS = [
    r'not\s+covered', r'out\s+of\s+scope', r'unrelated', r'no\s+match'
]
_negative_regex = re.compile('|'.join(_NEGATIVE_PATTERNS), re.IGNORECASE)


def _string_sim(a: str, b: str) -> float:
    """一个轻量的字符串相似度：Jaccard on tokens。"""
    toks_a = set(re.findall(r"[a-zA-Z0-9]+", a.lower()))
    toks_b = set(re.findall(r"[a-zA-Z0-9]+", b.lower()))
    if not toks_a or not toks_b:
        return 0.0
    inter = len(toks_a & toks_b)
    union = len(toks_a | toks_b)
    return inter / union


def refine_confidence(
    mappings_df: pd.DataFrame,
    mode: LLMMode = 'none',
) -> pd.DataFrame:
    """
    对 module_mappings 的 confidence 做可选的后处理。
    期望输入列：source_module_name、target_module_name、confidence。
    返回：复制后的 DataFrame（不修改原对象）。
    """
    if mode == 'none':
        return mappings_df.copy()

    df = mappings_df.copy()

    if mode == 'rule':
        inc, dec = 0.03, 0.05
        sims = df.apply(
            lambda r: _string_sim(str(r.get('source_module_name', '')), str(r.get('target_module_name', ''))), axis=1
        )
        df['confidence'] = (
            df['confidence']
            + inc * (sims > 0.6).astype(float)
            - dec * df['target_module_name'].astype(str).str.contains(_negative_regex).astype(float)
        ).clip(0.0, 1.0)
        return df

    if mode == 'openai':
        # 占位实现：避免在无网络/无密钥时出错，直接返回原值。
        # 如需启用，请在此处实现基于模块名称/描述的调用，并对返回结果进行融合。
        return df

    return df
