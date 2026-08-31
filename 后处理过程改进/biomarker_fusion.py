#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生物标志物与基线概率的确定性融合（无 torch / 模型依赖）。

与 BiomarkerEnhancedPredictor 中 _calculate_biomarker_boost + _combine_predictions 逻辑一致。
"""

from typing import Any, Dict, Tuple

import numpy as np


def apply_biomarker_fusion(
    base_probs: np.ndarray,
    n_biomarker_matches: int,
    max_confidence: float,
    biomarker_weight: float,
    min_biomarker_confidence: float = 0.6,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    base_probs : (2,) [P(Interictal), P(Preictal)]
    n_biomarker_matches : 匹配到的标志物条数（0 表示无匹配）
    max_confidence : 匹配项中的最大置信度；无匹配时应为 0
    """
    base_probs = np.asarray(base_probs, dtype=np.float64).reshape(2)
    boost: Dict[str, Any] = {
        "boost_strength": 0.0,
        "boost_direction": 0,
        "avg_confidence": 0.0,
        "max_confidence": float(max_confidence),
        "n_matches": int(n_biomarker_matches),
    }
    if n_biomarker_matches <= 0:
        return base_probs.copy(), boost

    max_c = float(max_confidence)
    boost["max_confidence"] = max_c
    if max_c >= min_biomarker_confidence:
        boost_strength = min(1.0, max_c * float(biomarker_weight))
        boost["boost_strength"] = boost_strength
        boost["boost_direction"] = 1
    else:
        boost_strength = 0.0

    enhanced = base_probs.copy()
    if boost_strength > 0 and boost["boost_direction"] == 1:
        transfer = boost_strength * base_probs[0]
        enhanced[0] = base_probs[0] - transfer
        enhanced[1] = base_probs[1] + transfer
    enhanced = np.clip(enhanced, 0.0, 1.0)
    enhanced = enhanced / (enhanced.sum() + 1e-10)
    return enhanced, boost
