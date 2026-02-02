# -*- coding: utf-8 -*-

# 核心映射算法模块
# 实现了"语义相似度 + 难度匹配修正"的映射逻辑

import pandas as pd
import numpy as np
from typing import Optional
from . import config
from .text_embed import get_embedder
from .llm_refine import refine_confidence, LLMMode

def perform_mapping(
    los_df: pd.DataFrame,
    modules_df: pd.DataFrame,
    source_system: str,
    target_system: str,
    embed_method: str = 'tfidf',
    llm_mode: LLMMode = 'none',
    similarity_scaling: str | None = None,
    scaling_alpha: float | None = None,
    top_k_target_lo: int | None = None
):
    """
    执行跨体系学习成果和模块的映射。

    Args:
        los_df (pd.DataFrame): 学习成果数据。
        modules_df (pd.DataFrame): 模块数据。
        source_system (str): 源体系名称。
        target_system (str): 目标体系名称。

    Returns:
        pd.DataFrame: 模块间的映射结果。
    """
    # 筛选源和目标体系的数据
    source_los = los_df[los_df['system'] == source_system].copy()
    target_los = los_df[los_df['system'] == target_system].copy()

    if source_los.empty or target_los.empty:
        raise ValueError(f"源体系 '{source_system}' 或目标体系 '{target_system}' 在数据中没有学习成果。")

    # 文本向量化
    embedder = get_embedder(method=embed_method, random_state=config.RANDOM_STATE)
    all_descriptions = pd.concat([source_los['description'], target_los['description']]).tolist()
    all_vectors = embedder.fit_transform(all_descriptions) # 先 fit 整个语料库
    
    # 分割向量：前 len(source_los) 个是源，后面是目标
    source_count = len(source_los)
    source_vectors = all_vectors[:source_count]
    target_vectors = all_vectors[source_count:]

    # 计算语义相似度
    semantic_similarity_matrix = embedder.calculate_similarity(source_vectors, target_vectors)

    # 可选：Top-K 过滤降低噪声稀释
    if top_k_target_lo is None:
        top_k_target_lo = getattr(config, 'TOP_K_TARGET_LO', 0)
    if top_k_target_lo and top_k_target_lo > 0:
        # 对每一行保留前 k 列最大值，其余置 0
        topk_indices = np.argpartition(-semantic_similarity_matrix, top_k_target_lo-1, axis=1)[:, :top_k_target_lo]
        mask = np.zeros_like(semantic_similarity_matrix, dtype=bool)
        rows_idx = np.arange(semantic_similarity_matrix.shape[0])[:, None]
        mask[rows_idx, topk_indices] = True
        semantic_similarity_matrix = np.where(mask, semantic_similarity_matrix, 0.0)

    # 相似度缩放
    if similarity_scaling is None:
        similarity_scaling = getattr(config, 'SIMILARITY_SCALING', 'none')
    if scaling_alpha is None:
        scaling_alpha = getattr(config, 'SCALING_ALPHA', 1.0)
    raw = semantic_similarity_matrix
    if similarity_scaling == 'power':
        # 提升低值：使用平方根型：raw ** alpha (alpha<1)，保持单调
        if scaling_alpha is None or scaling_alpha <= 0:
            scaling_alpha = 0.5
        scaled = np.power(raw, float(scaling_alpha))
        semantic_similarity_matrix = scaled
    elif similarity_scaling == 'logistic':
        k = getattr(config, 'LOGISTIC_K', 8.0)
        x0 = getattr(config, 'LOGISTIC_X0', 0.1)
        semantic_similarity_matrix = 1.0 / (1.0 + np.exp(-k * (raw - x0)))

    # 计算难度修正因子（范围 0~10）
    # 公式：difficulty_factor = 1 - |d_s - d_t| / DIFFICULTY_MAX
    # 当差异为 0 → 因子=1；差异最大=10 → 因子=0
    source_difficulty = np.array(source_los['difficulty_level'].tolist(), dtype=float).reshape(-1, 1)
    target_difficulty = np.array(target_los['difficulty_level'].tolist(), dtype=float).reshape(1, -1)
    diff_levels = np.abs(source_difficulty - target_difficulty)
    difficulty_factor = 1 - diff_levels / float(config.DIFFICULTY_MAX)

    # 计算最终置信度 (对应论文公式)
    # confidence = semantic * ( (1-w) + w * diff_factor )
    confidence_matrix = semantic_similarity_matrix * ((1 - config.DIFFICULTY_WEIGHT) + config.DIFFICULTY_WEIGHT * difficulty_factor)

    # 构建 LO-level 的映射结果
    lo_mapping_list = []
    for i, source_lo_id in enumerate(source_los['lo_id']):
        for j, target_lo_id in enumerate(target_los['lo_id']):
            lo_mapping_list.append({
                'source_lo_id': source_lo_id,
                'target_lo_id': target_lo_id,
                'confidence': confidence_matrix[i, j]
            })
    lo_mappings_df = pd.DataFrame(lo_mapping_list)

    # 聚合到模块级别
    # 合并模块信息
    lo_mappings_df = lo_mappings_df.merge(source_los[['lo_id', 'module_id', 'weight']], left_on='source_lo_id', right_on='lo_id')
    lo_mappings_df.rename(columns={'module_id': 'source_module', 'weight': 'source_weight'}, inplace=True)
    lo_mappings_df = lo_mappings_df.merge(target_los[['lo_id', 'module_id']], left_on='target_lo_id', right_on='lo_id')
    lo_mappings_df.rename(columns={'module_id': 'target_module'}, inplace=True)
    
    # 按 (source_module, target_module) 分组，计算加权平均置信度
    def weighted_avg(group):
        # 检查权重和是否为零
        if group['source_weight'].sum() == 0:
            return group['confidence'].mean()
        return np.average(group['confidence'], weights=group['source_weight'])

    module_mappings = lo_mappings_df.groupby(['source_module', 'target_module']).apply(weighted_avg).reset_index(name='confidence')
    
    # 统计每个模块对中包含的 LO 对数量
    pairs_count = lo_mappings_df.groupby(['source_module', 'target_module']).size().reset_index(name='pairs_count')
    module_mappings = module_mappings.merge(pairs_count, on=['source_module', 'target_module'])

    # 分类映射类型
    module_mappings['mapping_type'] = _classify_mapping(module_mappings['confidence'])
    
    # 合并模块名
    module_names = modules_df[['module_id', 'module_name']].drop_duplicates()
    module_mappings = module_mappings.merge(module_names.rename(columns={'module_id': 'source_module', 'module_name': 'source_module_name'}), on='source_module')
    module_mappings = module_mappings.merge(module_names.rename(columns={'module_id': 'target_module', 'module_name': 'target_module_name'}), on='target_module')

    # 调整列顺序
    module_mappings = module_mappings[[
        'source_module', 'source_module_name', 'target_module', 'target_module_name', 
        'confidence', 'mapping_type', 'pairs_count'
    ]]

    module_mappings = module_mappings.sort_values(by='confidence', ascending=False)

    # 可选 LLM/规则微调置信度
    try:
        refined = refine_confidence(module_mappings, mode=llm_mode)
        # 重新分类映射类型（因置信度可能变化）
        if llm_mode != 'none':
            refined['mapping_type'] = _classify_mapping(refined['confidence'])
        return refined
    except Exception:
        # 如果微调失败，回退到原始结果
        return module_mappings


def perform_course_mapping(module_mappings: pd.DataFrame, modules_df: pd.DataFrame, strategy: str | None = None) -> pd.DataFrame:
    """基于已计算的 module_mappings 聚合为 course 级别的映射结果。

    逻辑：
    - 将 source_module / target_module 映射到各自的 course_id。
    - 按 (source_course, target_course) 分组，对模块置信度做加权平均。
      权重 = source 模块的 module_weight × pairs_count （体现模块权重与覆盖的 LO 对数量）。
    - 重新分类 mapping_type 使用课程级置信度。

    返回列：
        source_course, target_course, confidence, mapping_type, contributing_modules
    """
    if module_mappings.empty:
        return pd.DataFrame(columns=['source_course','target_course','confidence','mapping_type','contributing_modules'])

    # 准备模块到课程映射表（确保 module_id 唯一）
    module_to_course = modules_df[['module_id','course_id']].drop_duplicates(subset=['module_id'], keep='last').set_index('module_id')['course_id']

    df = module_mappings.copy()
    df['source_course'] = df['source_module'].map(module_to_course)
    df['target_course'] = df['target_module'].map(module_to_course)

    # 合并模块权重（确保 module_id 唯一）
    module_weights = modules_df[['module_id','module_weight']].drop_duplicates(subset=['module_id'], keep='last').set_index('module_id')['module_weight']
    df['source_module_weight'] = df['source_module'].map(module_weights).fillna(0.0)

    # 计算聚合权重：模块权重 × pairs_count（若缺失则视为1）
    if 'pairs_count' not in df.columns:
        df['pairs_count'] = 1
    df['agg_weight'] = df['source_module_weight'] * df['pairs_count']
    # 避免全为 0 导致 np.average 报错
    df.loc[df['agg_weight'] == 0, 'agg_weight'] = 1e-6

    course_rows = []
    if strategy is None:
        strategy = getattr(config, 'COURSE_AGGREGATION_STRATEGY', 'weighted_mean')

    for (sc, tc), group in df.groupby(['source_course','target_course']):
        if strategy == 'weighted_mean':
            if group['agg_weight'].sum() == 0:
                conf = group['confidence'].mean()
            else:
                conf = float(np.average(group['confidence'], weights=group['agg_weight']))
        elif strategy == 'union':
            # 概率并集：1 - Π(1 - p_i)；p_i = confidence * weight_norm
            total_w = group['agg_weight'].sum()
            weight_norm = group['agg_weight'] / (total_w if total_w > 0 else 1.0)
            p = (group['confidence'] * weight_norm).clip(0, 1)
            conf = float(1.0 - np.prod(1.0 - p))
        else:
            # 回退
            conf = group['confidence'].mean()
        course_rows.append({
            'source_course': sc,
            'target_course': tc,
            'confidence': conf,
            'contributing_modules': len(group)
        })
    course_df = pd.DataFrame(course_rows)
    if course_df.empty:
        return course_df
    course_df['mapping_type'] = _classify_mapping(course_df['confidence'])
    course_df = course_df.sort_values(by='confidence', ascending=False)
    return course_df

def _classify_mapping(confidence_series: pd.Series) -> pd.Series:
    # 根据置信度为映射关系分类
    conditions = [
        confidence_series > config.THRESHOLD_EXACT,
        (confidence_series > config.THRESHOLD_PARTIAL) & (confidence_series <= config.THRESHOLD_EXACT),
        (confidence_series > config.THRESHOLD_ENRICH) & (confidence_series <= config.THRESHOLD_PARTIAL),
    ]
    choices = ['Exact', 'Partial', 'Enrichment']
    result = np.select(conditions, choices, default='Gap')
    return pd.Series(result, index=confidence_series.index)

def perform_aggregate_mapping(
    los_df: pd.DataFrame,
    modules_df: pd.DataFrame,
    source_system: str,
    target_system: str,
    source_aggregate_name: Optional[str] = None,
    target_aggregate_name: Optional[str] = None,
    embed_method: str = 'tfidf',
    llm_mode: LLMMode = 'none',
    similarity_scaling: Optional[str] = None,
    scaling_alpha: Optional[float] = None,
    top_k_target_lo: Optional[int] = None
):
    """
    执行聚合映射：将源体系和目标体系的所有 LOs 分别聚合为一个整体，计算整体间的映射。

    Args:
        los_df: 学习成果数据（已过滤到选定的 courses/modules）
        modules_df: 模块数据（已过滤）
        source_system: 源体系名称
        target_system: 目标体系名称
        source_aggregate_name: 源聚合名称
        target_aggregate_name: 目标聚合名称
        其他参数同 perform_mapping

    Returns:
        dict: 聚合映射结果，包含整体置信度、LO 对数量、覆盖率等
    """
    # 筛选源和目标体系的数据
    source_los = los_df[los_df['system'] == source_system].copy()
    target_los = los_df[los_df['system'] == target_system].copy()

    if source_los.empty or target_los.empty:
        raise ValueError(f"聚合映射失败：源体系 '{source_system}' 或目标体系 '{target_system}' 在过滤后的数据中没有学习成果。")

    # 设置默认聚合名称
    if not source_aggregate_name:
        source_aggregate_name = f"{source_system.upper()}_Aggregate"
    if not target_aggregate_name:
        target_aggregate_name = f"{target_system.upper()}_Aggregate"

    # 文本向量化
    embedder = get_embedder(method=embed_method, random_state=config.RANDOM_STATE)
    all_descriptions = pd.concat([source_los['description'], target_los['description']]).tolist()
    all_vectors = embedder.fit_transform(all_descriptions)
    
    source_count = len(source_los)
    source_vectors = all_vectors[:source_count]
    target_vectors = all_vectors[source_count:]

    # 计算所有 LO 对之间的相似度
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(source_vectors, target_vectors)

    # 应用 top-k 过滤
    if top_k_target_lo and top_k_target_lo > 0:
        for i in range(similarity_matrix.shape[0]):
            top_k_indices = np.argsort(similarity_matrix[i])[-top_k_target_lo:]
            mask = np.zeros(similarity_matrix.shape[1], dtype=bool)
            mask[top_k_indices] = True
            similarity_matrix[i, ~mask] = 0.0

    # 应用相似度缩放
    if similarity_scaling == 'power' and scaling_alpha:
        similarity_matrix = np.power(similarity_matrix, scaling_alpha)
    elif similarity_scaling == 'logistic' and scaling_alpha:
        k = config.LOGISTIC_K
        x0 = config.LOGISTIC_X0
        similarity_matrix = 1 / (1 + np.exp(-k * (similarity_matrix - x0)))

    # 应用难度修正
    source_difficulties = source_los['difficulty_level'].values
    target_difficulties = target_los['difficulty_level'].values
    
    for i in range(len(source_los)):
        for j in range(len(target_los)):
            diff_factor = 1 - abs(source_difficulties[i] - target_difficulties[j]) / config.DIFFICULTY_MAX
            adjusted_sim = similarity_matrix[i, j] * ((1 - config.DIFFICULTY_WEIGHT) + config.DIFFICULTY_WEIGHT * diff_factor)
            similarity_matrix[i, j] = np.clip(adjusted_sim, 0, 1)

    # LLM 后处理（可选）- 聚合模式下简化处理
    # 注意：聚合模式的 LLM 后处理需要特殊实现，这里暂时跳过
    # 如需启用，可构建临时 DataFrame 批量调用 refine_confidence
    if llm_mode == 'rule':
        # 简化的规则调整：对整体置信度矩阵应用小幅调整
        # 这里只做演示，实际可根据需求扩展
        similarity_matrix = np.clip(similarity_matrix * 1.02, 0, 1)  # 小幅提升

    # 聚合统计
    # 计算整体置信度：使用加权平均（按 LO 权重）
    source_weights = source_los['weight'].to_numpy()
    target_weights = target_los['weight'].to_numpy()
    
    # 归一化权重
    source_weights_sum = float(np.sum(source_weights))
    target_weights_sum = float(np.sum(target_weights))
    source_weights = source_weights / source_weights_sum
    target_weights = target_weights / target_weights_sum
    
    # 加权平均置信度
    weighted_confidences = []
    for i in range(len(source_los)):
        for j in range(len(target_los)):
            weight = source_weights[i] * target_weights[j]
            weighted_confidences.append(similarity_matrix[i, j] * weight)
    
    overall_confidence = float(np.sum(weighted_confidences))
    
    # 计算覆盖率统计
    threshold = config.THRESHOLD_ENRICH  # 至少 Enrichment 级别才算覆盖
    covered_source = np.any(similarity_matrix >= threshold, axis=1).sum()
    covered_target = np.any(similarity_matrix >= threshold, axis=0).sum()
    
    source_coverage_rate = float(covered_source / len(source_los))
    target_coverage_rate = float(covered_target / len(target_los))
    
    # 置信度分布统计
    all_confidences = similarity_matrix.flatten()
    exact_count = int(np.sum(all_confidences > config.THRESHOLD_EXACT))
    partial_count = int(np.sum((all_confidences > config.THRESHOLD_PARTIAL) & (all_confidences <= config.THRESHOLD_EXACT)))
    enrich_count = int(np.sum((all_confidences > config.THRESHOLD_ENRICH) & (all_confidences <= config.THRESHOLD_PARTIAL)))
    gap_count = int(np.sum(all_confidences <= config.THRESHOLD_ENRICH))
    
    # 计算映射类型
    if overall_confidence > config.THRESHOLD_EXACT:
        mapping_type = 'Exact'
    elif overall_confidence > config.THRESHOLD_PARTIAL:
        mapping_type = 'Partial'
    elif overall_confidence > config.THRESHOLD_ENRICH:
        mapping_type = 'Enrichment'
    else:
        mapping_type = 'Gap'
    
    # 构建详细的 LO 对列表（用于导出 CSV）
    lo_pairs = []
    source_lo_ids = source_los['lo_id'].tolist()
    target_lo_ids = target_los['lo_id'].tolist()
    source_descriptions = source_los['description'].tolist()
    target_descriptions = target_los['description'].tolist()
    
    for i in range(len(source_los)):
        for j in range(len(target_los)):
            conf = similarity_matrix[i, j]
            # 确定类型
            if conf > config.THRESHOLD_EXACT:
                pair_type = 'Exact'
            elif conf > config.THRESHOLD_PARTIAL:
                pair_type = 'Partial'
            elif conf > config.THRESHOLD_ENRICH:
                pair_type = 'Enrichment'
            else:
                pair_type = 'Gap'
            
            lo_pairs.append({
                'source_lo': source_lo_ids[i],
                'source_description': source_descriptions[i],
                'target_lo': target_lo_ids[j],
                'target_description': target_descriptions[j],
                'confidence': conf,
                'mapping_type': pair_type
            })
    
    # 返回聚合结果
    result = {
        'source_aggregate': source_aggregate_name,
        'target_aggregate': target_aggregate_name,
        'overall_confidence': overall_confidence,
        'mapping_type': mapping_type,
        'source_lo_count': len(source_los),
        'target_lo_count': len(target_los),
        'total_lo_pairs': len(source_los) * len(target_los),
        'source_coverage_rate': source_coverage_rate,
        'target_coverage_rate': target_coverage_rate,
        'exact_pairs': exact_count,
        'partial_pairs': partial_count,
        'enrichment_pairs': enrich_count,
        'gap_pairs': gap_count,
        'source_modules': modules_df[modules_df['system'] == source_system]['module_name'].tolist(),
        'target_modules': modules_df[modules_df['system'] == target_system]['module_name'].tolist(),
        'source_courses': modules_df[modules_df['system'] == source_system]['course_id'].unique().tolist(),
        'target_courses': modules_df[modules_df['system'] == target_system]['course_id'].unique().tolist(),
        'lo_pairs': lo_pairs  # 新增：详细的 LO 对列表
    }
    
    return result
