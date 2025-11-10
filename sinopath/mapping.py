# -*- coding: utf-8 -*-
"""
核心映射算法模块 (对应论文 Section IV-B)。
实现了“语义相似度 + 难度匹配修正”的映射逻辑。
"""
import pandas as pd
import numpy as np
from . import config
from .text_embed import get_embedder
from .llm_refine import refine_confidence, LLMMode

def perform_mapping(
    los_df: pd.DataFrame,
    modules_df: pd.DataFrame,
    source_system: str,
    target_system: str,
    embed_method: str = 'tfidf',
    llm_mode: LLMMode = 'none'
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
    # 1. 筛选源和目标体系的数据
    source_los = los_df[los_df['system'] == source_system].copy()
    target_los = los_df[los_df['system'] == target_system].copy()

    if source_los.empty or target_los.empty:
        raise ValueError(f"源体系 '{source_system}' 或目标体系 '{target_system}' 在数据中没有学习成果。")

    # 2. 文本向量化
    embedder = get_embedder(method=embed_method, random_state=config.RANDOM_STATE)
    all_descriptions = pd.concat([source_los['description'], target_los['description']]).tolist()
    all_vectors = embedder.fit_transform(all_descriptions) # 先 fit 整个语料库
    
    # 分割向量：前 len(source_los) 个是源，后面是目标
    source_count = len(source_los)
    source_vectors = all_vectors[:source_count]
    target_vectors = all_vectors[source_count:]

    # 3. 计算语义相似度
    semantic_similarity_matrix = embedder.calculate_similarity(source_vectors, target_vectors)

    # 4. 计算难度修正因子
    # 使用广播机制高效计算难度差异
    source_difficulty = np.array(source_los['difficulty_level'].tolist()).reshape(-1, 1)
    target_difficulty = np.array(target_los['difficulty_level'].tolist()).reshape(1, -1)
    diff_levels = np.abs(source_difficulty - target_difficulty)
    difficulty_factor = 1 - diff_levels / 4.0

    # 5. 计算最终置信度 (对应论文公式)
    # confidence = semantic * ( (1-w) + w * diff_factor )
    confidence_matrix = semantic_similarity_matrix * ((1 - config.DIFFICULTY_WEIGHT) + config.DIFFICULTY_WEIGHT * difficulty_factor)

    # 6. 构建 LO-level 的映射结果
    lo_mapping_list = []
    for i, source_lo_id in enumerate(source_los['lo_id']):
        for j, target_lo_id in enumerate(target_los['lo_id']):
            lo_mapping_list.append({
                'source_lo_id': source_lo_id,
                'target_lo_id': target_lo_id,
                'confidence': confidence_matrix[i, j]
            })
    lo_mappings_df = pd.DataFrame(lo_mapping_list)

    # 7. 聚合到模块级别
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

    # 8. 分类映射类型
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

    # 9. 可选 LLM/规则微调置信度
    try:
        refined = refine_confidence(module_mappings, mode=llm_mode)
        # 重新分类映射类型（因置信度可能变化）
        if llm_mode != 'none':
            refined['mapping_type'] = _classify_mapping(refined['confidence'])
        return refined
    except Exception:
        # 如果微调失败，回退到原始结果
        return module_mappings

def _classify_mapping(confidence_series: pd.Series) -> pd.Series:
    """根据置信度为映射关系分类。"""
    conditions = [
        confidence_series > config.THRESHOLD_EXACT,
        (confidence_series > config.THRESHOLD_PARTIAL) & (confidence_series <= config.THRESHOLD_EXACT),
        (confidence_series > config.THRESHOLD_ENRICH) & (confidence_series <= config.THRESHOLD_PARTIAL),
    ]
    choices = ['Exact', 'Partial', 'Enrichment']
    result = np.select(conditions, choices, default='Gap')
    return pd.Series(result, index=confidence_series.index)
