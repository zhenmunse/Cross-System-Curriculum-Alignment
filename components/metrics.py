# -*- coding: utf-8 -*-

# 指标计算模块
# 负责计算 Coverage, Alignment Accuracy, Gap Ratio, Bridge Efficiency 四项指标。

import pandas as pd
import numpy as np
from . import config

def calculate_all_metrics(module_mappings: pd.DataFrame, los_df: pd.DataFrame, modules_df: pd.DataFrame, source_system: str, target_system: str):
    """
    计算所有定义的指标，包括全局和分体系对。

    Args:
        module_mappings (pd.DataFrame): 模块映射结果。
        los_df (pd.DataFrame): 学习成果数据。
        modules_df (pd.DataFrame): 模块数据。
        source_system (str): 源体系名称。
        target_system (str): 目标体系名称。

    Returns:
        pd.DataFrame: 包含所有指标结果的 DataFrame。
    """
    metrics = {}
    
    # 准备数据
    target_modules = modules_df[modules_df['system'] == target_system]['module_id'].unique()
    source_los_weights = los_df[los_df['system'] == source_system][['module_id', 'lo_id', 'weight']].set_index('lo_id')['weight']

    # 1. Coverage (C)
    metrics['coverage'] = _calculate_coverage(module_mappings, target_modules)

    # 2. Alignment Accuracy (A)
    # 需要原始的 LO-level 映射来计算加权平均，这里我们用模块映射的 confidence 近似
    # 注意：更精确的实现需要回溯到 LO 对，但这里根据题目要求聚合后计算
    non_gap_mappings = module_mappings[module_mappings['mapping_type'] != 'Gap']
    metrics['alignment_accuracy'] = _calculate_alignment_accuracy(non_gap_mappings, los_df, source_system)

    # 3. Gap Ratio (G)
    metrics['gap_ratio'] = _calculate_gap_ratio(module_mappings, target_modules)

    # 4. Bridge Efficiency (B)
    metrics['bridge_efficiency'] = _calculate_bridge_efficiency(module_mappings, target_modules)

    # 组装成 DataFrame
    metrics_df = pd.DataFrame([metrics])
    metrics_df['source_system'] = source_system
    metrics_df['target_system'] = target_system
    
    # 调整列顺序
    metrics_df = metrics_df[['source_system', 'target_system', 'coverage', 'alignment_accuracy', 'gap_ratio', 'bridge_efficiency']]
    
    return metrics_df

def _calculate_coverage(module_mappings: pd.DataFrame, target_modules: np.ndarray) -> float:
    """计算覆盖率 (Coverage)。"""
    if len(target_modules) == 0:
        return 0.0
    
    covered_modules = module_mappings[module_mappings['mapping_type'] != 'Gap']['target_module'].nunique()
    return covered_modules / len(target_modules)

def _calculate_alignment_accuracy(non_gap_mappings: pd.DataFrame, los_df: pd.DataFrame, source_system: str) -> float:
    """计算对齐准确度 (Alignment Accuracy)。"""
    if non_gap_mappings.empty:
        return 0.0
    
    # 获取源模块的总权重
    source_module_weights = los_df[los_df['system'] == source_system].groupby('module_id')['weight'].sum()
    
    # 将模块权重合并到映射结果中
    weighted_mappings = non_gap_mappings.merge(source_module_weights.rename('source_module_weight'), left_on='source_module', right_index=True)
    
    if weighted_mappings['source_module_weight'].sum() == 0:
        # pandas 的 mean 返回 numpy.floating，显式转为 Python float 以满足类型检查
        return float(weighted_mappings['confidence'].mean())
        
    # 计算加权平均置信度
    weighted_accuracy = np.average(weighted_mappings['confidence'], weights=weighted_mappings['source_module_weight'])
    # np.average 返回 numpy.floating，显式转为 Python float
    return float(weighted_accuracy)

def _calculate_gap_ratio(module_mappings: pd.DataFrame, target_modules: np.ndarray) -> float:
    """计算差距比例 (Gap Ratio)。"""
    if len(target_modules) == 0:
        return 0.0
        
    # 找到每个目标模块的最佳映射
    best_mappings = module_mappings.loc[module_mappings.groupby('target_module')['confidence'].idxmax()]
    
    gap_modules = best_mappings[best_mappings['mapping_type'] == 'Gap']['target_module'].nunique()
    
    # 考虑那些在映射中从未出现的目标模块
    all_mapped_targets = module_mappings['target_module'].unique()
    unmapped_targets = set(target_modules) - set(all_mapped_targets)
    
    total_gap_modules = gap_modules + len(unmapped_targets)
    
    return total_gap_modules / len(target_modules)

def _calculate_bridge_efficiency(module_mappings: pd.DataFrame, target_modules: np.ndarray) -> float:
    """计算桥梁效率 (Bridge Efficiency)。"""
    if len(target_modules) == 0:
        return 0.0

    # 修复前的覆盖率
    coverage_before = _calculate_coverage(module_mappings, target_modules)
    if coverage_before == 1.0:
        return 0.0 # 已经完全覆盖，效率为0

    # 识别可修复的条目
    # 假设对 Enrichment 与 Gap 中 confidence 在特定区间的条目进行修复
    lower_bound, upper_bound = config.BRIDGE_CONFIDENCE_RANGE
    fixable_mask = (module_mappings['confidence'] >= lower_bound) & (module_mappings['confidence'] < upper_bound)
    
    # 模拟修复
    num_fixable = fixable_mask.sum()
    num_to_fix = int(num_fixable * config.BRIDGE_FIX_RATE)
    
    # 找出这些可修复的条目，并随机选择一部分进行“修复”
    fixable_indices = module_mappings[fixable_mask].index
    if num_to_fix > 0 and not fixable_indices.empty:
        # 使用固定的随机状态来选择
        np.random.seed(config.RANDOM_STATE)
        fixed_indices = np.random.choice(fixable_indices, size=min(num_to_fix, len(fixable_indices)), replace=False)
        
        # 创建一个修复后的副本
        mappings_after_fix = module_mappings.copy()
        # 将这些条目的类型提升为 Partial，使其不再是 Gap
        mappings_after_fix.loc[fixed_indices, 'mapping_type'] = 'Partial' 
    else:
        mappings_after_fix = module_mappings

    # 计算修复后的覆盖率
    coverage_after = _calculate_coverage(mappings_after_fix, target_modules)

    # 计算提升百分比
    if coverage_before == 0: # 避免除以零
        return 100.0 if coverage_after > 0 else 0.0
        
    efficiency = ((coverage_after - coverage_before) / coverage_before) * 100
    return efficiency
