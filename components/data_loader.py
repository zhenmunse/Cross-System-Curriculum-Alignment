# -*- coding: utf-8 -*-

# 数据加载与校验模块
# 负责从 CSV 或 Excel 文件中读取课程模块和学习成果数据，并进行预处理和校验

import pandas as pd
import os
import logging
from typing import Optional

def load_data(los_path: str, modules_path: str, 
              source_system: Optional[str] = None, target_system: Optional[str] = None,
              source_courses: Optional[list] = None, source_modules: Optional[list] = None,
              target_courses: Optional[list] = None, target_modules: Optional[list] = None):
    """
    加载并校验学习成果和模块数据。

    Args:
        los_path (str): 学习成果文件路径 (CSV or XLSX)。
        modules_path (str): 模块文件路径 (CSV or XLSX)。
        source_system (str): 源体系名称（用于过滤）。
        target_system (str): 目标体系名称（用于过滤）。
        source_courses (list): 源体系中要保留的课程列表。
        source_modules (list): 源体系中要保留的模块列表。
        target_courses (list): 目标体系中要保留的课程列表。
        target_modules (list): 目标体系中要保留的模块列表。

    Returns:
        tuple: (los_df, modules_df) 两个 Pandas DataFrame。
    """
    # 加载数据
    los_df = _read_file(los_path)
    modules_df = _read_file(modules_path)

    # 校验数据
    los_df, modules_df = _validate_and_clean_data(los_df, modules_df)

    # 应用 course/module 过滤
    los_df, modules_df = _apply_filters(
        los_df, modules_df,
        source_system, target_system,
        source_courses, source_modules,
        target_courses, target_modules
    )

    return los_df, modules_df

def _read_file(file_path: str) -> pd.DataFrame:
    # 根据文件扩展名读取 CSV 或 Excel 文件
    if not os.path.exists(file_path):
        logging.error(f"文件未找到: {file_path}")
        raise FileNotFoundError(f"文件未找到: {file_path}")

    _, extension = os.path.splitext(file_path)
    if extension == '.csv':
        return pd.read_csv(file_path)
    elif extension in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {extension}。请使用 CSV 或 Excel。")

def _validate_and_clean_data(los_df: pd.DataFrame, modules_df: pd.DataFrame):

    # 对加载的数据进行校验、清洗和类型转换。

    # 检查模块数据的必需字段
    required_module_cols = ['module_id', 'system', 'module_name', 'course_id']
    for col in required_module_cols:
        if col not in modules_df.columns:
            raise ValueError(f"模块文件缺失必需字段: '{col}'")

    # 检查学习成果数据的必需字段
    required_lo_cols = ['lo_id', 'module_id', 'system', 'description', 'difficulty_level', 'weight']
    for col in required_lo_cols:
        if col not in los_df.columns:
            raise ValueError(f"学习成果文件缺失必需字段: '{col}'")

    # 清洗和类型转换
    modules_df['system'] = modules_df['system'].str.strip().str.lower()
    los_df['system'] = los_df['system'].str.strip().str.lower()

    # 转换数据类型
    try:
        los_df['difficulty_level'] = pd.to_numeric(los_df['difficulty_level'])
        los_df['weight'] = pd.to_numeric(los_df['weight'])
    except ValueError as e:
        raise TypeError(f"学习成果文件中的 'difficulty_level' 或 'weight' 字段无法转换为数值类型: {e}")

    # 过滤无效数据
    initial_los_count = len(los_df)
    los_df.dropna(subset=['description'], inplace=True)
    if len(los_df) < initial_los_count:
        logging.warning(f"过滤掉 {initial_los_count - len(los_df)} 条描述为空的学习成果。")
    
    los_df = los_df[los_df['description'].str.strip() != '']

    # 难度范围校验（0~10）
    invalid_range = (los_df['difficulty_level'] < 0) | (los_df['difficulty_level'] > 10)
    if invalid_range.any():
        count_invalid = int(invalid_range.sum())
        logging.warning(f"发现 {count_invalid} 条 difficulty_level 超出 0~10 的范围，将进行裁剪。")
        los_df.loc[los_df['difficulty_level'] < 0, 'difficulty_level'] = 0
        los_df.loc[los_df['difficulty_level'] > 10, 'difficulty_level'] = 10

    # 检查关联完整性
    unmatched_los = los_df[~los_df['module_id'].isin(modules_df['module_id'])]
    if not unmatched_los.empty:
        unmatched_ids = unmatched_los['module_id'].unique()
        logging.warning(f"发现 {len(unmatched_los)} 条学习成果的 module_id 在模块文件中找不到匹配项。涉及的 module_id: {list(unmatched_ids)}")
        # 仅保留能匹配上的
        los_df = los_df[los_df['module_id'].isin(modules_df['module_id'])]

    logging.info(f"数据加载与校验完成。加载了 {len(modules_df)} 个模块和 {len(los_df)} 条学习成果。")

    # 处理 module_weight （可选列）
    # 规则：若不存在则为每个 course 下均分；若存在则按 course 内归一化到和为 1
    if 'module_weight' not in modules_df.columns:
        # 均分：每个 course 下的模块权重 = 1 / 模块数
        modules_df['module_weight'] = modules_df.groupby('course_id')['course_id'].transform(lambda g: 1.0 / len(g))
    else:
        # 尝试数值转换
        try:
            modules_df['module_weight'] = pd.to_numeric(modules_df['module_weight'])
        except ValueError as e:
            raise TypeError(f"模块文件中的 'module_weight' 字段无法转换为数值类型: {e}")
        # 归一化每个 course 下的权重，使其和为 1
        modules_df['module_weight'] = modules_df.groupby('course_id')['module_weight'].transform(lambda s: 0.0 if s.sum()==0 else s / s.sum())

    # 校验：每个 course 的权重和≈1
    sums = modules_df.groupby('course_id')['module_weight'].sum()
    bad_courses = sums[~((sums > 0.999) & (sums < 1.001))]
    if not bad_courses.empty:
        logging.warning(f"发现 {len(bad_courses)} 个 course 的 module_weight 和不为 1，已做归一化：{bad_courses.to_dict()}")
    
    return los_df, modules_df

def _apply_filters(los_df: pd.DataFrame, modules_df: pd.DataFrame,
                   source_system: Optional[str], target_system: Optional[str],
                   source_courses: Optional[list], source_modules: Optional[list],
                   target_courses: Optional[list], target_modules: Optional[list]):
    """
    根据指定的 system/course/module 过滤数据。

    优先级：module > course > system
    """
    initial_modules_count = len(modules_df)
    initial_los_count = len(los_df)

    # 构建过滤条件
    source_mask = pd.Series([False] * len(modules_df), index=modules_df.index)
    target_mask = pd.Series([False] * len(modules_df), index=modules_df.index)

    # 源体系过滤
    if source_system:
        source_mask = modules_df['system'] == source_system.lower()
        # 进一步细化过滤
        if source_courses:
            source_mask &= modules_df['course_id'].isin(source_courses)
        if source_modules:
            source_mask &= modules_df['module_name'].isin(source_modules)

    # 目标体系过滤
    if target_system:
        target_mask = modules_df['system'] == target_system.lower()
        # 进一步细化过滤
        if target_courses:
            target_mask &= modules_df['course_id'].isin(target_courses)
        if target_modules:
            target_mask &= modules_df['module_name'].isin(target_modules)

    # 合并：保留源体系或目标体系的数据（两者至少满足一个）
    filter_mask = source_mask | target_mask

    # 应用模块过滤
    modules_df = modules_df[filter_mask].copy()

    # 根据保留的模块过滤 LOs
    los_df = los_df[los_df['module_id'].isin(modules_df['module_id'])].copy()

    filtered_modules = initial_modules_count - len(modules_df)
    filtered_los = initial_los_count - len(los_df)

    if filtered_modules > 0 or filtered_los > 0:
        logging.info(f"应用 course/module 过滤: 过滤掉 {filtered_modules} 个模块和 {filtered_los} 条学习成果。")
        logging.info(f"保留: {len(modules_df)} 个模块, {len(los_df)} 条学习成果。")

    return los_df, modules_df
