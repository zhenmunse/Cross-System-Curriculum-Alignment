# -*- coding: utf-8 -*-
"""
数据加载与校验模块。
负责从 CSV 或 Excel 文件中读取课程模块和学习成果数据，并进行预处理和校验。
"""
import pandas as pd
import os
import logging

def load_data(los_path: str, modules_path: str):
    """
    加载并校验学习成果和模块数据。

    Args:
        los_path (str): 学习成果文件路径 (CSV or XLSX)。
        modules_path (str): 模块文件路径 (CSV or XLSX)。

    Returns:
        tuple: (los_df, modules_df) 两个 Pandas DataFrame。
    """
    # 加载数据
    los_df = _read_file(los_path)
    modules_df = _read_file(modules_path)

    # 校验数据
    los_df, modules_df = _validate_and_clean_data(los_df, modules_df)

    return los_df, modules_df

def _read_file(file_path: str) -> pd.DataFrame:
    """根据文件扩展名读取 CSV 或 Excel 文件。"""
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
    """
    对加载的数据进行校验、清洗和类型转换。
    """
    # 1. 检查模块数据的必需字段
    required_module_cols = ['module_id', 'system', 'module_name']
    for col in required_module_cols:
        if col not in modules_df.columns:
            raise ValueError(f"模块文件缺失必需字段: '{col}'")

    # 2. 检查学习成果数据的必需字段
    required_lo_cols = ['lo_id', 'module_id', 'system', 'description', 'difficulty_level', 'weight']
    for col in required_lo_cols:
        if col not in los_df.columns:
            raise ValueError(f"学习成果文件缺失必需字段: '{col}'")

    # 3. 清洗和类型转换
    # 统一 system 字段为小写，便于匹配
    modules_df['system'] = modules_df['system'].str.strip().str.lower()
    los_df['system'] = los_df['system'].str.strip().str.lower()

    # 转换数据类型
    try:
        los_df['difficulty_level'] = pd.to_numeric(los_df['difficulty_level'])
        los_df['weight'] = pd.to_numeric(los_df['weight'])
    except ValueError as e:
        raise TypeError(f"学习成果文件中的 'difficulty_level' 或 'weight' 字段无法转换为数值类型: {e}")

    # 4. 过滤无效数据
    # 过滤掉描述为空的学习成果
    initial_los_count = len(los_df)
    los_df.dropna(subset=['description'], inplace=True)
    if len(los_df) < initial_los_count:
        logging.warning(f"过滤掉 {initial_los_count - len(los_df)} 条描述为空的学习成果。")
    
    los_df = los_df[los_df['description'].str.strip() != '']

    # 5. 检查关联完整性
    unmatched_los = los_df[~los_df['module_id'].isin(modules_df['module_id'])]
    if not unmatched_los.empty:
        unmatched_ids = unmatched_los['module_id'].unique()
        logging.warning(f"发现 {len(unmatched_los)} 条学习成果的 module_id 在模块文件中找不到匹配项。涉及的 module_id: {list(unmatched_ids)}")
        # 仅保留能匹配上的
        los_df = los_df[los_df['module_id'].isin(modules_df['module_id'])]

    logging.info(f"数据加载与校验完成。加载了 {len(modules_df)} 个模块和 {len(los_df)} 条学习成果。")
    
    return los_df, modules_df
