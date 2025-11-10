# -*- coding: utf-8 -*-
"""
报告生成模块。
负责将实验结果（映射表、指标、图表）和运行日志保存到文件。
"""
import os
import logging
import pandas as pd
from datetime import datetime
from . import config

def setup_logging(log_dir: str):
    """配置日志记录器，将日志同时输出到控制台和文件。"""
    log_file = os.path.join(log_dir, 'run.log')
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )

def log_run_details(args, los_count, modules_count):
    """记录本次运行的参数和基本信息。"""
    logging.info("="*50)
    logging.info(f"SinoPath Experiment Run")
    logging.info(f"Timestamp: {datetime.now().isoformat()}")
    logging.info("="*50)
    logging.info("Parameters:")
    for arg, value in vars(args).items():
        logging.info(f"  --{arg}: {value}")
    logging.info("-" * 50)
    logging.info(f"Data Loaded: {modules_count} modules, {los_count} learning outcomes.")
    logging.info(f"Source System: {args.source_system}")
    logging.info(f"Target System: {args.target_system}")
    logging.info("-" * 50)


def save_results(mappings_df: pd.DataFrame, metrics_df: pd.DataFrame, out_dir: str, source_system: str, target_system: str):
    """
    将映射结果和指标保存为 CSV 文件。

    Args:
        mappings_df (pd.DataFrame): 模块映射结果。
        metrics_df (pd.DataFrame): 指标计算结果。
        out_dir (str): 输出目录。
        source_system (str): 源体系名称。
        target_system (str): 目标体系名称。
    """
    # 确保输出目录存在
    os.makedirs(out_dir, exist_ok=True)

    # 1. 保存映射结果
    mappings_filename = f"mappings_{source_system}_to_{target_system}.csv"
    mappings_path = os.path.join(out_dir, mappings_filename)
    mappings_df.to_csv(mappings_path, index=False)
    logging.info(f"映射结果已保存至: {mappings_path}")

    # 2. 保存指标结果
    metrics_filename = f"metrics_{source_system}_to_{target_system}.csv"
    metrics_path = os.path.join(out_dir, metrics_filename)
    metrics_df.to_csv(metrics_path, index=False)
    logging.info(f"指标结果已保存至: {metrics_path}")
