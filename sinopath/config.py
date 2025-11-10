# -*- coding: utf-8 -*-
"""
配置模块：统一管理超参数、路径和阈值。
"""
import logging

# --- 随机种子 ---
RANDOM_STATE = 42

# --- 映射算法阈值 (对应论文 Section IV-B) ---
THRESHOLD_EXACT = 0.85
THRESHOLD_PARTIAL = 0.65
THRESHOLD_ENRICH = 0.45

# --- 难度修正权重 ---
DIFFICULTY_WEIGHT = 0.3

# --- 指标计算参数 (对应论文 Section V-B) ---
BRIDGE_FIX_RATE = 0.5
BRIDGE_CONFIDENCE_RANGE = (0.40, 0.65)

# --- 可视化参数 ---
FIGURE_DPI = 300
FIGURE_FONT_SIZE = 10

# --- 日志配置 ---
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# --- 文件路径 (通常由命令行参数覆盖) ---
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_LOS_PATH = "data/learning_outcomes.csv"
DEFAULT_MODULES_PATH = "data/modules.csv"
