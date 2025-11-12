# -*- coding: utf-8 -*-
"""
配置模块：统一管理超参数、路径和阈值。
"""
import logging

# --- 随机种子 ---
RANDOM_STATE = 42

# --- 映射算法阈值 ---
THRESHOLD_EXACT = 0.70
THRESHOLD_PARTIAL = 0.50
THRESHOLD_ENRICH = 0.30

# --- 难度修正参数 ---
# 难度取值范围扩大为 0~10：
# 0 = 完全无需前置知识即可学习
# 10 = 通识/系统化教育无法讲授的高度专门内容
DIFFICULTY_MAX = 10
DIFFICULTY_WEIGHT = 0.35  # 用于融合难度修正因子

# --- 相似度缩放与过滤参数（用于提高低值的可分辨性） ---
# 说明：原值可能非常稀疏且偏低，以下参数用于对其进行单调放缩或截断以提升课程级聚合结果。
# SIMILARITY_SCALING: 'none' | 'power' | 'logistic'
SIMILARITY_SCALING = 'power'
# SCALING_ALPHA: 若采用 'power'，在 (0,1) 时提升小值 (例如 0.5 -> 平方根)；=1 不变
SCALING_ALPHA = 0.25
# LOGISTIC_K 与 LOGISTIC_X0: 若采用 'logistic'，则 new = 1/(1+exp(-k*(raw - x0)))
LOGISTIC_K = 8.0
LOGISTIC_X0 = 0.1
# TOP_K_TARGET_LO: 每个源 LO 只保留前 k 个目标 LO（其余视为 0），减少噪声稀释
TOP_K_TARGET_LO = 5

# --- 课程级聚合策略 ---
# 'weighted_mean': 原来的加权平均
# 'union': 概率并集 1 - Π(1 - conf*weight_norm)，在多高质量模块时提高整体课程相关度
COURSE_AGGREGATION_STRATEGY = 'union'

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
