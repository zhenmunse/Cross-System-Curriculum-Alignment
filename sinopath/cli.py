# -*- coding: utf-8 -*-

# 命令行接口模块
# 使用 argparse 定义和解析命令行参数。

import argparse
from . import config

def parse_arguments():

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="SinoPath: 跨教育体系学习成果映射实验框架")

    parser.add_argument(
        '--los',
        type=str,
        default=config.DEFAULT_LOS_PATH,
        help='学习成果数据文件路径 (CSV/XLSX)。'
    )
    parser.add_argument(
        '--modules',
        type=str,
        default=config.DEFAULT_MODULES_PATH,
        help='课程模块数据文件路径 (CSV/XLSX)。'
    )
    parser.add_argument(
        '--source_system',
        type=str,
        required=True,
        help='源教育体系的名称 (例如 "A-level")。'
    )
    parser.add_argument(
        '--target_system',
        type=str,
        required=True,
        help='目标教育体系的名称 (例如 "University")。'
    )
    parser.add_argument(
        '--source_courses',
        type=str,
        default=None,
        help='源体系中要映射的课程列表，逗号分隔 (例如 "Course1,Course2")。留空则使用整个源体系。'
    )
    parser.add_argument(
        '--source_modules',
        type=str,
        default=None,
        help='源体系中要映射的模块列表，逗号分隔 (例如 "Module1,Module2")。留空则使用整个源体系。'
    )
    parser.add_argument(
        '--target_courses',
        type=str,
        default=None,
        help='目标体系中要映射的课程列表，逗号分隔 (例如 "Course3,Course4")。留空则使用整个目标体系。'
    )
    parser.add_argument(
        '--target_modules',
        type=str,
        default=None,
        help='目标体系中要映射的模块列表，逗号分隔 (例如 "Module3,Module4")。留空则使用整个目标体系。'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default=config.DEFAULT_OUTPUT_DIR,
        help='输出结果的目录。'
    )
    parser.add_argument(
        '--embed_method',
        type=str,
        default='tfidf',
        choices=['tfidf', 'hash'],
        help='文本向量化方法（本地）：tfidf 或 hash。默认 tfidf。'
    )
    parser.add_argument(
        '--llm_mode',
        type=str,
        default='none',
        choices=['none', 'rule', 'openai'],
        help='是否启用 LLM/规则微调：none（默认）、rule、openai（占位）。'
    )
    parser.add_argument(
        '--similarity_scaling',
        type=str,
        default=config.SIMILARITY_SCALING,
        choices=['none','power','logistic'],
        help='余弦相似度缩放方式：none/power/logistic，用于提升低相关度的分辨性。'
    )
    parser.add_argument(
        '--scaling_alpha',
        type=float,
        default=config.SCALING_ALPHA,
        help='缩放参数：power 模式下使用 (raw ** alpha 或 raw ** (1/alpha) 取决于数值区间)。'
    )
    parser.add_argument(
        '--top_k_target_lo',
        type=int,
        default=config.TOP_K_TARGET_LO,
        help='每个源 LO 仅保留最高的 k 个目标 LO 相似度，其余置 0。'
    )
    parser.add_argument(
        '--course_agg_strategy',
        type=str,
        default=config.COURSE_AGGREGATION_STRATEGY,
        choices=['weighted_mean','union'],
        help='课程级聚合策略：weighted_mean 或 union 并集提升模式。'
    )
    parser.add_argument(
        '--threshold_exact',
        type=float,
        default=config.THRESHOLD_EXACT,
        help='"Exact" 映射的置信度阈值。'
    )
    parser.add_argument(
        '--threshold_partial',
        type=float,
        default=config.THRESHOLD_PARTIAL,
        help='"Partial" 映射的置信度阈值。'
    )
    parser.add_argument(
        '--threshold_enrich',
        type=float,
        default=config.THRESHOLD_ENRICH,
        help='"Enrichment" 映射的置信度阈值。'
    )
    parser.add_argument(
        '--bridge_fix_rate',
        type=float,
        default=config.BRIDGE_FIX_RATE,
        help='桥梁效率计算中的修复率。'
    )
    parser.add_argument(
        '--aggregate_mode',
        action='store_true',
        help='启用聚合映射模式：将选中的源 courses/modules 聚合为一个整体，与目标整体进行映射。'
    )
    parser.add_argument(
        '--source_aggregate_name',
        type=str,
        default=None,
        help='聚合模式下源体系聚合后的名称（如 "A-level Math Complete"）。'
    )
    parser.add_argument(
        '--target_aggregate_name',
        type=str,
        default=None,
        help='聚合模式下目标体系聚合后的名称（如 "University Calculus Series"）。'
    )

    args = parser.parse_args()
    
    # 将命令行传入的阈值更新到 config 模块
    config.THRESHOLD_EXACT = args.threshold_exact
    config.THRESHOLD_PARTIAL = args.threshold_partial
    config.THRESHOLD_ENRICH = args.threshold_enrich
    config.BRIDGE_FIX_RATE = args.bridge_fix_rate
    
    # 将体系名称统一转换为小写
    args.source_system = args.source_system.lower()
    args.target_system = args.target_system.lower()

    # 解析 course/module 列表参数
    if args.source_courses:
        args.source_courses = [c.strip() for c in args.source_courses.split(',') if c.strip()]
    if args.source_modules:
        args.source_modules = [m.strip() for m in args.source_modules.split(',') if m.strip()]
    if args.target_courses:
        args.target_courses = [c.strip() for c in args.target_courses.split(',') if c.strip()]
    if args.target_modules:
        args.target_modules = [m.strip() for m in args.target_modules.split(',') if m.strip()]

    return args
