# -*- coding: utf-8 -*-
"""
命令行接口模块。
使用 argparse 定义和解析命令行参数。
"""
import argparse
from . import config

def parse_arguments():
    """
    解析命令行参数。
    """
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
        '--out_dir',
        type=str,
        default=config.DEFAULT_OUTPUT_DIR,
        help='输出结果的目录。'
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

    args = parser.parse_args()
    
    # 将命令行传入的阈值更新到 config 模块，以便全局访问
    config.THRESHOLD_EXACT = args.threshold_exact
    config.THRESHOLD_PARTIAL = args.threshold_partial
    config.THRESHOLD_ENRICH = args.threshold_enrich
    config.BRIDGE_FIX_RATE = args.bridge_fix_rate
    
    # 将体系名称统一转换为小写
    args.source_system = args.source_system.lower()
    args.target_system = args.target_system.lower()

    return args
