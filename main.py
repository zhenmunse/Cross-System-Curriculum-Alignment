# -*- coding: utf-8 -*-
"""
SinoPath 实验框架主入口。
"""
import sys
import os
import logging

# 将项目根目录添加到 Python 路径中，以便导入 sinopath 包
# 这使得我们可以直接从顶层运行 `python main.py`
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from sinopath.cli import parse_arguments
from sinopath.data_loader import load_data
from sinopath.mapping import perform_mapping
from sinopath.metrics import calculate_all_metrics
from sinopath.visualization import generate_visualizations
from sinopath.reporter import setup_logging, log_run_details, save_results

def main():
    """
    主执行函数。
    """
    # 1. 解析命令行参数
    args = parse_arguments()

    # 2. 设置日志记录
    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(args.out_dir)

    try:
        # 3. 加载和校验数据
        los_df, modules_df = load_data(args.los, args.modules)
        log_run_details(args, len(los_df), len(modules_df))

        # 4. 执行映射算法
        logging.info(f"开始执行从 '{args.source_system}' 到 '{args.target_system}' 的映射...")
        module_mappings = perform_mapping(
            los_df,
            modules_df,
            args.source_system,
            args.target_system,
            embed_method=getattr(args, 'embed_method', 'tfidf'),
            llm_mode=getattr(args, 'llm_mode', 'none')
        )
        logging.info("映射完成。")

        # 5. 计算指标
        logging.info("开始计算指标...")
        metrics_df = calculate_all_metrics(module_mappings, los_df, modules_df, args.source_system, args.target_system)
        logging.info("指标计算完成。")
        logging.info(f"\n--- 指标结果 ---\n{metrics_df.to_string(index=False)}\n----------------")

        # 6. 保存结果到文件
        save_results(module_mappings, metrics_df, args.out_dir, args.source_system, args.target_system)

        # 7. 生成可视化图表
        logging.info("开始生成可视化图表...")
        generate_visualizations(module_mappings, args.out_dir, args.source_system, args.target_system)
        logging.info(f"图表已保存至: {args.out_dir}")

        logging.info("="*50)
        logging.info("SinoPath 实验运行成功完成！")
        logging.info("="*50)

    except Exception as e:
        logging.error(f"实验过程中发生错误: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
