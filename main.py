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
from sinopath.mapping import perform_mapping, perform_course_mapping, perform_aggregate_mapping
from sinopath.metrics import calculate_all_metrics
from sinopath.visualization import generate_visualizations, generate_course_visualizations, generate_aggregate_visualizations
from sinopath.reporter import setup_logging, log_run_details, save_results
import json

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
        los_df, modules_df = load_data(
            args.los, 
            args.modules,
            source_system=args.source_system,
            target_system=args.target_system,
            source_courses=getattr(args, 'source_courses', None),
            source_modules=getattr(args, 'source_modules', None),
            target_courses=getattr(args, 'target_courses', None),
            target_modules=getattr(args, 'target_modules', None)
        )
        log_run_details(args, len(los_df), len(modules_df))

        # 4. 检查是否启用聚合映射模式
        if getattr(args, 'aggregate_mode', False):
            # 聚合映射模式：将所有选中的 LOs 聚合为整体进行映射
            logging.info(f"启用聚合映射模式: '{args.source_system}' 整体 -> '{args.target_system}' 整体")
            aggregate_result = perform_aggregate_mapping(
                los_df,
                modules_df,
                args.source_system,
                args.target_system,
                source_aggregate_name=getattr(args, 'source_aggregate_name', None),
                target_aggregate_name=getattr(args, 'target_aggregate_name', None),
                embed_method=getattr(args, 'embed_method', 'tfidf'),
                llm_mode=getattr(args, 'llm_mode', 'none'),
                similarity_scaling=getattr(args, 'similarity_scaling', None),
                scaling_alpha=getattr(args, 'scaling_alpha', None),
                top_k_target_lo=getattr(args, 'top_k_target_lo', None)
            )
            logging.info("聚合映射完成。")
            
            # 打印聚合结果
            logging.info(f"\n{'='*50}")
            logging.info("聚合映射结果")
            logging.info(f"{'='*50}")
            logging.info(f"源聚合: {aggregate_result['source_aggregate']}")
            logging.info(f"  - 包含 {len(aggregate_result['source_courses'])} 个课程: {', '.join(aggregate_result['source_courses'])}")
            logging.info(f"  - 包含 {len(aggregate_result['source_modules'])} 个模块")
            logging.info(f"  - 包含 {aggregate_result['source_lo_count']} 个学习成果")
            logging.info(f"目标聚合: {aggregate_result['target_aggregate']}")
            logging.info(f"  - 包含 {len(aggregate_result['target_courses'])} 个课程: {', '.join(aggregate_result['target_courses'])}")
            logging.info(f"  - 包含 {len(aggregate_result['target_modules'])} 个模块")
            logging.info(f"  - 包含 {aggregate_result['target_lo_count']} 个学习成果")
            logging.info(f"{'-'*50}")
            logging.info(f"整体置信度: {aggregate_result['overall_confidence']:.4f}")
            logging.info(f"映射类型: {aggregate_result['mapping_type']}")
            logging.info(f"源体系覆盖率: {aggregate_result['source_coverage_rate']:.2%}")
            logging.info(f"目标体系覆盖率: {aggregate_result['target_coverage_rate']:.2%}")
            logging.info(f"{'-'*50}")
            logging.info(f"LO 对分布 (共 {aggregate_result['total_lo_pairs']} 对):")
            logging.info(f"  - Exact: {aggregate_result['exact_pairs']}")
            logging.info(f"  - Partial: {aggregate_result['partial_pairs']}")
            logging.info(f"  - Enrichment: {aggregate_result['enrichment_pairs']}")
            logging.info(f"  - Gap: {aggregate_result['gap_pairs']}")
            logging.info(f"{'='*50}\n")
            
            # 保存聚合结果到 JSON
            aggregate_output_path = os.path.join(args.out_dir, f'aggregate_mapping_{args.source_system}_to_{args.target_system}.json')
            with open(aggregate_output_path, 'w', encoding='utf-8') as f:
                json.dump(aggregate_result, f, indent=2, ensure_ascii=False)
            logging.info(f"聚合映射结果已保存至: {aggregate_output_path}")
            
            # 生成聚合模式的可视化图表
            logging.info("开始生成聚合映射可视化图表...")
            generate_aggregate_visualizations(aggregate_result, args.out_dir)
            logging.info(f"聚合映射图表已保存至: {args.out_dir}")
            logging.info("  - aggregate_pair_type_pie.png: LO对类型分布饼图")
            logging.info("  - aggregate_coverage_comparison.png: 覆盖率对比柱状图")
            logging.info("  - aggregate_summary.csv: 整体指标摘要表格")
            
        else:
            # 常规映射模式：一对一映射
            logging.info(f"开始执行从 '{args.source_system}' 到 '{args.target_system}' 的映射...")
            module_mappings = perform_mapping(
                los_df,
                modules_df,
                args.source_system,
                args.target_system,
                embed_method=getattr(args, 'embed_method', 'tfidf'),
                llm_mode=getattr(args, 'llm_mode', 'none'),
                similarity_scaling=getattr(args, 'similarity_scaling', None),
                scaling_alpha=getattr(args, 'scaling_alpha', None),
                top_k_target_lo=getattr(args, 'top_k_target_lo', None)
            )
            logging.info("映射完成。")

            # 4.1 课程级聚合映射
            course_mappings = perform_course_mapping(module_mappings, modules_df, strategy=getattr(args, 'course_agg_strategy', None))
            if course_mappings.empty:
                logging.info("课程级映射结果为空。")
            else:
                logging.info(f"课程级映射生成，共 {len(course_mappings)} 条。")

            # 5. 计算指标
            logging.info("开始计算指标...")
            metrics_df = calculate_all_metrics(module_mappings, los_df, modules_df, args.source_system, args.target_system)
            logging.info("指标计算完成。")
            logging.info(f"\n--- 指标结果 ---\n{metrics_df.to_string(index=False)}\n----------------")

            # 6. 保存结果到文件
            save_results(module_mappings, metrics_df, args.out_dir, args.source_system, args.target_system, course_mappings_df=course_mappings)

            # 7. 生成可视化图表
            logging.info("开始生成可视化图表...")
            generate_visualizations(module_mappings, args.out_dir, args.source_system, args.target_system)
            generate_course_visualizations(course_mappings, args.out_dir, args.source_system, args.target_system)
            logging.info(f"图表已保存至: {args.out_dir}")

        logging.info("="*50)
        logging.info("SinoPath 实验运行成功完成！")
        logging.info("="*50)

    except Exception as e:
        logging.error(f"实验过程中发生错误: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
