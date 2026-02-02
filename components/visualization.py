# -*- coding: utf-8 -*-

# 可视化模块
# 使用 matplotlib 生成用于论文的图表

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from . import config

def generate_visualizations(module_mappings: pd.DataFrame, out_dir: str, source_system: str, target_system: str):
    """
    生成所有定义的可视化图表。

    Args:
        module_mappings (pd.DataFrame): 模块映射结果。
        out_dir (str): 输出目录。
        source_system (str): 源体系名称。
        target_system (str): 目标体系名称。
    """
    # 设置 matplotlib 字体：中文使用宋体，西文使用 Times New Roman
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 宋体用于中文
    plt.rcParams['font.serif'] = ['Times New Roman']  # Times New Roman 用于西文
    plt.rcParams['font.family'] = ['serif', 'sans-serif']  # 优先使用 serif 字体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

    # 生成覆盖度热力图
    _generate_coverage_heatmap(module_mappings, out_dir, source_system, target_system)

    # 生成 Gap 分布图
    _generate_gap_distribution(module_mappings, out_dir, target_system)

    # 生成映射类型饼图
    _generate_pair_type_pie(module_mappings, out_dir)


def generate_course_visualizations(course_mappings: pd.DataFrame, out_dir: str, source_system: str, target_system: str):
    """生成课程级的可视化图表。"""
    if course_mappings.empty:
        return
    _generate_course_heatmap(course_mappings, out_dir, source_system, target_system)
    _generate_course_gap_distribution(course_mappings, out_dir, target_system)
    _generate_course_pair_type_pie(course_mappings, out_dir)

def _generate_coverage_heatmap(module_mappings: pd.DataFrame, out_dir: str, source_system: str, target_system: str):
    """生成模块间映射置信度的热力图。"""
    pivot_table = module_mappings.pivot_table(
        index='source_module_name', 
        columns='target_module_name', 
        values='confidence'
    )
    
    if pivot_table.empty:
        return

    # 文字换行处理函数
    def wrap_text(text, max_length=15):
        """将长文本按空格或连字符分割并换行"""
        if len(text) <= max_length:
            return text
        words = text.replace('-', '- ').split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_len = len(word)
            if current_length + word_len <= max_length:
                current_line.append(word)
                current_length += word_len + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_len + 1
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    # 处理标签文字
    x_labels = [wrap_text(str(col)) for col in pivot_table.columns]
    y_labels = [wrap_text(str(idx)) for idx in pivot_table.index]
    
    # 根据标签数量动态调整图像大小
    fig_width = max(10, len(pivot_table.columns) * 1.5)
    fig_height = max(8, len(pivot_table.index) * 1.2)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cax = ax.matshow(pivot_table, cmap='viridis')
    
    # 添加颜色条
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=config.FIGURE_FONT_SIZE)

    # 设置刻度位置
    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_yticks(np.arange(len(pivot_table.index)))

    # 设置标签，X轴标签旋转45度
    ax.set_xticklabels(x_labels, rotation=45, ha='center', fontsize=config.FIGURE_FONT_SIZE)
    ax.set_yticklabels(y_labels, fontsize=config.FIGURE_FONT_SIZE)
    
    # 将X轴刻度移到底部
    ax.xaxis.set_ticks_position('bottom')

    ax.set_title(f'Confidence Heatmap: {source_system} to {target_system}', 
                 fontsize=config.FIGURE_FONT_SIZE + 2, pad=20)
    ax.set_xlabel(f'Target Modules ({target_system})', fontsize=config.FIGURE_FONT_SIZE, labelpad=10)
    ax.set_ylabel(f'Source Modules ({source_system})', fontsize=config.FIGURE_FONT_SIZE, labelpad=10)

    # 使用tight_layout并增加边距，确保所有文字都可见
    plt.tight_layout(pad=2.0)
    save_path = os.path.join(out_dir, f'coverage_heatmap_{source_system}_to_{target_system}.png')
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

def _generate_course_heatmap(course_mappings: pd.DataFrame, out_dir: str, source_system: str, target_system: str):
    pivot_table = course_mappings.pivot_table(index='source_course', columns='target_course', values='confidence')
    if pivot_table.empty:
        return
    fig_width = max(8, len(pivot_table.columns) * 1.2)
    fig_height = max(6, len(pivot_table.index) * 1.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cax = ax.matshow(pivot_table, cmap='viridis')
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=config.FIGURE_FONT_SIZE)
    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_xticklabels(list(pivot_table.columns), rotation=45, ha='center', fontsize=config.FIGURE_FONT_SIZE)
    ax.set_yticklabels(list(pivot_table.index), fontsize=config.FIGURE_FONT_SIZE)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_title(f'Course Confidence Heatmap: {source_system} to {target_system}', fontsize=config.FIGURE_FONT_SIZE + 2, pad=20)
    ax.set_xlabel(f'Target Courses ({target_system})', fontsize=config.FIGURE_FONT_SIZE, labelpad=10)
    ax.set_ylabel(f'Source Courses ({source_system})', fontsize=config.FIGURE_FONT_SIZE, labelpad=10)
    plt.tight_layout(pad=2.0)
    save_path = os.path.join(out_dir, f'course_heatmap_{source_system}_to_{target_system}.png')
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

def _generate_gap_distribution(module_mappings: pd.DataFrame, out_dir: str, target_system: str):
    """按 target_module 统计 Gap 数。"""
    # 找到每个目标模块的最佳映射
    best_mappings = module_mappings.loc[module_mappings.groupby('target_module_name')['confidence'].idxmax()]
    gap_counts = best_mappings[best_mappings['mapping_type'] == 'Gap'].groupby('target_module_name').size()

    if gap_counts.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    gap_counts.plot(kind='bar', ax=ax)

    ax.set_title(f'Gap Distribution in Target System ({target_system})', fontsize=config.FIGURE_FONT_SIZE + 2)
    ax.set_xlabel('Target Module', fontsize=config.FIGURE_FONT_SIZE)
    ax.set_ylabel('Number of Gaps (Best mapping is "Gap")', fontsize=config.FIGURE_FONT_SIZE)
    plt.xticks(rotation=45, ha='right', fontsize=config.FIGURE_FONT_SIZE)
    plt.yticks(fontsize=config.FIGURE_FONT_SIZE)
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, f'gap_distribution_{target_system}.png')
    plt.savefig(save_path, dpi=config.FIGURE_DPI)
    plt.close(fig)

def _generate_course_gap_distribution(course_mappings: pd.DataFrame, out_dir: str, target_system: str):
    best = course_mappings.loc[course_mappings.groupby('target_course')['confidence'].idxmax()]
    gap_counts = best[best['mapping_type'] == 'Gap'].groupby('target_course').size()
    if gap_counts.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    gap_counts.plot(kind='bar', ax=ax)
    ax.set_title(f'Gap Distribution in Target Courses ({target_system})', fontsize=config.FIGURE_FONT_SIZE + 2)
    ax.set_xlabel('Target Course', fontsize=config.FIGURE_FONT_SIZE)
    ax.set_ylabel('Number of Gaps (Best mapping is "Gap")', fontsize=config.FIGURE_FONT_SIZE)
    plt.xticks(rotation=45, ha='right', fontsize=config.FIGURE_FONT_SIZE)
    plt.yticks(fontsize=config.FIGURE_FONT_SIZE)
    plt.tight_layout()
    save_path = os.path.join(out_dir, f'course_gap_distribution_{target_system}.png')
    plt.savefig(save_path, dpi=config.FIGURE_DPI)
    plt.close(fig)

def _generate_pair_type_pie(module_mappings: pd.DataFrame, out_dir: str):
    # 生成映射关系类型占比饼图
    type_counts = module_mappings['mapping_type'].value_counts()

    if type_counts.empty:
        return

    # 导出详细数据 CSV：列出所有映射对，按准确度倒序（Exact → Partial → Enrichment → Gap）
    type_order = ['Exact', 'Partial', 'Enrichment', 'Gap']
    # 创建类型排序的映射
    type_rank = {t: i for i, t in enumerate(type_order)}
    # 添加排序列
    mappings_with_rank = module_mappings.copy()
    mappings_with_rank['type_rank'] = mappings_with_rank['mapping_type'].map(type_rank)
    # 按类型排序，然后按置信度倒序
    mappings_sorted = mappings_with_rank.sort_values(['type_rank', 'confidence'], ascending=[True, False])
    # 删除辅助列
    mappings_sorted = mappings_sorted.drop(columns=['type_rank'])
    
    csv_path = os.path.join(out_dir, 'pair_type_distribution.csv')
    mappings_sorted.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 按数量降序，利于阅读（仅用于图表显示）
    type_counts = type_counts.sort_values(ascending=False)
    labels = type_counts.index.tolist()
    values = type_counts.to_numpy()

    fig, ax = plt.subplots(figsize=(8, 8))

    # 仅绘制百分比，隐藏扇区内标签，避免堆叠；小于1%不显示
    def _fmt(p: float) -> str:
        return f"{p:.1f}%" if p >= 1 else ''

    pie_ret = ax.pie(
        values,
        labels=None,
        autopct=_fmt,
        startangle=90,
        pctdistance=0.7,
        wedgeprops={'linewidth': 0.6, 'edgecolor': 'white'}
    )
    wedges = pie_ret[0]

    # 让饼图为正圆
    ax.axis('equal')

    # 右侧图例：显示 标签 + 计数 + 百分比
    total = float(values.sum())
    legend_labels = [f"{lab} ({int(cnt)}, {cnt / total * 100:.1f}%)" for lab, cnt in zip(labels, values)]
    ax.legend(wedges, legend_labels, title='Mapping Type', loc='center left', bbox_to_anchor=(1.0, 0.5),
              fontsize=config.FIGURE_FONT_SIZE, title_fontsize=config.FIGURE_FONT_SIZE)

    ax.set_title('Distribution of Mapping Types', fontsize=config.FIGURE_FONT_SIZE + 2)

    plt.tight_layout()
    save_path = os.path.join(out_dir, 'pair_type_pie.png')
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

def _generate_course_pair_type_pie(course_mappings: pd.DataFrame, out_dir: str):
    type_counts = course_mappings['mapping_type'].value_counts()
    if type_counts.empty:
        return

    # 导出详细数据 CSV：列出所有课程映射对，按准确度倒序（Exact → Partial → Enrichment → Gap）
    type_order = ['Exact', 'Partial', 'Enrichment', 'Gap']
    # 创建类型排序的映射
    type_rank = {t: i for i, t in enumerate(type_order)}
    # 添加排序列
    mappings_with_rank = course_mappings.copy()
    mappings_with_rank['type_rank'] = mappings_with_rank['mapping_type'].map(type_rank)
    # 按类型排序，然后按置信度倒序
    mappings_sorted = mappings_with_rank.sort_values(['type_rank', 'confidence'], ascending=[True, False])
    # 删除辅助列
    mappings_sorted = mappings_sorted.drop(columns=['type_rank'])
    
    csv_path = os.path.join(out_dir, 'course_pair_type_distribution.csv')
    mappings_sorted.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 按数量降序，利于阅读（仅用于图表显示）
    type_counts = type_counts.sort_values(ascending=False)
    labels = type_counts.index.tolist()
    values = type_counts.to_numpy()

    fig, ax = plt.subplots(figsize=(8, 8))

    def _fmt(p: float) -> str:
        return f"{p:.1f}%" if p >= 1 else ''

    pie_ret = ax.pie(
        values,
        labels=None,
        autopct=_fmt,
        startangle=90,
        pctdistance=0.7,
        wedgeprops={'linewidth': 0.6, 'edgecolor': 'white'}
    )
    wedges = pie_ret[0]

    ax.axis('equal')

    total = float(values.sum())
    legend_labels = [f"{lab} ({int(cnt)}, {cnt / total * 100:.1f}%)" for lab, cnt in zip(labels, values)]
    ax.legend(wedges, legend_labels, title='Course Mapping Type', loc='center left', bbox_to_anchor=(1.0, 0.5),
              fontsize=config.FIGURE_FONT_SIZE, title_fontsize=config.FIGURE_FONT_SIZE)

    ax.set_title('Distribution of Course Mapping Types', fontsize=config.FIGURE_FONT_SIZE + 2)
    plt.tight_layout()
    save_path = os.path.join(out_dir, 'course_pair_type_pie.png')
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

def generate_aggregate_visualizations(aggregate_result: dict, out_dir: str):
    """
    生成聚合映射模式的可视化图表。

    Args:
        aggregate_result: perform_aggregate_mapping 返回的聚合结果字典
        out_dir: 输出目录
    """
    # 设置 matplotlib 字体
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.family'] = ['serif', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    source_name = aggregate_result['source_aggregate']
    target_name = aggregate_result['target_aggregate']

    # 1. LO 对类型分布饼图
    _generate_aggregate_pair_type_pie(aggregate_result, out_dir)

    # 2. 覆盖率对比柱状图
    _generate_aggregate_coverage_bar(aggregate_result, out_dir, source_name, target_name)

    # 3. 整体指标摘要 CSV
    _generate_aggregate_summary_csv(aggregate_result, out_dir, source_name, target_name)

def _generate_aggregate_pair_type_pie(aggregate_result: dict, out_dir: str):
    """生成聚合模式的 LO 对类型分布饼图。"""
    # 导出详细的 LO 对数据 CSV：按准确度倒序（Exact → Partial → Enrichment → Gap）
    if 'lo_pairs' in aggregate_result and aggregate_result['lo_pairs']:
        lo_pairs_df = pd.DataFrame(aggregate_result['lo_pairs'])
        
        # 按类型和置信度排序
        type_order = ['Exact', 'Partial', 'Enrichment', 'Gap']
        type_rank = {t: i for i, t in enumerate(type_order)}
        lo_pairs_df['type_rank'] = lo_pairs_df['mapping_type'].map(type_rank)
        lo_pairs_sorted = lo_pairs_df.sort_values(['type_rank', 'confidence'], ascending=[True, False])
        lo_pairs_sorted = lo_pairs_sorted.drop(columns=['type_rank'])
        
        csv_path = os.path.join(out_dir, 'aggregate_pair_type_distribution.csv')
        lo_pairs_sorted.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 提取统计数据
    data = {
        'Exact': aggregate_result['exact_pairs'],
        'Partial': aggregate_result['partial_pairs'],
        'Enrichment': aggregate_result['enrichment_pairs'],
        'Gap': aggregate_result['gap_pairs']
    }
    
    # 过滤掉数量为 0 的类型
    data = {k: v for k, v in data.items() if v > 0}
    
    if not data:
        return

    labels = list(data.keys())
    values = np.array(list(data.values()))
    
    fig, ax = plt.subplots(figsize=(8, 8))

    def _fmt(p: float) -> str:
        return f"{p:.1f}%" if p >= 1 else ''

    pie_ret = ax.pie(
        values,
        labels=None,
        autopct=_fmt,
        startangle=90,
        pctdistance=0.7,
        wedgeprops={'linewidth': 0.6, 'edgecolor': 'white'},
        colors=['#2ecc71', '#3498db', '#f39c12', '#e74c3c']  # Exact, Partial, Enrichment, Gap
    )
    wedges = pie_ret[0]

    ax.axis('equal')

    total = float(np.sum(values))
    legend_labels = [f"{lab} ({int(cnt)}, {cnt / total * 100:.1f}%)" for lab, cnt in zip(labels, values)]
    ax.legend(wedges, legend_labels, title='LO Pair Types', loc='center left', bbox_to_anchor=(1.0, 0.5),
              fontsize=config.FIGURE_FONT_SIZE, title_fontsize=config.FIGURE_FONT_SIZE)

    ax.set_title('Aggregate Mapping: LO Pair Type Distribution', fontsize=config.FIGURE_FONT_SIZE + 2)
    plt.tight_layout()
    save_path = os.path.join(out_dir, 'aggregate_pair_type_pie.png')
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

def _generate_aggregate_coverage_bar(aggregate_result: dict, out_dir: str, source_name: str, target_name: str):
    # 生成聚合模式的覆盖率对比柱状图
    categories = ['Source Coverage', 'Target Coverage']
    coverage_rates = [
        aggregate_result['source_coverage_rate'] * 100,
        aggregate_result['target_coverage_rate'] * 100
    ]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(categories, coverage_rates, color=['#3498db', '#e74c3c'], width=0.5)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=config.FIGURE_FONT_SIZE)
    
    ax.set_ylabel('Coverage Rate (%)', fontsize=config.FIGURE_FONT_SIZE)
    ax.set_title('Aggregate Mapping: Coverage Rate Comparison', fontsize=config.FIGURE_FONT_SIZE + 2)
    ax.set_ylim(0, 110)  # 留出标签空间
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.xticks(fontsize=config.FIGURE_FONT_SIZE)
    plt.yticks(fontsize=config.FIGURE_FONT_SIZE)
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, 'aggregate_coverage_comparison.png')
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

def _generate_aggregate_summary_csv(aggregate_result: dict, out_dir: str, source_name: str, target_name: str):
    # 生成聚合映射的整体指标摘要 CSV 文件
    import csv
    
    save_path = os.path.join(out_dir, 'aggregate_summary.csv')
    
    with open(save_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        
        # 写入标题
        writer.writerow(['Aggregate Mapping Summary'])
        writer.writerow([])  # 空行
        
        # 源体系信息
        writer.writerow(['Source Aggregate', source_name])
        writer.writerow(['Source Courses Count', len(aggregate_result['source_courses'])])
        writer.writerow(['Source Modules Count', len(aggregate_result['source_modules'])])
        writer.writerow(['Source LOs Count', aggregate_result['source_lo_count']])
        writer.writerow(['Source Courses', ', '.join(aggregate_result['source_courses'])])
        writer.writerow([])  # 空行
        
        # 目标体系信息
        writer.writerow(['Target Aggregate', target_name])
        writer.writerow(['Target Courses Count', len(aggregate_result['target_courses'])])
        writer.writerow(['Target Modules Count', len(aggregate_result['target_modules'])])
        writer.writerow(['Target LOs Count', aggregate_result['target_lo_count']])
        writer.writerow(['Target Courses', ', '.join(aggregate_result['target_courses'])])
        writer.writerow([])  # 空行
        
        # 核心指标
        writer.writerow(['Overall Confidence', f"{aggregate_result['overall_confidence']:.4f}"])
        writer.writerow(['Mapping Type', aggregate_result['mapping_type']])
        writer.writerow([])  # 空行
        
        # 覆盖率
        writer.writerow(['Source Coverage Rate', f"{aggregate_result['source_coverage_rate'] * 100:.2f}%"])
        writer.writerow(['Target Coverage Rate', f"{aggregate_result['target_coverage_rate'] * 100:.2f}%"])
        writer.writerow([])  # 空行
        
        # LO 对分布
        writer.writerow(['Total LO Pairs', aggregate_result['total_lo_pairs']])
        writer.writerow(['Exact Pairs', aggregate_result['exact_pairs']])
        writer.writerow(['Partial Pairs', aggregate_result['partial_pairs']])
        writer.writerow(['Enrichment Pairs', aggregate_result['enrichment_pairs']])
        writer.writerow(['Gap Pairs', aggregate_result['gap_pairs']])

