# -*- coding: utf-8 -*-
"""
可视化模块。
使用 matplotlib 生成用于论文的图表，禁止使用 seaborn。
"""
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

    # 1. 生成覆盖度热力图
    _generate_coverage_heatmap(module_mappings, out_dir, source_system, target_system)

    # 2. 生成 Gap 分布图
    _generate_gap_distribution(module_mappings, out_dir, target_system)

    # 3. 生成映射类型饼图
    _generate_pair_type_pie(module_mappings, out_dir)

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

    # 设置标签，X轴标签旋转45度并右对齐以避免重叠
    ax.set_xticklabels(x_labels, rotation=45, ha='left', fontsize=config.FIGURE_FONT_SIZE)
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

def _generate_pair_type_pie(module_mappings: pd.DataFrame, out_dir: str):
    """生成映射关系类型占比饼图。"""
    type_counts = module_mappings['mapping_type'].value_counts()

    if type_counts.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(type_counts, labels=type_counts.index.tolist(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    ax.set_title('Distribution of Mapping Types', fontsize=config.FIGURE_FONT_SIZE + 2)
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, 'pair_type_pie.png')
    plt.savefig(save_path, dpi=config.FIGURE_DPI)
    plt.close(fig)
