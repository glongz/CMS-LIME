#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成期刊论文示意图：因果一致性扰动机制

展示Primitive Mask、有向通道图、因果闭包、扰动算子和后处理的完整流程
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
from matplotlib import cm
import seaborn as sns
import os
from torchvision import transforms

# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字体也使用类似Times New Roman的样式
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# 设置全局参数
DPI = 300
FIG_SIZE = (14, 10)  # 适合期刊的宽高比


def create_main_pipeline_figure(save_individual=True, save_dir='pictures/pipeline'):
    """
    创建主流程图：展示从原始数据到扰动后数据的完整pipeline
    
    Parameters:
    -----------
    save_individual : bool
        是否单独保存每个子图
    save_dir : str
        保存目录路径
    """
    # 创建保存目录
    if save_individual:
        os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 11), dpi=DPI)
    # 重新设计布局：4行，让图a和图h都占据整行
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.25,
                          height_ratios=[1.2, 1, 1, 1.2], width_ratios=[1, 1, 1, 1])
    
    # 优化的颜色方案（更专业、更协调）
    colors = {
        'original': '#1E88E5',      # 明亮蓝色 - 原始数据
        'mask': '#7B1FA2',          # 深紫色 - Mask
        'graph': '#FB8C00',         # 明亮橙色 - 图
        'closure': '#D32F2F',       # 深红色 - 闭包
        'perturb': '#43A047',       # 绿色 - 扰动
        'postprocess': '#E53935',   # 红色 - 后处理
        'output': '#00897B',        # 青绿色 - 输出
        'parent': '#FFB74D',        # 浅橙色 - 父节点
        'child': '#81C784'          # 浅绿色 - 子节点
    }
    
    # ========== 步骤1: 原始EEG数据 ==========
    ax1 = fig.add_subplot(gs[0, :])  # 占据整行，和图h一样大
    ax1.set_title('(a) Original EEG Signal\n$\\mathbf{X} \\in \\mathbb{R}^{C \\times T}$', 
                  fontsize=16, fontweight='bold', pad=15)
    
    # 绘制EEG数据矩阵可视化
    # 设置固定随机种子以确保可重复性
    np.random.seed(42)
    C, T = 4, 12  # 缩短时间长度，使图更简洁
    eeg_data = np.random.randn(C, T)
    # 确保每个通道显示为一行
    # 使用暖黄色colormap（YlOrBr - 黄到橙棕色）
    im1 = ax1.imshow(eeg_data, aspect='auto', cmap='YlOrBr', 
                     vmin=-2, vmax=2, interpolation='nearest', origin='upper')
    
    # 添加通道和时间轴标签
    ax1.set_ylabel('Channels $c$', fontsize=14, fontweight='bold')
    # 将Time标签移到时间轴末端（图内右下方）
    ax1.text(T-1, -0.2, 'Time $t$', fontsize=14, fontweight='bold', ha='right', va='top')
    # 设置正确的tick位置（对应数组索引），更简洁的刻度
    ax1.set_xticks(np.linspace(0, T-1, 4))
    ax1.set_xticklabels(['1', f'{T//3}', f'{2*T//3}', f'{T}'])
    ax1.set_yticks(range(C))
    ax1.set_yticklabels([f'$c_{i+1}$' for i in range(C)])
    
    # 添加颜色条
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Amplitude', fontsize=13)
    cbar1.ax.tick_params(labelsize=12)
    
    # 单独保存子图a
    if save_individual:
        fig_a = plt.figure(figsize=(12, 3), dpi=DPI)
        ax_a = fig_a.add_subplot(111)
        ax_a.set_title('(a) Original EEG Signal\n$\\mathbf{X} \\in \\mathbb{R}^{C \\times T}$', 
                      fontsize=16, fontweight='bold', pad=15)
        im_a = ax_a.imshow(eeg_data, aspect='auto', cmap='YlOrBr', 
                         vmin=-2, vmax=2, interpolation='nearest', origin='upper')
        ax_a.set_ylabel('Channels $c$', fontsize=14, fontweight='bold')
        # 将Time标签移到时间轴末端（图内右下方）
        ax_a.text(T-1, -0.2, 'Time $t$', fontsize=14, fontweight='bold', ha='right', va='top')
        ax_a.set_xticks(np.linspace(0, T-1, 4))
        ax_a.set_xticklabels(['1', f'{T//3}', f'{2*T//3}', f'{T}'])
        ax_a.set_yticks(range(C))
        ax_a.set_yticklabels([f'$c_{i+1}$' for i in range(C)])
        cbar_a = plt.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
        cbar_a.set_label('Amplitude', fontsize=13)
        cbar_a.ax.tick_params(labelsize=12)
        fig_a.savefig(os.path.join(save_dir, 'pipeline_a.pdf'), dpi=DPI, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        fig_a.savefig(os.path.join(save_dir, 'pipeline_a.png'), dpi=DPI, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        plt.close(fig_a)
    
    # ========== 步骤2: Primitive Mask ==========
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title('(b) Primitive Mask\n$\\mathrm{mask}(u) = \\mathcal{C}_u \\times [t_u^{\\mathrm{start}}, t_u^{\\mathrm{end}}]$',
                  fontsize=14, fontweight='bold', pad=12)
    
    # 绘制mask区域（使用类似e图的风格：YlOrRd colormap + 黑色边框）
    mask_data = np.zeros((C, T))
    mask_channels = [1, 2]  # C_u = {2, 3}
    mask_time = [3, 8]      # [t_start, t_end] - 调整以适应新的时间长度
    
    # 创建mask强度图（类似e图的扰动强度图）
    for c in mask_channels:
        mask_data[c, mask_time[0]:mask_time[1]] = 0.6  # 设置mask强度
    
    im2 = ax2.imshow(mask_data, aspect='auto', cmap='YlOrRd', 
                     vmin=0, vmax=0.9, interpolation='nearest', origin='upper')
    
    # 高亮mask区域（使用黑色边框，类似e图）
    for c in mask_channels:
        rect = Rectangle((mask_time[0]-0.5, c-0.5), 
                       mask_time[1]-mask_time[0], 1,
                       linewidth=2, edgecolor='#333333', 
                       facecolor='none', linestyle='--', alpha=0.8)
        ax2.add_patch(rect)
    
    ax2.set_ylabel('Channels $c$', fontsize=13)
    # 将Time标签移到时间轴末端（图内右下方）
    ax2.text(T-1, -0.2, 'Time $t$', fontsize=13, ha='right', va='top')
    ax2.set_xticks([0, T//2, T-1])
    ax2.set_xticklabels(['1', f'{T//2}', f'{T}'])
    ax2.set_yticks(range(C))
    ax2.set_yticklabels([f'$c_{i+1}$' for i in range(C)])
    
    # 添加标注（使用更柔和的颜色）
    ax2.text(mask_time[0]+2, mask_channels[0]-0.3, 
            f'$\\mathcal{{C}}_u = \\{{c_2, c_3\\}}$', 
            fontsize=11, color='#333333', fontweight='bold')
    ax2.text(mask_time[0]+2, mask_channels[1]+0.3, 
            f'$[{mask_time[0]}, {mask_time[1]}]$', 
            fontsize=11, color='#333333', fontweight='bold')
    
    # 添加颜色条
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Mask Strength', fontsize=12)
    cbar2.ax.tick_params(labelsize=11)
    
    # 单独保存子图b
    if save_individual:
        fig_b = plt.figure(figsize=(4, 3), dpi=DPI)
        ax_b = fig_b.add_subplot(111)
        ax_b.set_title('(b) Primitive Mask\n$\\mathrm{mask}(u) = \\mathcal{C}_u \\times [t_u^{\\mathrm{start}}, t_u^{\\mathrm{end}}]$',
                      fontsize=14, fontweight='bold', pad=12)
        # 创建mask强度图（类似e图的风格）
        mask_data_save = np.zeros((C, T))
        for c_save in mask_channels:
            mask_data_save[c_save, mask_time[0]:mask_time[1]] = 0.6
        
        im_b = ax_b.imshow(mask_data_save, aspect='auto', cmap='YlOrRd', 
                         vmin=0, vmax=0.9, interpolation='nearest', origin='upper')
        for c in mask_channels:
            rect_b = Rectangle((mask_time[0]-0.5, c-0.5), 
                           mask_time[1]-mask_time[0], 1,
                           linewidth=2, edgecolor='#333333', 
                           facecolor='none', linestyle='--', alpha=0.8)
            ax_b.add_patch(rect_b)
        ax_b.set_ylabel('Channels $c$', fontsize=13)
        # 将Time标签移到时间轴末端（图内右下方）
        ax_b.text(T-1, -0.2, 'Time $t$', fontsize=13, ha='right', va='top')
        ax_b.set_xticks([0, T//2, T-1])
        ax_b.set_xticklabels(['1', f'{T//2}', f'{T}'])
        ax_b.set_yticks(range(C))
        ax_b.set_yticklabels([f'$c_{i+1}$' for i in range(C)])
        ax_b.text(mask_time[0]+2, mask_channels[0]-0.3, 
                f'$\\mathcal{{C}}_u = \\{{c_2, c_3\\}}$', 
                fontsize=11, color='#333333', fontweight='bold')
        ax_b.text(mask_time[0]+2, mask_channels[1]+0.3, 
                f'$[{mask_time[0]}, {mask_time[1]}]$', 
                fontsize=11, color='#333333', fontweight='bold')
        # 添加颜色条
        cbar_b = plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
        cbar_b.set_label('Mask Strength', fontsize=12)
        cbar_b.ax.tick_params(labelsize=11)
        fig_b.savefig(os.path.join(save_dir, 'pipeline_b.pdf'), dpi=DPI, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        fig_b.savefig(os.path.join(save_dir, 'pipeline_b.png'), dpi=DPI, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        plt.close(fig_b)
    
    # ========== 步骤3: 有向通道图 ==========
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title('(c) Directed Channel Graph\n$\\mathcal{G} = (\\mathcal{V}, \\mathcal{E})$',
                  fontsize=14, fontweight='bold', pad=12)
    ax3.set_xlim(-0.5, 3.5)
    ax3.set_ylim(-0.5, 3.5)
    ax3.axis('off')
    
    # 定义节点位置
    node_pos = {
        0: (0.5, 2.5),
        1: (2.5, 2.5),
        2: (0.5, 0.5),
        3: (2.5, 0.5)
    }
    
    # 绘制有向边（因果关系）
    edges = [(0, 1, 1), (0, 2, 1), (1, 3, 1), (2, 3, 1)]  # (source, target, lag)
    
    for src, tgt, lag in edges:
        x1, y1 = node_pos[src]
        x2, y2 = node_pos[tgt]
        
        # 绘制箭头
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=20,
                               color=colors['graph'], linewidth=2,
                               connectionstyle='arc3,rad=0.1')
        ax3.add_patch(arrow)
        
        # 添加滞后标签
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        ax3.text(mid_x+0.1, mid_y+0.1, f'$\\ell={lag}$', 
                fontsize=11, color=colors['graph'], 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 绘制节点
    for i, (x, y) in node_pos.items():
        circle = Circle((x, y), 0.3, facecolor=colors['graph'], 
                       fill=True, alpha=0.7, zorder=10)
        ax3.add_patch(circle)
        ax3.text(x, y, f'$c_{i+1}$', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
    
    # 添加图例说明
    ax3.text(1.5, -0.2, 'Learned via Granger/NOTEARS', 
            ha='center', fontsize=11, style='italic')
    
    # 单独保存子图c
    if save_individual:
        fig_c = plt.figure(figsize=(4, 3), dpi=DPI)
        ax_c = fig_c.add_subplot(111)
        ax_c.set_title('(c) Directed Channel Graph\n$\\mathcal{G} = (\\mathcal{V}, \\mathcal{E})$',
                      fontsize=14, fontweight='bold', pad=12)
        ax_c.set_xlim(-0.5, 3.5)
        ax_c.set_ylim(-0.5, 3.5)
        ax_c.axis('off')
        for src, tgt, lag in edges:
            x1, y1 = node_pos[src]
            x2, y2 = node_pos[tgt]
            arrow_c = FancyArrowPatch((x1, y1), (x2, y2),
                                   arrowstyle='->', mutation_scale=20,
                                   color=colors['graph'], linewidth=2,
                                   connectionstyle='arc3,rad=0.1')
            ax_c.add_patch(arrow_c)
            mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
            ax_c.text(mid_x+0.1, mid_y+0.1, f'$\\ell={lag}$', 
                    fontsize=11, color=colors['graph'], 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        for i, (x, y) in node_pos.items():
            circle_c = Circle((x, y), 0.3, facecolor=colors['graph'], 
                           fill=True, alpha=0.7, zorder=10)
            ax_c.add_patch(circle_c)
            ax_c.text(x, y, f'$c_{i+1}$', ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='white')
        ax_c.text(1.5, -0.2, 'Learned via Granger/NOTEARS', 
                ha='center', fontsize=11, style='italic')
        fig_c.savefig(os.path.join(save_dir, 'pipeline_c.pdf'), dpi=DPI, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        fig_c.savefig(os.path.join(save_dir, 'pipeline_c.png'), dpi=DPI, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        plt.close(fig_c)
    
    # ========== 步骤4: 因果闭包 ==========
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_title('(d) Causal Closure\n$\\mathrm{closure}(\\mathcal{C}_u)$',
                  fontsize=14, fontweight='bold', pad=12)
    ax4.set_xlim(-0.5, 3.5)
    ax4.set_ylim(-0.5, 3.5)
    ax4.axis('off')
    
        # 高亮闭包中的节点（基于mask_channels = [1, 2]）
    closure_nodes = {
        0: (0.5, 2.5, -1, 'parent'),   # c1是c2的父节点
        1: (2.5, 2.5, 0, 'seed'),      # c2是种子节点
        2: (0.5, 0.5, 0, 'seed'),      # c3是种子节点
        3: (2.5, 0.5, 1, 'child')      # c4是c2和c3的子节点
    }
    
    # 绘制相关边
    for src, tgt, lag in edges:
        x1, y1 = node_pos[src]
        x2, y2 = node_pos[tgt]
        
        # 判断边是否在闭包中
        in_closure = (src in [0] and tgt in [1, 2]) or (src in [1, 2] and tgt in [3])
        
        if in_closure:
            arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                   arrowstyle='->', mutation_scale=20,
                                   color=colors['closure'], linewidth=3,
                                   connectionstyle='arc3,rad=0.1', alpha=0.8)
            ax4.add_patch(arrow)
        else:
            arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                   arrowstyle='->', mutation_scale=15,
                                   color='gray', linewidth=1, alpha=0.3,
                                   linestyle='--', connectionstyle='arc3,rad=0.1')
            ax4.add_patch(arrow)
    
    # 绘制节点，根据类型使用不同颜色
    for i, (x, y, lag_offset, node_type) in closure_nodes.items():
        if node_type == 'seed':
            color = colors['closure']
            alpha = 1.0
            edge_color = colors['closure']
            edge_width = 3
        elif node_type == 'parent':
            color = colors.get('parent', '#FFB74D')  # 浅橙色表示父节点
            alpha = 0.8
            edge_color = colors['closure']
            edge_width = 2
        else:  # child
            color = colors.get('child', '#81C784')  # 浅绿色表示子节点
            alpha = 0.8
            edge_color = colors['closure']
            edge_width = 2
        
        circle = Circle((x, y), 0.3, facecolor=color, 
                       fill=True, alpha=alpha, zorder=10,
                       edgecolor=edge_color, linewidth=edge_width)
        ax4.add_patch(circle)
        
        label = f'$c_{i+1}$'
        if lag_offset != 0:
            label += f'\n$\\Delta={lag_offset:+d}$'
        ax4.text(x, y, label, ha='center', va='center', 
                fontsize=12, fontweight='bold', color='black')
    
    # 添加闭包说明
    ax4.text(1.5, -0.2, 'Seeds + Parents + Children', 
            ha='center', fontsize=11, style='italic')
    
    # 单独保存子图d
    if save_individual:
        fig_d = plt.figure(figsize=(4, 3), dpi=DPI)
        ax_d = fig_d.add_subplot(111)
        ax_d.set_title('(d) Causal Closure\n$\\mathrm{closure}(\\mathcal{C}_u)$',
                      fontsize=14, fontweight='bold', pad=12)
        ax_d.set_xlim(-0.5, 3.5)
        ax_d.set_ylim(-0.5, 3.5)
        ax_d.axis('off')
        for src, tgt, lag in edges:
            x1, y1 = node_pos[src]
            x2, y2 = node_pos[tgt]
            in_closure = (src in [0] and tgt in [1, 2]) or (src in [1, 2] and tgt in [3])
            if in_closure:
                arrow_d = FancyArrowPatch((x1, y1), (x2, y2),
                                       arrowstyle='->', mutation_scale=20,
                                       color=colors['closure'], linewidth=3,
                                       connectionstyle='arc3,rad=0.1', alpha=0.8)
                ax_d.add_patch(arrow_d)
            else:
                arrow_d = FancyArrowPatch((x1, y1), (x2, y2),
                                       arrowstyle='->', mutation_scale=15,
                                       color='gray', linewidth=1, alpha=0.3,
                                       linestyle='--', connectionstyle='arc3,rad=0.1')
                ax_d.add_patch(arrow_d)
        for i, (x, y, lag_offset, node_type) in closure_nodes.items():
            if node_type == 'seed':
                color_d = colors['closure']
                alpha_d = 1.0
                edge_color_d = colors['closure']
                edge_width_d = 3
            elif node_type == 'parent':
                color_d = colors.get('parent', '#FFB74D')
                alpha_d = 0.8
                edge_color_d = colors['closure']
                edge_width_d = 2
            else:
                color_d = colors.get('child', '#81C784')
                alpha_d = 0.8
                edge_color_d = colors['closure']
                edge_width_d = 2
            circle_d = Circle((x, y), 0.3, facecolor=color_d, 
                           fill=True, alpha=alpha_d, zorder=10,
                           edgecolor=edge_color_d, linewidth=edge_width_d)
            ax_d.add_patch(circle_d)
            label_d = f'$c_{i+1}$'
            if lag_offset != 0:
                label_d += f'\n$\\Delta={lag_offset:+d}$'
            ax_d.text(x, y, label_d, ha='center', va='center', 
                    fontsize=12, fontweight='bold', color='black')
        ax_d.text(1.5, -0.2, 'Seeds + Parents + Children', 
                ha='center', fontsize=11, style='italic')
        fig_d.savefig(os.path.join(save_dir, 'pipeline_d.pdf'), dpi=DPI, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        fig_d.savefig(os.path.join(save_dir, 'pipeline_d.png'), dpi=DPI, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        plt.close(fig_d)
    
    # ========== 步骤5: 扰动应用 ==========
    ax5 = fig.add_subplot(gs[2, 2:])  # 占据第3行的后两列
    ax5.set_title('(f) Perturbation Application\n$\\mathcal{P}_u(\\mathbf{X}; \\xi)$',
                  fontsize=12, fontweight='bold', pad=12)
    
    # 对闭包中的每个节点应用扰动（定义closure_info供后续使用）
    closure_info = [
        (0, -1, mask_time[0]-1, mask_time[1]-1),  # parent c1, lag=-1
        (1, 0, mask_time[0], mask_time[1]),       # seed c2, lag=0
        (2, 0, mask_time[0], mask_time[1]),       # seed c3, lag=0
        (3, 1, mask_time[0]+1, mask_time[1]+1)   # child c4, lag=+1
    ]
    
    # 绘制实际扰动后的数据（与图h使用相同的扰动）
    np.random.seed(123)  # 使用与图h相同的随机种子
    perturbed_data = eeg_data.copy()
    for ch, lag, t_start, t_end in closure_info:
        if 0 <= ch < C:
            t_start_clip = max(0, min(t_start, T-1))
            t_end_clip = max(t_start_clip+1, min(t_end, T))
            if t_start_clip < t_end_clip:
                segment = perturbed_data[ch, t_start_clip:t_end_clip]
                perturbation_strength = 0.8  # 与图h一致
                perturbation = np.random.normal(0, perturbation_strength * np.std(segment), len(segment))
                perturbed_data[ch, t_start_clip:t_end_clip] += perturbation
    
    # 创建扰动强度图（类似details_3的风格）
    perturbation_map_e = np.zeros((C, T))
    for ch, lag, t_start, t_end in closure_info:
        if 0 <= ch < C:
            t_start_clip = max(0, min(t_start, T-1))
            t_end_clip = max(t_start_clip+1, min(t_end, T))
            if t_start_clip < t_end_clip:
                # 根据扰动强度设置值
                perturbation_map_e[ch, t_start_clip:t_end_clip] = 0.6
    
    # 使用YlOrRd colormap显示扰动强度（类似details_3）
    # 不使用extent，直接使用数组索引，确保边框和扰动区域重合
    im5 = ax5.imshow(perturbation_map_e, aspect='auto', cmap='YlOrRd', 
                    vmin=0, vmax=0.9, interpolation='nearest', origin='upper')
    
    # 添加扰动区域标注（使用details_3的风格）
    # 确保边框坐标与imshow的数组索引坐标一致
    for ch, lag, t_start, t_end in closure_info:
        if 0 <= ch < C:
            t_start_clip = max(0, min(t_start, T-1))
            t_end_clip = max(t_start_clip+1, min(t_end, T))
            if t_start_clip < t_end_clip:
                # 使用更柔和的边框颜色，坐标直接使用数组索引
                rect = Rectangle((t_start_clip-0.5, ch-0.5), 
                               t_end_clip-t_start_clip, 1,
                               linewidth=2, edgecolor='#333333', 
                               facecolor='none', linestyle='--', alpha=0.8)
                ax5.add_patch(rect)
                
                # 添加滞后标注（更柔和的样式）
                mid_t = (t_start_clip + t_end_clip) / 2
                ax5.text(mid_t, ch, f'$\\Delta={lag:+d}$', 
                        ha='center', va='center', fontsize=10,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white', alpha=0.9))
    
    ax5.set_ylabel('Channels $c$', fontsize=13)
    # 将Time标签移到时间轴末端（图内右下方）
    ax5.text(T-1, -0.2, 'Time $t$', fontsize=13, ha='right', va='top')
    # 设置正确的tick位置（对应数组索引）
    ax5.set_xticks(np.linspace(0, T-1, 4))
    ax5.set_xticklabels(['1', f'{T//3}', f'{2*T//3}', f'{T}'])
    ax5.set_yticks(range(C))
    ax5.set_yticklabels([f'$c_{i+1}$' for i in range(C)])
    
    # 添加颜色条（类似details_3）
    cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    cbar5.set_label('Perturbation Strength', fontsize=13)
    cbar5.ax.tick_params(labelsize=12)
    
    # 单独保存子图e
    if save_individual:
        fig_e = plt.figure(figsize=(8, 3), dpi=DPI)
        ax_e = fig_e.add_subplot(111)
        ax_e.set_title('(f) Perturbation Application\n$\\mathcal{P}_u(\\mathbf{X}; \\xi)$',
                      fontsize=12, fontweight='bold', pad=12)
        # 创建扰动强度图（类似details_3的风格）
        perturbation_map_e_save = np.zeros((C, T))
        for ch_save, lag_save, t_start_save, t_end_save in closure_info:
            if 0 <= ch_save < C:
                t_start_clip_save = max(0, min(t_start_save, T-1))
                t_end_clip_save = max(t_start_clip_save+1, min(t_end_save, T))
                if t_start_clip_save < t_end_clip_save:
                    perturbation_map_e_save[ch_save, t_start_clip_save:t_end_clip_save] = 0.6
        
        im_e = ax_e.imshow(perturbation_map_e_save, aspect='auto', cmap='YlOrRd', 
                    vmin=0, vmax=0.9, interpolation='nearest', origin='upper')
        for ch, lag, t_start, t_end in closure_info:
            if 0 <= ch < C:
                t_start_clip = max(0, min(t_start, T-1))
                t_end_clip = max(t_start_clip+1, min(t_end, T))
                if t_start_clip < t_end_clip:
                    rect_e = Rectangle((t_start_clip-0.5, ch-0.5), 
                                   t_end_clip-t_start_clip, 1,
                                   linewidth=2, edgecolor='#333333', 
                                   facecolor='none', linestyle='--', alpha=0.8)
                    ax_e.add_patch(rect_e)
                    mid_t = (t_start_clip + t_end_clip) / 2
                    ax_e.text(mid_t, ch, f'$\\Delta={lag:+d}$', 
                            ha='center', va='center', fontsize=10,
                            fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='white', alpha=0.9))
        ax_e.set_ylabel('Channels $c$', fontsize=13)
        # 将Time标签移到时间轴末端（图内右下方）
        ax_e.text(T-1, -0.2, 'Time $t$', fontsize=13, ha='right', va='top')
        # 设置正确的tick位置（对应数组索引）
        ax_e.set_xticks(np.linspace(0, T-1, 4))
        ax_e.set_xticklabels(['1', f'{T//3}', f'{2*T//3}', f'{T}'])
        ax_e.set_yticks(range(C))
        ax_e.set_yticklabels([f'$c_{i+1}$' for i in range(C)])
        cbar_e = plt.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.04)
        cbar_e.set_label('Perturbation Strength', fontsize=13)
        cbar_e.ax.tick_params(labelsize=12)
        fig_e.savefig(os.path.join(save_dir, 'pipeline_e.pdf'), dpi=DPI, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        fig_e.savefig(os.path.join(save_dir, 'pipeline_e.png'), dpi=DPI, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        plt.close(fig_e)
    
    # ========== 步骤6: 混合创新过程 ==========
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.set_title('(g) Hybrid Innovation\n$\\varepsilon_t = \\beta_G \\varepsilon_t^G + (1-\\beta_G) \\varepsilon_t^{AR}$',
                  fontsize=14, fontweight='bold', pad=12)
    
    # 生成示例扰动信号
    t_perturb = np.linspace(0, 10, 50)
    np.random.seed(42)
    
    # 高斯噪声
    epsilon_g = np.random.normal(0, 0.5, len(t_perturb))
    
    # AR(1)过程
    phi = 0.7  # AR系数
    nu = np.random.normal(0, 0.5, len(t_perturb))
    epsilon_ar = np.zeros_like(nu)
    epsilon_ar[0] = nu[0]
    for i in range(1, len(epsilon_ar)):
        epsilon_ar[i] = phi * epsilon_ar[i-1] + nu[i]
    
    # 混合
    beta_g = 0.7
    epsilon_hybrid = beta_g * epsilon_g + (1 - beta_g) * epsilon_ar
    
    ax6.plot(t_perturb, epsilon_g, label='$\\varepsilon_t^G$ (Gaussian)', 
            linewidth=1.5, alpha=0.7, linestyle='--')
    ax6.plot(t_perturb, epsilon_ar, label='$\\varepsilon_t^{AR}$ (AR(1))', 
            linewidth=1.5, alpha=0.7, linestyle=':')
    ax6.plot(t_perturb, epsilon_hybrid, label='$\\varepsilon_t$ (Hybrid)', 
            linewidth=2, color=colors['perturb'])
    
    ax6.axhline(y=0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
    ax6.set_xlabel('Time', fontsize=13)
    ax6.set_ylabel('Innovation $\\varepsilon_t$', fontsize=13)
    ax6.legend(fontsize=11, loc='upper right')
    ax6.tick_params(labelsize=12)
    ax6.grid(True, alpha=0.3)
    
    # 单独保存子图f
    if save_individual:
        fig_f = plt.figure(figsize=(4, 3), dpi=DPI)
        ax_f = fig_f.add_subplot(111)
        ax_f.set_title('(g) Hybrid Innovation\n$\\varepsilon_t = \\beta_G \\varepsilon_t^G + (1-\\beta_G) \\varepsilon_t^{AR}$',
                      fontsize=14, fontweight='bold', pad=12)
        ax_f.plot(t_perturb, epsilon_g, label='$\\varepsilon_t^G$ (Gaussian)', 
                linewidth=1.5, alpha=0.7, linestyle='--')
        ax_f.plot(t_perturb, epsilon_ar, label='$\\varepsilon_t^{AR}$ (AR(1))', 
                linewidth=1.5, alpha=0.7, linestyle=':')
        ax_f.plot(t_perturb, epsilon_hybrid, label='$\\varepsilon_t$ (Hybrid)', 
                linewidth=2, color=colors['perturb'])
        ax_f.axhline(y=0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
        ax_f.set_xlabel('Time', fontsize=13)
        ax_f.set_ylabel('Innovation $\\varepsilon_t$', fontsize=13)
        ax_f.legend(fontsize=11, loc='upper right')
        ax_f.tick_params(labelsize=12)
        ax_f.grid(True, alpha=0.3)
        fig_f.savefig(os.path.join(save_dir, 'pipeline_f.pdf'), dpi=DPI, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        fig_f.savefig(os.path.join(save_dir, 'pipeline_f.png'), dpi=DPI, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        plt.close(fig_f)
    
    # ========== 步骤7: 统计特性保持 ==========
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.set_title('(g) Statistics Preservation\nMoment Matching',
                  fontsize=14, fontweight='bold', pad=12)
    
    # 绘制统计特性对比
    channels = range(C)
    orig_mean = np.mean(eeg_data, axis=1)
    orig_std = np.std(eeg_data, axis=1)
    
    # 模拟扰动后的统计特性
    pert_mean = orig_mean + np.random.normal(0, 0.1, C)
    pert_std = orig_std + np.random.normal(0, 0.05, C)
    
    # 调整后的统计特性（应该接近原始）
    adj_mean = orig_mean.copy()
    adj_std = orig_std.copy()
    
    x = np.arange(C)
    width = 0.25
    
    ax7.bar(x - width, orig_mean, width, label='Original $\\mu$', 
           color=colors['original'], alpha=0.7)
    ax7.bar(x, pert_mean, width, label='Perturbed $\\mu\'$', 
           color=colors['perturb'], alpha=0.7)
    ax7.bar(x + width, adj_mean, width, label='Adjusted $\\mu\'\'$', 
           color=colors['postprocess'], alpha=0.7)
    
    ax7.set_xlabel('Channel', fontsize=13)
    ax7.set_ylabel('Mean Value', fontsize=13)
    ax7.set_xticks(x)
    ax7.set_xticklabels([f'$c_{i+1}$' for i in range(C)])
    ax7.legend(fontsize=11)
    ax7.tick_params(labelsize=12)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 单独保存子图g
    if save_individual:
        fig_g = plt.figure(figsize=(4, 3), dpi=DPI)
        ax_g = fig_g.add_subplot(111)
        ax_g.set_title('(h) Statistics Preservation\nMoment Matching',
                      fontsize=14, fontweight='bold', pad=12)
        ax_g.bar(x - width, orig_mean, width, label='Original $\\mu$', 
               color=colors['original'], alpha=0.7)
        ax_g.bar(x, pert_mean, width, label='Perturbed $\\mu\'$', 
               color=colors['perturb'], alpha=0.7)
        ax_g.bar(x + width, adj_mean, width, label='Adjusted $\\mu\'\'$', 
               color=colors['postprocess'], alpha=0.7)
        ax_g.set_xlabel('Channel', fontsize=13)
        ax_g.set_ylabel('Mean Value', fontsize=13)
        ax_g.set_xticks(x)
        ax_g.set_xticklabels([f'$c_{i+1}$' for i in range(C)])
        ax_g.legend(fontsize=11)
        ax_g.tick_params(labelsize=12)
        ax_g.grid(True, alpha=0.3, axis='y')
        fig_g.savefig(os.path.join(save_dir, 'pipeline_g.pdf'), dpi=DPI, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        fig_g.savefig(os.path.join(save_dir, 'pipeline_g.png'), dpi=DPI, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        plt.close(fig_g)
    
    # ========== 步骤8: 最终输出 ==========
    ax8 = fig.add_subplot(gs[3, :])  # 调整到第4行
    ax8.set_title('(I) Final Perturbed Signal\n$\\mathbf{X}\'\' \\in \\mathbb{R}^{C \\times T}$',
                  fontsize=16, fontweight='bold', pad=15)
    
    # 生成最终扰动后的数据
    # 使用固定随机种子确保扰动可重复且可见
    np.random.seed(123)  # 使用不同的种子生成扰动
    final_data = eeg_data.copy().astype(float)  # 确保是浮点类型
    
    # 对闭包中的每个节点应用更强的扰动
    for ch, lag, t_start, t_end in closure_info:
        if 0 <= ch < C:
            t_start_clip = max(0, min(t_start, T-1))
            t_end_clip = max(t_start_clip+1, min(t_end, T))
            if t_start_clip < t_end_clip:
                # 添加更强的扰动（增加扰动强度系数）
                segment = final_data[ch, t_start_clip:t_end_clip]
                # 使用更大的扰动强度，确保扰动明显可见
                perturbation_strength = 1.2  # 增加到1.2，使扰动更明显
                segment_std = np.std(segment) if np.std(segment) > 1e-10 else 1.0
                perturbation = np.random.normal(0, perturbation_strength * segment_std, len(segment))
                final_data[ch, t_start_clip:t_end_clip] += perturbation
    
    # 应用统计特性保持后处理（模拟）
    # 为了在示意图中清晰显示扰动效果，我们只进行部分调整
    # 在实际应用中，这会完全恢复统计特性，但这里我们保留部分扰动效果以便可视化
    for ch in range(C):
        orig_mean = np.mean(eeg_data[ch, :])
        orig_std = np.std(eeg_data[ch, :])
        pert_mean = np.mean(final_data[ch, :])
        pert_std = np.std(final_data[ch, :])
        
        # 部分调整：混合原始统计特性和扰动后的数据
        # 这样可以同时看到扰动效果和统计特性保持的效果
        if pert_std > 1e-10:
            # 标准化扰动后的数据
            normalized = (final_data[ch, :] - pert_mean) / pert_std
            # 重新缩放，但只应用部分调整以保留扰动效果
            adjusted = normalized * orig_std * 0.5 + orig_mean * 0.5
            # 混合：50%调整后的数据 + 50%原始扰动数据
            final_data[ch, :] = adjusted * 0.5 + final_data[ch, :] * 0.5
    
    im8 = ax8.imshow(final_data, aspect='auto', cmap='YlOrBr', 
                    vmin=-2, vmax=2, interpolation='nearest', origin='upper')
    
    # 高亮扰动区域
    for ch, lag, t_start, t_end in closure_info:
        if 0 <= ch < C:
            t_start_clip = max(0, min(t_start, T-1))
            t_end_clip = max(t_start_clip+1, min(t_end, T))
            if t_start_clip < t_end_clip:
                # 使用与e图一致的边框颜色
                rect = Rectangle((t_start_clip-0.5, ch-0.5), 
                               t_end_clip-t_start_clip, 1,
                               linewidth=2, edgecolor='#333333', 
                               facecolor='none', linestyle='--', alpha=0.8)
                ax8.add_patch(rect)
    
    ax8.set_ylabel('Channels $c$', fontsize=14, fontweight='bold')
    # 将Time标签移到时间轴末端（图内右下方）
    ax8.text(T-1, -0.2, 'Time $t$', fontsize=14, fontweight='bold', ha='right', va='top')
    # 设置正确的tick位置（对应数组索引），更简洁的刻度
    ax8.set_xticks(np.linspace(0, T-1, 4))
    ax8.set_xticklabels(['1', f'{T//3}', f'{2*T//3}', f'{T}'])
    ax8.set_yticks(range(C))
    ax8.set_yticklabels([f'$c_{i+1}$' for i in range(C)])
    
    cbar8 = plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
    cbar8.set_label('Amplitude', fontsize=13)
    cbar8.ax.tick_params(labelsize=12)
    
    # 单独保存子图h
    if save_individual:
        fig_h = plt.figure(figsize=(12, 3), dpi=DPI)
        ax_h = fig_h.add_subplot(111)
        ax_h.set_title('(I) Final Perturbed Signal\n$\\mathbf{X}\'\' \\in \\mathbb{R}^{C \\times T}$',
                      fontsize=16, fontweight='bold', pad=15)
        im_h = ax_h.imshow(final_data, aspect='auto', cmap='YlOrBr', 
                    vmin=-2, vmax=2, interpolation='nearest', origin='upper')
        for ch, lag, t_start, t_end in closure_info:
            if 0 <= ch < C:
                t_start_clip = max(0, min(t_start, T-1))
                t_end_clip = max(t_start_clip+1, min(t_end, T))
                if t_start_clip < t_end_clip:
                    # 使用与e图一致的边框颜色
                    rect_h = Rectangle((t_start_clip-0.5, ch-0.5), 
                                   t_end_clip-t_start_clip, 1,
                                   linewidth=2, edgecolor='#333333', 
                                   facecolor='none', linestyle='--', alpha=0.8)
                    ax_h.add_patch(rect_h)
        ax_h.set_ylabel('Channels $c$', fontsize=14, fontweight='bold')
        # 将Time标签移到时间轴末端（图内右下方）
        ax_h.text(T-1, -0.2, 'Time $t$', fontsize=14, fontweight='bold', ha='right', va='top')
        ax_h.set_xticks(np.linspace(0, T-1, 4))
        ax_h.set_xticklabels(['1', f'{T//3}', f'{2*T//3}', f'{T}'])
        ax_h.set_yticks(range(C))
        ax_h.set_yticklabels([f'$c_{i+1}$' for i in range(C)])
        cbar_h = plt.colorbar(im_h, ax=ax_h, fraction=0.046, pad=0.04)
        cbar_h.set_label('Amplitude', fontsize=13)
        cbar_h.ax.tick_params(labelsize=12)
        fig_h.savefig(os.path.join(save_dir, 'pipeline_h.pdf'), dpi=DPI, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        fig_h.savefig(os.path.join(save_dir, 'pipeline_h.png'), dpi=DPI, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        plt.close(fig_h)
    
    # 添加流程箭头（调整位置以适应新布局）
    # 第一行：图a到图b
    fig.text(0.125, 0.72, '→', fontsize=18, ha='center', va='center', 
            color='#666666', fontweight='bold', alpha=0.6)
    fig.text(0.375, 0.72, '→', fontsize=18, ha='center', va='center', 
            color='#666666', fontweight='bold', alpha=0.6)
    fig.text(0.625, 0.72, '→', fontsize=18, ha='center', va='center', 
            color='#666666', fontweight='bold', alpha=0.6)
    
    # 添加英文描述文本
    # description_text = (
    #     "This figure illustrates the graph-conditioned causal perturbation pipeline for EEG signal interpretation. "
    #     "(a) Original EEG signal $\\mathbf{X}$ with $C$ channels and $T$ time points. "
    #     "(b) Primitive mask defines the spatial-temporal region $\\mathrm{mask}(u)$ for each primitive $u$. "
    #     "(c) Directed channel graph $\\mathcal{G}$ learned via Granger causality or NOTEARS captures causal relationships. "
    #     "(d) First-order causal closure includes seed channels, their direct parents (with negative lag offsets), and children (with positive lag offsets). "
    #     "(e) Perturbation operator $\\mathcal{P}_u$ applies additive innovations to all nodes in the closure with time-aligned windows. "
    #     "(f) Hybrid innovation process combines Gaussian noise and AR(1)-like correlated perturbations. "
    #     "(g) Statistics-preserving post-processing maintains original mean and variance per channel. "
    #     "(h) Final perturbed signal $\\mathbf{X}''$ with causally consistent modifications."
    # )
    
    # 在底部添加描述文本
    # fig.text(0.5, 0.02, description_text, ha='center', va='bottom',
    #         fontsize=9, wrap=True, style='italic',
    #         bbox=dict(boxstyle='round,pad=0.8', facecolor='#F5F5F5',
    #                  edgecolor='#CCCCCC', linewidth=1, alpha=0.9))
    
    plt.suptitle('Graph-Conditioned Causal Perturbation Pipeline', 
                fontsize=18, fontweight='bold', y=0.985)
    
    return fig


def create_detail_figure(save_individual=True, save_dir='pictures/pipeline'):
    """
    创建细节图：展示因果闭包和扰动机制的详细过程
    
    Parameters:
    -----------
    save_individual : bool
        是否单独保存每个子图
    save_dir : str
        保存目录路径
    """
    # 创建保存目录
    if save_individual:
        os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(12, 8), dpi=DPI)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # 使用与主图一致的颜色方案
    colors = {
        'seed': '#D32F2F',      # 深红色 - 种子节点
        'parent': '#FFB74D',    # 浅橙色 - 父节点
        'child': '#81C784',     # 浅绿色 - 子节点
        'graph': '#FB8C00',     # 明亮橙色 - 图
        'perturb': '#43A047'    # 绿色 - 扰动
    }
    
    # ========== 左图：因果闭包详细说明 ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('(d)Causal Closure Construction\n$\\mathrm{closure}(\\mathcal{C}_u)$',
    fontsize = 14, fontweight = 'bold', pad = 12)
    ax1.set_xlim(-1, 4)
    ax1.set_ylim(-1, 4)
    ax1.axis('off')
    
    # 节点位置
    pos = {
        0: (1, 3),   # c1 (parent)
        1: (3, 3),   # c2 (seed)
        2: (1, 1),   # c3 (seed)
        3: (3, 1)    # c4 (child)
    }
    
    # 绘制完整图结构（浅色）
    edges_all = [(0, 1, -1), (0, 2, -1), (1, 3, 1), (2, 3, 1)]
    for src, tgt, lag in edges_all:
        x1, y1 = pos[src]
        x2, y2 = pos[tgt]
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=15,
                               color='lightgray', linewidth=1, alpha=0.3,
                               linestyle='--')
        ax1.add_patch(arrow)
    
    # 高亮闭包中的边
    closure_edges = [(0, 1, -1), (0, 2, -1), (1, 3, 1), (2, 3, 1)]
    for src, tgt, lag in closure_edges:
        x1, y1 = pos[src]
        x2, y2 = pos[tgt]
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=20,
                               color=colors['graph'], linewidth=2.5,
                               connectionstyle='arc3,rad=0.15')
        ax1.add_patch(arrow)
        
        # 添加滞后标签
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        offset_x = 0.15 if lag < 0 else -0.15
        ax1.text(mid_x+offset_x, mid_y+0.15, f'$\\ell={abs(lag)}$', 
                fontsize=9, color=colors['graph'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=colors['graph'], linewidth=1.5))
    
    # 绘制节点
    node_types = {
        0: ('parent', colors['parent'], -1),
        1: ('seed', colors['seed'], 0),
        2: ('seed', colors['seed'], 0),
        3: ('child', colors['child'], 1)
    }
    
    for i, (x, y) in pos.items():
        node_type, color, lag_offset = node_types[i]
        circle = Circle((x, y), 0.4, facecolor=color, fill=True, 
                       alpha=0.8, zorder=10, edgecolor='black', linewidth=2)
        ax1.add_patch(circle)
        
        label = f'$c_{i+1}$'
        if lag_offset != 0:
            label += f'\n$\\Delta={lag_offset:+d}$'
        ax1.text(x, y, label, ha='center', va='center', 
                fontsize=11, fontweight='bold', color='black')
    
    # 添加图例
    legend_elements = [
        mpatches.Patch(facecolor=colors['seed'], alpha=0.8, label='Seed channels $(c, 0)$'),
        mpatches.Patch(facecolor=colors['parent'], alpha=0.8, label='Parents $(p, -\\ell)$'),
        mpatches.Patch(facecolor=colors['child'], alpha=0.8, label='Children $(q, +\\ell)$')
    ]
    ax1.legend(handles=legend_elements, loc='lower left', fontsize=9, 
              frameon=True, fancybox=True, shadow=True)
    
    # 添加公式说明
    # formula_text = ('$\\mathrm{closure}(\\mathcal{C}_u) = \\{(c,0) : c \\in \\mathcal{C}_u\\}$\n'
    #                '$\\cup \\{(p,-\\ell) : p \\to c \\in \\mathcal{E}, c \\in \\mathcal{C}_u\\}$\n'
    #                '$\\cup \\{(q,+\\ell) : c \\to q \\in \\mathcal{E}, c \\in \\mathcal{C}_u\\}$')
    # ax1.text(0.5, -0.5, formula_text, ha='center', fontsize=10,
    #         bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5),
    #         transform=ax1.transAxes)
    
    # 单独保存子图1（因果闭包构建）
    if save_individual:
        fig_d1 = plt.figure(figsize=(5, 5), dpi=DPI)
        ax_d1 = fig_d1.add_subplot(111)
        ax_d1.set_title('(d)Causal Closure Construction\n$\\mathrm{closure}(\\mathcal{C}_u)$',
        fontsize = 16, fontweight = 'bold', pad = 12)
        ax_d1.set_xlim(-1.5, 4.5)  # 扩大x范围，为图例留出空间
        ax_d1.set_ylim(-1, 4)
        ax_d1.axis('off')
        for src, tgt, lag in edges_all:
            x1, y1 = pos[src]
            x2, y2 = pos[tgt]
            arrow_d1 = FancyArrowPatch((x1, y1), (x2, y2),
                                   arrowstyle='->', mutation_scale=15,
                                   color='lightgray', linewidth=1, alpha=0.3,
                                   linestyle='--')
            ax_d1.add_patch(arrow_d1)
        for src, tgt, lag in closure_edges:
            x1, y1 = pos[src]
            x2, y2 = pos[tgt]
            arrow_d1 = FancyArrowPatch((x1, y1), (x2, y2),
                                   arrowstyle='->', mutation_scale=20,
                                   color=colors['graph'], linewidth=2.5,
                                   connectionstyle='arc3,rad=0.15')
            ax_d1.add_patch(arrow_d1)
            mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
            offset_x = 0.15 if lag < 0 else -0.15
            ax_d1.text(mid_x+offset_x, mid_y+0.15, f'$\\ell={abs(lag)}$', 
                    fontsize=13, color=colors['graph'], fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor=colors['graph'], linewidth=1.5))
        for i, (x, y) in pos.items():
            node_type, color, lag_offset = node_types[i]
            circle_d1 = Circle((x, y), 0.4, facecolor=color, fill=True, 
                           alpha=0.8, zorder=10, edgecolor='black', linewidth=2)
            ax_d1.add_patch(circle_d1)
            label_d1 = f'$c_{i+1}$'
            if lag_offset != 0:
                label_d1 += f'\n$\\Delta={lag_offset:+d}$'
                # c1 (i=0) 有lag_offset=-1，将标签放在圆圈左侧
                if i == 0:
                    ax_d1.text(x - 0.6, y, label_d1, ha='right', va='center', 
                            fontsize=13, fontweight='bold', color='black')
                else:
                    # c4 (i=3) 有lag_offset=1，将标签放在圆圈右侧
                    ax_d1.text(x + 0.6, y, label_d1, ha='left', va='center', 
                            fontsize=13, fontweight='bold', color='black')
            else:
                # c2和c3没有lag_offset
                if i == 1:
                    # c2移到圆圈右边
                    ax_d1.text(x + 0.6, y, label_d1, ha='left', va='center', 
                            fontsize=13, fontweight='bold', color='black')
                else:  # i == 2, c3
                    # c3移到圆圈左边
                    ax_d1.text(x - 0.6, y, label_d1, ha='right', va='center', 
                            fontsize=13, fontweight='bold', color='black')
        legend_d1 = [
            mpatches.Patch(facecolor=colors['seed'], alpha=0.8, label='Seed channels $(c, 0)$'),
            mpatches.Patch(facecolor=colors['parent'], alpha=0.8, label='Parents $(p, -\\ell)$'),
            mpatches.Patch(facecolor=colors['child'], alpha=0.8, label='Children $(q, +\\ell)$')
        ]
        ax_d1.legend(handles=legend_d1, loc='upper left', fontsize=12, 
                  frameon=True, fancybox=True, shadow=True,
                  bbox_to_anchor=(-0.3, 0.75), borderpad=1.0)
        # ax_d1.text(0.5, -0.5, formula_text, ha='center', fontsize=11,
        #         bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5),
        #         transform=ax_d1.transAxes)
        fig_d1.savefig(os.path.join(save_dir, 'details_1.pdf'), dpi=DPI, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        fig_d1.savefig(os.path.join(save_dir, 'details_1.png'), dpi=DPI, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        plt.close(fig_d1)
    
    # ========== 右图：时间对齐和扰动应用 ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Time-Aligned Perturbation Windows', 
                  fontsize=12, fontweight='bold', pad=15)
    
    # 绘制时间轴和扰动窗口
    T_total = 20
    t_start_orig = 5
    t_end_orig = 12
    
    channels_detail = [
        (0, -1, 'Parent $c_1$', colors['parent']),
        (1, 0, 'Seed $c_2$', colors['seed']),
        (2, 0, 'Seed $c_3$', colors['seed']),
        (3, 1, 'Child $c_4$', colors['child'])
    ]
    
    y_positions = [3, 2, 1, 0]
    
    for idx, (ch, lag, label, color) in enumerate(channels_detail):
        y = y_positions[idx]
        
        # 计算对齐后的时间窗口
        t_start_aligned = max(0, t_start_orig + lag)
        t_end_aligned = min(T_total, t_end_orig + lag)
        
        # 绘制时间轴
        ax2.plot([0, T_total], [y, y], 'k-', linewidth=1, alpha=0.3)
        
        # 绘制原始时间窗口（虚线）
        ax2.plot([t_start_orig, t_end_orig], [y+0.1, y+0.1], 
                'k--', linewidth=1.5, alpha=0.5, label='Original window' if idx == 0 else '')
        
        # 绘制对齐后的扰动窗口（实线矩形）
        if t_start_aligned < t_end_aligned:
            rect = Rectangle((t_start_aligned, y-0.15), 
                           t_end_aligned - t_start_aligned, 0.3,
                           facecolor=color, alpha=0.6, edgecolor='black', linewidth=2)
            ax2.add_patch(rect)
            
            # 添加标签
            mid_t = (t_start_aligned + t_end_aligned) / 2
            ax2.text(mid_t, y, label, ha='center', va='center', 
                    fontsize=9, fontweight='bold')
            
            # 添加滞后标注
            if lag != 0:
                ax2.annotate('', xy=(t_start_aligned, y-0.4), 
                           xytext=(t_start_orig, y-0.4),
                           arrowprops=dict(arrowstyle='<->', color='red', 
                                         lw=2, alpha=0.7))
                ax2.text((t_start_orig + t_start_aligned)/2, y-0.5, 
                        f'$\\Delta={lag:+d}$', ha='center', fontsize=8,
                        color='red', fontweight='bold')
    
    ax2.set_xlim(-1, T_total+1)
    ax2.set_ylim(-1, 4)
    ax2.set_ylabel('Channel', fontsize=11, fontweight='bold')
    # 将Time标签移到时间轴末端（图内右下方）
    ax2.text(T_total, -0.6, 'Time $t$', fontsize=11, fontweight='bold', ha='right', va='top')
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([f'$c_{i+1}$' for i in range(4)])
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 添加说明
    ax2.text(T_total/2, 3.5, 'Aligned windows: $\\mathcal{T}\'(u, \\Delta) = [\\max(1, t_u^{\\mathrm{start}}+\\Delta), \\min(T, t_u^{\\mathrm{end}}+\\Delta)]$',
            ha='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    # 单独保存子图2（时间对齐扰动窗口）
    if save_individual:
        fig_d2 = plt.figure(figsize=(6, 5), dpi=DPI)
        ax_d2 = fig_d2.add_subplot(111)
        ax_d2.set_title('(e)Time-Aligned Perturbation Windows',
                      fontsize=18, fontweight='bold', pad=15)
        for idx, (ch, lag, label, color) in enumerate(channels_detail):
            y = y_positions[idx]
            t_start_aligned = max(0, t_start_orig + lag)
            t_end_aligned = min(T_total, t_end_orig + lag)
            ax_d2.plot([0, T_total], [y, y], 'k-', linewidth=1, alpha=0.3)
            ax_d2.plot([t_start_orig, t_end_orig], [y+0.1, y+0.1], 
                    'k--', linewidth=1.5, alpha=0.5, label='Original window' if idx == 0 else '')
            if t_start_aligned < t_end_aligned:
                rect_d2 = Rectangle((t_start_aligned, y-0.15), 
                               t_end_aligned - t_start_aligned, 0.3,
                               facecolor=color, alpha=0.6, edgecolor='black', linewidth=2)
                ax_d2.add_patch(rect_d2)
                mid_t = (t_start_aligned + t_end_aligned) / 2
                ax_d2.text(mid_t, y, label, ha='center', va='center', 
                        fontsize=10, fontweight='bold')
                if lag != 0:
                    # ax_d2.annotate('', xy=(t_start_aligned, y-0.4),
                    #            xytext=(t_start_orig, y-0.4),
                    #            arrowprops=dict(arrowstyle='<->', color='red',
                    #                          lw=2, alpha=0.7))
                    ax_d2.text((t_start_orig + t_start_aligned)/2, y-0.5, 
                            f'$\\Delta={lag:+d}$', ha='center', fontsize=9,
                            color='red', fontweight='bold')
        ax_d2.set_xlim(-1, T_total+1)
        ax_d2.set_ylim(-1, 4)
        ax_d2.set_ylabel('Channel', fontsize=13, fontweight='bold')
        # 将Time标签移到时间轴末端（图内右下方）
        ax_d2.text(T_total, -0.6, 'Time $t$', fontsize=13, fontweight='bold', ha='right', va='top')
        ax_d2.set_yticks(y_positions)
        ax_d2.set_yticklabels([f'$c_{i+1}$' for i in range(4)])
        ax_d2.grid(True, alpha=0.3, axis='x')
        ax_d2.text(T_total/2, 3.3, '$\\mathcal{T}\'(u, \\Delta) = [\\max(1, t_u^{\\mathrm{start}}+\\Delta), \\min(T, t_u^{\\mathrm{end}}+\\Delta)]$',
                ha='center', fontsize=14, color='black')
        fig_d2.savefig(os.path.join(save_dir, 'details_2.pdf'), dpi=DPI, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        fig_d2.savefig(os.path.join(save_dir, 'details_2.png'), dpi=DPI, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        plt.close(fig_d2)
    
    # ========== 下图：扰动叠加过程 ==========
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_title('Multi-Primitive Composition\n$\\mathcal{P}_{\\mathrm{set}}(\\mathbf{X}; \\mathbf{z}) = \\mathcal{P}_{u_{i_M}}(\\cdots \\mathcal{P}_{u_{i_1}}(\\mathbf{X}; \\xi_{i_1})\\cdots; \\xi_{i_M})$',
                  fontsize=12, fontweight='bold', pad=15)
    
    # 模拟多个primitive的扰动叠加
    C_comp, T_comp = 4, 20
    base_signal = np.random.randn(C_comp, T_comp)
    
    # 定义多个primitives
    primitives = [
        {'channels': [1], 'time': [3, 8], 'strength': 0.3},
        {'channels': [2], 'time': [6, 11], 'strength': 0.3},
        {'channels': [0, 3], 'time': [10, 15], 'strength': 0.3}
    ]
    
    # 创建累积扰动可视化
    perturbation_map = np.zeros((C_comp, T_comp))
    for prim in primitives:
        for ch in prim['channels']:
            t_start, t_end = prim['time']
            perturbation_map[ch, t_start:t_end] += prim['strength']
    
    # 绘制叠加效果
    im3 = ax3.imshow(perturbation_map, aspect='auto', cmap='YlOrRd', 
                    vmin=0, vmax=0.9, interpolation='nearest')
    
    # 添加每个primitive的边界
    for i, prim in enumerate(primitives):
        for ch in prim['channels']:
            t_start, t_end = prim['time']
            rect = Rectangle((t_start-0.5, ch-0.5), t_end-t_start, 1,
                           linewidth=2, edgecolor=f'C{i}', 
                           facecolor='none', linestyle='--', alpha=0.8)
            ax3.add_patch(rect)
            
            # 添加primitive标签
            mid_t = (t_start + t_end) / 2
            ax3.text(mid_t, ch, f'$u_{i+1}$', ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    ax3.set_ylabel('Channels $c$', fontsize=11, fontweight='bold')
    # 将Time标签移到时间轴末端（右下方）
    # 将Time标签移到时间轴末端（图内右下方）
    ax3.text(T_comp-1, -0.2, 'Time $t$', fontsize=11, fontweight='bold', ha='right', va='top')
    ax3.set_xticks([0, T_comp//4, T_comp//2, 3*T_comp//4, T_comp-1])
    ax3.set_xticklabels(['1', f'{T_comp//4}', f'{T_comp//2}', f'{3*T_comp//4}', f'{T_comp}'])
    ax3.set_yticks(range(C_comp))
    ax3.set_yticklabels([f'$c_{i+1}$' for i in range(C_comp)])
    
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('Cumulative Perturbation Strength', fontsize=10)
    
    # 添加说明文本
    ax3.text(T_comp/2, -1.5, 
            'Overlapping masks result in additive superposition: $X\'_{v,t} = X_{v,t} + \\sum_{u: (v,t) \\in \\mathrm{mask}(u)} \\varepsilon_{v,t}^{(u)}$',
            ha='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.5))
    
    # 单独保存子图3（多基元组合）
    if save_individual:
        fig_d3 = plt.figure(figsize=(10, 4), dpi=DPI)
        ax_d3 = fig_d3.add_subplot(111)
        ax_d3.set_title('Multi-Primitive Composition\n$\\mathcal{P}_{\\mathrm{set}}(\\mathbf{X}; \\mathbf{z}) = \\mathcal{P}_{u_{i_M}}(\\cdots \\mathcal{P}_{u_{i_1}}(\\mathbf{X}; \\xi_{i_1})\\cdots; \\xi_{i_M})$',
                      fontsize=14, fontweight='bold', pad=15)
        im_d3 = ax_d3.imshow(perturbation_map, aspect='auto', cmap='YlOrRd', 
                    vmin=0, vmax=0.9, interpolation='nearest')
        for i, prim in enumerate(primitives):
            for ch in prim['channels']:
                t_start, t_end = prim['time']
                rect_d3 = Rectangle((t_start-0.5, ch-0.5), t_end-t_start, 1,
                               linewidth=2, edgecolor=f'C{i}', 
                               facecolor='none', linestyle='--', alpha=0.8)
                ax_d3.add_patch(rect_d3)
                mid_t = (t_start + t_end) / 2
                ax_d3.text(mid_t, ch, f'$u_{i+1}$', ha='center', va='center',
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        ax_d3.set_ylabel('Channels $c$', fontsize=13, fontweight='bold')
        # 将Time标签移到时间轴末端（图内右下方）
        ax_d3.text(T_comp-1, -0.2, 'Time $t$', fontsize=13, fontweight='bold', ha='right', va='top')
        ax_d3.set_xticks([0, T_comp//4, T_comp//2, 3*T_comp//4, T_comp-1])
        ax_d3.set_xticklabels(['1', f'{T_comp//4}', f'{T_comp//2}', f'{3*T_comp//4}', f'{T_comp}'])
        ax_d3.set_yticks(range(C_comp))
        ax_d3.set_yticklabels([f'$c_{i+1}$' for i in range(C_comp)])
        cbar_d3 = plt.colorbar(im_d3, ax=ax_d3, fraction=0.046, pad=0.04)
        cbar_d3.set_label('Cumulative Perturbation Strength', fontsize=12)
        cbar_d3.ax.tick_params(labelsize=11)
        ax_d3.text(T_comp/2, -1.5, 
                'Overlapping masks result in additive superposition: $X\'_{v,t} = X_{v,t} + \\sum_{u: (v,t) \\in \\mathrm{mask}(u)} \\varepsilon_{v,t}^{(u)}$',
                ha='center', fontsize=10, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.5))
        fig_d3.savefig(os.path.join(save_dir, 'details_3.pdf'), dpi=DPI, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        fig_d3.savefig(os.path.join(save_dir, 'details_3.png'), dpi=DPI, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        plt.close(fig_d3)
    
    # 添加英文描述
    description_text_detail = (
        "This figure provides detailed views of the causal closure construction and perturbation mechanism. "
        
        "Left: The causal closure $\\mathrm{closure}(\\mathcal{C}_u)$ includes seed channels (red), "
        "their direct parents (orange, with negative lag offsets $\\Delta<0$), and direct children (green, with positive lag offsets $\\Delta>0$). "
        "Top right: Time-aligned perturbation windows show how lag offsets shift the perturbation time intervals. "
        "Bottom: Multi-primitive composition demonstrates how overlapping masks result in additive superposition of perturbations."
    )
    
    fig.text(0.5, 0.02, description_text_detail, ha='center', va='bottom', 
            fontsize=9, wrap=True, style='italic',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#F5F5F5', 
                     edgecolor='#CCCCCC', linewidth=1, alpha=0.9))
    
    plt.suptitle('Causal Closure and Perturbation Mechanism Details', 
                fontsize=15, fontweight='bold', y=0.98)
    
    return fig


if __name__ == '__main__':
    print("生成期刊论文示意图...")
    
    # 生成主流程图（单独保存每个子图）
    print("1. 生成主流程图...")
    save_directory = r'pictures\pipline'  # 注意：用户要求的是\pipline（拼写）
    fig1 = create_main_pipeline_figure(save_individual=True, save_dir=save_directory)
    print(f"   已单独保存子图到: {save_directory}")
    # 尝试保存完整图（如果文件未被占用）
    try:
        fig1.savefig('causal_perturbation_pipeline.pdf', dpi=DPI, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        fig1.savefig('causal_perturbation_pipeline.png', dpi=DPI, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print("   已保存完整图: causal_perturbation_pipeline.pdf/png")
    except PermissionError:
        print("   警告: 完整图文件被占用，跳过保存（子图已单独保存）")
    plt.close(fig1)
    
    # 生成细节图（单独保存每个子图）
    print("2. 生成细节图...")
    save_directory = r'pictures\pipline'  # 与主图保存在同一目录
    fig2 = create_detail_figure(save_individual=True, save_dir=save_directory)
    print(f"   已单独保存子图到: {save_directory}")
    try:
        fig2.savefig('causal_perturbation_details.pdf', dpi=DPI, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        fig2.savefig('causal_perturbation_details.png', dpi=DPI, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print("   已保存完整图: causal_perturbation_details.pdf/png")
    except PermissionError:
        print("   警告: 细节图文件被占用，跳过保存（子图已单独保存）")
    plt.close(fig2)
    
    print("\n所有示意图已生成完成！")
    print("文件格式: PDF (矢量图，适合论文) 和 PNG (高分辨率位图)")
    
    # 可选：显示图像
    # plt.show()
