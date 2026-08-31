#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基元库比较工具
比较多个基元库的信息，展示基元相似性和使用频率
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict, Counter
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PrimitiveLibraryComparator:
    def __init__(self):
        self.libraries = {}
        self.comparison_results = {}
        
    def load_library(self, library_path, library_name):
        """加载基元库"""
        try:
            with open(library_path, 'rb') as f:
                library_data = pickle.load(f)
            
            # 处理不同的数据结构
            if isinstance(library_data, dict):
                if 'primitives' in library_data:
                    primitives = library_data['primitives']
                    if isinstance(primitives, dict):
                        # 将字典转换为列表
                        primitives_list = []
                        for prim_id, prim_obj in primitives.items():
                            # 提取基元信息
                            prim_info = {
                                'id': prim_id,
                                'type': getattr(prim_obj, 'primitive_type', 'unknown'),
                                'channels': getattr(prim_obj, 'channels', []),
                                'time_range': getattr(prim_obj, 'time_range', (0, 0)),
                                'quality': float(getattr(prim_obj, 'quality_score', 0)),
                                'usage': int(getattr(prim_obj, 'usage_count', 0))
                            }
                            primitives_list.append(prim_info)
                        
                        # 创建简化的库对象
                        class SimpleLibrary:
                            def __init__(self, primitives):
                                self.primitives = primitives
                        
                        self.libraries[library_name] = SimpleLibrary(primitives_list)
                    else:
                        self.libraries[library_name] = SimpleLibrary(list(primitives))
                else:
                    print(f"⚠️  未找到'primitives'键: {library_name}")
                    return False
            else:
                # 假设是直接的库对象
                self.libraries[library_name] = library_data
            
            print(f"✅ 成功加载基元库: {library_name}")
            print(f"   基元总数: {len(self.libraries[library_name].primitives)}")
            
            # 统计各类型基元数量
            type_counts = defaultdict(int)
            for primitive in self.libraries[library_name].primitives:
                ptype = primitive.get('type', 'unknown') if isinstance(primitive, dict) else getattr(primitive, 'type', 'unknown')
                type_counts[ptype] += 1
            
            print(f"   类型分布: {dict(type_counts)}")
            return True
            
        except Exception as e:
            print(f"❌ 加载基元库失败 {library_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_primitive_features(self, primitive):
        """提取基元特征用于比较"""
        features = {
            'type': primitive.get('type', 'unknown'),
            'quality': primitive.get('quality', 0),
            'usage': primitive.get('usage', 0),
            'channels': len(primitive.get('channels', [])),
            'time_range': primitive.get('time_range', (0, 0)),
            'duration': 0
        }
        
        # 计算持续时间
        if features['time_range'] and len(features['time_range']) >= 2:
            features['duration'] = features['time_range'][1] - features['time_range'][0]
        
        return features
    
    def find_similar_primitives(self, threshold=0.8):
        """查找相似的基元"""
        print(f"\n🔍 查找相似基元 (相似度阈值: {threshold})...")
        
        similar_groups = []
        library_names = list(self.libraries.keys())
        
        for i, lib1_name in enumerate(library_names):
            for j, lib2_name in enumerate(library_names[i+1:], i+1):
                lib1 = self.libraries[lib1_name]
                lib2 = self.libraries[lib2_name]
                
                similar_pairs = []
                
                for p1_idx, p1 in enumerate(lib1.primitives):
                    for p2_idx, p2 in enumerate(lib2.primitives):
                        similarity = self.calculate_similarity(p1, p2)
                        
                        if similarity >= threshold:
                            similar_pairs.append({
                                'lib1': lib1_name,
                                'lib2': lib2_name,
                                'p1_idx': p1_idx,
                                'p2_idx': p2_idx,
                                'p1_id': p1.get('id', f'p1_{p1_idx}'),
                                'p2_id': p2.get('id', f'p2_{p2_idx}'),
                                'similarity': similarity,
                                'type': p1.get('type', 'unknown')
                            })
                
                if similar_pairs:
                    similar_groups.append({
                        'libraries': (lib1_name, lib2_name),
                        'pairs': similar_pairs
                    })
        
        self.comparison_results['similar_groups'] = similar_groups
        return similar_groups
    
    def calculate_similarity(self, p1, p2):
        """计算两个基元的相似度"""
        # 基本相似度计算
        similarity = 0.0
        
        # 类型匹配 (权重: 0.4)
        if p1.get('type') == p2.get('type'):
            similarity += 0.4
        
        # 质量相似度 (权重: 0.2)
        q1 = p1.get('quality', 0)
        q2 = p2.get('quality', 0)
        if q1 > 0 and q2 > 0:
            quality_sim = 1 - abs(q1 - q2) / max(q1, q2)
            similarity += 0.2 * quality_sim
        
        # 通道数相似度 (权重: 0.2)
        c1 = len(p1.get('channels', []))
        c2 = len(p2.get('channels', []))
        if c1 > 0 and c2 > 0:
            channel_sim = 1 - abs(c1 - c2) / max(c1, c2)
            similarity += 0.2 * channel_sim
        
        # 持续时间相似度 (权重: 0.2)
        t1 = p1.get('time_range', (0, 0))
        t2 = p2.get('time_range', (0, 0))
        if len(t1) >= 2 and len(t2) >= 2:
            d1 = t1[1] - t1[0]
            d2 = t2[1] - t2[0]
            if d1 > 0 and d2 > 0:
                duration_sim = 1 - abs(d1 - d2) / max(d1, d2)
                similarity += 0.2 * duration_sim
        
        return similarity
    
    def analyze_usage_patterns(self):
        """分析使用模式"""
        print("\n📊 分析使用模式...")
        
        usage_data = []
        
        for lib_name, library in self.libraries.items():
            for i, primitive in enumerate(library.primitives):
                usage_data.append({
                    'library': lib_name,
                    'primitive_id': primitive.get('id', f'p_{i}'),
                    'type': primitive.get('type', 'unknown'),
                    'usage': primitive.get('usage', 0),
                    'quality': primitive.get('quality', 0)
                })
        
        self.comparison_results['usage_data'] = usage_data
        return usage_data
    
    def create_comparison_visualizations(self, save_dir='library_comparison_results'):
        """创建比较可视化图表"""
        print(f"\n🎨 创建可视化图表...")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 基元库概览
        self.plot_library_overview(save_dir)
        
        # 2. 相似性热力图
        self.plot_similarity_heatmap(save_dir)
        
        # 3. 使用频率分析
        self.plot_usage_analysis(save_dir)
        
        # 4. 基元类型分布
        self.plot_type_distribution(save_dir)
        
        # 5. 质量vs使用频率散点图
        self.plot_quality_usage_scatter(save_dir)
        
        print(f"✅ 所有图表已保存到: {save_dir}")
    
    def plot_library_overview(self, save_dir):
        """绘制基元库概览"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('基元库概览比较', fontsize=16, fontweight='bold')
        
        # 基元总数比较
        lib_names = list(self.libraries.keys())
        lib_counts = [len(lib.primitives) for lib in self.libraries.values()]
        
        axes[0, 0].bar(range(len(lib_names)), lib_counts, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('基元总数比较')
        axes[0, 0].set_xlabel('基元库')
        axes[0, 0].set_ylabel('基元数量')
        axes[0, 0].set_xticks(range(len(lib_names)))
        axes[0, 0].set_xticklabels([name.split('_')[-1] for name in lib_names], rotation=45)
        
        # 添加数值标签
        for i, count in enumerate(lib_counts):
            axes[0, 0].text(i, count + max(lib_counts)*0.01, str(count), 
                           ha='center', va='bottom', fontweight='bold')
        
        # 类型分布堆叠柱状图
        type_data = defaultdict(list)
        for lib_name, library in self.libraries.items():
            type_counts = defaultdict(int)
            for primitive in library.primitives:
                type_counts[primitive.get('type', 'unknown')] += 1
            
            for ptype in ['microstate', 'shapelet', 'timefreq', 'unknown']:
                type_data[ptype].append(type_counts.get(ptype, 0))
        
        bottom = np.zeros(len(lib_names))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, (ptype, counts) in enumerate(type_data.items()):
            axes[0, 1].bar(range(len(lib_names)), counts, bottom=bottom, 
                          label=ptype, color=colors[i % len(colors)], alpha=0.8)
            bottom += np.array(counts)
        
        axes[0, 1].set_title('基元类型分布')
        axes[0, 1].set_xlabel('基元库')
        axes[0, 1].set_ylabel('基元数量')
        axes[0, 1].set_xticks(range(len(lib_names)))
        axes[0, 1].set_xticklabels([name.split('_')[-1] for name in lib_names], rotation=45)
        axes[0, 1].legend()
        
        # 平均质量比较
        avg_qualities = []
        for library in self.libraries.values():
            qualities = [p.get('quality', 0) for p in library.primitives if p.get('quality', 0) > 0]
            avg_qualities.append(np.mean(qualities) if qualities else 0)
        
        axes[1, 0].bar(range(len(lib_names)), avg_qualities, color='lightcoral', alpha=0.7)
        axes[1, 0].set_title('平均基元质量')
        axes[1, 0].set_xlabel('基元库')
        axes[1, 0].set_ylabel('平均质量')
        axes[1, 0].set_xticks(range(len(lib_names)))
        axes[1, 0].set_xticklabels([name.split('_')[-1] for name in lib_names], rotation=45)
        
        # 添加数值标签
        for i, quality in enumerate(avg_qualities):
            axes[1, 0].text(i, quality + max(avg_qualities)*0.01, f'{quality:.2f}', 
                           ha='center', va='bottom', fontweight='bold')
        
        # 使用频率分布
        all_usages = []
        for library in self.libraries.values():
            usages = [p.get('usage', 0) for p in library.primitives]
            all_usages.extend(usages)
        
        axes[1, 1].hist(all_usages, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('使用频率分布')
        axes[1, 1].set_xlabel('使用次数')
        axes[1, 1].set_ylabel('基元数量')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'library_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_similarity_heatmap(self, save_dir):
        """绘制相似性热力图"""
        if 'similar_groups' not in self.comparison_results:
            self.find_similar_primitives()
        
        lib_names = list(self.libraries.keys())
        n_libs = len(lib_names)
        
        # 创建相似性矩阵
        similarity_matrix = np.zeros((n_libs, n_libs))
        
        for group in self.comparison_results['similar_groups']:
            lib1, lib2 = group['libraries']
            lib1_idx = lib_names.index(lib1)
            lib2_idx = lib_names.index(lib2)
            
            # 计算平均相似度
            similarities = [pair['similarity'] for pair in group['pairs']]
            avg_similarity = np.mean(similarities) if similarities else 0
            
            similarity_matrix[lib1_idx, lib2_idx] = avg_similarity
            similarity_matrix[lib2_idx, lib1_idx] = avg_similarity
        
        # 对角线设为1
        np.fill_diagonal(similarity_matrix, 1.0)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, 
                   xticklabels=[name.split('_')[-1] for name in lib_names],
                   yticklabels=[name.split('_')[-1] for name in lib_names],
                   annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': '相似度'})
        
        plt.title('基元库间相似性热力图', fontsize=14, fontweight='bold')
        plt.xlabel('基元库')
        plt.ylabel('基元库')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'similarity_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_usage_analysis(self, save_dir):
        """绘制使用频率分析"""
        if 'usage_data' not in self.comparison_results:
            self.analyze_usage_patterns()
        
        usage_df = pd.DataFrame(self.comparison_results['usage_data'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('基元使用频率分析', fontsize=16, fontweight='bold')
        
        # 1. 各库高使用频率基元数量
        high_usage_threshold = usage_df['usage'].quantile(0.8)  # 前20%
        high_usage_counts = usage_df[usage_df['usage'] >= high_usage_threshold].groupby('library').size()
        
        lib_short_names = [name.split('_')[-1] for name in high_usage_counts.index]
        axes[0, 0].bar(range(len(lib_short_names)), high_usage_counts.values, 
                      color='orange', alpha=0.7)
        axes[0, 0].set_title(f'高使用频率基元数量\n(使用次数 ≥ {high_usage_threshold:.1f})')
        axes[0, 0].set_xlabel('基元库')
        axes[0, 0].set_ylabel('基元数量')
        axes[0, 0].set_xticks(range(len(lib_short_names)))
        axes[0, 0].set_xticklabels(lib_short_names, rotation=45)
        
        # 添加数值标签
        for i, count in enumerate(high_usage_counts.values):
            axes[0, 0].text(i, count + max(high_usage_counts.values)*0.01, str(count), 
                           ha='center', va='bottom', fontweight='bold')
        
        # 2. 各类型基元使用频率箱线图
        usage_df_filtered = usage_df[usage_df['usage'] > 0]  # 只显示有使用的基元
        if not usage_df_filtered.empty:
            sns.boxplot(data=usage_df_filtered, x='type', y='usage', ax=axes[0, 1])
            axes[0, 1].set_title('各类型基元使用频率分布')
            axes[0, 1].set_xlabel('基元类型')
            axes[0, 1].set_ylabel('使用次数')
        
        # 3. 各库平均使用频率
        avg_usage = usage_df.groupby('library')['usage'].mean()
        lib_short_names = [name.split('_')[-1] for name in avg_usage.index]
        
        axes[1, 0].bar(range(len(lib_short_names)), avg_usage.values, 
                      color='lightblue', alpha=0.7)
        axes[1, 0].set_title('各库平均使用频率')
        axes[1, 0].set_xlabel('基元库')
        axes[1, 0].set_ylabel('平均使用次数')
        axes[1, 0].set_xticks(range(len(lib_short_names)))
        axes[1, 0].set_xticklabels(lib_short_names, rotation=45)
        
        # 添加数值标签
        for i, usage in enumerate(avg_usage.values):
            axes[1, 0].text(i, usage + max(avg_usage.values)*0.01, f'{usage:.2f}', 
                           ha='center', va='bottom', fontweight='bold')
        
        # 4. 使用频率Top10基元
        top_used = usage_df.nlargest(10, 'usage')
        if not top_used.empty:
            y_pos = np.arange(len(top_used))
            axes[1, 1].barh(y_pos, top_used['usage'], color='green', alpha=0.7)
            axes[1, 1].set_title('使用频率Top10基元')
            axes[1, 1].set_xlabel('使用次数')
            axes[1, 1].set_ylabel('基元')
            
            # 设置y轴标签
            labels = [f"{row['library'].split('_')[-1]}:{row['type']}:{row['primitive_id'][:8]}" 
                     for _, row in top_used.iterrows()]
            axes[1, 1].set_yticks(y_pos)
            axes[1, 1].set_yticklabels(labels, fontsize=8)
            axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'usage_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_type_distribution(self, save_dir):
        """绘制基元类型分布"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('基元类型分布分析', fontsize=16, fontweight='bold')
        
        # 1. 各库类型分布饼图
        lib_names = list(self.libraries.keys())
        n_libs = len(lib_names)
        
        # 计算子图布局
        cols = min(2, n_libs)
        rows = (n_libs + cols - 1) // cols
        
        fig2, axes2 = plt.subplots(rows, cols, figsize=(12, 4*rows))
        if n_libs == 1:
            axes2 = [axes2]
        elif rows == 1:
            axes2 = axes2.reshape(1, -1)
        
        for i, (lib_name, library) in enumerate(self.libraries.items()):
            type_counts = defaultdict(int)
            for primitive in library.primitives:
                type_counts[primitive.get('type', 'unknown')] += 1
            
            row, col = i // cols, i % cols
            ax = axes2[row, col] if rows > 1 else axes2[col]
            
            if type_counts:
                wedges, texts, autotexts = ax.pie(type_counts.values(), 
                                                 labels=type_counts.keys(),
                                                 autopct='%1.1f%%', 
                                                 startangle=90)
                ax.set_title(f'{lib_name.split("_")[-1]}\n({sum(type_counts.values())}个基元)')
        
        # 隐藏多余的子图
        for i in range(n_libs, rows * cols):
            row, col = i // cols, i % cols
            ax = axes2[row, col] if rows > 1 else axes2[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'type_distribution_pies.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 类型分布对比
        type_data = defaultdict(list)
        for lib_name, library in self.libraries.items():
            type_counts = defaultdict(int)
            for primitive in library.primitives:
                type_counts[primitive.get('type', 'unknown')] += 1
            
            for ptype in ['microstate', 'shapelet', 'timefreq', 'unknown']:
                type_data[ptype].append(type_counts.get(ptype, 0))
        
        # 堆叠柱状图
        lib_short_names = [name.split('_')[-1] for name in lib_names]
        bottom = np.zeros(len(lib_names))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, (ptype, counts) in enumerate(type_data.items()):
            axes[0].bar(range(len(lib_names)), counts, bottom=bottom, 
                       label=ptype, color=colors[i % len(colors)], alpha=0.8)
            bottom += np.array(counts)
        
        axes[0].set_title('基元类型分布对比')
        axes[0].set_xlabel('基元库')
        axes[0].set_ylabel('基元数量')
        axes[0].set_xticks(range(len(lib_names)))
        axes[0].set_xticklabels(lib_short_names, rotation=45)
        axes[0].legend()
        
        # 3. 类型比例对比
        proportions = []
        for lib_name, library in self.libraries.items():
            type_counts = defaultdict(int)
            total = len(library.primitives)
            for primitive in library.primitives:
                type_counts[primitive.get('type', 'unknown')] += 1
            
            lib_proportions = []
            for ptype in ['microstate', 'shapelet', 'timefreq', 'unknown']:
                lib_proportions.append(type_counts.get(ptype, 0) / total if total > 0 else 0)
            proportions.append(lib_proportions)
        
        proportions = np.array(proportions)
        
        x = np.arange(len(lib_names))
        width = 0.2
        
        for i, ptype in enumerate(['microstate', 'shapelet', 'timefreq', 'unknown']):
            axes[1].bar(x + i*width, proportions[:, i], width, 
                       label=ptype, color=colors[i], alpha=0.8)
        
        axes[1].set_title('基元类型比例对比')
        axes[1].set_xlabel('基元库')
        axes[1].set_ylabel('比例')
        axes[1].set_xticks(x + width * 1.5)
        axes[1].set_xticklabels(lib_short_names, rotation=45)
        axes[1].legend()
        axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'type_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_quality_usage_scatter(self, save_dir):
        """绘制质量vs使用频率散点图"""
        if 'usage_data' not in self.comparison_results:
            self.analyze_usage_patterns()
        
        usage_df = pd.DataFrame(self.comparison_results['usage_data'])
        
        plt.figure(figsize=(12, 8))
        
        # 为每个库使用不同颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.libraries)))
        
        for i, (lib_name, color) in enumerate(zip(self.libraries.keys(), colors)):
            lib_data = usage_df[usage_df['library'] == lib_name]
            plt.scatter(lib_data['quality'], lib_data['usage'], 
                       c=[color], label=lib_name.split('_')[-1], 
                       alpha=0.6, s=50)
        
        plt.xlabel('基元质量')
        plt.ylabel('使用频率')
        plt.title('基元质量 vs 使用频率', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 添加趋势线
        if len(usage_df) > 1:
            z = np.polyfit(usage_df['quality'], usage_df['usage'], 1)
            p = np.poly1d(z)
            plt.plot(usage_df['quality'], p(usage_df['quality']), "r--", alpha=0.8, 
                    label=f'趋势线 (斜率: {z[0]:.3f})')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'quality_usage_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comparison_report(self, save_dir='library_comparison_results'):
        """生成比较报告"""
        print("\n📝 生成比较报告...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'libraries_compared': list(self.libraries.keys()),
            'summary': {},
            'detailed_analysis': {}
        }
        
        # 基本统计
        for lib_name, library in self.libraries.items():
            lib_stats = {
                'total_primitives': len(library.primitives),
                'type_distribution': {},
                'avg_quality': 0,
                'avg_usage': 0,
                'high_usage_count': 0
            }
            
            type_counts = defaultdict(int)
            qualities = []
            usages = []
            
            for primitive in library.primitives:
                ptype = primitive.get('type', 'unknown')
                type_counts[ptype] += 1
                
                quality = primitive.get('quality', 0)
                if quality > 0:
                    qualities.append(quality)
                
                usage = primitive.get('usage', 0)
                usages.append(usage)
                if usage >= 5:  # 定义高使用频率阈值
                    lib_stats['high_usage_count'] += 1
            
            lib_stats['type_distribution'] = dict(type_counts)
            lib_stats['avg_quality'] = np.mean(qualities) if qualities else 0
            lib_stats['avg_usage'] = np.mean(usages) if usages else 0
            
            report['summary'][lib_name] = lib_stats
        
        # 相似性分析
        if 'similar_groups' in self.comparison_results:
            similarity_summary = []
            for group in self.comparison_results['similar_groups']:
                similarity_summary.append({
                    'libraries': group['libraries'],
                    'similar_pairs_count': len(group['pairs']),
                    'avg_similarity': np.mean([p['similarity'] for p in group['pairs']])
                })
            report['detailed_analysis']['similarity'] = similarity_summary
        
        # 保存报告
        report_path = os.path.join(save_dir, 'comparison_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 比较报告已保存: {report_path}")
        return report

def main():
    """主函数"""
    print("🔍 基元库比较分析工具")
    print("=" * 50)
    
    # 定义要比较的基元库
    libraries_to_compare = {
        'PATIENT1': r'primitive_libraries\enhanced_cms_lime_library_20250923_111248-PATIENT1\enhanced_cms_lime_primitives.pkl',
        'PATIENT23': r'primitive_libraries\enhanced_cms_lime_library_20250923_134926-PATIENT23\enhanced_cms_lime_primitives.pkl',
        'LIB_205803': r'primitive_libraries\enhanced_cms_lime_library_20250923_205803\enhanced_cms_lime_primitives.pkl',
        'LIB_161715': r'primitive_libraries\enhanced_cms_lime_library_20250923_161715\enhanced_cms_lime_primitives.pkl'
    }
    
    # 创建比较器
    comparator = PrimitiveLibraryComparator()
    
    # 加载基元库
    loaded_count = 0
    for lib_name, lib_path in libraries_to_compare.items():
        if os.path.exists(lib_path):
            if comparator.load_library(lib_path, lib_name):
                loaded_count += 1
        else:
            print(f"⚠️  基元库文件不存在: {lib_path}")
    
    if loaded_count < 2:
        print("❌ 至少需要2个基元库才能进行比较")
        return
    
    print(f"\n✅ 成功加载 {loaded_count} 个基元库")
    
    # 执行分析
    print("\n🔄 开始分析...")
    
    # 查找相似基元
    comparator.find_similar_primitives(threshold=0.7)
    
    # 分析使用模式
    comparator.analyze_usage_patterns()
    
    # 创建可视化
    save_dir = f'library_comparison_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    comparator.create_comparison_visualizations(save_dir)
    
    # 生成报告
    comparator.generate_comparison_report(save_dir)
    
    print("\n🎉 分析完成！")
    print(f"📁 结果保存在: {save_dir}")
    
    # 打印简要总结
    print("\n📊 简要总结:")
    for lib_name, library in comparator.libraries.items():
        print(f"  {lib_name}: {len(library.primitives)} 个基元")
    
    if 'similar_groups' in comparator.comparison_results:
        total_similar_pairs = sum(len(group['pairs']) for group in comparator.comparison_results['similar_groups'])
        print(f"\n🔗 发现 {total_similar_pairs} 对相似基元")

if __name__ == '__main__':
    main()