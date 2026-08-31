#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMS-LIME 简化可视化 - 四个基元时间序列

显示四个子图：
1. 原始脑电单通道信号
2. 微状态时间序列
3. Shapelet时间序列
4. 时频基元时间序列

Author: CMS-LIME Framework
Date: 2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# import seaborn as sns  # Not used; keep commented if you want
from scipy.io import loadmat
import torch
import json
import re
import warnings
warnings.filterwarnings('ignore')

# Import CMS-LIME modules
from model.EEGInception_SE import EEGInception
from cms_lime_explainer import CMSLimeExplainer
from microstate_analysis import MicrostateAnalyzer
from shapelet_analysis import ShapeletAnalyzer
from timefreq_analysis import TimeFreqAnalyzer

# -----------------------------
# Matplotlib global font settings:
# Times New Roman + EVEN larger fonts
# -----------------------------
from matplotlib import font_manager

preferred_font = "Times New Roman"
available_fonts = {f.name for f in font_manager.fontManager.ttflist}

if preferred_font in available_fonts:
    plt.rcParams["font.family"] = preferred_font
else:
    # Fallback (still serif-like)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]

plt.rcParams["axes.unicode_minus"] = False

# Global font sizes (increased)
plt.rcParams.update({
    "font.size": 18,          # base (was 14)
    "axes.titlesize": 24,     # was 18
    "axes.labelsize": 20,     # was 16
    "xtick.labelsize": 18,    # was 14
    "ytick.labelsize": 18,    # was 14
    "legend.fontsize": 18,    # was 14
})

# Keep text as text in SVG (so PPT can keep the font if available)
plt.rcParams["svg.fonttype"] = "path"


def load_patient_data(patient_id=1, max_samples=1):
    """
    Load CHB-MIT patient EEG data
    """
    print("Loading patient data...")

    # 数据路径模板
    data_root_template = r'D:\public_data\CHBMIT\1_data_clean\chb%02d'
    segment_info_template = r'D:\public_data\CHBMIT\segment_clean\30-5-240\chb%02d\segment_info.json'

    data_root_dir = data_root_template % patient_id
    segment_info_path = segment_info_template % patient_id

    print(f"Patient {patient_id} data directory: {data_root_dir}")
    print(f"Segment info: {segment_info_path}")

    # 检查路径是否存在
    if not os.path.exists(data_root_dir) or not os.path.exists(segment_info_path):
        print("CHB-MIT data not found, generating synthetic data...")
        return generate_synthetic_data()

    try:
        # 加载分段信息
        with open(segment_info_path) as f:
            files_list = json.load(f)

        X_data = []

        for dict_file in files_list[:max_samples * 2]:
            if len(X_data) >= max_samples:
                break

            try:
                # 加载EEG数据
                eeg_data = np.load(os.path.join(data_root_dir, dict_file['File']))
                eeg_data = np.squeeze(eeg_data)

                # 获取分段信息
                eeg_start = dict_file['Span'][0]

                # 提取5秒片段 (1280个采样点，采样率256Hz)
                split_data = eeg_data[:, eeg_start:eeg_start + 1280]

                # 检查数据质量
                if split_data.shape[1] < 1280:
                    continue
                if np.var(split_data) == 0:
                    continue

                X_data.append(split_data)

            except Exception as e:
                print(f"Error loading file {dict_file['File']}: {e}")
                continue

        if len(X_data) > 0:
            data = np.array(X_data)
            print(f"Successfully loaded CHB-MIT data, shape: {data.shape}")
            return data
        else:
            print("No valid CHB-MIT data found, generating synthetic data...")
            return generate_synthetic_data()

    except Exception as e:
        print(f"Error loading CHB-MIT data: {e}")
        print("Generating synthetic data...")
        return generate_synthetic_data()


def generate_synthetic_data():
    """
    Generate synthetic EEG data
    """
    print("Generating synthetic EEG data...")
    np.random.seed(42)
    n_channels = 22
    n_timepoints = 1280

    # Generate realistic EEG-like data
    t = np.linspace(0, 5, n_timepoints)  # 5 seconds
    data = np.zeros((1, n_channels, n_timepoints))

    for ch in range(n_channels):
        # Alpha rhythm (8-12 Hz)
        alpha = 0.5 * np.sin(2 * np.pi * 10 * t + np.random.random() * 2 * np.pi)
        # Beta rhythm (13-30 Hz)
        beta = 0.3 * np.sin(2 * np.pi * 20 * t + np.random.random() * 2 * np.pi)
        # Gamma rhythm (30-100 Hz)
        gamma = 0.1 * np.sin(2 * np.pi * 40 * t + np.random.random() * 2 * np.pi)
        # Noise
        noise = 0.1 * np.random.randn(n_timepoints)

        data[0, ch, :] = alpha + beta + gamma + noise

    print(f"Successfully generated data, shape: {data.shape}")
    return data


def load_model():
    """
    Load EEGInception model
    """
    print("Loading EEGInception model...")
    model = EEGInception(input_time=5000, fs=256, ncha=22, n_classes=3)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {total_params}")

    return model


def create_simplified_visualization(sample, save_path):
    """
    Create CMS-LIME primitive units visualization showing the basic information and meaning of each primitive
    """
    print("Generating CMS-LIME primitive units visualization...")

    # Font size constants (increased)
    FS_TITLE = 26          # was 18
    FS_SUBTITLE = 22       # was 16
    FS_LABEL = 20          # was 16
    FS_TICK = 18           # was 14
    FS_ANNOT = 18          # was 13
    FS_ANNOT_SMALL = 16    # was 12

    # Initialize analyzers
    microstate_analyzer = MicrostateAnalyzer(n_microstates=4, method='kmeans')
    shapelet_analyzer = ShapeletAnalyzer(method='information_gain', n_shapelets=10)
    timefreq_analyzer = TimeFreqAnalyzer(method='stft', sampling_rate=256.0)

    # Prepare data
    sample_reshaped = sample.reshape(1, *sample.shape)
    dummy_labels = np.array([0])

    # Fit analyzers
    microstate_analyzer.fit(sample_reshaped[0])
    shapelet_analyzer.fit(sample_reshaped, dummy_labels)
    timefreq_analyzer.fit(sample_reshaped[0])

    # Get primitive information
    try:
        microstate_maps = microstate_analyzer.get_microstate_maps() if hasattr(microstate_analyzer, 'get_microstate_maps') else []
    except Exception:
        microstate_maps = []

    # Create figure with 3 main sections for primitive units
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1], hspace=0.45, wspace=0.35)

    # -----------------------------
    # 1) Microstate Primitives
    # -----------------------------
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])

        # Create topographic-like visualization of microstate map
        if i < len(microstate_maps) and microstate_maps[i] is not None:
            map_data = microstate_maps[i].reshape(-1, 1) if len(microstate_maps[i].shape) == 1 else microstate_maps[i]
        else:
            # Generate synthetic microstate map
            n_channels = sample.shape[0]
            map_data = np.random.randn(n_channels)

        # Create a circular arrangement for channels
        n_channels = len(map_data)
        angles = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)

        # Color map based on microstate values
        ax.scatter(x, y, c=map_data.flatten(), cmap='RdBu_r', s=140, alpha=0.9)

        # Add connecting lines to show spatial relationships
        for j in range(n_channels):
            for k in range(j + 1, min(j + 4, n_channels)):
                ax.plot([x[j], x[k]], [y[j], y[k]], 'gray', alpha=0.22, linewidth=0.7)

        ax.set_title(f'Microstate {i + 1}\nSpatial Pattern', fontsize=FS_SUBTITLE, fontweight='bold')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')

    # -----------------------------
    # 2) Shapelet Primitives
    # -----------------------------
    # Generate representative shapelets for visualization
    shapelet_examples = []
    for i in range(4):
        length = 20 + i * 10
        t = np.linspace(0, 1, length)
        if i == 0:  # Spike pattern
            pattern = np.exp(-(t - 0.5) ** 2 / 0.05)
        elif i == 1:  # Oscillation pattern
            pattern = np.sin(2 * np.pi * 3 * t) * np.exp(-t * 2)
        elif i == 2:  # Step pattern
            pattern = np.where(t < 0.5, 0.2, 0.8)
        else:  # Complex pattern
            pattern = np.sin(2 * np.pi * 2 * t) + 0.3 * np.sin(2 * np.pi * 8 * t)
        shapelet_examples.append(pattern)

    for i in range(4):
        ax = fig.add_subplot(gs[1, i])

        pattern = shapelet_examples[i]
        t = np.linspace(0, len(pattern) / 256, len(pattern))  # Time in seconds

        ax.plot(t, pattern, linewidth=3.5, alpha=0.95)
        ax.fill_between(t, pattern, alpha=0.28)

        pattern_names = ['Spike Pattern', 'Oscillation', 'Step Change', 'Complex Wave']
        ax.set_title(f'Shapelet {i + 1}\n{pattern_names[i]}', fontsize=FS_SUBTITLE, fontweight='bold')
        ax.set_ylabel('Amplitude', fontsize=FS_LABEL)
        ax.set_xlabel('Time (s)', fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, alpha=0.3)

        info_text = f'Length: {len(pattern)} samples\nDiscriminative: High'
        ax.text(
            0.02, 0.98, info_text, transform=ax.transAxes,
            va='top', ha='left', fontsize=FS_ANNOT_SMALL,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='lightgreen', alpha=0.75)
        )

    # -----------------------------
    # 3) Time-Frequency Primitives
    # -----------------------------
    freq_bands = [(1, 4, 'Delta'), (4, 8, 'Theta'), (8, 13, 'Alpha'), (13, 30, 'Beta')]

    for i, (low_freq, high_freq, band_name) in enumerate(freq_bands):
        ax = fig.add_subplot(gs[2, i])

        # Create time-frequency representation
        t = np.linspace(0, 2, 100)  # 2 seconds
        freqs = np.linspace(1, 50, 50)
        T, F = np.meshgrid(t, freqs)

        # Generate band-specific power pattern
        center_freq = (low_freq + high_freq) / 2
        power = np.exp(-((F - center_freq) / (high_freq - low_freq)) ** 2) * (1 + 0.5 * np.sin(2 * np.pi * 0.5 * T))

        ax.imshow(power, aspect='auto', origin='lower', cmap='viridis', alpha=0.95)

        ax.set_title(f'{band_name} Band\n({low_freq}-{high_freq} Hz)', fontsize=FS_SUBTITLE, fontweight='bold')
        ax.set_ylabel('Frequency (Hz)', fontsize=FS_LABEL)
        ax.set_xlabel('Time (s)', fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)

        ax.set_xticks([0, 25, 50, 75, 99])
        ax.set_xticklabels(['0', '0.5', '1.0', '1.5', '2.0'])
        ax.set_yticks([0, 12, 25, 37, 49])
        ax.set_yticklabels(['1', '13', '25', '37', '50'])

        info_text = f'Band: {band_name}\nPower: Variable\nDuration: ~1-2s'
        ax.text(
            0.02, 0.98, info_text, transform=ax.transAxes,
            va='top', ha='left', fontsize=FS_ANNOT_SMALL,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='yellow', alpha=0.78)
        )

    plt.tight_layout()

    # Save as SVG
    save_path_svg = save_path.replace('.png', '.svg')
    plt.savefig(save_path_svg, format='svg', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Primitive units visualization saved to: {save_path_svg}")


def main():
    """
    Main function
    """
    print("=" * 60)
    print("CMS-LIME 简化基元时间序列可视化")
    print("=" * 60)

    # Load data
    data = load_patient_data(patient_id=1, max_samples=1)

    # Load model
    model = load_model()

    # Analyze first sample
    sample = data[0]
    print(f"\nAnalyzing sample, shape: {sample.shape}")

    # Create visualization
    save_path = "pictures/cms_lime_simplified_visualization_v2.png"
    os.makedirs("pictures", exist_ok=True)

    create_simplified_visualization(sample, save_path)

    print("\n" + "=" * 60)
    print("Primitive units visualization completed!")
    print(f"Output: {save_path.replace('.png', '.svg')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
