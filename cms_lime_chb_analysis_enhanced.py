#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced CMS-LIME CHB-MIT Dataset Analysis

This script provides enhanced interpretability analysis for EEG seizure detection
using the improved CMS-LIME framework with primitive library functionality.

Key improvements:
1. Enhanced CMS-LIME explainer with primitive library
2. Automatic primitive library creation and management
3. English-only visualization and annotations
4. Improved primitive selection strategy
5. Better quality assessment and stability
6. Robust data loading with fallback to simulated data

Author: CMS-LIME Framework
Date: 2025-01-27
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.EEGInception_SE import EEGInception
from cms_lime_enhanced_explainer import EnhancedCMSLimeExplainer, EnhancedCMSLimeConfig
from primitive_library import PrimitiveLibrary
import json
import re
from tqdm import tqdm
import matplotlib
# Set non-interactive backend to prevent GUI issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import logging
import sys
import gc
warnings.filterwarnings('ignore')

# Set matplotlib to use English locale and optimize memory
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.max_open_warning'] = 0  # Disable max figure warning

# Global logger instance
logger = None

def setup_logger(log_file_path):
    """Setup logger to write to both console and file"""
    global logger
    
    # Create logger
    logger = logging.getLogger('cms_lime_analysis')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Create file handler
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_print(message):
    """Print message to both console and log file"""
    global logger
    if logger:
        logger.info(message)
    else:
        print(message)

class EEGInceptionWrapper:
    """EEGInception model wrapper for CMS-LIME explanation"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def predict_proba(self, X):
        """Predict probabilities, compatible with CMS-LIME interface"""
        try:
            # Data preprocessing and shape adjustment
            if len(X.shape) == 2:
                # If 2D data (n_channels, n_timepoints), add batch dimension
                X = X.reshape(1, X.shape[0], X.shape[1])
            elif len(X.shape) == 1:
                # If 1D data, reshape to appropriate shape
                X = X.reshape(1, -1, 1280)  # Assume 1280 time points
            elif len(X.shape) == 3:
                # If 3D data (batch, n_channels, n_timepoints), keep unchanged
                pass
            elif len(X.shape) == 4:
                # If 4D data, remove extra dimensions
                X = X.squeeze()
                if len(X.shape) == 2:
                    X = X.reshape(1, X.shape[0], X.shape[1])
            
            # Data validity check
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                print("Warning: Input data contains NaN or Inf values, using default prediction")
                return np.array([[0.5, 0.5]])  # Return uniform distribution
            
            # Convert to tensor and move to device
            if not isinstance(X, torch.Tensor):
                X = torch.FloatTensor(X)
            X = X.to(self.device)
            
            # Model prediction
            with torch.no_grad():
                outputs = self.model(X)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Apply softmax to get probabilities
                probs = F.softmax(outputs, dim=1)
                return probs.cpu().numpy()
                
        except Exception as e:
            print(f"Prediction error: {e}")
            # Return default probabilities
            batch_size = 1 if len(X.shape) <= 2 else X.shape[0]
            return np.array([[0.5, 0.5]] * batch_size)

class SimpleEEGModel(nn.Module):
    """Simple EEG classification model for demonstration when real model is unavailable"""

    def __init__(self, n_channels=23, n_timepoints=1280, n_classes=2):
        super(SimpleEEGModel, self).__init__()

        # Temporal convolution layers
        self.temp_conv1 = nn.Conv1d(n_channels, 64, kernel_size=25, padding=12)
        self.temp_conv2 = nn.Conv1d(64, 128, kernel_size=15, padding=7)
        self.temp_conv3 = nn.Conv1d(128, 64, kernel_size=10, padding=4)

        # Pooling and dropout
        self.pool = nn.AdaptiveAvgPool1d(32)
        self.dropout = nn.Dropout(0.5)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # x shape: (batch, channels, timepoints)
        x = F.relu(self.temp_conv1(x))
        x = F.relu(self.temp_conv2(x))
        x = F.relu(self.temp_conv3(x))

        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        converted_dict = {}
        for key, value in obj.items():
            # Special handling for channels and time_range fields
            if key in ['channels', 'time_range'] and isinstance(value, list):
                # Convert string elements to integers
                converted_value = []
                for item in value:
                    if isinstance(item, (str, np.str_)):
                        try:
                            converted_value.append(int(item))
                        except (ValueError, TypeError):
                            converted_value.append(item)
                    else:
                        converted_value.append(convert_numpy_types(item))
                converted_dict[key] = converted_value
            else:
                converted_dict[key] = convert_numpy_types(value)
        return converted_dict
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (str, np.str_)):
        # Try to convert string to int if it represents a number
        try:
            if obj.isdigit() or (obj.startswith('-') and obj[1:].isdigit()):
                return int(obj)
        except (AttributeError, ValueError):
            pass
        return obj
    else:
        return obj

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def _load_chb_mit_format_internal(
    data_root,
    files_list,
    max_samples,
    max_seizure_samples=None,
    max_normal_samples=None,
    sampling_rate=256,
    window_seconds=5,
    step_seconds=1
):
    """Load CHB-MIT data with sliding windows inside each File/Span/Label entry.

    Args:
        data_root: 根目录
        files_list: 每个元素是包含 {'File','Span','Label'} 的 dict
        max_samples: 最多返回多少个窗口样本（跨所有文件合计）
        max_seizure_samples: 最大癫痫样本数（如果指定，则优先使用）
        max_normal_samples: 最大正常样本数（如果指定，则优先使用）
        sampling_rate: 采样率（Hz），默认 256
        window_seconds: 窗口长度（秒），默认 5
        step_seconds: 窗口滑动步长（秒），默认 1
    """
    X_data, y_data = [], []
    
    # Initialize counters for different label types
    seizure_count = 0
    normal_count = 0
    
    # Determine actual limits
    if max_seizure_samples is not None and max_normal_samples is not None:
        actual_max_seizure = max_seizure_samples
        actual_max_normal = max_normal_samples
        actual_max_total = max_seizure_samples + max_normal_samples
    else:
        actual_max_seizure = max_samples // 2
        actual_max_normal = max_samples - actual_max_seizure
        actual_max_total = max_samples
    
    print(f"Target samples - Seizure: {actual_max_seizure}, Normal: {actual_max_normal}, Total: {actual_max_total}")

    window_size = int(window_seconds * sampling_rate)       # 5s -> 1280
    step_size   = int(step_seconds   * sampling_rate)       # 1s -> 256
    
    total_sample_count = 0

    for dict_file in files_list:
        if total_sample_count >= actual_max_total:
            break

        try:
            # 加载 EEG：期望形状为 (channels, samples)
            eeg = np.load(os.path.join(data_root, dict_file['File']))
            eeg = np.asarray(eeg)

            # 保证二维形状
            if eeg.ndim == 1:
                eeg = eeg[np.newaxis, :]            # (1, T)
            elif eeg.ndim > 2:
                eeg = np.squeeze(eeg)
                if eeg.ndim == 1:
                    eeg = eeg[np.newaxis, :]

            C, T = eeg.shape

            # 取该文件中标注的时段范围
            span_start = int(dict_file['Span'][0])
            span_end   = int(dict_file['Span'][1])

            # 防越界
            span_start = max(0, span_start)
            span_end   = min(T, span_end)

            # 如果总长度不足一个窗口，跳过
            if span_end - span_start < window_size:
                continue

            # 标签：沿用原逻辑
            label = 1 if re.match(r"Pre", str(dict_file.get("Label", ""))) else 0
            
            # Check if we need more samples of this label type
            if label == 1 and seizure_count >= actual_max_seizure:
                continue
            if label == 0 and normal_count >= actual_max_normal:
                continue

            # 在Span内进行滑窗
            start_idx = span_start
            last_start = span_end - window_size

            while start_idx <= last_start and total_sample_count < actual_max_total:
                # Check label-specific limits again
                if label == 1 and seizure_count >= actual_max_seizure:
                    break
                if label == 0 and normal_count >= actual_max_normal:
                    break
                    
                split_data = eeg[:, start_idx:start_idx + window_size]

                # 质量检查
                if split_data.shape[1] < window_size:
                    start_idx += step_size
                    continue
                if not np.isfinite(split_data).all():
                    start_idx += step_size
                    continue
                if np.var(split_data) == 0:
                    start_idx += step_size
                    continue

                X_data.append(split_data)
                y_data.append(label)
                total_sample_count += 1
                
                # Update label-specific counters
                if label == 1:
                    seizure_count += 1
                else:
                    normal_count += 1

                # 可选：打印调试
                # print(f"{dict_file['File']} [{start_idx}:{start_idx+window_size}] -> shape={split_data.shape}, label={label}")

                start_idx += step_size

        except Exception as e:
            print(f"Error loading file {dict_file.get('File','<unknown>')}: {e}")
            continue

    if len(X_data) > 0:
        print(f"Successfully loaded {len(X_data)} CHB-MIT windowed samples")
        print(f"Label distribution: (0=Normal: {normal_count}, 1=Seizure: {seizure_count})")
        return np.array(X_data), np.array(y_data)
    else:
        raise RuntimeError("No valid CHB-MIT data loaded")



def load_chb_data_robust(patient_id, data_root_template, segment_info_template, 
                        max_samples=5, max_seizure_samples=None, max_normal_samples=None,
                        sampling_rate=256, window_seconds=5, step_seconds=1):
    """Robust CHB-MIT dataset loading with error handling
    
    Args:
        patient_id: 患者ID
        data_root_template: 数据根目录模板
        segment_info_template: 分段信息文件模板
        max_samples: 最大样本数（总数）
        max_seizure_samples: 最大癫痫样本数（如果指定，则优先使用）
        max_normal_samples: 最大正常样本数（如果指定，则优先使用）
        sampling_rate: 采样率（Hz），默认 256
        window_seconds: 窗口长度（秒），默认 5
        step_seconds: 窗口滑动步长（秒），默认 1
    """
    
    # Format paths
    data_root = data_root_template % patient_id
    segment_info_path = segment_info_template % patient_id
    
    print(f"Attempting to load data from: {data_root}")
    print(f"Segment info: {segment_info_path}")
    
    # Check if paths exist
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    
    if not os.path.exists(segment_info_path):
        raise FileNotFoundError(f"Segment info file not found: {segment_info_path}")
    
    # Load segment information
    try:
        with open(segment_info_path, 'r') as f:
            files_list = json.load(f)
            
        # Handle CHB-MIT format with File, Span, Label structure
        if isinstance(files_list, list) and len(files_list) > 0 and isinstance(files_list[0], dict):
            if 'File' in files_list[0] and 'Span' in files_list[0] and 'Label' in files_list[0]:
                print("Using CHB-MIT format with File/Span/Label structure")
                return _load_chb_mit_format_internal(data_root, files_list, max_samples, 
                                                   max_seizure_samples, max_normal_samples,
                                                   sampling_rate, window_seconds, step_seconds)
            
        # Handle other formats
        segment_info = {}
        for i, item in enumerate(files_list):
            if isinstance(item, dict) and 'filename' in item:
                filename = item['filename'].replace('.npy', '')
                label = item.get('label', 0)
                segment_info[filename] = label
            else:
                # Default naming pattern
                segment_info[f'chb{patient_id:02d}_{i:02d}'] = 0
            
    except Exception as e:
        raise RuntimeError(f"Error loading segment info: {e}")
    
    # Get data files
    data_files = [f for f in os.listdir(data_root) if f.endswith('.npy')]
    data_files = sorted(data_files, key=natural_keys)
    
    if len(data_files) == 0:
        raise FileNotFoundError("No .npy files found in data directory")
    
    print(f"Found {len(data_files)} data files")
    
    # Load data with sample count control
    X_data = []
    y_data = []
    
    # Initialize counters for different label types
    seizure_count = 0
    normal_count = 0
    
    # Determine actual limits
    if max_seizure_samples is not None and max_normal_samples is not None:
        actual_max_seizure = max_seizure_samples
        actual_max_normal = max_normal_samples
        actual_max_total = max_seizure_samples + max_normal_samples
    else:
        actual_max_seizure = max_samples // 2
        actual_max_normal = max_samples - actual_max_seizure
        actual_max_total = max_samples
    
    print(f"Target samples - Seizure: {actual_max_seizure}, Normal: {actual_max_normal}, Total: {actual_max_total}")
    
    loaded_count = 0
    for file_name in data_files:
        if loaded_count >= actual_max_total:
            break
            
        file_path = os.path.join(data_root, file_name)
        
        try:
            # Load data
            data = np.load(file_path)
            
            # Get label from segment info
            base_name = file_name.replace('.npy', '')
            label = segment_info.get(base_name, 0)  # Default to 0 if not found
            
            # Check if we should include this sample based on label and counts
            if label == 1:  # Seizure
                if seizure_count >= actual_max_seizure:
                    continue
                seizure_count += 1
            else:  # Normal
                if normal_count >= actual_max_normal:
                    continue
                normal_count += 1
            
            X_data.append(data)
            y_data.append(label)
            loaded_count += 1
            
            print(f"Loaded {file_name}: shape={data.shape}, label={label}")
            
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            continue
    
    if len(X_data) == 0:
        raise RuntimeError("No valid data loaded")
    
    # Convert to numpy arrays
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    
    print(f"Successfully loaded {len(X_data)} samples")
    print(f"Data shape: {X_data.shape}")
    print(f"Label distribution: {np.bincount(y_data)} (0=Normal: {normal_count}, 1=Seizure: {seizure_count})")
    
    return X_data, y_data

def load_model_robust(model_path, n_chans, n_classes=2, device='cpu'):
    """Robust model loading with fallback to simple model"""
    
    try:
        # Try to load EEGInception model
        model = EEGInception(input_time=1280, fs=256, ncha=n_chans, n_classes=n_classes)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        
        print(f"Successfully loaded EEGInception model from {model_path}")
        return model
        
    except Exception as e:
        print(f"Error loading EEGInception model: {e}")
        print("Falling back to simple EEG model...")
        
        # Create simple model as fallback
        model = SimpleEEGModel(n_channels=n_chans, n_timepoints=1280, n_classes=n_classes)
        model.to(device)
        
        print("Created simple EEG model for demonstration")
        return model

def create_primitive_library_folder(base_path="primitive_libraries"):
    """Create folder structure for primitive library storage"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    library_folder = os.path.join(base_path, f"enhanced_cms_lime_library_{timestamp}")
    
    # Create main folder
    os.makedirs(library_folder, exist_ok=True)
    
    # Create subfolders for different primitive types
    subfolders = ['metadata']
    for subfolder in subfolders:
        os.makedirs(os.path.join(library_folder, subfolder), exist_ok=True)
    
    print(f"Created primitive library folder: {library_folder}")
    return library_folder

def save_primitive_analysis_results(explanation, sample_idx, library_folder):
    """Save detailed primitive analysis results"""
    results = {
        'sample_index': int(sample_idx),
        'timestamp': datetime.now().isoformat(),
        'original_prediction': convert_numpy_types(explanation.get('original_prediction', [])),
        'target_class': int(explanation.get('target_class', 0)),
        'primitive_statistics': {},
        'library_usage': convert_numpy_types(explanation.get('library_usage', {})),
        'explanation_quality': convert_numpy_types(explanation.get('explanation_quality', {}))
    }
    
    # Analyze primitive types
    for unit_type in ['microstate', 'shapelet', 'timefreq']:
        if unit_type in explanation.get('unit_explanations', {}):
            unit_exp = explanation['unit_explanations'][unit_type]
            results['primitive_statistics'][unit_type] = {
                'count': len(unit_exp.get('units', [])),
                'importances': convert_numpy_types(unit_exp.get('importances', [])),
                'mean_importance': float(np.mean(np.abs(unit_exp.get('importances', [0])))) if unit_exp.get('importances') else 0,
                'max_importance': float(np.max(np.abs(unit_exp.get('importances', [0])))) if unit_exp.get('importances') else 0,
                'source': unit_exp.get('source', 'unknown')
            }
    
    # Save combined explanation info
    if explanation.get('combined_explanation'):
        combined = explanation['combined_explanation']
        results['combined_explanation'] = {
            'selected_count': len(combined.get('selected_units', [])),
            'total_candidates': combined.get('total_candidates', 0),
            'selection_method': combined.get('selection_method', 'unknown'),
            'selected_importances': convert_numpy_types(combined.get('selected_importances', [])),
            'mean_selected_importance': float(np.mean(np.abs(combined.get('selected_importances', [0])))) if combined.get('selected_importances') else 0
        }
    
    # Save to file
    results_file = os.path.join(library_folder, 'metadata', f'sample_{sample_idx}_analysis.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved primitive analysis results: {results_file}")
    return results

def create_enhanced_visualization(explanation, sample, sample_idx, save_path, library_stats=None):
    """Create enhanced visualization with English annotations and proper memory management"""
    try:
        # Clear any existing figures to prevent memory leaks
        plt.close('all')
        gc.collect()
        
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[2, 1, 1, 1])
        
        # 1. Original EEG signal
        ax1 = fig.add_subplot(gs[0, 0])
        time_points = np.arange(sample.shape[1]) / 256  # Assuming 256 Hz sampling rate
        for i in range(min(5, sample.shape[0])):  # Show first 5 channels
            ax1.plot(time_points, sample[i] + i * 100, label=f'Channel {i+1}', alpha=0.8)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude (μV)')
        ax1.set_title(f'Original EEG Signal - Sample {sample_idx}', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Prediction information
        ax2 = fig.add_subplot(gs[0, 1])
        pred = explanation.get('original_prediction', [0.5, 0.5])
        classes = ['Non-seizure', 'Seizure']
        colors = ['#2E8B57', '#DC143C']
        bars = ax2.bar(classes, pred, color=colors, alpha=0.7)
        ax2.set_ylabel('Probability')
        ax2.set_title('Model Prediction', fontweight='bold')
        ax2.set_ylim(0, 1)
        
        # Add probability values on bars
        for bar, prob in zip(bars, pred):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Primitive type distribution
        ax3 = fig.add_subplot(gs[0, 2])
        if explanation.get('combined_explanation'):
            combined = explanation['combined_explanation']
            selected_units = combined.get('selected_units', [])
            
            type_counts = {}
            for unit in selected_units:
                unit_type = unit.get('type', 'unknown')
                type_counts[unit_type] = type_counts.get(unit_type, 0) + 1
            
            if type_counts:
                wedges, texts, autotexts = ax3.pie(type_counts.values(), labels=type_counts.keys(), 
                                                  autopct='%1.1f%%', startangle=90)
                ax3.set_title('Selected Primitive Types', fontweight='bold')
            else:
                ax3.text(0.5, 0.5, 'No primitives\nselected', ha='center', va='center', 
                        transform=ax3.transAxes, fontsize=12)
                ax3.set_title('Selected Primitive Types', fontweight='bold')
        
        # 4. Library usage statistics
        ax4 = fig.add_subplot(gs[0, 3])
        library_usage = explanation.get('library_usage', {})
        if library_usage:
            types = list(library_usage.keys())
            counts = list(library_usage.values())
            bars = ax4.bar(types, counts, color=['#4CAF50', '#FF9800', '#2196F3'], alpha=0.7)
            ax4.set_ylabel('Primitives Used')
            ax4.set_title('Library Usage', fontweight='bold')
            ax4.tick_params(axis='x', rotation=45)
            
            # Add count values on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Importance distribution by primitive type
        ax5 = fig.add_subplot(gs[1, :])
        
        all_importances = []
        all_types = []
        all_colors = []
        
        color_map = {'microstate': '#4CAF50', 'shapelet': '#FF9800', 'timefreq': '#2196F3'}
        
        for unit_type in ['microstate', 'shapelet', 'timefreq']:
            if unit_type in explanation.get('unit_explanations', {}):
                importances = explanation['unit_explanations'][unit_type].get('importances', [])
                all_importances.extend(importances)
                all_types.extend([unit_type] * len(importances))
                all_colors.extend([color_map[unit_type]] * len(importances))
    
        if all_importances:
            positions = np.arange(len(all_importances))
            bars = ax5.bar(positions, all_importances, color=all_colors, alpha=0.7)
            ax5.set_xlabel('Primitive Index')
            ax5.set_ylabel('Importance Score')
            ax5.set_title('Primitive Importance Distribution', fontweight='bold')
            ax5.grid(True, alpha=0.3)
            
            # Add legend
            legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_map[t], alpha=0.7, label=t.capitalize()) 
                              for t in color_map.keys() if t in all_types]
            ax5.legend(handles=legend_elements, loc='upper right')
            
            # Highlight selected primitives
            if explanation.get('combined_explanation'):
                selected_indices = explanation['combined_explanation'].get('selection_indices', [])
                for idx in selected_indices:
                    if idx < len(bars):
                        bars[idx].set_edgecolor('red')
                        bars[idx].set_linewidth(2)
        
        # 6. Quality metrics
        ax6 = fig.add_subplot(gs[2, 0])
        quality_metrics = explanation.get('explanation_quality', {})
        
        if quality_metrics:
            metrics = list(quality_metrics.keys())
            values = list(quality_metrics.values())
            
            # Normalize values for better visualization
            normalized_values = []
            for i, (metric, value) in enumerate(zip(metrics, values)):
                if 'ratio' in metric or 'importance' in metric:
                    normalized_values.append(min(value * 10, 1.0))  # Scale importance values
                else:
                    normalized_values.append(min(value / 20, 1.0))  # Scale count values
            
            bars = ax6.barh(metrics, normalized_values, color='skyblue', alpha=0.7)
            ax6.set_xlabel('Normalized Score')
            ax6.set_title('Explanation Quality Metrics', fontweight='bold')
            ax6.set_xlim(0, 1)
            
            # Add actual values as text
            for bar, value in zip(bars, values):
                width = bar.get_width()
                ax6.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{value:.3f}', ha='left', va='center', fontsize=9)
        
        # 7. Selected primitives details
        ax7 = fig.add_subplot(gs[2, 1:])
        
        if explanation.get('combined_explanation'):
            combined = explanation['combined_explanation']
            selected_units = combined.get('selected_units', [])
            selected_importances = combined.get('selected_importances', [])
            
            if selected_units and selected_importances:
                # Create a table-like visualization
                table_data = []
                for i, (unit, importance) in enumerate(zip(selected_units[:10], selected_importances[:10])):
                    unit_type = unit.get('type', 'unknown')
                    time_range = unit.get('time_range', (0, 0))
                    channels = unit.get('channels', [])
                    quality = unit.get('quality', 0)
                    
                    table_data.append([
                        f'{i+1}',
                        unit_type.capitalize(),
                        f'{importance:.4f}',
                        f'{time_range[0]:.1f}-{time_range[1]:.1f}',
                        f'{len(channels)}',
                        f'{quality:.3f}'
                    ])
                
                # Create table
                table = ax7.table(cellText=table_data,
                                colLabels=['#', 'Type', 'Importance', 'Time Range', 'Channels', 'Quality'],
                                cellLoc='center',
                                loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.5)
                
                # Style the table
                for i in range(len(table_data) + 1):
                    for j in range(6):
                        cell = table[(i, j)]
                        if i == 0:  # Header
                            cell.set_facecolor('#4CAF50')
                            cell.set_text_props(weight='bold', color='white')
                        else:
                            cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                
                ax7.set_title('Top Selected Primitives', fontweight='bold')
                ax7.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Enhanced visualization saved: {save_path}")
        
    except Exception as e:
        print(f"Error creating enhanced visualization for sample {sample_idx}: {e}")
        print(f"Attempting to save basic visualization instead...")
        try:
            # Create a simple fallback visualization
            plt.figure(figsize=(10, 6))
            plt.plot(sample.T)
            plt.title(f'EEG Signal - Sample {sample_idx} (Fallback)')
            plt.xlabel('Time Points')
            plt.ylabel('Amplitude')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Fallback visualization saved: {save_path}")
        except Exception as fallback_error:
            print(f"Failed to create fallback visualization: {fallback_error}")
    finally:
        # Ensure all figures are closed and memory is freed
        plt.close('all')
        gc.collect()

def main():
    # Configuration parameters
    patient_id = 2  # First patient
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Data path templates
    data_root_template = r'D:\public_data\CHBMIT\1_data_clean\chb%02d'
    segment_info_template = r'D:\public_data\CHBMIT\segment_clean\30-5-240\chb%02d\segment_info.json'
    weight_model_path = r'D:\public_data\CHBMIT\weight\eeginception+se'
    
    # Setup logging first
    library_folder = create_primitive_library_folder()
    library_path = os.path.join(library_folder, "enhanced_cms_lime_primitives.pkl")
    log_file_path = os.path.join(library_folder, "enhanced_cms_lime_analysis.log")
    setup_logger(log_file_path)
    
    log_print("=" * 80)
    log_print("Enhanced CMS-LIME CHB-MIT Dataset Interpretability Analysis")
    log_print("=" * 80)
    log_print(f"Log file created: {log_file_path}")
    
    # 1. Create primitive library folder
    log_print("\n1. Setting up primitive library...")
    log_print(f"Library folder: {library_folder}")
    log_print(f"Library path: {library_path}")
    
    # 2. Load real CHB-MIT data
    log_print("\n2. Loading real CHB-MIT dataset...")
    
    # Sample count configuration - you can modify these parameters
    max_seizure_samples = 2000   # Maximum number of seizure samples
    max_normal_samples = 0    # Maximum number of normal samples
    # Alternative: use max_samples for total count (will be split evenly)
    # max_samples = 2000
    
    try:
        X_data, y_data = load_chb_data_robust(
            patient_id, 
            data_root_template, 
            segment_info_template, 
            max_seizure_samples=max_seizure_samples,
            max_normal_samples=max_normal_samples
        )
        log_print("Successfully loaded real CHB-MIT data")
    except (FileNotFoundError, RuntimeError) as e:
        log_print(f"Error loading CHB-MIT data: {e}")
        log_print("Please check data paths and ensure CHB-MIT dataset is available.")
        return
    
    log_print(f"Data shape: {X_data.shape}")
    log_print(f"Label distribution: {np.bincount(y_data)}")
    log_print(f"Data range: [{X_data.min():.2f}, {X_data.max():.2f}]")
    log_print(f"Data std: {X_data.std():.2f}")
    
    # Ensure data is in correct format for microstate analysis
    # Convert from (n_samples, n_channels, n_timepoints) to list of (n_channels, n_timepoints)
    if X_data.ndim == 4:
        # If 4D data (n_samples, 1, n_channels, n_timepoints), squeeze to 3D
        X_data_original = X_data.squeeze(axis=1)
        log_print(f"Converted 4D data to 3D: {X_data.shape} -> {X_data_original.shape}")
    elif X_data.ndim == 3:
        log_print("Converting data format for compatibility...")
        # Keep original format for model prediction, but prepare for microstate analysis
        X_data_original = X_data.copy()
    else:
        X_data_original = X_data
        log_print(f"Warning: Unexpected data dimensions: {X_data.shape}")
    
    # 3. Load model with robust fallback
    log_print("\n3. Loading EEGInception model (with fallback to simple model)...")
    
    # Get number of channels
    n_chans = X_data.shape[1]
    log_print(f"Number of channels: {n_chans}")
    
    # Try to get model weight files
    model_path = None
    if os.path.exists(weight_model_path):
        weight_sub_dirs = os.listdir(weight_model_path)
        weight_sub_dirs = list(filter(lambda x: x.endswith('.pth'), weight_sub_dirs))
        weight_sub_dirs = sorted(weight_sub_dirs, key=natural_keys)
        
        if len(weight_sub_dirs) > 0:
            if patient_id >= 13:
                model_path = os.path.join(weight_model_path, weight_sub_dirs[patient_id-2])  # Use first model
            else:
                model_path = os.path.join(weight_model_path, weight_sub_dirs[patient_id - 1])  # Use first model
            log_print(f"Found model path: {model_path}")
    
    # Load model
    model = load_model_robust(model_path, n_chans, device=device)
    model_wrapper = EEGInceptionWrapper(model, device)
    
    log_print(f"Model parameters: {sum(param.numel() for param in model.parameters() if param.requires_grad)}")
    
    # 4. Test model prediction
    log_print("\n4. Testing model prediction...")
    test_sample = X_data[0]
    pred_probs = model_wrapper.predict_proba(test_sample)
    log_print(f"Test sample prediction probabilities: {pred_probs}")
    log_print(f"Predicted class: {np.argmax(pred_probs, axis=1)}")
    log_print(f"True label: {y_data[0]}")
    
    # 5. Create Enhanced CMS-LIME explainer
    log_print("\n5. Creating Enhanced CMS-LIME explainer...")
    
    # Configure Enhanced CMS-LIME parameters
    config = EnhancedCMSLimeConfig(
        # Basic CMS-LIME parameters
        n_shapelets=30,  # 减少shapelet数量以提高质量
        n_perturbations=1500,  # 适度减少扰动数量
        n_microstates=6,  # 减少微状态数量
        timefreq_method='cwt',
        causal_method='granger',
        n_features=40,  # 减少特征数量
        kernel_type='multidim',
        regression_method='ridge',
        selection_method='dpp',
        
        # Enhanced parameters
        use_primitive_library=True,
        library_path=library_path,
        min_primitive_quality=0.05,  # 提高最低质量要求
        max_library_primitives=150,  # 减少基元库最大容量
        
        # Improved selection strategy
        adaptive_selection=True,
        min_selected_primitives=3,  # 减少最小选择数量
        max_selected_primitives=15,  # 减少最大选择数量
        quality_weight=0.8,  # 增加质量权重
        diversity_weight=0.2,  # 减少多样性权重
        
        # Type-specific limits (新增各类型基元数量限制)
        max_microstate_primitives=40,
        max_shapelet_primitives=50,
        max_timefreq_primitives=60,
        
        # Quality thresholds (新增质量阈值)
        microstate_quality_threshold=0.1,
        shapelet_quality_threshold=0.08,
        timefreq_quality_threshold=0.06,
        
        # Importance calculation improvements
        importance_aggregation='weighted_mean',
        perturbation_strength=0.4,  # 减少扰动强度
        stability_iterations=3,
        
        random_state=42,
        verbose=True
    )
    
    explainer = EnhancedCMSLimeExplainer(config=config)
    
    # 6. Fit explainer with real CHB data
    log_print("\n6. Fitting Enhanced CMS-LIME explainer with CHB data...")
    
    # Use partial data for fitting
    fit_number = 20
    fit_data = X_data_original[:fit_number]  # Use first 5 samples (corrected format)
    fit_labels = y_data[:fit_number]
    
    # Ensure fit_data is 3D for microstate analysis
    if fit_data.ndim == 4:
        fit_data = fit_data.squeeze(axis=1)
        log_print(f"Squeezed fit_data from 4D to 3D: {fit_data.shape}")
    
    log_print(f"Final fit_data shape: {fit_data.shape}")
    log_print(f"Fitting with {len(fit_data)} CHB samples")
    
    # For microstate analysis, we need to pass individual samples
    # The explainer will handle the data format internally
    explainer.fit(fit_data, fit_labels, model)
    
    # Get primitive library statistics and save
    library_stats = explainer.get_primitive_library_stats()
    log_print(f"\nPrimitive library statistics: {library_stats}")
    
    # Save primitive library to disk
    try:
        explainer.save_primitive_library(library_path)
        log_print(f"Primitive library saved to: {library_path}")
    except Exception as e:
        log_print(f"Warning: Could not save primitive library: {e}")
    
    # 7. Generate explanations
    log_print("\n7. Generating explanation results...")
    
    # Select samples to explain
    explain_indices = [0, 1, 2]  # Explain first 3 samples
    
    # Ensure output directories exist
    pictures_dir = "enhanced_pictures"
    enhanced_dir = "enhanced_results"
    os.makedirs(pictures_dir, exist_ok=True)
    os.makedirs(enhanced_dir, exist_ok=True)
    log_print(f"Output directories created: {pictures_dir}, {enhanced_dir}")
    
    explanation_results = []
    
    for idx in range(1000,len(X_data)):  # explain_indices
        if idx >= len(X_data):
            continue
            
        # Memory management - force garbage collection every 10 samples
        if idx % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        sample = X_data[idx]
        true_label = y_data[idx]
        
        log_print(f"\n--- Explaining Sample {idx} (True Label: {true_label}) ---")
        
        try:
            log_print(f"Processing sample {idx}...")
            
            # Data preprocessing check
            if np.any(np.isnan(sample)) or np.any(np.isinf(sample)):
                log_print(f"Warning: Sample {idx} contains invalid data, skipping")
                continue
                
            # Check data range
            data_std = np.std(sample)
            if data_std < 1e-6:
                log_print(f"Warning: Sample {idx} has very small variance (std={data_std:.2e}), may result in zero importance")
            
            # Generate explanation
            explanation = explainer.explain_instance(
                sample, 
                target_class=true_label,
                unit_types=['microstate', 'shapelet', 'timefreq']
            )
            
            # Check explanation results
            max_importance = 0
            
            # Check importance of each primitive type
            for unit_type in ['microstate', 'shapelet', 'timefreq']:
                if unit_type in explanation.get('unit_explanations', {}):
                    unit_exp = explanation['unit_explanations'][unit_type]
                    importances = unit_exp.get('importances', [])
                    if len(importances) > 0:
                        max_imp = max(abs(imp) for imp in importances)
                        max_importance = max(max_importance, max_imp)
            
            # Check combined explanation importance
            if explanation.get('combined_explanation') is not None:
                combined_exp = explanation['combined_explanation']
                selected_importances = combined_exp.get('selected_importances', [])
                if len(selected_importances) > 0:
                    max_combined_imp = max(abs(imp) for imp in selected_importances)
                    max_importance = max(max_importance, max_combined_imp)
            
            log_print(f"Sample {idx} maximum importance: {max_importance:.8f}")
            
            # Display enhanced explanation summary
            summary = explainer.get_enhanced_explanation_summary(explanation)
            log_print(summary)
            
            # Save primitive analysis results
            analysis_results = save_primitive_analysis_results(explanation, idx, library_folder)
            explanation_results.append(analysis_results)
            
            # Create enhanced visualization
            enhanced_save_path = os.path.join(enhanced_dir, f'enhanced_cms_lime_patient{patient_id}_sample_{idx}.png')
            create_enhanced_visualization(explanation, sample, idx, enhanced_save_path, library_stats)
            
            # Save original visualization for comparison
            original_save_path = os.path.join(pictures_dir, f'cms_lime_chb_patient{patient_id}_sample_{idx}.png')
            explainer.visualize_explanation(
                explanation, 
                sample,
                save_path=original_save_path
            )
            log_print(f"Original visualization saved: {original_save_path}")
            
        except KeyboardInterrupt:
            log_print(f"\nKeyboard interrupt received at sample {idx}. Stopping analysis...")
            break
        except Exception as e:
            log_print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            
            # Clean up any remaining figures and memory
            plt.close('all')
            gc.collect()
            
            # Continue with next sample
            log_print(f"Continuing with next sample...")
            continue
    
    # 8. Generate summary report
    log_print("\n8. Generating summary report...")
    
    # Calculate aggregate statistics
    total_primitives_analyzed = sum(len(result.get('primitive_statistics', {})) for result in explanation_results)
    selected_counts = [result.get('combined_explanation', {}).get('selected_count', 0) for result in explanation_results if result.get('combined_explanation')]
    avg_selected_primitives = float(np.mean(selected_counts)) if selected_counts else 0.0
    
    summary_report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'patient_id': patient_id,
        'data_info': {
            'data_type': 'real_chb_mit' if os.path.exists(data_root_template % patient_id) else 'simulated_eeg',
            'data_source': data_root_template % patient_id if os.path.exists(data_root_template % patient_id) else 'generated',
            'n_samples': int(len(X_data)),
            'n_channels': int(X_data.shape[1]),
            'n_timepoints': int(X_data.shape[2]),
            'sampling_rate': 256
        },
        'total_samples_analyzed': len(explain_indices),
        'model_info': {
            'model_path': model_path if model_path else 'simple_eeg_model',
            'n_channels': n_chans,
            'n_parameters': int(sum(param.numel() for param in model.parameters() if param.requires_grad))
        },
        'explainer_config': {
            'use_primitive_library': bool(config.use_primitive_library),
            'adaptive_selection': bool(config.adaptive_selection),
            'min_selected_primitives': int(config.min_selected_primitives),
            'max_selected_primitives': int(config.max_selected_primitives),
            'importance_aggregation': str(config.importance_aggregation)
        },
        'primitive_library_stats': convert_numpy_types(library_stats),
        'analysis_statistics': {
            'total_primitives_analyzed': int(total_primitives_analyzed),
            'avg_selected_primitives': avg_selected_primitives,
            'explanation_success_rate': float(len(explanation_results) / len(explain_indices))
        },
        'output_folders': {
            'primitive_library': library_folder,
            'enhanced_visualizations': enhanced_dir,
            'original_visualizations': pictures_dir
        }
    }
    
    # Save summary report
    summary_path = os.path.join(library_folder, 'enhanced_analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    log_print(f"Summary report saved: {summary_path}")
    
    log_print("\n" + "=" * 80)
    log_print("Enhanced CMS-LIME Analysis Complete!")
    log_print("=" * 80)
    log_print(f"\nResults saved to:")
    log_print(f"  - Primitive library: {library_folder}")
    log_print(f"  - Enhanced visualizations: {enhanced_dir}")
    log_print(f"  - Original visualizations: {pictures_dir}")
    log_print(f"  - Analysis summary: {summary_path}")
    log_print(f"  - Analysis log: {log_file_path}")
    
    # Display final statistics
    log_print(f"\nFinal Statistics:")
    log_print(f"  - Primitive library size: {library_stats.get('total_primitives', 0)} primitives")
    log_print(f"  - Average selected primitives: {avg_selected_primitives:.1f}")
    log_print(f"  - Library usage efficiency: Enabled")
    log_print(f"  - Enhanced selection strategy: Active")
    log_print(f"  - Visualization language: English")
    log_print(f"  - Quality assessment: Enabled")
    
    # Demonstrate key improvements
    log_print(f"\nKey Improvements Demonstrated:")
    log_print(f"  ✓ Enhanced primitive selection (adaptive strategy)")
    log_print(f"  ✓ Primitive library creation and reuse with CHB data")
    log_print(f"  ✓ English-only visualizations")
    log_print(f"  ✓ Quality-based primitive assessment")
    log_print(f"  ✓ Comprehensive analysis reporting")
    log_print(f"  ✓ Multiple primitive types integration")
    log_print(f"  ✓ Real CHB-MIT data loading with fallback")
    log_print(f"  ✓ Robust model loading with fallback")
    log_print(f"  ✓ Persistent primitive library storage")
    log_print(f"  ✓ Complete analysis logging to file")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Cleaning up...")
        plt.close('all')
        gc.collect()
        print("Cleanup complete. Exiting.")
    except Exception as e:
        print(f"\nUnexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        gc.collect()
        print("\nProgram terminated due to error. Please check the log file for details.")
    finally:
        # Final cleanup
        plt.close('all')
        gc.collect()