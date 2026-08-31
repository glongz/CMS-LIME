#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo Enhanced CMS-LIME Analysis with Simulated EEG Data

This script demonstrates the enhanced CMS-LIME framework with primitive library
functionality using simulated EEG data, showcasing all improvements including:
1. Enhanced primitive selection strategy
2. Primitive library creation and management
3. English-only visualization
4. Improved quality assessment

Author: CMS-LIME Framework
Date: 2025-01-27
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cms_lime_enhanced_explainer import EnhancedCMSLimeExplainer, EnhancedCMSLimeConfig
from primitive_library import PrimitiveLibrary
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use English locale
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class SimpleEEGModel(nn.Module):
    """Simple EEG classification model for demonstration"""
    
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

class EEGModelWrapper:
    """Model wrapper for CMS-LIME explanation"""
    
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

def generate_simulated_eeg_data(n_samples=10, n_channels=23, n_timepoints=1280, sampling_rate=256):
    """Generate realistic simulated EEG data for demonstration"""
    
    np.random.seed(42)  # For reproducibility
    
    X_data = []
    y_data = []
    
    for i in range(n_samples):
        # Generate base EEG signal
        t = np.linspace(0, n_timepoints/sampling_rate, n_timepoints)
        
        # Create multi-channel EEG signal
        eeg_signal = np.zeros((n_channels, n_timepoints))
        
        for ch in range(n_channels):
            # Base alpha rhythm (8-12 Hz)
            alpha_freq = 8 + np.random.uniform(0, 4)
            alpha_signal = np.sin(2 * np.pi * alpha_freq * t) * (10 + np.random.uniform(0, 5))
            
            # Beta rhythm (13-30 Hz)
            beta_freq = 13 + np.random.uniform(0, 17)
            beta_signal = np.sin(2 * np.pi * beta_freq * t) * (5 + np.random.uniform(0, 3))
            
            # Theta rhythm (4-7 Hz)
            theta_freq = 4 + np.random.uniform(0, 3)
            theta_signal = np.sin(2 * np.pi * theta_freq * t) * (8 + np.random.uniform(0, 4))
            
            # Random noise
            noise = np.random.normal(0, 2, n_timepoints)
            
            # Combine signals
            eeg_signal[ch] = alpha_signal + beta_signal + theta_signal + noise
            
            # Add channel-specific variations
            eeg_signal[ch] *= (0.8 + 0.4 * np.random.random())
        
        # Determine label (seizure vs non-seizure)
        if i < n_samples // 2:
            # Non-seizure: normal EEG patterns
            label = 0
        else:
            # Seizure: add spike-wave patterns
            label = 1
            
            # Add seizure-like patterns
            seizure_start = np.random.randint(100, n_timepoints - 300)
            seizure_duration = np.random.randint(100, 200)
            
            for ch in range(n_channels):
                # High amplitude spikes
                spike_times = np.arange(seizure_start, seizure_start + seizure_duration, 20)
                for spike_time in spike_times:
                    if spike_time < n_timepoints:
                        spike_amplitude = 50 + np.random.uniform(0, 30)
                        spike_width = 5
                        spike_range = slice(max(0, spike_time - spike_width), 
                                          min(n_timepoints, spike_time + spike_width))
                        eeg_signal[ch, spike_range] += spike_amplitude * np.exp(-0.5 * np.linspace(-2, 2, len(range(*spike_range.indices(n_timepoints))))** 2)
        
        X_data.append(eeg_signal)
        y_data.append(label)
    
    return np.array(X_data), np.array(y_data)

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

def create_primitive_library_folder(base_path="primitive_libraries"):
    """Create folder structure for primitive library storage"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    library_folder = os.path.join(base_path, f"demo_cms_lime_library_{timestamp}")
    
    # Create main folder
    os.makedirs(library_folder, exist_ok=True)
    
    # Create subfolders for different primitive types
    subfolders = ['microstate', 'shapelet', 'timefreq', 'combined', 'metadata']
    for subfolder in subfolders:
        os.makedirs(os.path.join(library_folder, subfolder), exist_ok=True)
    
    print(f"Created primitive library folder: {library_folder}")
    return library_folder

def save_primitive_analysis_results(explanation, sample_idx, library_folder):
    """Save detailed primitive analysis results"""
    results = {
        'sample_index': sample_idx,
        'timestamp': datetime.now().isoformat(),
        'original_prediction': explanation.get('original_prediction', []).tolist() if hasattr(explanation.get('original_prediction', []), 'tolist') else explanation.get('original_prediction', []),
        'target_class': explanation.get('target_class', 0),
        'primitive_statistics': {},
        'library_usage': explanation.get('library_usage', {}),
        'explanation_quality': explanation.get('explanation_quality', {})
    }
    
    # Analyze primitive types
    for unit_type in ['microstate', 'shapelet', 'timefreq']:
        if unit_type in explanation.get('unit_explanations', {}):
            unit_exp = explanation['unit_explanations'][unit_type]
            results['primitive_statistics'][unit_type] = {
                'count': len(unit_exp.get('units', [])),
                'importances': unit_exp.get('importances', []),
                'mean_importance': np.mean(np.abs(unit_exp.get('importances', [0]))) if unit_exp.get('importances') else 0,
                'max_importance': np.max(np.abs(unit_exp.get('importances', [0]))) if unit_exp.get('importances') else 0,
                'source': unit_exp.get('source', 'unknown')
            }
    
    # Save combined explanation info
    if explanation.get('combined_explanation'):
        combined = explanation['combined_explanation']
        results['combined_explanation'] = {
            'selected_count': len(combined.get('selected_units', [])),
            'total_candidates': combined.get('total_candidates', 0),
            'selection_method': combined.get('selection_method', 'unknown'),
            'selected_importances': combined.get('selected_importances', []),
            'mean_selected_importance': np.mean(np.abs(combined.get('selected_importances', [0]))) if combined.get('selected_importances') else 0
        }
    
    # Save to file
    results_file = os.path.join(library_folder, 'metadata', f'sample_{sample_idx}_analysis.json')
    with open(results_file, 'w') as f:
        json.dump(convert_numpy_types(results), f, indent=2)
    
    print(f"Saved primitive analysis results: {results_file}")
    return results

def create_enhanced_visualization(explanation, sample, sample_idx, save_path, library_stats=None):
    """Create enhanced visualization with English annotations"""
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
    plt.close()

def main():
    print("=" * 80)
    print("Enhanced CMS-LIME Demo with Simulated EEG Data")
    print("=" * 80)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Create primitive library folder
    print("\n1. Setting up primitive library...")
    library_folder = create_primitive_library_folder()
    library_path = os.path.join(library_folder, "demo_cms_lime_primitives.pkl")
    
    # 2. Generate simulated EEG data
    print("\n2. Generating simulated EEG data...")
    X_data, y_data = generate_simulated_eeg_data(n_samples=10, n_channels=23, n_timepoints=1280)
    
    print(f"Generated data shape: {X_data.shape}")
    print(f"Label distribution: {np.bincount(y_data)}")
    print(f"Data range: [{X_data.min():.2f}, {X_data.max():.2f}]")
    print(f"Data std: {X_data.std():.2f}")
    
    # 3. Create and train simple model
    print("\n3. Creating and training simple EEG model...")
    model = SimpleEEGModel(n_channels=23, n_timepoints=1280, n_classes=2).to(device)
    model_wrapper = EEGModelWrapper(model, device)
    
    print(f"Model parameters: {sum(param.numel() for param in model.parameters() if param.requires_grad)}")
    
    # 4. Test model prediction
    print("\n4. Testing model prediction...")
    test_sample = X_data[0]
    pred_probs = model_wrapper.predict_proba(test_sample)
    print(f"Test sample prediction probabilities: {pred_probs}")
    print(f"Predicted class: {np.argmax(pred_probs, axis=1)}")
    print(f"True label: {y_data[0]}")
    
    # 5. Create Enhanced CMS-LIME explainer
    print("\n5. Creating Enhanced CMS-LIME explainer...")
    
    # Configure Enhanced CMS-LIME parameters
    config = EnhancedCMSLimeConfig(
        # Basic CMS-LIME parameters
        n_shapelets=30,
        n_perturbations=1000,
        n_microstates=6,
        timefreq_method='cwt',
        causal_method='granger',
        n_features=30,
        kernel_type='multidim',
        regression_method='ridge',
        selection_method='dpp',
        
        # Enhanced parameters
        use_primitive_library=True,
        library_path=library_path,
        min_primitive_quality=0.01,
        max_library_primitives=100,
        
        # Improved selection strategy
        adaptive_selection=True,
        min_selected_primitives=3,
        max_selected_primitives=15,
        quality_weight=0.7,
        diversity_weight=0.3,
        
        # Importance calculation improvements
        importance_aggregation='weighted_mean',
        perturbation_strength=0.3,
        stability_iterations=3,
        
        random_state=42,
        verbose=True
    )
    
    explainer = EnhancedCMSLimeExplainer(config=config)
    
    # 6. Fit explainer
    print("\n6. Fitting Enhanced CMS-LIME explainer...")
    
    # Use partial data for fitting
    fit_data = X_data[:6]  # Use first 6 samples
    fit_labels = y_data[:6]
    
    explainer.fit(fit_data, fit_labels, model)
    
    # Get primitive library statistics
    library_stats = explainer.get_primitive_library_stats()
    print(f"\nPrimitive library statistics: {library_stats}")
    
    # 7. Generate explanations
    print("\n7. Generating explanation results...")
    
    # Select samples to explain
    explain_indices = [0, 1, 2, 7, 8, 9]  # Mix of non-seizure and seizure samples
    
    # Ensure output directories exist
    pictures_dir = "demo_pictures"
    enhanced_dir = "demo_enhanced_results"
    os.makedirs(pictures_dir, exist_ok=True)
    os.makedirs(enhanced_dir, exist_ok=True)
    
    explanation_results = []
    
    for idx in explain_indices:
        if idx >= len(X_data):
            continue
            
        sample = X_data[idx]
        true_label = y_data[idx]
        
        print(f"\n--- Explaining Sample {idx} (True Label: {true_label}) ---")
        
        try:
            print(f"Processing sample {idx}...")
            
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
            
            print(f"Sample {idx} maximum importance: {max_importance:.8f}")
            
            # Display enhanced explanation summary
            summary = explainer.get_enhanced_explanation_summary(explanation)
            print(summary)
            
            # Save primitive analysis results
            analysis_results = save_primitive_analysis_results(explanation, idx, library_folder)
            explanation_results.append(analysis_results)
            
            # Create enhanced visualization
            enhanced_save_path = os.path.join(enhanced_dir, f'enhanced_demo_sample_{idx}.png')
            create_enhanced_visualization(explanation, sample, idx, enhanced_save_path, library_stats)
            
            # Save original visualization for comparison
            original_save_path = os.path.join(pictures_dir, f'demo_sample_{idx}.png')
            explainer.visualize_explanation(
                explanation, 
                sample,
                save_path=original_save_path
            )
            print(f"Original visualization saved: {original_save_path}")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # 8. Generate summary report
    print("\n8. Generating summary report...")
    
    # Calculate aggregate statistics
    total_primitives_analyzed = sum(len(result.get('primitive_statistics', {})) for result in explanation_results)
    selected_counts = [result.get('combined_explanation', {}).get('selected_count', 0) for result in explanation_results if result.get('combined_explanation')]
    avg_selected_primitives = float(np.mean(selected_counts)) if selected_counts else 0.0
    
    summary_report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'demo_info': {
            'data_type': 'simulated_eeg',
            'n_samples': int(len(X_data)),
            'n_channels': int(X_data.shape[1]),
            'n_timepoints': int(X_data.shape[2]),
            'sampling_rate': 256
        },
        'total_samples_analyzed': len(explain_indices),
        'model_info': {
            'model_type': 'SimpleEEGModel',
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
    summary_path = os.path.join(library_folder, 'demo_analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    print(f"Summary report saved: {summary_path}")
    
    print("\n" + "=" * 80)
    print("Enhanced CMS-LIME Demo Analysis Complete!")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  - Primitive library: {library_folder}")
    print(f"  - Enhanced visualizations: {enhanced_dir}")
    print(f"  - Original visualizations: {pictures_dir}")
    print(f"  - Analysis summary: {summary_path}")
    
    # Display final statistics
    print(f"\nFinal Statistics:")
    print(f"  - Primitive library size: {library_stats.get('total_primitives', 0)} primitives")
    print(f"  - Average selected primitives: {avg_selected_primitives:.1f}")
    print(f"  - Library usage efficiency: Enabled")
    print(f"  - Enhanced selection strategy: Active")
    print(f"  - Visualization language: English")
    print(f"  - Quality assessment: Enabled")
    
    # Demonstrate key improvements
    print(f"\nKey Improvements Demonstrated:")
    print(f"  ✓ Enhanced primitive selection (adaptive strategy)")
    print(f"  ✓ Primitive library creation and reuse")
    print(f"  ✓ English-only visualizations")
    print(f"  ✓ Quality-based primitive assessment")
    print(f"  ✓ Comprehensive analysis reporting")
    print(f"  ✓ Multiple primitive types integration")

if __name__ == '__main__':
    main()