import os
import re
import glob
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from scipy import stats
from scipy.stats import mannwhitneyu, wilcoxon
from torchvision import models

from skimage.measure import regionprops, label
from config import (
    COLOR_MUTANT,
    COLOR_WILDTYPE,
    FONT_SIZE_TITLE,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK
)

def extract_morphology(segmented_image_path):
    image = cv2.imread(segmented_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Error: Image not found or cannot be loaded.")
    
    body_mask = (image == 150).astype(np.uint8)
    spikes_mask = (image == 200).astype(np.uint8)

    if not np.any(body_mask):
        raise ValueError("Error: No virus body detected in the image.")
    if not np.any(spikes_mask):
        print("Warning: No spikes detected in the image.")
    
    body_labels = label(body_mask)
    spike_labels = label(spikes_mask)

    body_props = regionprops(body_labels)
    spike_props = regionprops(spike_labels)

    if not body_props:
        body_data = [{
            "ID": 0, "Area": 0, "Perimeter": 0,
            "Major Axis Length": 0, "Minor Axis Length": 0
        }]
    else:
        body_data = [{
            "ID": i + 1,
            "Area": prop.area,
            "Perimeter": prop.perimeter,
            "Major Axis Length": prop.major_axis_length,
            "Minor Axis Length": prop.minor_axis_length
        } for i, prop in enumerate(body_props)]

    if not spike_props:
        spike_data = [{
            "ID": 0, "Area": 0, "Perimeter": 0,
            "Major Axis Length": 0, "Minor Axis Length": 0
        }]
    else:
        spike_data = [{
            "ID": i + 1,
            "Area": prop.area,
            "Perimeter": prop.perimeter,
            "Centroid X": prop.centroid[0],
            "Centroid Y": prop.centroid[1]
        } for i, prop in enumerate(spike_props)]

    body_df = pd.DataFrame(body_data)
    spike_df = pd.DataFrame(spike_data)

    return body_df, spike_df

def make_morphology_df(segmented_image_path):
    file_path_ls = []
    label_ls = []
    proc_file_paths = []
    proc_labels = []

    for root, _, files in os.walk(segmented_image_path):
        for file in files:
            if file.lower().endswith('.png'):
                file_path = os.path.join(root, file)
                file_path_ls.append(file_path)
                label = 'mut' if 'A2_MHV' in file else 'wt' if 'MHVWT' in file else 'unknown'
                label_ls.append(label)

    df_input = pd.DataFrame({'filepath': file_path_ls, 'class': label_ls})
    print(df_input)

    sum_spike_area = []
    avg_spike_perim = []
    sum_body_area = []
    avg_body_perim = []

    for i in range(len(df_input)):
        file_path = df_input.iloc[i, 0]
        file_label = df_input.iloc[i, 1]
        try:
            body_df, spike_df = extract_morphology(file_path)
        except ValueError as e:
            print(f"Skipping {file_path}: {e}")
            continue
        sum_spike_area.append(np.sum(spike_df['Area']))
        avg_spike_perim.append(np.mean(spike_df['Perimeter']))
        sum_body_area.append(np.sum(body_df['Area']))
        avg_body_perim.append(np.mean(body_df['Perimeter']))
        proc_file_paths.append(file_path)
        proc_labels.append(file_label)

    df = pd.DataFrame({
        'Total_Spike_Area_Per_Particle': sum_spike_area,
        'Total_Body_Area_Per_Particle': sum_body_area,
        'Class': proc_labels,
        'file_path': proc_file_paths
    })

    df['Image_Index'] = df.groupby('Class').cumcount()
    return df

def display_individual_morph_stats(df):
    df['Image_Index'] = df.groupby('Class').cumcount()
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    sns.kdeplot(
        data=df, x='Total_Spike_Area_Per_Particle', hue='Class',
        fill=True, common_norm=False, alpha=0.4, linewidth=2, ax=ax[0]
    )
    ax[0].set_title('Spike Area Distribution')
    ax[0].set_xlabel('Spike Area per Particle')
    ax[0].set_ylabel('Density')
    ax[0].set_xlim(0, df['Total_Spike_Area_Per_Particle'].quantile(0.99))
    ax[0].grid(True)

    sns.kdeplot(
        data=df, x='Total_Body_Area_Per_Particle', hue='Class',
        fill=True, common_norm=False, alpha=0.4, linewidth=2, ax=ax[1]
    )
    ax[1].set_title('Body Area Distribution')
    ax[1].set_xlabel('Body Area per Particle')
    ax[1].set_ylabel('Density')
    ax[1].set_xlim(0, df['Total_Body_Area_Per_Particle'].quantile(0.99))
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

def get_base_file_map(directory, suffix_pattern):
    file_map = {}
    for file in os.listdir(directory):
        if file.endswith('.png'):
            base = re.sub(suffix_pattern, '', os.path.splitext(file)[0])
            file_map[base] = os.path.join(directory, file)
    return file_map

def compare_methods(manual_method_dir, automatic_method_dir, manual_method_suffix, automatic_method_suffix):
    files_manual = get_base_file_map(manual_method_dir, manual_method_suffix)
    files_auto = get_base_file_map(automatic_method_dir, automatic_method_suffix)
    common_keys = sorted(set(files_manual.keys()) & set(files_auto.keys()))
    print(f"Matched {len(common_keys)} images.")

    spike_areas_manual, spike_areas_auto, image_ids = [], [], []

    for key in common_keys:
        try:
            _, spike_df_manual = extract_morphology(files_manual[key])
            _, spike_df_auto = extract_morphology(files_auto[key])
            spike_areas_manual.append(spike_df_manual['Area'].sum())
            spike_areas_auto.append(spike_df_auto['Area'].sum())
            image_ids.append(key)
        except Exception as e:
            print(f"Skipping {key}: {e}")
            continue

    df_compare = pd.DataFrame({
        'Image_ID': image_ids,
        'Spike Area Manual': spike_areas_manual,
        'Spike Area Automatic': spike_areas_auto
    })

    df_compare['Class'] = df_compare['Image_ID'].apply(
        lambda x: 'mutant' if 'A2_MHV' in x else 'wildtype'
    )

    df_compare.to_csv('/home/ascott10/documents/projects/capstone_viruses/results/df_compare.csv')
    return df_compare

def compare_methods_plotting(save_dir):
    compare_files = glob.glob(os.path.join(save_dir, "df_compare_*.csv"))
    most_recent_file = max(compare_files, key=os.path.getmtime)
    df_compare = pd.read_csv(most_recent_file)

    df_long = df_compare.melt(
        id_vars=['Image_ID', 'Class'],
        value_vars=['Spike Area Manual', 'Spike Area Automatic'],
        var_name='Method',
        value_name='Spike Area'
    )

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    sns.swarmplot(data=df_long, x='Class', y='Spike Area', hue='Method',
                  dodge=True, alpha=0.6, size=3, ax=ax[0])
    ax[0].set_title('Spike Area by Method and Class')
    ax[0].set_ylabel("Spike Area (log scale)")
    ax[0].set_xlabel('Class')
    ax[0].set_yscale("log")
    ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)

    sns.swarmplot(data=df_long, x='Method', y='Spike Area', hue='Class',
                  dodge=True, alpha=0.6, size=3, ax=ax[1])
    ax[1].set_title('Spike Area by Class and Method')
    ax[1].set_ylabel("Spike Area (log scale)")
    ax[1].set_xlabel('Method')
    ax[1].set_yscale("log")
    ax[1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)

    plt.tight_layout()
    plt.show()

def mean_ci(data, confidence=0.95):
    m, se = np.mean(data), stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    return m, m - h, m + h

def compare_methods_stats(save_dir, plot_yes=None):
    latest_file = max(glob.glob(os.path.join(save_dir, "df_compare_*.csv")), key=os.path.getmtime)
    df = pd.read_csv(latest_file)

    df_long = df.melt(id_vars=['Image_ID', 'Class'],
                      value_vars=['Spike Area Manual', 'Spike Area Automatic'],
                      var_name='Method', value_name='Spike Area').dropna()

    df_long = df_long.groupby(['Image_ID', 'Class', 'Method'])['Spike Area'].mean().reset_index()
    df_wide = df_long.pivot(index=['Image_ID', 'Class'], columns='Method', values='Spike Area').reset_index()
    df_wide.columns.name = None
    df_wide = df_wide.rename(columns={'Spike Area Manual': 'Manual', 'Spike Area Automatic': 'Auto'})

    df_wide['Percent_Error'] = np.where(df_wide['Manual'] != 0,
                                        np.abs(df_wide['Auto'] - df_wide['Manual']) / df_wide['Manual'] * 100,
                                        np.nan)
    df_wide['Abs_Difference'] = np.abs(df_wide['Manual'] - df_wide['Auto'])

    paired_df = df_wide.dropna(subset=['Manual', 'Auto'])
    stat, p = wilcoxon(paired_df['Manual'], paired_df['Auto'])

    print("Wilcoxon Signed-Rank Test: Manual vs Automatic Spike Area")
    print(f"  Statistic = {stat:.4f}")
    print(f"  p-value   = {p:.6f}")
    if p < 0.05:
        print("  There is a statistically significant difference between Manual and Automatic methods.")
    else:
        print("  No significant difference between Manual and Automatic methods.")

    summary = {}
    for cls in ['mutant', 'wildtype']:
        pe = df_wide[df_wide['Class'] == cls]['Percent_Error'].dropna()
        ad = df_wide[df_wide['Class'] == cls]['Abs_Difference'].dropna()
        summary[cls] = {
            'Percent_Error': mean_ci(pe),
            'Abs_Difference': mean_ci(ad)
        }

    if plot_yes is not None:
        classes = ['mutant', 'wildtype']
        means = []
        errors = []

        for cls in classes:
            data = df_wide[df_wide['Class'] == cls]['Percent_Error'].dropna()
            mean, lower, upper = mean_ci(data)
            means.append(mean)
            errors.append(mean - lower)

        plt.bar(classes, means, yerr=errors, capsize=5)
        plt.ylabel('Mean Percent Error (%)')
        plt.title('Difference in Calculation Between Manual and Automatic')
        plt.show()

    return df_wide, stat, p, summary

def compare_classes_stats(df_ground_truth_morph, plot_yes=None, save_dir=None, save_name="class_comparison_stats.csv"):
    metrics = ['Total_Spike_Area_Per_Particle', 'Spike_Count']
    results = {}

    for metric in metrics:
        data_mutant = df_ground_truth_morph[df_ground_truth_morph['Class'] == 'mutant'][metric].dropna()
        data_wildtype = df_ground_truth_morph[df_ground_truth_morph['Class'] == 'wildtype'][metric].dropna()

        stat, p = mannwhitneyu(data_mutant, data_wildtype, alternative='two-sided')
        results[metric] = {
            'Mutant_Mean': np.mean(data_mutant),
            'Wildtype_Mean': np.mean(data_wildtype),
            'Mann-Whitney_Statistic': stat,
            'p-value': p
        }

        print(f"Comparison for {metric}:")
        print(f"  Mann-Whitney U statistic = {stat:.8f}")
        print(f"  p-value = {p:.8f}")
        if p < 0.05:
            print("  Statistically significant difference between mutant and wildtype.")
        else:
            print("  No significant difference between mutant and wildtype.")
        print()

    if plot_yes:
        for metric in metrics:
            plt.figure(figsize=(6, 5))
            sns.boxplot(data=df_ground_truth_morph, x='Class', y=metric)
            plt.title(f'Comparison of {metric} by Class')
            plt.ylabel(metric)
            plt.xlabel('Class')
            plt.tight_layout()
            plt.show()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name)
        pd.DataFrame(results).T.to_csv(save_path)
        print(f"Saved stats to: {save_path}")

    return results

def compare_classes_plotting(df_ground_truth_morph, save_dir=None, save_name="class_comparison_boxplot.png", show_plot=False):
    metrics = ['Total_Spike_Area_Per_Particle', 'Spike_Count']
    class_palette = {'mutant': COLOR_MUTANT, 'wildtype': COLOR_WILDTYPE}

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    sns.boxplot(data=df_ground_truth_morph, x='Class', y='Total_Spike_Area_Per_Particle', ax=ax[0], palette=class_palette)
    ax[0].set_title('Total Spike Area by Class', fontsize=FONT_SIZE_TITLE)
    ax[0].set_ylabel("Total Spike Area", fontsize=FONT_SIZE_LABEL)
    ax[0].set_xlabel('Class', fontsize=FONT_SIZE_LABEL)
    ax[0].tick_params(labelsize=FONT_SIZE_TICK)

    sns.boxplot(data=df_ground_truth_morph, x='Class', y='Spike_Count', ax=ax[1], palette=class_palette)
    ax[1].set_title('Spike Count by Class', fontsize=FONT_SIZE_TITLE)
    ax[1].set_ylabel("Spike Count", fontsize=FONT_SIZE_LABEL)
    ax[1].set_xlabel('Class', fontsize=FONT_SIZE_LABEL)
    ax[1].tick_params(labelsize=FONT_SIZE_TICK)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
