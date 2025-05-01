import pandas as pd
import numpy as np
import re
from code.unet_training_setup import load_segmented_ims
from code.morphology import extract_morphology
from pathlib import Path

import os


import numpy as np
import os
import re

import matplotlib.pyplot as plt
import seaborn as sns
from code.unet_training_setup import load_segmented_ims
from code.morphology import extract_morphology

import pandas as pd
import numpy as np
import os
import re
from config import *
from code.unet_training_setup import load_segmented_ims
from code.morphology import extract_morphology

def compare_load_segmented_ims(input_path):
    """Load segmented image paths and labels based on filename patterns."""

    image_filepaths = []
    image_labels = []
    image_ids = []

    mut_labels = ['A2_MHV', 'muimage']
    wt_labels = ['MHVWT', 'wtimage']

    
            
    for files in os.listdir(input_path):
        if files.lower().endswith(('.png', '.tif', '.tiff')):
            file_path = os.path.join(input_path, files)
            image_id = Path(files).stem

            if any(tag in image_id for tag in mut_labels):
                label = 'mutant'
            elif any(tag in image_id for tag in wt_labels):
                label = 'wildtype'
            else:
                label = 'unknown'

            image_labels.append(label)
            image_filepaths.append(file_path)
            image_ids.append(image_id)


            

    return image_labels, image_filepaths, image_ids



def compare_combine_dfs(convert_df, SEGMENTED_MASKS_WT, SEGMENTED_MASKS_MUT, AUTO_SEGMENTED_MASKS_WT, AUTO_SEGMENTED_MASKS_MUT, SAVE_DIR):
    manual_labels_muts, manual_filepaths_muts, manual_ids_muts = compare_load_segmented_ims(SEGMENTED_MASKS_MUT)
    manual_labels_wts, manual_filepaths_wts, manual_ids_wts = compare_load_segmented_ims(SEGMENTED_MASKS_WT)
    auto_image_labels_muts, auto_image_filepaths_muts, auto_image_ids_muts = compare_load_segmented_ims(AUTO_SEGMENTED_MASKS_MUT)
    auto_image_labels_wts, auto_image_filepaths_wts, auto_image_ids_wts = compare_load_segmented_ims(AUTO_SEGMENTED_MASKS_WT)

    all_manual_files = pd.DataFrame({
        'im_id': manual_ids_muts + manual_ids_wts,
        'manual_segmented_file_path': manual_filepaths_muts + manual_filepaths_wts,
        'class': manual_labels_muts + manual_labels_wts
    }).drop_duplicates()

    all_auto_files = pd.DataFrame({
        'im_id': auto_image_ids_muts + auto_image_ids_wts,
        'auto_segmented_file_path': auto_image_filepaths_muts + auto_image_filepaths_wts,
        'class': auto_image_labels_muts + auto_image_labels_wts
    }).drop_duplicates()

    convert_df['Modified Name'] = convert_df['Modified Name'].astype(str).str.replace(r'_corrected.*', '', regex=True)
    all_manual_files['im_id'] = all_manual_files['im_id'].astype(str).str.replace(r'_corrected.*', '', regex=True)
    all_auto_files['im_id'] = all_auto_files['im_id'].astype(str).str.replace(r'_seg_ver2.*', '', regex=True)

    print('all auto files', all_auto_files)
    print('all manual files', all_manual_files)

    manual_segmented_with_convert = all_manual_files.merge(convert_df, left_on='im_id', right_on='Modified Name')
    print('manual_segmented_with_convert', manual_segmented_with_convert)
    merged_df = manual_segmented_with_convert.merge(all_auto_files, left_on='File_name', right_on='im_id')

    final_df = pd.DataFrame({'im_id': merged_df['im_id_y'], 
                             'manual_segmented_file_path': merged_df['manual_segmented_file_path'],
                             'auto_segmented_file_path': merged_df['auto_segmented_file_path'],
                             'class': merged_df['class_x']})
    
    return final_df


    


def build_compare_df_from_auto_manual(all_df):
    df_compare = all_df[['im_id', 'class', 'manual_segmented_file_path', 'auto_segmented_file_path']].copy()
    df_compare.rename(columns={
        'im_id': 'Image_ID',
        'class': 'Class',
        'manual_segmented_file_path': 'manual_path',
        'auto_segmented_file_path': 'auto_path'
    }, inplace=True)

    df_compare['Spike Area Manual'] = np.nan
    df_compare['Spike Area Automatic'] = np.nan
    df_compare['Body Area Manual'] = np.nan
    df_compare['Body Area Automatic'] = np.nan
    df_compare['Body Perimeter Manual'] = np.nan
    df_compare['Body Perimeter Automatic'] = np.nan

    for idx, row in df_compare.iterrows():
        try:
            body_manual, spike_manual = extract_morphology(row['manual_path'])
            body_auto, spike_auto = extract_morphology(row['auto_path'])

            # Spike areas
            df_compare.at[idx, 'Spike Area Manual'] = spike_manual['Area'].sum()
            df_compare.at[idx, 'Spike Area Automatic'] = spike_auto['Area'].sum()

            # Body areas & perimeters
            df_compare.at[idx, 'Body Area Manual'] = body_manual['Area'].sum()
            df_compare.at[idx, 'Body Area Automatic'] = body_auto['Area'].sum()
            df_compare.at[idx, 'Body Perimeter Manual'] = body_manual['Perimeter'].sum()
            df_compare.at[idx, 'Body Perimeter Automatic'] = body_auto['Perimeter'].sum()

        except Exception as e:
            print(f"Skipping {row['Image_ID']}: {e}")
            continue

    df_compare = df_compare.dropna(subset=['Spike Area Manual', 'Spike Area Automatic'])
    return df_compare





def compare_methods_plotting(df_compare):
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
    plt.savefig("spike_area_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, wilcoxon


# 95% CI helper
def mean_ci(data, confidence=0.95):
    m, se = np.mean(data), stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., len(data)-1)
    return m, m-h, m+h

def compare_methods_stats(df_compare, SAVE_DIR, plot_yes=None):
    if df_compare.empty:
        print(" No data to analyze.")
        return

    df_long = df_compare.melt(
        id_vars=['Image_ID', 'Class'],
        value_vars=['Spike Area Manual', 'Spike Area Automatic'],
        var_name='Method',
        value_name='Spike Area'
    )

    df_long = df_long.groupby(['Image_ID', 'Class', 'Method'])['Spike Area'].mean().reset_index()

    try:
        df_wide = df_long.pivot(index=['Image_ID', 'Class'], columns='Method', values='Spike Area').reset_index()
        df_wide.columns.name = None
    except Exception as e:
        print(f"Pivot error: {e}")
        return

    df_wide = df_wide.rename(columns={
        'Spike Area Manual': 'Manual',
        'Spike Area Automatic': 'Auto'})

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

    if plot_yes:
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
        plt.savefig(os.path.join(SAVE_DIR, "percent_error_barplot.png"), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(SAVE_DIR, 'percent_error_barplot.png')}")
        plt.show()

    return df_wide, stat, p, summary


    
def compare_classes_stats(df_ground_truth_morph, plot_yes=None, SAVE_DIR=None, save_name="class_comparison_stats.csv"):
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

    if SAVE_DIR:
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_path = os.path.join(SAVE_DIR, save_name)
        pd.DataFrame(results).T.to_csv(save_path)
        print(f"Saved stats to: {save_path}")

    return results


def plot_spike_vs_body_area(df_compare, SAVE_DIR, base_filename="spike_vs_body_area"):
    os.makedirs(SAVE_DIR, exist_ok=True)
    results = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for i, (area_col, spike_col, title, ax) in enumerate([
        ('Body Area Manual', 'Spike Area Manual', 'Manual Segmentation', axes[0]),
        ('Body Area Automatic', 'Spike Area Automatic', 'Automatic Segmentation', axes[1])
    ]):
        for cls in ['mutant', 'wildtype']:
            subset = df_compare[df_compare['Class'] == cls]
            sns.regplot(
                data=subset,
                x=area_col,
                y=spike_col,
                scatter=True,
                label=cls,
                ax=ax,
                scatter_kws={'alpha': 0.6},
                line_kws={'linewidth': 1.5}
            )
            try:
                r, p = pearsonr(subset[area_col], subset[spike_col])
                results.append({'Method': title, 'Class': cls, 'Pearson_r': r, 'p_value': p})
            except Exception as e:
                print(f"Pearson error: {e}")
                results.append({'Method': title, 'Class': cls, 'Pearson_r': None, 'p_value': None})

        ax.set_title(title)
        ax.set_xlabel('Body Area')
        if i == 0:
            ax.set_ylabel('Spike Area')
        else:
            ax.set_ylabel('')
        ax.legend(title='Class')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{base_filename}_linear.png"), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(SAVE_DIR, f'{base_filename}_linear.png')}")
    plt.show()

    # Log version
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for i, (area_col, spike_col, title, ax) in enumerate([
        ('Body Area Manual', 'Spike Area Manual', 'Manual Segmentation (Log Scale)', axes[0]),
        ('Body Area Automatic', 'Spike Area Automatic', 'Automatic Segmentation (Log Scale)', axes[1])
    ]):
        for cls in ['mutant', 'wildtype']:
            subset = df_compare[df_compare['Class'] == cls]
            sns.scatterplot(data=subset, x=area_col, y=spike_col, hue='Class', ax=ax, alpha=0.6)

        ax.set_title(title)
        ax.set_xlabel('Body Area')
        if i == 0:
            ax.set_ylabel('Spike Area')
        else:
            ax.set_ylabel('')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(title='Class')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{base_filename}_logscale.png"), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(SAVE_DIR, f'{base_filename}_logscale.png')}")
    plt.show()

    pd.DataFrame(results).to_csv(os.path.join(SAVE_DIR, f"{base_filename}_pearson_stats.csv"), index=False)
    print(f"Saved Pearson r statistics to {os.path.join(SAVE_DIR, f'{base_filename}_pearson_stats.csv')}")
def export_full_morphology_stats(df_compare, SAVE_DIR, filename="all_morphology_stats.csv"):
    columns_to_export = [
        'Image_ID', 'Class',
        'Spike Area Manual', 'Body Area Manual', 'Body Perimeter Manual',
        'Spike Area Automatic', 'Body Area Automatic', 'Body Perimeter Automatic'
    ]
    df_export = df_compare[columns_to_export].copy()
    df_export.columns = [
        'Image_ID', 'Class',
        'Spike_Area_Manual', 'Body_Area_Manual', 'Body_Perimeter_Manual',
        'Spike_Area_Auto', 'Body_Area_Auto', 'Body_Perimeter_Auto'
    ]
    save_path = os.path.join(SAVE_DIR, filename)
    df_export.to_csv(save_path, index=False)
    print(f"✅ Saved full morphology stats to: {save_path}")