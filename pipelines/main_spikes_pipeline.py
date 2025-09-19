################ Import Libraries ################
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import *
from code.analyze_all import (
    analyze_dataset,
    summarize_results,
    plot_kde_comparisons
)

################ Setup ################

# collect mutant and wildtype segmented image files
mutant_files = [os.path.join(SEGMENTED_MASKS_MUT, f) 
                for f in os.listdir(SEGMENTED_MASKS_MUT) 
                if f.lower().endswith((".png", ".tif", ".jpg"))]

wt_files = [os.path.join(SEGMENTED_MASKS_WT, f) 
            for f in os.listdir(SEGMENTED_MASKS_WT) 
            if f.lower().endswith((".png", ".tif", ".jpg"))]

################ Run Analysis ################

results = []
analyze_dataset(mutant_files, "mutant", results)
analyze_dataset(wt_files, "wildtype", results)

# convert results → DataFrame
df = pd.DataFrame(results)

# save raw morphology stats
out_csv = os.path.join(TABLES_DIR, "all_distance_data.csv")
df.to_csv(out_csv, index=False)
print(f"Saved raw data to {out_csv}")

# summary statistics
summary = summarize_results(df)
out_sum = os.path.join(TABLES_DIR, "summary_stats.csv")
summary.to_csv(out_sum, index=False)
print(f"Saved summary stats to {out_sum}")

# plots
plot_kde_comparisons(df, FIGURES_DIR)
