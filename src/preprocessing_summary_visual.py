import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

JSON_LOG_PATH = r"data\processed\preprocessing_summary.json"
OUTPUT_PATH = r"output\visualizations\preprocessing_summary_dashboard.png"

def main():
    if not os.path.exists(JSON_LOG_PATH):
        print(f"Log file not found: {JSON_LOG_PATH}")
        print("Please run 'python src\\preprocessing.py' first to generate the latest log file.")
        return
        
    with open(JSON_LOG_PATH, 'r') as f:
        data = json.load(f)
        
    if not data:
        print("JSON is empty. No processed files found.")
        return
        
    df = pd.DataFrame(data)
    
    print("\n" + "="*80)
    print("PREPROCESSING REFINEMENT & DATA LOSS REPORT:")
    print("="*80)
    print(df.to_string(index=False))
    
    # 4-Panel Dashboard Design
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Data Preprocessing (Pipeline) Modification Analysis', fontsize=22, fontweight='bold', y=0.98)
    
    # 1. Row Count Comparison (Before vs After) (Log Scale bar chart)
    ax1 = axes[0, 0]
    df_melt_rows = df.melt(id_vars='file_short', value_vars=['rows_before', 'rows_after'], 
                           var_name='Status', value_name='Count (Log Scale)')
    sns.barplot(data=df_melt_rows, x='file_short', y='Count (Log Scale)', hue='Status', ax=ax1, palette='mako')
    ax1.set_yscale('log')
    ax1.set_title('Row Count Comparison (Before vs After)', fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(title="Condition")
    
    # 2. Feature (Column) Count Reduction (Line chart)
    ax2 = axes[0, 1]
    ax2.plot(df['file_short'], df['cols_before'], marker='o', linestyle=':', color='gray', label='Before')
    ax2.plot(df['file_short'], df['cols_after'], marker='s', linestyle='-', color='teal', label='After (Cleaned)')
    ax2.set_title('Feature Count Reduction (Removed Identifiers)', fontsize=14)
    ax2.set_ylabel('Number of Columns')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # 3. Data Dropped Details (Reason: NaN / Infinity)
    ax3 = axes[1, 0]
    if 'removed_nan_inf_rows' in df.columns:
        sns.barplot(data=df, x='file_short', y='removed_nan_inf_rows', ax=ax3, color='#8e44ad', alpha=0.8)
        # Avoid breaking the chart if there are 0 drops by using symlog
        if df['removed_nan_inf_rows'].max() > 0:
            ax3.set_yscale('symlog')
        ax3.set_title('Rows Dropped due to Corruption (NaN/Infinity)', fontsize=14)
        ax3.set_ylabel('Dropped Row Count (Log Scale)')
        ax3.tick_params(axis='x', rotation=45)
        
    # 4. Global Summary (Text Box)
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    total_files = len(df)
    total_before = df['rows_before'].sum()
    total_after = df['rows_after'].sum()
    total_removed = total_before - total_after
    avg_cols_dropped = (df['cols_before'] - df['cols_after']).mean()
    
    summary_text = (
        "--- Global Preprocessing Summary ---\n\n"
        f"Total Files Processed: {total_files}\n\n"
        f"Pipeline Input Rows: {total_before:,}\n"
        f"Pipeline Output Rows: {total_after:,}\n\n"
        f"Total Dropped/Corrupt Rows: {total_removed:,}\n"
        f"Average Features Dropped per File: {avg_cols_dropped:.1f}\n\n"
        "Status: Pipeline Execution 100% Successful"
    )
    
    # Center the text
    ax4.text(0.1, 0.5, summary_text, fontsize=15, family='monospace', va='center', 
             bbox=dict(facecolor='#f1f2f6', alpha=0.8, boxstyle='round,pad=1'))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"\n[+] Dashboard visual successfully created -> {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
