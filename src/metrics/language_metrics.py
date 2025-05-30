from google.colab import drive # Keep this if you're primarily running in Google Colab
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import functions from the metrics module
from metrics import get_type_metrics # Ensure metrics.py is in the same folder or in PYTHONPATH

# Mount Google Drive (if running in Colab)
try:
    drive.mount('/content/drive', force_remount=True)
except Exception:
    print("Google Drive not mounted. Assuming local execution or data available locally.")
    # You might want to adjust 'path' if running locally without Google Drive.
    # For local execution, you might need to change `path` to a local directory
    # or ensure your Excel files are in the same folder as this script.


def plot_language_bias(df_full: pd.DataFrame, plot_title: str):
    """
    Generates a scatter plot to compare bias scores in English
    and Spanish (matched and full) for different models.

    Args:
        df_full (pd.DataFrame): DataFrame containing 'model',
                                'bias_english', 'bias_spanish', and 'bias_spanish_full' columns.
                                (Column names are dynamically adjusted
                                to be generic, 'bias_english' and 'bias_spanish'
                                for comparison data, and 'bias_spanish_full'
                                for full Spanish data).
        plot_title (str): Plot title (e.g., 'Ambiguous Bias Scores' or 'Disambiguated Bias Scores').
    """
    x_pos = np.arange(len(df_full))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#36A2EB','#FFCE56', '#228B22','#FF6384', '#9966FF', '#A0522D'] * 2 # Colors for the models

    for i, (idx, row) in enumerate(df_full.iterrows()):
        color = colors[i % len(colors)]

        # English - Circle
        ax.scatter(x_pos[i], row.iloc[2], # Assuming the 3rd column is 'bias_english'
                   color=color, s=100, zorder=3,
                   edgecolors='black', linewidth=1.2, label='English' if i == 0 else "")

        # Spanish (matched) - Square
        ax.scatter(x_pos[i], row.iloc[3], # Assuming the 4th column is 'bias_spanish'
                   color=color, marker='s', s=100, zorder=3,
                   edgecolors='black', linewidth=1.2, label='Spanish (matched)' if i == 0 else "")

        # Spanish Full Line (horizontal, model color)
        ax.hlines(y=row.iloc[4], # Assuming the 5th column is 'bias_spanish_full'
                   xmin=x_pos[i] - 0.1, xmax=x_pos[i] + 0.1,
                   colors=color, linestyles='-', linewidth=2, zorder=1, label='Spanish (full)' if i == 0 else "")

        # Arrow between English and Spanish (matched)
        ax.annotate('',
                    xy=(x_pos[i], row.iloc[3]), # Spanish (matched)
                    xytext=(x_pos[i], row.iloc[2]), # English
                    arrowprops=dict(facecolor=color, arrowstyle='->', lw=2),
                    zorder=2)

        # Display text values
        # Square text (Spanish matched)
        if row.iloc[3] < row.iloc[2]:
            ax.text(x_pos[i], row.iloc[3] - 0.04,
                    f"{row.iloc[3]:.2f}", fontsize=11,
                    ha='center', va='top')
        else:
            ax.text(x_pos[i], row.iloc[3] + 0.04,
                    f"{row.iloc[3]:.2f}", fontsize=11,
                    ha='center', va='bottom')

        # Circle text (English) → below or above depending on relative position
        if row.iloc[2] < row.iloc[3]:
            va_pos = 'top'
            offset = -0.04
        else:
            va_pos = 'bottom'
            offset = 0.04

        ax.text(x_pos[i], row.iloc[2] + offset,
                f"{row.iloc[2]:.2f}", fontsize=11,
                ha='center', va=va_pos)

        # Spanish full (horizontal) → to the right, centered
        ax.text(x_pos[i] + 0.15, row.iloc[4],
                f"{row.iloc[4]:.2f}", fontsize=11,
                ha='left', va='center')

    # X-axis: model names
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_full['model'], rotation=45, ha='right', fontsize=12)

    # Base horizontal line at 0
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)

    # Vertical limits
    # Ensure column names exist before accessing them
    col_names = [df_full.columns[2], df_full.columns[3], df_full.columns[4]]
    y_min = df_full[col_names].min().min() - 0.2
    y_max = df_full[col_names].max().max() + 0.2
    ax.set_ylim(y_min, y_max)

    # Labels and title
    ax.set_ylabel('Bias Score', fontsize=12)
    ax.set_title(plot_title, fontsize=14)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='English',
                   markerfacecolor='gray', markeredgecolor='black', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='Spanish (matched)',
                   markerfacecolor='gray', markeredgecolor='black', markersize=10),
        plt.Line2D([0], [0], color='black', lw=2, label='Spanish (full)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Grid
    ax.grid(axis='y', linestyle='--', alpha=0.2)

    ax.set_xlim(-0.5, len(df_full) - 0.5 + 0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Define the models
    models = ['GPT-4o mini', 'Llama 3.1 Instruct', 'Llama 3.1 Uncensored', 'DeepSeek R1', 'Gemini 2.0 Flash', 'Claude 3.5 Haiku']

    # Path to your Excel files
    path = '/content/drive/Shareddrives/Trees - Fairness en LLMs/Resultados/V2' # Adjust path as needed

    # Load DataFrames
    try:
        df_english = pd.read_excel(f'{path}/Resultados inglés.xlsx', sheet_name='Sheet1')
        df_spanish = pd.read_excel(f'{path}/Resultados inglés.xlsx', sheet_name='Sheet2')
        # df_main is necessary for df_type_ambig and df_type_disamb
        df_main = pd.read_excel(f'{path}/Resultados_agregados_vf_T075.xlsx')
    except FileNotFoundError:
        print(f"Error: Make sure your Excel files are in the specified path: {path}")
        print("If you're running locally, adjust the 'path' variable or place the files in the same directory as the script.")
        exit() # Exit if files are not found

    # Ensure df_main has the 'tipo' column if needed for get_type_metrics
    # (The original notebook code already had it, but it's a good check)
    if 'tipo' not in df_main.columns:
        print("Warning: 'tipo' column not found in df_main. Ensure the 'tipo' column exists.")
        # You could add a dummy column or handle the error differently if critical.
        df_main['tipo'] = 'xpooled' # Default value if it doesn't exist

    # Calculate metrics for English and Spanish
    print("Calculating metrics for English and Spanish...")
    df_en_disamb, df_en_ambig = get_type_metrics(df_english, models)
    df_es_disamb, df_es_ambig = get_type_metrics(df_spanish, models)
    df_type_disamb, df_type_ambig = get_type_metrics(df_main, models) # Necessary for 'spanish_full'

    # --- Analysis and Plotting for Ambiguous Bias Scores ---
    print("\nProcessing and plotting ambiguous bias metrics...")
    en_amb_xpooled = df_en_ambig.query("type == 'xpooled'")
    es_amb_xpooled = df_es_ambig.query("type == 'xpooled'")
    es_full_amb_xpooled = df_type_ambig.query("type == 'xpooled'")

    # Merge DataFrames for ambiguous plot
    enes_amb_merged = en_amb_xpooled.merge(es_amb_xpooled, on=['model', 'type'], suffixes=('_en', '_es'))
    enes_amb_full = enes_amb_merged.merge(
        es_full_amb_xpooled[['model', 'type', 'bias_score']],
        on=['model', 'type'],
        how='left'
    )
    enes_amb_full.rename(columns={
        'bias_score_en': 'bias_english',
        'bias_score_es': 'bias_spanish',
        'bias_score': 'bias_spanish_full'
    }, inplace=True)

    plot_language_bias(enes_amb_full, 'Ambiguous Bias Scores: English vs Spanish (Matched & Full)')

    # --- Analysis and Plotting for Disambiguated Bias Scores ---
    print("\nProcessing and plotting disambiguated bias metrics...")
    en_disamb_xpooled = df_en_disamb.query("type == 'xpooled'")
    es_disamb_xpooled = df_es_disamb.query("type == 'xpooled'")
    es_full_disamb_xpooled = df_type_disamb.query("type == 'xpooled'")

    # Merge DataFrames for disambiguated plot
    enes_disamb_merged = en_disamb_xpooled.merge(es_disamb_xpooled, on=['model', 'type'], suffixes=('_en', '_es'))
    enes_disamb_full = enes_disamb_merged.merge(
        es_full_disamb_xpooled[['model', 'type', 'bias_score']],
        on=['model', 'type'],
        how='left'
    )
    enes_disamb_full.rename(columns={
        'bias_score_en': 'bias_english',
        'bias_score_es': 'bias_spanish',
        'bias_score': 'bias_spanish_full'
    }, inplace=True)

    plot_language_bias(enes_disamb_full, 'Disambiguated Bias Scores: English vs Spanish (Matched & Full)')

    print("\nScript execution complete. Language comparison plots generated.")
