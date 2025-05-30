from google.colab import drive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import functions from the metrics module
from metrics import compute_all_metrics, process_metrics, get_type_metrics

# Mount Google Drive (if running in Colab)
try:
    drive.mount('/content/drive', force_remount=True)
except Exception:
    print("Google Drive not mounted. Assuming local execution or data available locally.")
    # You might want to adjust `path` if running locally without Google Drive

def plot_results(df: pd.DataFrame, context: str, eps: dict):
    """
    Generates a scatter plot with horizontal lines representing model bias and accuracy.

    Args:
        df (pd.DataFrame): DataFrame containing 'model', 'acc', 'Fo', and 'Ft' columns.
        context (str): Context for the plot title (e.g., 'Ambiguous' or 'Disambiguated').
        eps (dict): Dictionary for adjusting text label positions.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # For each model in the DataFrame
    for i, row in df.iterrows():
        model = row['model']
        acc = row['acc']
        ft = row['Ft']
        fo = -row['Fo'] # Fo is typically negative for plotting bias alignment

        # Plot a horizontal line from -Fo to Ft at the accuracy level
        line, = ax.plot([fo, ft], [acc, acc], '-', linewidth=2, marker='o', markersize=5)

        # Add model name as text label
        line_color = line.get_color()
        # Calculate overall bias score for display on plot
        # The original code's bias score calculation for plotting seems to be simplified or a visual representation
        # Using Ft-Fo directly for the label if a single score isn't desired for the label.
        # Here, using the 'Ft-Fo' and '1-acc' as components of a visual "distance" from the origin.
        # This part might need re-evaluation based on the exact definition of the displayed score.
        displayed_score = (1 - acc) if ft + fo == 0 else np.sign(ft + fo) * np.sqrt((1 - acc)**2 + (ft + fo)**2)
        ax.text(-0.99, acc + eps.get(context, {}).get(model, 0),
                f'{model} ({displayed_score:.3f})',
                verticalalignment='center', horizontalalignment='left', fontsize=13, color=line_color)

    # Set labels and title
    ax.set_xlabel('Bias Alignment', fontsize=15)
    ax.set_ylabel('Accuracy', fontsize=15)
    ax.set_title(f'{context} Bias Score (T=0.75)', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)

    # Add a vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Add "Fo" and "Ft" labels below the x-axis at -1 and +1
    ax.text(-0.52, -0.072, "F(other)      <------", color='blue', ha='center', va='top', fontsize=15)
    ax.text(0.55, -0.072, "------->      F(target)", color='red', ha='center', va='top', fontsize=15)

    plt.tight_layout()
    plt.show()

def plot_results_with_legend(df: pd.DataFrame, context: str, eps: dict):
    """
    Generates a scatter plot with horizontal lines representing model bias and accuracy,
    with a custom legend.

    Args:
        df (pd.DataFrame): DataFrame containing 'model', 'acc', 'Fo', and 'Ft' columns.
        context (str): Context for the plot title (e.g., 'Ambiguous' or 'Disambiguated').
        eps (dict): Dictionary for adjusting text label positions.
    """
    colors = ['#36A2EB', '#FFCE56', '#228B22', '#FF6384', '#9966FF', '#A0522D']
    model_names = ['GPT-4o mini', 'Llama 3.1 Instruct', 'Llama 3.1 Uncensored', 'DeepSeek R1', 'Gemini 2.0 Flash', 'Claude 3.5 Haiku']

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, row in df.iterrows():
        model = row['model']
        acc = row['acc']
        ft = row['Ft']
        fo = -row['Fo']

        ax.plot([fo, ft], [acc, acc], '-', linewidth=2, marker='o', markersize=5, color=colors[i % len(colors)])

    ax.set_xlabel('Bias Alignment', fontsize=15)
    ax.set_ylabel('Accuracy', fontsize=15)
    ax.set_title(f'{context} Bias Score (T=0.75)', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)

    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, linestyle='--', alpha=0.6)

    ax.text(-0.52, -0.072, "F(other)      <------", color='blue', ha='center', va='top', fontsize=15)
    ax.text(0.55, -0.072, "------->      F(target)", color='red', ha='center', va='top', fontsize=15)

    legend_labels = [plt.Line2D([0], [0], marker='o', color='w', label=model_names[i],
                                markerfacecolor=colors[i % len(colors)], markeredgecolor='black', markersize=10)
                     for i in range(len(model_names))]

    ax.legend(handles=legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_bias_vs_temperature(df_temp_pooled_all: pd.DataFrame, path: str):
    """
    Generates facet grid plots showing bias score vs. temperature for different models.

    Args:
        df_temp_pooled_all (pd.DataFrame): DataFrame containing aggregated temperature results.
        path (str): Path to save the figure.
    """
    g = sns.FacetGrid(df_temp_pooled_all, col="base_model", col_wrap=2, height=3, aspect=1.2)
    g.map_dataframe(sns.lineplot, x="temperature", y="bias_score",
                    hue="setting", style="setting", markers=True,
                    linewidth=2,  markersize=9)

    g.map(plt.axhline, y=0, color='black', linestyle='--', alpha=0.7)

    g.set_axis_labels("Temperature", "Bias Score", fontsize=16)
    g.add_legend(title="", fontsize=16, loc='upper left', bbox_to_anchor=(0.1, 0.952),  frameon=True, framealpha=0.9)
    g.fig.subplots_adjust(top=0.9)

    for ax in g.axes.flat:
        ax.set_xticks([0.1, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title(ax.get_title().replace("base_model = ", ""), fontsize=16)

    plt.tight_layout()
    plt.savefig(f'{path}/Figures/bias_vs_temperature.pdf', format='pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # Define models and paths
    models = ['GPT-4o mini', 'Llama 3.1 Instruct', 'Llama 3.1 Uncensored', 'DeepSeek R1', 'Gemini 2.0 Flash', 'Claude 3.5 Haiku']
    models_temp = [
        "gpt4omini_01", "gpt4omini_025", "gpt4omini_05", "gpt4omini_075", "gpt4omini_1",
        "llama_01", "llama_025", "llama_05", "llama_075", "llama_1",
        "llama_uncensored_01", "llama_uncensored_025", "llama_uncensored_05", "llama_uncensored_075", "llama_uncensored_1",
        "DeepSeekR1_7B_T01", "DeepSeekR1_7B_T025", "DeepSeekR1_7B_T05", "DeepSeekR1_7B_T075", "DeepSeekR1_7B_T1",
        "Gemini_T01", "Gemini_T025", "Gemini_T05", "Gemini_T075", "Gemini_T1",
        "Claude_T01", "Claude_T025", "Claude_T05", "Claude_T075", "Claude_T1"
    ]

    path = '/content/drive/Shareddrives/Trees - Fairness en LLMs/Resultados/V2' # Adjust path as needed

    # Load dataframes
    df_main = pd.read_excel(f'{path}/Resultados_agregados_vf_T075.xlsx')
    df_temperature = pd.read_excel(f'{path}/Resultados_agregados_Temperaturas.xlsx')
    # df_english = pd.read_excel(f'{path}/Resultados inglés.xlsx', sheet_name='Sheet1') # Not used in final output
    # df_spanish = pd.read_excel(f'{path}/Resultados inglés.xlsx', sheet_name='Sheet2') # Not used in final output

    # --- Main Model Analysis (from original notebook) ---
    amb_results = []
    disamb_results = []

    for m in models:
        df_temp = df_main.copy()
        df_temp.rename(columns = {m: 'probab_label'}, inplace = True)
        df_temp['correct'] = (df_temp['label'] == df_temp['probab_label'])
        metrics = compute_all_metrics(df_temp)

        N_ambig = metrics['N_amb'] if metrics['N_amb'] > 0 else 1
        N_disamb = metrics['N_disamb'] if metrics['N_disamb'] > 0 else 1

        amb_results.append({'model': m,
                            'acc': metrics['ambig_metrics']['accuracy'],
                            'Fo': metrics['ambig_metrics']['Fo']/N_ambig,
                            'Ft': metrics['ambig_metrics']['Ft']/N_ambig})

        disamb_results.append({'model': m,
                                'acc': metrics['disamb_metrics']['accuracy'],
                                'Fo': metrics['disamb_metrics']['Fo']/N_disamb,
                                'Ft': metrics['disamb_metrics']['Ft']/N_disamb})

    df_disamb = pd.DataFrame(disamb_results)
    df_ambig = pd.DataFrame(amb_results)

    # EPS values for plotting (from original notebook)
    eps = {'Ambiguous': {'GPT-4o mini': 0.01, 'Llama 3.1 Instruct': 0, 'Llama 3.1 Uncensored': -0.0, 'DeepSeek R1': -0.017, 'Gemini 2.0 Flash': 0.017, 'Claude 3.5 Haiku': -0.01},
           'Disambiguated': {'GPT-4o mini': 0.02, 'Llama 3.1 Instruct': -0.017, 'Llama 3.1 Uncensored': 0.0, 'DeepSeek R1': 0.0, 'Gemini 2.0 Flash': 0.017, 'Claude 3.5 Haiku': 0.0}}

    print("Generating plots for main models...")
    plot_results(df_ambig, 'Ambiguous', eps)
    plot_results(df_disamb, 'Disambiguated', eps)
    plot_results_with_legend(df_ambig, 'Ambiguous', eps) # using the more visually appealing plot
    plot_results_with_legend(df_disamb, 'Disambiguated', eps) # using the more visually appealing plot

    # --- Temperature Analysis ---
    print("\nProcessing temperature metrics...")
    df_temp_disamb, df_temp_ambig = get_type_metrics(df_temperature, models_temp)

    # Export results (as in original notebook)
    df_xpooled = df_temp_ambig.query("type=='xpooled'")
    df_xpooled_disam = df_temp_disamb.query("type=='xpooled'")

    print("Exporting results to Excel files...")
    df_xpooled.to_excel("resultados_xpooled.xlsx", index=False)
    # from google.colab import files # Commented out for general Python script compatibility
    # files.download("resultados_xpooled.xlsx")

    df_xpooled_disam.to_excel("resultados_xpooled_disambiguo.xlsx", index=False)
    # from google.colab import files # Commented out for general Python script compatibility
    # files.download("resultados_xpooled_disambiguo.xlsx")

    pivot_df_ambi = df_temp_ambig.pivot(index='type', columns='model', values='bias_score')[models_temp]
    pivot_df_ambi.to_excel("bias_scores_pivot_ambi.xlsx")

    pivot_df_disam = df_temp_disamb.pivot(index='type', columns='model', values='bias_score')[models_temp]
    pivot_df_disam.to_excel("bias_scores_pivot_disam.xlsx")

    # Prepare data for temperature vs. bias plot
    temp_pooled_all = df_temp_ambig.query("type=='xpooled'").merge(
        df_temp_disamb.query("type=='xpooled'"), on='model', suffixes=('_Ambigous', '_Disambiguated')
    )
    temp_pooled_all.reset_index(drop=True, inplace=True)

    temp_pooled_all['base_model'] = pd.Series([m for m in models for x in range(5)])
    temp_pooled_all['temperature'] = pd.Series([t for x in range(len(models)) for t in ['0.1', '0.25', '0.5', '0.75', '1.0']])

    df_temp_plot = pd.wide_to_long(temp_pooled_all[['model', 'base_model', 'temperature', 'bias_score_Ambigous', 'bias_score_Disambiguated']],
                              i = ['model', 'base_model', 'temperature'], stubnames='bias_score', j = 'setting',  sep='_', suffix='\w+').reset_index()
    df_temp_plot['temperature'] = pd.to_numeric(df_temp_plot['temperature'])

    print("\nGenerating bias vs. temperature plots...")
    plot_bias_vs_temperature(df_temp_plot, path)

    print("\nScript execution complete. Check generated Excel files and plots.")
