import os # For directory creation
import glob
import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # For DataFrame and easier plotting
from matplotlib.lines import Line2D
from viz_math_plot import filter_low_percentage_languages, plot_language_heatmaps
# --- Configuration & Setup ---
region_to_country = {
    "Africa": ["Morocco", "Egypt", "Zimbabwe", "South Africa", "Nigeria"],
    "North America": ["United States", "Canada"],
    "South America": ["Chile", "Mexico", "Brazil", "Peru", "Argentina"],
    "East Asia": ["South Korea", "Japan", "Taiwan", "Hong Kong", "China"],
    "South Asia": ["India", "Bangladesh", "Nepal", "Pakistan"],
    "Southeast Asia": ["Vietnam", "Malaysia", "Philippines", "Indonesia", "Singapore", "Thailand"],
    "Middle East/West Asia": ["Iran", "Israel", "Lebanon", "Saudi Arabia", "Turkey"],
    "East Europe": ["Ukraine", "Czech Republic", "Romania", "Poland"],
    "North Europe": ["United Kingdom", "Russia"],
    "South Europe": ["Spain", "Italy"],
    "West Europe": ["Netherlands", "France", "Germany"],
    "Oceania": ["New Zealand", "Australia"]
}
# Define country language info
country_languages = {
    "United States": "English", "United Kingdom": "English", "Australia": "English",
    "Canada": "English", "New Zealand": "English", "Singapore": "English",
    "Zimbabwe": "English", "South Africa": "English", "Nigeria": "English",
    "China": "Chinese", "Japan": "Japanese", "South Korea": "Korean",
    "France": "French", "Germany": "German", "Spain": "Spanish",
    "Italy": "Italian", "Russia": "Russian", "Brazil": "Portuguese",
    "India": "Hindi", "Mexico": "Spanish", "Netherlands": "Dutch",
    "Thailand": "Thai", "Vietnam": "Vietnamese", "Indonesia": "Indonesian",
    "Malaysia": "Malay", "Hong Kong": "Chinese", "Taiwan": "Chinese",
    "Saudi Arabia": "Arabic", "Iran": "Persian", "Israel": "Hebrew",
    "Turkey": "Turkish", "Lebanon": "Arabic", "Pakistan": "Urdu",
    "Bangladesh": "Bengali", "Nepal": "Nepali", "Philippines": "Filipino",
    "Argentina": "Spanish", "Chile": "Spanish", "Peru": "Spanish",
    "Poland": "Polish", "Czech Republic": "Czech", "Romania": "Romanian",
    "Ukraine": "Ukrainian", "Morocco": "Arabic", "Egypt": "Arabic"
}

# --- Core Data Processing Functions ---
# (These functions remain largely the same as before)
def process_file(filepath, stats_dict, condition_key_correct, country_languages_map):
    question_choices = defaultdict(lambda: defaultdict(dict))
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from line in {filepath}: {line.strip()}")
                    continue

                country = row.get('country')
                qid = row.get('question_idx')
                choice_id = row.get('data_idx')

                if not all([country, qid is not None, choice_id is not None, 'answer_extracted' in row, 'answer' in row]):
                    print(f"Warning: Skipping row with missing data in {filepath}: {row}")
                    continue

                if country not in stats_dict:
                    stats_dict[country] = {
                        "en_correct": [],
                        "multi_correct": [],
                        "language": country_languages_map.get(country, "Unknown")
                    }

                pred = str(row['answer_extracted']).strip().lower()
                gt = str(row['answer']).strip().lower()
                question_choices[country][qid][choice_id] = (pred == gt)

        for country, questions in question_choices.items():
            for qid, choices in questions.items():
                all_correct = all(choices.values())
                stats_dict[country][condition_key_correct].append(all_correct)
    except FileNotFoundError:
        print(f"Error: File not found {filepath}. Please check the path and model name.")
    except Exception as e:
        print(f"An unexpected error occurred while processing {filepath}: {e}")


def calculate_accuracies_and_improvement(stats_dict):
    for country, stats in stats_dict.items():
        if stats.get("en_correct"):
            stats["en_accuracy"] = sum(stats["en_correct"]) / len(stats["en_correct"])
        else:
            stats["en_accuracy"] = 0.0

        if stats.get("multi_correct"):
            stats["multi_accuracy"] = sum(stats["multi_correct"]) / len(stats["multi_correct"])
        else:
            stats["multi_accuracy"] = 0.0

        stats["improvement"] = stats["multi_accuracy"] - stats["en_accuracy"]
        stats["is_english_speaking"] = stats["language"] == "English"
    return stats_dict

def old_plot_method(model_name_mapping, all_models_data):
    if not all_models_data:
        print("No data was processed. Exiting visualization phase. Please check your model_names, file paths, and file contents.")
    else:
        df_results = pd.DataFrame(all_models_data)
        df_results['plot_model_name'] = df_results['model_actual'].map(lambda x: model_name_mapping.get(x, x))

        print("\nDataFrame created with processed results (includes 'plot_model_name'):")
        print(df_results.head())

        output_plot_dir = "viz/culture_plots"
        os.makedirs(output_plot_dir, exist_ok=True)
        print(f"\nSaving plots to ./{output_plot_dir}/")

        # --- VISUALIZATION GENERATION ---
        print("Generating visualizations...")

        common_plot_height = 7  # Define common height for Plot 4 and Plot 6

        # Plot 1, 2, 3 (as previously defined - no height change needed unless desired)
        # ... (Plot 1, 2, 3 code remains here) ...
        # Plot 1: Sorted Bar Chart of Improvement by Country
        if not df_results.empty:
            for model_actual_name in df_results['model_actual'].unique():
                plot_model_display_name = model_name_mapping.get(model_actual_name, model_actual_name)
                df_model_specific = df_results[df_results['model_actual'] == model_actual_name].sort_values(by="improvement", ascending=False)

                if df_model_specific.empty:
                    print(f"No data to plot for model {plot_model_display_name} in Plot 1.")
                    continue

                plt.figure(figsize=(12, max(8, len(df_model_specific['country']) * 0.3)))
                sns.barplot(x="improvement", y="country", data=df_model_specific, palette="vlag", dodge=False)
                plt.title(f'Improvement with Native Prefill by Country ({plot_model_display_name})', fontsize=16)
                plt.xlabel("Accuracy Improvement (Native - English)", fontsize=12)
                plt.ylabel("Country", fontsize=12)
                plt.axvline(0, color='grey', lw=1, linestyle='--')
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                plt.tight_layout()
                filename_base = f"1_improvement_by_country_{plot_model_display_name.replace(' ', '_').replace('/', '_')}"
                plt.savefig(os.path.join(output_plot_dir, f"{filename_base}.png"), dpi=300)
                plt.savefig(os.path.join(output_plot_dir, f"{filename_base}.pdf"), format="pdf")
                plt.close()
            print("Plot 1 (Improvement by Country) generated for PNG and PDF.")


        # Plot 2: Grouped Bar Chart: English vs. Native Accuracy by Country
        if not df_results.empty:
            for model_actual_name in df_results['model_actual'].unique():
                plot_model_display_name = model_name_mapping.get(model_actual_name, model_actual_name)
                df_model_specific = df_results[df_results['model_actual'] == model_actual_name].copy()
                if df_model_specific.empty:
                    print(f"No data to plot for model {plot_model_display_name} in Plot 2.")
                    continue

                df_model_specific_sorted = df_model_specific.sort_values(by="improvement", ascending=False)
                print(model_actual_name)
                print(df_model_specific_sorted.head())
                df_melted = df_model_specific_sorted.melt(
                                        id_vars=['country', 'plot_model_name', 'language', 'improvement'],
                                        value_vars=['en_accuracy', 'multi_accuracy'],
                                        var_name='Prefill Type', value_name='Accuracy')
                df_melted['Prefill Type'] = df_melted['Prefill Type'].map(
                                        {'en_accuracy': 'English Prefill',
                                         'multi_accuracy': 'Native Prefill'})
                plt.figure(figsize=(max(14, len(df_model_specific_sorted['country']) * 0.5), 7))
                sns.barplot(x="country", y="Accuracy", hue="Prefill Type", data=df_melted,
                            order=df_model_specific_sorted['country'])
                plt.title(f'English vs. Native Prefill Accuracy by Country ({plot_model_display_name})', fontsize=16)
                plt.xlabel("Country", fontsize=12)
                plt.ylabel("Accuracy", fontsize=12)
                plt.xticks(rotation=60, ha="right", fontsize=10)
                plt.yticks(fontsize=10)
                plt.legend(title="Prefill Type", fontsize=10, title_fontsize=12)
                plt.ylim(0, 1.05)
                plt.tight_layout()
                filename_base = f"2_grouped_accuracy_by_country_{plot_model_display_name.replace(' ', '_').replace('/', '_')}"
                plt.savefig(os.path.join(output_plot_dir, f"{filename_base}.png"), dpi=300)
                plt.savefig(os.path.join(output_plot_dir, f"{filename_base}.pdf"), format="pdf")
                plt.close()
            print("Plot 2 (Grouped Accuracy by Country) generated for PNG and PDF.")

        # Plot 3: Scatter Plot: Improvement vs. Baseline English Accuracy
        if not df_results.empty:
            for model_actual_name in df_results['model_actual'].unique():
                plot_model_display_name = model_name_mapping.get(model_actual_name, model_actual_name)
                df_model_specific = df_results[df_results['model_actual'] == model_actual_name]
                if df_model_specific.empty:
                    print(f"No data to plot for model {plot_model_display_name} in Plot 3.")
                    continue

                plt.figure(figsize=(10, 7))
                sns.scatterplot(x="en_accuracy", y="improvement", hue="is_english_speaking",
                                size="improvement", data=df_model_specific, sizes=(30, 150), alpha=0.7,
                                palette={True: "blue", False: "red"})
                plt.title(f'Improvement vs. Baseline English Accuracy ({plot_model_display_name})', fontsize=16)
                plt.xlabel("Baseline English Prefill Accuracy", fontsize=12)
                plt.ylabel("Improvement (Native - English)", fontsize=12)
                plt.axhline(0, color='grey', lw=1, linestyle='--')
                plt.legend(title='Is English Speaking', fontsize=10, title_fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                plt.tight_layout()
                filename_base = f"3_improvement_vs_baseline_{plot_model_display_name.replace(' ', '_').replace('/', '_')}"
                plt.savefig(os.path.join(output_plot_dir, f"{filename_base}.png"), dpi=300)
                plt.savefig(os.path.join(output_plot_dir, f"{filename_base}.pdf"), format="pdf")
                plt.close()
            print("Plot 3 (Improvement vs. Baseline) generated for PNG and PDF.")


        # Plot 4: Box Plot of Improvement by English-Native Status
        if not df_results.empty:
            if len(df_results['model_actual'].unique()) > 1:
                # Use common_plot_height for the figure height
                plt.figure(figsize=(max(8, len(df_results['plot_model_name'].unique()) * 2.5), common_plot_height))
                sns.boxplot(x="plot_model_name", y="improvement", hue="is_english_speaking", data=df_results, palette="Set2")
                plt.title('Improvement Distribution by Native Language Type (Per Model)', fontsize=16)
                plt.xlabel('')
                plt.ylabel("Improvement (Native - English)", fontsize=12)
                plt.xticks(rotation=25, ha="right", fontsize=10)
                plt.yticks(fontsize=10)
                plt.legend(title='Is English Speaking', labels=['Non-English Native', 'English Native'], fontsize=10, title_fontsize=12)
                plt.tight_layout()
                filename_base = "4_boxplot_improvement_by_english_status_per_model"
                plt.savefig(os.path.join(output_plot_dir, f"{filename_base}.png"), dpi=300)
                plt.savefig(os.path.join(output_plot_dir, f"{filename_base}.pdf"), format="pdf")
                plt.close()
            elif not df_results.empty:
                model_actual_name = df_results['model_actual'].unique()[0]
                plot_model_display_name = model_name_mapping.get(model_actual_name, model_actual_name)
                # Use common_plot_height for the figure height
                plt.figure(figsize=(8, common_plot_height))
                sns.boxplot(x="is_english_speaking", y="improvement", data=df_results[df_results['model_actual'] == model_actual_name], palette="Set2")
                plt.title(f'Improvement Distribution by Native Language Type ({plot_model_display_name})', fontsize=16)
                plt.xlabel("Is Native Language English", fontsize=12)
                plt.ylabel("Improvement (Native - English)", fontsize=12)
                plt.xticks([False, True], ['Non-English Native', 'English Native'], fontsize=10)
                plt.yticks(fontsize=10)
                # plt.
                plt.tight_layout()
                filename_base = f"4_boxplot_improvement_by_english_status_{plot_model_display_name.replace(' ', '_').replace('/', '_')}"
                plt.savefig(os.path.join(output_plot_dir, f"{filename_base}.png"), dpi=300)
                plt.savefig(os.path.join(output_plot_dir, f"{filename_base}.pdf"), format="pdf")
                plt.close()
            print("Plot 4 (Boxplot Improvement by English Status) generated for PNG and PDF.")


        # Plot 5: Heatmap (Models vs. Countries, showing improvement)
        # ... (Plot 5 code remains here - height is typically dynamic based on number of models/countries) ...
        if len(df_results['model_actual'].unique()) > 1 and not df_results.empty:
            try:
                pivot_df = df_results.pivot_table(index="plot_model_name", columns="country", values="improvement")

                plt.figure(figsize=(max(15, len(pivot_df.columns) * 0.6), max(6, len(pivot_df.index) * 0.6)))
                sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="vlag", center=0, linewidths=.5)
                plt.title('Improvement Heatmap: Models vs. Countries', fontsize=16)
                plt.xlabel("Country", fontsize=12)
                plt.ylabel("Model", fontsize=12)
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                plt.tight_layout()
                filename_base = "5_heatmap_improvement_models_vs_countries"
                plt.savefig(os.path.join(output_plot_dir, f"{filename_base}.png"), dpi=300)
                plt.savefig(os.path.join(output_plot_dir, f"{filename_base}.pdf"), format="pdf")
                plt.close()
                print("Plot 5 (Heatmap Models vs. Countries) generated for PNG and PDF.")
            except Exception as e:
                print(f"Could not generate heatmap, possibly due to data shape or missing values: {e}")
        elif len(df_results['model_actual'].unique()) <= 1:
            print("Skipping Plot 5 (Heatmap) as it requires multiple models for comparison.")


        # --- Plot 6: Average Accuracy (English vs Native) per Model with Annotations ---
        if not df_results.empty:
            df_avg_model_acc = df_results.groupby(['model_actual', 'plot_model_name']) \
                                       .agg(avg_en_accuracy=('en_accuracy', 'mean'),
                                            avg_multi_accuracy=('multi_accuracy', 'mean')) \
                                       .reset_index()

            df_avg_melted = df_avg_model_acc.melt(
                id_vars=['model_actual', 'plot_model_name'],
                value_vars=['avg_en_accuracy', 'avg_multi_accuracy'],
                var_name='Prefill Setting',
                value_name='Average Accuracy'
            )

            df_avg_melted['Prefill Setting'] = df_avg_melted['Prefill Setting'].map({
                'avg_en_accuracy': 'Avg. English Prefill',
                'avg_multi_accuracy': 'Avg. Native Prefill'
            })

            model_order = [model_name_mapping.get(m, m) for m in model_names if m in df_results['model_actual'].unique()]
            df_avg_melted['plot_model_name'] = pd.Categorical(df_avg_melted['plot_model_name'], categories=model_order, ordered=True)
            df_avg_melted = df_avg_melted.sort_values('plot_model_name')

            # Use common_plot_height for the figure height
            fig, ax = plt.subplots(figsize=(max(8, len(df_avg_model_acc['plot_model_name'].unique()) * 2.5), common_plot_height))
            barplot = sns.barplot(x='plot_model_name', y='Average Accuracy', hue='Prefill Setting', data=df_avg_melted, ax=ax)

            for p in barplot.patches:
                if abs(p.get_height()) < 0.00001:
                    continue
                barplot.annotate(f"{p.get_height():.3f}",
                               (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha='center', va='center',
                               xytext=(0, 5),
                               textcoords='offset points',
                               fontsize=8,
                               color='black')

            ax.set_title('Average Model Accuracy (English vs. Native Prefill)', fontsize=16)
            ax.set_xlabel('')
            ax.set_ylabel('Average Accuracy Across Countries', fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=10)
            ax.tick_params(axis='y', labelsize=10)
            ax.legend(title='Prefill Setting', fontsize=10, title_fontsize=12, loc='best')

            if not df_avg_melted.empty:
                max_val_in_plot = df_avg_melted['Average Accuracy'].max()
                padding = max_val_in_plot * 0.15 if max_val_in_plot > 0.2 else 0.05
                ax.set_ylim(0, min(1.05, max_val_in_plot + padding))
            else:
                ax.set_ylim(0, 1.05)

            plt.tight_layout()

            filename_base = "6_average_accuracy_english_vs_native_per_model"
            plt.savefig(os.path.join(output_plot_dir, f"{filename_base}.png"), dpi=300)
            plt.savefig(os.path.join(output_plot_dir, f"{filename_base}.pdf"), format="pdf")
            plt.close(fig)
            print("Plot 6 (Average Accuracy per Model with Annotations) generated for PNG and PDF.")

        print("All visualizations generated and saved.")

def create_heatmap_visualization(all_models_data):
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(all_models_data)

    if df.empty:
        print("No data to visualize. Please check your input files and processing.")
        return

    # Get display names for models
    df['model'] = df['model_actual'].map(lambda x: model_name_mapping.get(x, x))
    count = 0
    total = 0
    for idx, row in df.iterrows() :
        if row['improvement'] > 0 and not row['is_english_speaking']:
            count += 1
        if not row['is_english_speaking']:
            total += 1
    print(count, total)

    # Create a pivot table for the heatmap
    pivot_data = df.pivot_table(
        index='model',
        columns='country',
        values='improvement'
    )

    # Sort countries by language (English-speaking first, then others)
    english_countries = [c for c in pivot_data.columns if country_languages.get(c) == "English"]
    non_english_countries = [c for c in pivot_data.columns if c not in english_countries]

    # Sort alphabetically within each group for better readability
    english_countries.sort()
    non_english_countries.sort()
    # Reorder columns to group English and non-English countries
    ordered_columns = english_countries + non_english_countries
    pivot_data = pivot_data[ordered_columns]

    # Determine appropriate vmin/vmax based on data
    max_abs_value = max(abs(pivot_data.min().min()), abs(pivot_data.max().max()))
    vmin = -max_abs_value
    vmax = max_abs_value

    # Create figure with appropriate size
    # Adjust figure size based on number of countries and models
    plt.figure(figsize=(min(30, max(20, len(pivot_data.columns) * 0.5)),
                        min(10, max(5, len(df['model'].unique()) * 0.8 + 2))))

    # Create a custom colormap (green for negative, white for zero, red for positive)
    # Using more saturated colors for better visibility
    cmap = sns.diverging_palette(240, 10, s=99, l=50, as_cmap=True)  # Green to Red

    # Create the heatmap
    ax = sns.heatmap(
        pivot_data,
        cmap=cmap,
        center=0,
        annot=True,  # Show values in cells
        fmt=".2f",   # Format as 2 decimal places
        linewidths=.5,
        vmin=vmin,   # Dynamic range based on data
        vmax=vmax
    )

    # Add a vertical line to separate English from non-English countries
    if english_countries and non_english_countries:
        plt.axvline(x=len(english_countries), color='black', linewidth=2)

    # Set titles and labels
    plt.title('Improvement from English to Native Language Prompts by Country and Model', fontsize=16)
    plt.xlabel('Countries', fontsize=12)
    plt.ylabel('Models', fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add annotations for language groups
    if english_countries:
        plt.text(
            len(english_countries) / 2,
            -0.8,
            'English-speaking Countries',
            ha='center',
            fontsize=10,
            fontweight='bold'
        )

    if non_english_countries:
        plt.text(
            len(english_countries) + len(non_english_countries) / 2,
            -0.8,
            'Non-English-speaking Countries',
            ha='center',
            fontsize=10,
            fontweight='bold'
        )

    # Add a colorbar label
    cbar = ax.collections[0].colorbar
    cbar.set_label('Improvement (Multilingual - English)', rotation=270, labelpad=10)

    # Tight layout to ensure everything fits
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)

    # Save the figure
    plt.savefig('viz/cultural_language_improvement_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig('viz/cultural_language_improvement_heatmap.pdf', format="pdf", dpi=300, bbox_inches='tight')
    print("Heatmap visualization saved to output/language_improvement_heatmap.png")

    # # Show the plot
    # plt.show()


def create_regional_comparison_boxplot(all_models_data):
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(all_models_data)

    if df.empty:
        print("No data to visualize. Please check your input files and processing.")
        return

    # Define the region to country mapping
    region_to_country = {
        "Africa": ["Morocco", "Egypt", "Zimbabwe", "South Africa", "Nigeria"],
        "North America": ["United States", "Canada"],
        "South America": ["Chile", "Mexico", "Brazil", "Peru", "Argentina"],
        "East Asia": ["South Korea", "Japan", "Taiwan", "Hong Kong", "China"],
        "South Asia": ["India", "Bangladesh", "Nepal", "Pakistan"],
        "Southeast Asia": ["Vietnam", "Malaysia", "Philippines", "Indonesia", "Singapore", "Thailand"],
        "Middle East/West Asia": ["Iran", "Israel", "Lebanon", "Saudi Arabia", "Turkey"],
        "East Europe": ["Ukraine", "Czech Republic", "Romania", "Poland"],
        "North Europe": ["United Kingdom", "Russia"],
        "South Europe": ["Spain", "Italy"],
        "West Europe": ["Netherlands", "France", "Germany"],
        "Oceania": ["New Zealand", "Australia"]
    }

    # Create a reverse mapping from country to region
    country_to_region = {}
    for region, countries in region_to_country.items():
        for country in countries:
            country_to_region[country] = region

    # Add region column to the dataframe
    df['region'] = df['country'].map(lambda x: country_to_region.get(x, "Other"))

    # Filter out rows where either en_accuracy or multi_accuracy is 0
    df = df[(df['multi_accuracy'] > 0) & (df['en_accuracy'] > 0)]

    # Get display names for models
    df['model'] = df['model_actual'].map(lambda x: model_name_mapping.get(x, x))

    # Reshape the data to have 'metric' and 'value' columns
    # This format works better for side-by-side boxplots
    melted_df = pd.melt(
        df,
        id_vars=['region', 'country', 'model', 'model_actual'],
        value_vars=['en_accuracy', 'multi_accuracy'],
        var_name='metric',
        value_name='accuracy'
    )

    # Rename metrics for better display
    melted_df['metric'] = melted_df['metric'].replace({
        'en_accuracy': 'Prefill English',
        'multi_accuracy': 'Prefill Target Language'
    })
    
    # Calculate average scores for each model, region, and metric
    region_model_avg = melted_df.groupby(['region', 'model', 'metric'])['accuracy'].mean().reset_index()
    
    # Calculate improvement percentages for specific regions
    # Group by region and metric, then calculate mean accuracy
    region_metric_avg = melted_df.groupby(['region', 'metric'])['accuracy'].mean().reset_index()
    
    # Create a pivot table to easily compare English vs Target Language for each region
    region_comparison = region_metric_avg.pivot(index='region', columns='metric', values='accuracy').reset_index()
    region_comparison['improvement'] = region_comparison['Prefill Target Language'] - region_comparison['Prefill English']
    region_comparison['improvement_pct'] = (region_comparison['improvement'] / region_comparison['Prefill English']) * 100
    
    # Print the improvements for South Europe and Oceania
    south_europe_improvement = region_comparison.loc[region_comparison['region'] == 'South Europe', 'improvement_pct'].values[0]
    oceania_improvement = region_comparison.loc[region_comparison['region'] == 'Oceania', 'improvement_pct'].values[0]
    
    print(f"South Europe improvement: {south_europe_improvement:.1f}%")
    print(f"Oceania improvement: {oceania_improvement:.1f}%")
    
    # Calculate East Asia performance difference between Chinese and non-Chinese models
    east_asia_data = melted_df[melted_df['region'] == 'East Asia']
    
    # Define Chinese-based models
    chinese_models = ['DeepSeek-R1-Distill-Qwen-14B', 'Qwen3-30B-A3B']
    east_asia_data['model_origin'] = east_asia_data['model'].apply(
        lambda x: 'Chinese' if x in chinese_models else 'Other')
    
    # Calculate average performance by model origin and metric
    east_asia_origin_avg = east_asia_data.groupby(['model_origin', 'metric'])['accuracy'].mean().reset_index()
    
    # Create pivot table for easy comparison
    east_asia_comparison = east_asia_origin_avg.pivot(index='model_origin', columns='metric', values='accuracy')
    
    # Calculate the performance difference between Chinese and non-Chinese models
    chinese_en = east_asia_comparison.loc['Chinese', 'Prefill English']
    other_en = east_asia_comparison.loc['Other', 'Prefill English']
    chinese_native = east_asia_comparison.loc['Chinese', 'Prefill Target Language']
    other_native = east_asia_comparison.loc['Other', 'Prefill Target Language']
    
    en_diff = chinese_en - other_en
    native_diff = chinese_native - other_native
    avg_diff = (en_diff + native_diff) / 2
    
    print(f"East Asia performance difference (Chinese vs Other models): {avg_diff:.1f} percentage points")
    
    # Calculate average performance in South Asia
    south_asia_avg = melted_df[melted_df['region'] == 'South Asia']['accuracy'].mean()
    print(f"South Asia mean performance: {south_asia_avg:.1f}%")

    # Define model colors and markers
    model_colors = {
        'QwQ-32B': "#4285F4",                       # Blue
        'Qwen3-30B-A3B': "#EA4335",                 # Red
        'DeepSeek-Qwen-14B': "#FBBC05",  # Yellow
        'DeepSeek-Llama-8B': "#34A853"               # Green
    }

    model_markers = {
        'QwQ-32B': "o",                            # Circle
        'Qwen3-30B-A3B': "^",                      # Triangle
        'DeepSeek-Qwen-14B': "s",       # Square
        'DeepSeek-Llama-8B': "*"                    # Star
    }

    # Order regions (customize as needed)
    region_order = [
        "North America", "South America", "East Europe", "South Europe",
        "West Europe", "Africa",
        "Middle East/West Asia", "East Asia", "South Asia", "Southeast Asia", "Oceania"
    ]

    # Filter only the regions that have data
    filtered_regions = [r for r in region_order if r in region_model_avg["region"].unique()]

    # Count samples per region for labels
    region_counts = {
        "North America": 27,
        "South America": 150,
        "East Europe": 115,
        "South Europe": 76,
        "West Europe": 96,
        "Africa": 134,
        "Middle East/West Asia": 127,
        "East Asia": 211,
        "South Asia": 106,
        "Southeast Asia": 159,
        "Oceania": 26
    }

    # Create figure with appropriate size
    plt.figure(figsize=(16, 8))

    # Create boxplot
    ax = sns.boxplot(
        x="region",
        y="accuracy",
        hue="metric",
        data=melted_df[melted_df['region'].isin(filtered_regions)],
        order=filtered_regions,
        palette={"Prefill English": "#ADD8E6", "Prefill Target Language": "#98FB98"},  # Light blue for English, light green for Native
        width=0.7,
        showfliers=False  # Don't show outliers as they'll be represented by model points
    )

    # Add scatter points for each model's average
    for model in sorted(region_model_avg['model'].unique()):
        for metric_idx, metric in enumerate(['Prefill English', 'Prefill Target Language']):
            # Offset for English vs Native Language points
            offset = -0.2 if metric == 'Prefill English' else 0.2

            # First filter model data to only include regions in filtered_regions
            model_data = region_model_avg[(region_model_avg["model"] == model) &
                                         (region_model_avg["metric"] == metric) &
                                         (region_model_avg["region"].isin(filtered_regions))]

            # Plot each model's points
            if not model_data.empty:
                # Get x-positions for each region
                x_positions = [filtered_regions.index(region) for region in model_data["region"]]

                # Add jitter for better visibility within each metric group
                jitter = np.random.normal(0, 0.05, size=len(x_positions))

                plt.scatter(
                    x=np.array(x_positions) + offset + jitter,
                    y=model_data["accuracy"],
                    color=model_colors.get(model, "#000000"),
                    marker=model_markers.get(model, "o"),
                    s=80,
                    alpha=0.8,
                    zorder=3,
                    label=f"{model} ({metric})" if metric_idx == 0 else "_nolegend_"
                )

    # Format x-axis labels to include counts
    x_labels = [f"{region}\n(N={region_counts.get(region, 0)})" for region in filtered_regions]
    ax.set_xticklabels(x_labels)
    plt.tick_params(axis='x', which='major', labelsize=16)  # Larger font for x-axis values

    # Set the title and labels
    plt.title("Models performance by region: Prefill with English vs. Target Language", fontsize=16)
    plt.xlabel("")
    plt.ylabel("Accuracy (%)", fontsize=20)

    # Create a custom legend for models
    model_legend_elements = [
        Line2D([0], [0], marker=model_markers.get(model, "o"),
               color=model_colors.get(model, "#000000"),
               label=model, markersize=8, linestyle='None')
        for model in sorted(region_model_avg['model'].unique())
    ]

    # Add metric legend elements
    metric_legend_elements = [
        Line2D([0], [0], marker='s', color="#ADD8E6", label='Prefill English', markersize=12, linestyle='None'),
        Line2D([0], [0], marker='s', color="#98FB98", label='Prefill Target Language', markersize=12, linestyle='None')
    ]

    # Place the legends
    # First legend for metrics
    first_legend = plt.legend(handles=metric_legend_elements, loc='upper right',
                                bbox_to_anchor=(0.99, 0.99))
                            #  bbox_to_anchor=(0.01, 0.99))
    # Add the first legend manually to the plot
    ax.add_artist(first_legend)

    # Second legend for models
    plt.legend(handles=model_legend_elements, loc='upper left',
               bbox_to_anchor=(0.01, 0.99))

    # Set y-axis limits
    plt.ylim(0, 100)

    # Add gridlines
    plt.grid(axis='y', linestyle='-', alpha=0.2)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=35, ha='center')

    # Tight layout
    plt.tight_layout()

    # Ensure output directories exist
    os.makedirs('viz', exist_ok=True)

    # Save the figure
    plt.savefig('viz/cultural_regional_comparison_boxplot.png', dpi=300, bbox_inches='tight')
    plt.savefig('viz/cultural_regional_comparison_boxplot.pdf', format="pdf", dpi=300, bbox_inches='tight')
    print("Regional comparison boxplot saved to viz/cultural_regional_comparison_boxplot.png and .pdf")



def plot_language_distributions():
    valid_models = {
        'Distill-Llama-8B',
        'Distill-Qwen-14B',
        'QwQ-32B',
        'Qwen3-30B-A3B',
    }

    model_stats = {}
    for jsonl_filename in glob.glob("log/toxic_bench/*/*B.jsonl"):
        matched = False
        matched_model = None
        for model in valid_models:
            if model in jsonl_filename:
                matched = True
                matched_model = model
                break
        if not matched:
            continue
        print(matched_model)
        if matched_model not in model_stats:
            model_stats[matched_model] = {}
        input_language = jsonl_filename.split('/')[2]
        if input_language not in model_stats[matched_model]:
            model_stats[matched_model][input_language] = {}
        reason_stats = defaultdict(int)
        answer_stats = defaultdict(int)
        with open(jsonl_filename, 'r') as f:
            for line in f:
                row = json.loads(line)
                output = row['output']
                try:
                    if '</think>' in output:
                        reasoning_section = output.split('</think>')[0]
                        reasoning_section = reasoning_section.replace('<think>','')
                        answer_section = output.split('</think>')[-1]
                        reason_lang = langid.predict(reasoning_section)
                        if len(answer_section) == 0:
                            answer_lang = 'unk'
                        else:
                            answer_lang = langid.predict(answer_section)
                        if reason_lang == 'zh-hans':
                            reason_lang = 'zh-CN'
                        if answer_lang == 'zh-hans':
                            answer_lang = 'zh-CN'
                        reason_stats[reason_lang] += 1
                        answer_stats[answer_lang] += 1
                except KeyboardInterrupt as e:
                    raise e
                except: # LangDetectException
                    reason_stats['unk'] += 1
                    answer_stats['unk'] += 1
        model_stats[matched_model][input_language] = {
            'reason_distribution': reason_stats,
            'answer_distribution': answer_stats,
        }
    filtered_model_stats = filter_low_percentage_languages(model_stats, threshold_percentage=2)
    plot_language_heatmaps(filtered_model_stats,
            model_order=['QwQ-32B', 'Qwen3-30B-A3B', 'Distill-Llama-8B', 'Distill-Qwen-14B'],
            distribution_types=['reason_distribution', 'answer_distribution']
        )

# --- Main Analysis Script ---
if __name__ == "__main__":
    # plot_language_distributions()
    # === CONFIGURATION: SET YOUR MODEL NAMES, MAPPING, AND PATHS HERE ===
    model_names = [
        'together__QwQ-32B',
        'openai__Qwen3-30B-A3B',
        'openai__DeepSeek-R1-Distill-Qwen-14B',
        'openai__DeepSeek-R1-Distill-Llama-8B',
    ]

    # Map actual model names (from model_names list) to display names for plots
    model_name_mapping = {
        'together__QwQ-32B': 'QwQ-32B', # Example display name
        'openai__Qwen3-30B-A3B': 'Qwen3-30B-A3B',
        'openai__DeepSeek-R1-Distill-Qwen-14B': 'DeepSeek-Qwen-14B',
        'openai__DeepSeek-R1-Distill-Llama-8B': 'DeepSeek-Llama-8B',
    }
    # If a model_name is not in model_name_mapping, its original name will be used.

    log_file_base_template = 'log/culture_bench/en/answer_extracted/{model_name}'

    # === DATA PROCESSING ===
    all_models_data = []
    print("Starting data processing...")
    for model_name_actual in model_names:
        print(f"Processing model identifier: {model_name_actual}")

        model_log_path_prefix = log_file_base_template.format(model_name=model_name_actual)

        country_stats_for_model = {}

        english_filepath = f'{model_log_path_prefix}__thinking_prefill-Okay.jsonl'
        process_file(
            english_filepath,
            country_stats_for_model,
            "en_correct",
            country_languages
        )
        multilingual_filepath = f'{model_log_path_prefix}__thinking_prefill-multi.jsonl'
        process_file(
            multilingual_filepath,
            country_stats_for_model,
            "multi_correct",
            country_languages
        )

        model_results = calculate_accuracies_and_improvement(country_stats_for_model)

        if not model_results:
            print(f"Warning: No results processed for model identifier {model_name_actual}. Check file paths and content.")
            continue

        for country, stats in model_results.items():
            if country not in {'Canada', "United States"}:
            # if not stats.get("is_english_speaking", False):
                all_models_data.append({
                    "model_actual": model_name_actual,
                    "country": country,
                    "language": stats.get("language", "Unknown"),
                    "en_accuracy": stats.get("en_accuracy", 0.0)*100,
                    "multi_accuracy": stats.get("multi_accuracy", 0.0)*100,
                    "improvement": stats.get("improvement", 0.0)*100,
                    "is_english_speaking": stats.get("is_english_speaking", False)
                })
    # old_plot_method(model_name_mapping, all_models_data)
    create_regional_comparison_boxplot(all_models_data)
    print("Data processing complete.")
