import glob
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from fastlangid.langid import LID
from langdetect import detect
from prefill_tokens import prefill_tokens
import matplotlib.pyplot as plt
langid = LID()

plt.rcParams.update({
    "figure.titlesize": 24,   # overall suptitle –– bigger than everything else
    "axes.titlesize": 20,     # per-axes titles (your model names)
    "axes.labelsize": 18,     # x- and y-axis labels
    "xtick.labelsize": 14,    # tick labels
    "ytick.labelsize": 14,
    "legend.fontsize": 14,    # legend text
})
# Filter out languages with very low percentages
def filter_low_percentage_languages(model_stats, threshold_percentage=1.0, dist_types=['reason_distribution', 'answer_distribution']):
    """
    Filter out detected languages that have a percentage below the threshold.
    
    Args:
        model_stats: The original model_stats dictionary
        threshold_percentage: The minimum percentage to keep a language (default: 1.0%)
        
    Returns:
        A filtered copy of the model_stats dictionary
    """
    filtered_stats = {}
    
    for model, input_lang_data in model_stats.items():
        filtered_stats[model] = {}
        
        for input_lang, distributions in input_lang_data.items():
            filtered_stats[model][input_lang] = {r: {} for r in dist_types}
            
            # Process both reason and answer distributions
            for dist_type in dist_types:
                # Get original distribution
                orig_dist = distributions[dist_type]
                
                # Skip empty distributions
                if not orig_dist:
                    continue
                
                # Calculate total counts for percentage
                total_count = sum(orig_dist.values())
                
                # Only keep languages above threshold
                if total_count > 0:
                    for lang, count in orig_dist.items():
                        percentage = (count / total_count) * 100
                        if percentage >= threshold_percentage:
                            filtered_stats[model][input_lang][dist_type][lang] = count
    return filtered_stats



# Create language heatmaps for each model
def plot_language_heatmaps(model_stats, model_order=[], distribution_types=['reason_distribution', 'answer_distribution'], titles=[]):
    # Get all unique languages across all models and input languages
    all_languages = set()
    all_input_languages = set()
    for model, input_lang_data in model_stats.items():
        all_input_languages.update(input_lang_data.keys())
        for input_lang, distributions in input_lang_data.items():
            for key in distribution_types:
                all_languages.update(distributions[key].keys())
            # all_languages.update(distributions['answer_distribution'].keys())
    # Define the priority order for languages to create diagonal pattern
    priority_langs = ["en", "es", "ja", "ko", "ru", "sw", "te", "zh-CN"]
    # Sort input languages to match priority order
    all_input_languages = sorted(list(all_input_languages), 
                                key=lambda x: (priority_langs.index(x) if x in priority_langs else float('inf'), x))
    # Sort detected languages with priority languages first, then others alphabetically
    def lang_sort_key(lang):
        if lang in priority_langs:
            return (0, priority_langs.index(lang))
        else:
            return (1, lang)  # Non-priority languages come after, sorted alphabetically
    all_languages = sorted(list(all_languages), key=lang_sort_key)
    
    # Define the model order (with Distill-Qwen-14B at the end)
    print(model_order)
    # if len(model_order):
    #     model_order = []
    #     for model in model_stats:
    #         if model != "Distill-Qwen-14B":
    #             model_order.append(model)
    #     # Add Distill-Qwen-14B at the end if it exists
    #     if "Distill-Qwen-14B" in model_stats:
    #         model_order.append("Distill-Qwen-14B")

    # Create figures for reasoning and answer language distributions
    for d_idx, distribution_type in enumerate(distribution_types):
        fig, axes = plt.subplots(1, len(model_stats), figsize=(5*len(model_stats), 6), sharey=True)
        if d_idx < len(titles):
            fig.suptitle(titles[d_idx], fontsize=25)
        else:
            fig.suptitle(f'Language Distribution in {distribution_type.split("_")[0].capitalize()} Section', fontsize=25)
        
        if len(model_stats) == 1:
            axes = [axes]  # Handle the case of single model
        
        # Iterate through models in the specified order
        for i, model in enumerate(model_order):
            print(model)
            input_lang_data = model_stats[model]
            
            # Create a matrix for the heatmap
            heatmap_data = np.zeros((len(all_input_languages), len(all_languages)))
            
            # Fill the matrix with language distribution data
            for row_idx, input_lang in enumerate(all_input_languages):
                if input_lang in input_lang_data:
                    for col_idx, lang in enumerate(all_languages):
                        if lang in input_lang_data[input_lang][distribution_type]:
                            heatmap_data[row_idx, col_idx] = input_lang_data[input_lang][distribution_type][lang]
            
            # Create a DataFrame for the heatmap
            df = pd.DataFrame(heatmap_data, index=all_input_languages, columns=all_languages)
            
            # Normalize by row (input language) to get percentages
            row_sums = df.sum(axis=1)
            df_normalized = df.div(row_sums, axis=0).fillna(0) * 100
            
            # Plot the heatmap
            ax = axes[i]
            
            sns.heatmap(df_normalized, annot=True, cmap="viridis", ax=ax, fmt=".1f", 
                    cbar=(i == len(axes)-1),  # Only show colorbar for last plot
                    cbar_kws={'label': 'Percentage (%)'})
            ax.set_title(model, fontsize=20, pad=12)   # pad adds a little vertical breathing room
            if i == 0:
                ax.set_ylabel("Input Language", fontsize=20)
            ax.set_xlabel("Detected Language", fontsize=20)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"viz/{distribution_type}_heatmap.pdf", bbox_inches='tight', dpi=300)
        plt.savefig(f"viz/{distribution_type}_heatmap.png", bbox_inches='tight', dpi=300)
        # plt.show()


def plot_language_distributions():
    valid_models = {
        'Distill-Llama-8B',
        'Distill-Qwen-14B',
        'QwQ-32B',
        'Qwen3-30B-A3B',
    }

    # model_stats = {}
    # for jsonl_filename in glob.glob("log/MATH-500/*/*B.jsonl"):
    #     matched = False
    #     matched_model = None
    #     for model in valid_models:
    #         if model in jsonl_filename:
    #             matched = True
    #             matched_model = model
    #             break
    #     if not matched:
    #         continue
    #     print(matched_model)
    #     if matched_model not in model_stats:
    #         model_stats[matched_model] = {}
    #     input_language = jsonl_filename.split('/')[2]
    #     if input_language not in model_stats[matched_model]:
    #         model_stats[matched_model][input_language] = {}
    #     reason_stats = defaultdict(int)
    #     answer_stats = defaultdict(int)
    #     with open(jsonl_filename, 'r') as f:
    #         for line in f:
    #             row = json.loads(line)
    #             output = row['output']
    #             try:
    #                 if '</think>' in output:
    #                     reasoning_section = output.split('</think>')[0]
    #                     reasoning_section = reasoning_section.replace('<think>','')
    #                     answer_section = output.split('</think>')[-1]
    #                     reason_lang = langid.predict(reasoning_section)
    #                     if len(answer_section) == 0:
    #                         answer_lang = 'unk'
    #                     else:
    #                         answer_lang = langid.predict(answer_section)
    #                     if reason_lang == 'zh-hans':
    #                         reason_lang = 'zh-CN'
    #                     if answer_lang == 'zh-hans':
    #                         answer_lang = 'zh-CN'
    #                     reason_stats[reason_lang] += 1
    #                     answer_stats[answer_lang] += 1
    #             except KeyboardInterrupt as e:
    #                 raise e
    #             except: # LangDetectException
    #                 reason_stats['unk'] += 1
    #                 answer_stats['unk'] += 1
    #     model_stats[matched_model][input_language] = {
    #         'reason_distribution': reason_stats,
    #         'answer_distribution': answer_stats,
    #     }
    # filtered_model_stats = filter_low_percentage_languages(model_stats, threshold_percentage=2)
    # plot_language_heatmaps(filtered_model_stats,
    #         model_order=['QwQ-32B', 'Qwen3-30B-A3B', 'Distill-Llama-8B', 'Distill-Qwen-14B'],
    #         distribution_types=['reason_distribution', 'answer_distribution']
    #     )
    token2lang = {}
    for model_name, lang2token_mapping in prefill_tokens.items():
        for lang, prefill in lang2token_mapping.items():
            token2lang[prefill] = lang

    token2lang['Ili Kup'] = 'sw'
    token2lang['To evaluate'] = 'en'
    token2lang['题目'] = 'zh-hans'
    token2lang['与えられた問題'] = 'ja'
    token2lang['Para encontrar'] = 'es'
    token2lang['ఇక్కడ'] = 'te'
    token2lang['Для'] = 'ru'
    token2lang['주어진'] = 'ko'
    token2lang['Jibu'] = 'sw'
    token2lang['실제'] = 'sw'

    model_stats = {}
    for jsonl_filename in glob.glob("log/MATH-500/*/*B*thinking_prefill-*.jsonl"):
        input_language = jsonl_filename.split('/')[2]
        if input_language != 'en' and 'Okay' in jsonl_filename:
            continue
        prefill_phrase = jsonl_filename.split('thinking_prefill-')[-1].replace('.jsonl','')
        print(token2lang[prefill_phrase])
        if token2lang[prefill_phrase] != input_language:
            continue
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
            'prefilled_reason_distribution': reason_stats,
            'prefilled_answer_distribution': answer_stats,
        }
    filtered_model_stats = filter_low_percentage_languages(model_stats,
                        threshold_percentage=2,
                        dist_types=['prefilled_reason_distribution', 'prefilled_answer_distribution']
                    )
    plot_language_heatmaps(filtered_model_stats,
            model_order=['QwQ-32B', 'Qwen3-30B-A3B', 'Distill-Llama-8B', 'Distill-Qwen-14B'],
            distribution_types=['prefilled_reason_distribution', 'prefilled_answer_distribution'],
            titles=['Language Distribution in Reasoning After Prefill Target Language', 'Language Distribution in Answer After Prefill Target Language']
        )

if __name__ == "__main__":
    plot_language_distributions()
    # import hashlib
    # import json
    # import glob
    # import os
    # import pandas as pd
    # import numpy as np
    # from datasets import load_dataset
    # from collections import defaultdict
    # from analyze_utils import calc_acc_v2
    # from prefill_tokens import prefill_tokens
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # from collections import defaultdict
    # token2lang = {}
    # for model_name, lang2token_mapping in prefill_tokens.items():
    #     for lang, prefill in lang2token_mapping.items():
    #         token2lang[prefill] = lang
    # token2lang['Ili Kup'] = 'sw'
    # model_names = [
    #     'Qwen-14B',
    #     'QwQ-32B',
    #     'Qwen3-30B-A3B',
    # ]
    # model_name_mapping = {
    #     'QwQ-32B': 'QwQ-32B', # Example display name
    #     'Qwen3-30B-A3B': 'Qwen3-30B-A3B',
    #     'Qwen-14B': 'DeepSeek-R1-Distill-Qwen-14B',
    #     # 'super_model_v3_final': 'SuperNet v3',
    # }
    # # TODO: iterate all model names
    # model_name = 'Qwen-14B'
    # pretty_model_name = model_name_mapping[model_name]


    # data_list = []
    # for jsonl_filename in glob.glob(f"log/MATH-500-*/*/answer_extracted/*{model_name}*.jsonl"):
    #     if 'prompt' in jsonl_filename:
    #         continue
    #     print(jsonl_filename)
    #     if 'thinking_prefill' in jsonl_filename:
    #         prefill_phrase = jsonl_filename.split('thinking_prefill-')[-1].replace('.jsonl','')
    #         lang = token2lang[prefill_phrase]
    #     else:
    #         lang = jsonl_filename.split('/')[2]
    #     print(lang)
    #     average_budgets = []
    #     with open(jsonl_filename.replace('answer_extracted/',''), 'r') as f:
    #         for line in f:
    #             row = json.loads(line)
    #             outputs = row['reasoning_output_tokens']#+row['num_output_tokens']
    #             average_budgets.append(outputs)
    #     acc, count = calc_acc_v2(jsonl_filename)
    #     token_budgets = np.max(average_budgets)
    #     print(token_budgets, acc, len(average_budgets))
    #     # Append data to list for DataFrame
    #     data_list.append({
    #         'language': lang,
    #         'accuracy': acc,
    #         'token_budget': token_budgets,
    #         'file': jsonl_filename,
    #         'count': count
    #     })
    # # TODO : there might be duplicated language setting, ignore the ones with lowest count
    # df = pd.DataFrame(data_list)
    # df = df.sort_values(by='language').reset_index(drop=True)
    
    # # Create the plot with token_budget on x-axis and accuracy on y-axis
    # plt.figure(figsize=(12, 8))

    # # Plot using Seaborn lineplot with languages as different lines
    # sns.lineplot(data=df, x='token_budget', y='accuracy', 
    #             hue='language', style='language', 
    #             markers=True, markersize=10, linewidth=2,
    #             palette='deep', legend='full')

    # # Customize the plot
    # plt.xlabel('Token Budget', fontsize=14)
    # plt.ylabel('Accuracy', fontsize=14)
    # plt.title(pretty_model_name, fontsize=16, fontweight='bold')
    # plt.legend(title='Language', bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # # TODO: save to viz/math-500-scaling/*.pdf