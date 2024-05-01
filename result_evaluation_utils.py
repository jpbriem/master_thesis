import os
import json
import ast
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import seaborn as sns
from IPython.display import clear_output

from itertools import combinations
import matplotlib.pyplot as plt
from holoviews.plotting.util import process_cmap
from adjustText import adjust_text
from tot.methods.arc_utils import load_arc_tasks
import warnings
warnings.filterwarnings("ignore")

SOTA = {
    'arc': {
        'categories_cnt': {'training': 23},
        'success_cnt': 23,
        'categories_rate': {'training': 0.46},
        'success_rate': 0.46
    },
    'arc_h_v': {
        'categories_cnt': {
            'fill_h': 48, 'fill_v': 49, 'move_h': 21, 'move_v': 20, 'pile_h': 42, 'pile_v': 37
        },
        'success_cnt': 217,
        'categories_rate': {
            'fill_h': 0.96, 'fill_v': 0.98, 'move_h': 0.42, 'move_v': 0.4, 'pile_h': 0.84, 'pile_v': 0.74
        },
        'success_rate': 0.7233333333333334
    },
    'arc_1D': {
        'categories_cnt': {
            '1d_denoising_1c': 48, '1d_denoising_mc': 50, '1d_fill': 49, '1d_flip': 50,
            '1d_hollow': 48, '1d_mirror': 13, '1d_move_1p': 50, '1d_move_2p': 50,
            '1d_move_2p_dp': 50, '1d_move_3p': 49, '1d_move_dp': 37, '1d_padded_fill': 44,
            '1d_pcopy_1c': 45, '1d_pcopy_mc': 47, '1d_recolor_cmp': 28, '1d_recolor_cnt': 40,
            '1d_recolor_oe': 13, '1d_scale_dp': 46
        },
        'success_cnt': 757,
        'categories_rate': {
            '1d_denoising_1c': 0.96, '1d_denoising_mc': 1.0, '1d_fill': 0.98, '1d_flip': 1.0,
            '1d_hollow': 0.96, '1d_mirror': 0.26, '1d_move_1p': 1.0, '1d_move_2p': 1.0,
            '1d_move_2p_dp': 1.0, '1d_move_3p': 0.98, '1d_move_dp': 0.74, '1d_padded_fill': 0.88,
            '1d_pcopy_1c': 0.9, '1d_pcopy_mc': 0.94, '1d_recolor_cmp': 0.56, '1d_recolor_cnt': 0.8,
            '1d_recolor_oe': 0.26, '1d_scale_dp': 0.92
        },
        'success_rate': 0.8411111111111111
    }
}

##############################################
# Create Log Summaries
##############################################
def flatten_dict(d, parent_key='', sep='_'):
    """
    Flatten a nested dictionary. Concatenate keys for nested elements.
    """
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif k == "new_representation":
            if v is None:
                items.append((new_key, "0 - 9"))
            elif "." in v:
                items.append((new_key, "'.', 'a' - 'i'"))
            elif "a" in v:
                items.append((new_key, "'a' - 'j'"))

            
        else:
            items.append((new_key, v))
    return dict(items)

def get_avg_task_complexity(tasks_jsons, tasks_names, solved_tasks=None):  
    # derive complexity => number of pixels in the test input
    complexity = []
    for task_json in tasks_jsons:
        array = np.array(task_json["test"][0]["input"])
        shape = array.shape
        complexity.append(np.prod(shape))

    df = pd.DataFrame({'task_name': tasks_names, 'complexity': complexity})
    
    if solved_tasks is None:
        return df.complexity.mean()  
       
    sum_complexity = 0
    for solved_task in solved_tasks:
        index = np.where( solved_task[0] == df['task_name'])
        sum_complexity += df['complexity'].iloc[index[0][0]]
    if sum_complexity == 0:
        return 0
    return sum_complexity / len(solved_tasks)
    
def read_and_parse_tasks_log(path, save_to_csv=False):
    data = []

    if not os.path.isdir(path):
        print(f"The provided path {path} is not a directory.")
        return pd.DataFrame()
    
    # derive task
    task = path.split("/")[-1]
    # get all tasks
    if task == "arc":
        data_dir = "ARC_datasets/ARC"
    elif task == "arc_1D":
        data_dir = "ARC_datasets/1D-ARC/dataset"
    elif task == "arc_h_v":
        data_dir = "ARC_datasets/arc_new"
    #data_dir = os.path.join("tree-of-thought-llm/src/tot/data", task)
    tasks_jsons, tasks_names, _ = load_arc_tasks(data_dir, task)

    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            log_file = os.path.join(item_path, 'all_tasks_log.json')
            if os.path.isfile(log_file):
                try:
                    with open(log_file, 'r') as file:
                        log_data = json.load(file)[0]  # Read only the first item
                        flattened_data = flatten_dict(log_data)
                        solved_tasks = flattened_data["solved_tasks"]
                        if len(solved_tasks) >= 3:
                            avg_complexity = get_avg_task_complexity(tasks_jsons, tasks_names, flattened_data["solved_tasks"])
                        else:
                            avg_complexity = 0 
                        flattened_data["avg_complexity"] = avg_complexity
                        if "solved_tasks_str_comparison" not in flattened_data:
                            flattened_data["success_cnt_w/o_str_comparison"] = 0
                            flattened_data["success_rate_w/o_str_comparison"] = 0
                        else:
                            flattened_data["success_cnt_w/o_str_comparison"] = flattened_data["success_cnt"] - len(flattened_data["solved_tasks_str_comparison"])
                            if flattened_data["success_cnt"] == 0:
                                flattened_data["success_rate_w/o_str_comparison"] = 0
                            else:
                                flattened_data["success_rate_w/o_str_comparison"] = flattened_data["success_cnt_w/o_str_comparison"] / (flattened_data["success_cnt"] / flattened_data["success_rate"])
                        data.append(flattened_data)
                except Exception as e:
                    print(f"Error reading {log_file}: {e}")
    df = pd.DataFrame(data)
    if save_to_csv:
        df.to_csv(path+"/summary.csv", index=False)
        print("saved to "+path+"/summary.csv")
    return df

##############################################
# Plotting Functions
##############################################

model_names = {
    "Llama-2-70b-Chat-GPTQ": "LLaMa 2 Chat - 70B",
    "Llama-2-13B-chat-GPTQ": "LLaMa 2 Chat - 13B",
    "Llama-2-7b-chat-hf": "LLaMa 2 Chat - 7B",
    "Llama-2-70B-GPTQ": "LLaMa 2 - 70B",
    "Llama-2-13B-GPTQ": "LLaMa 2 - 13B",
    "Llama-2-7B-GPTQ": "LLaMa 2 - 7B",
    "Platypus2-70B-GPTQ": "Platypus 2 - 70B",
    "Platypus2-13B-GPTQ": "Platypus 2 - 13B",
    "Mistral-7B-v0.1": "Mistral",
    "Mistral-7B-Instruct-v0.1": "Mistral Instruct",
    "Mixtral-8x7B-v0.1": "Mixtral",
    "Mixtral-8x7B-Instruct-v0.1": "Mixtral Instruct",
    "Mixtral-8x7B-Instruct-v0.1_rep": "Mixtral Instruct - Objects",
    "vicuna-7B-v1.5-GPTQ": "Vicuna - 7B",
    "vicuna-13B-v1.5-GPTQ": "Vicuna - 13B",
    "Qwen-7B-Chat-Int4": "Qwen Chat - 7B",
    "Qwen-72B-Chat-Int4": "Qwen Chat - 72B",
    "gpt-4-1106-preview": "GPT-4 Turbo",
    "gpt-4-1106-preview_rep": "GPT-4 Turbo - Object",
    "gpt-4-1106-preview_arc_h_v": "GPT-4 Turbo: ARC variant:\nobject orientation",
    "gpt-4-1106-preview_arc": "GPT-4 Turbo: ARC subset",
    "gpt-4-1106-preview_arc_1D": "GPT-4 Turbo: 1D-ARC",
    "gpt-3.5-turbo-1106": "GPT-3.5 Turbo",
    "gpt-3.5-turbo-1106_rep": "GPT-3.5 Turbo - Objects",
    "SOTA": "SOTA",
    "CoT": "CoT",
    "ToT": "ToT"
}

def plot_model_performance_across_runs(dataframe, exp_column, score_column, run_column="model", task="", top_n_models=None, SOTA=None, value_ticks="highest_only", small_fig=False):
    def plot_single_dataframe(df, ax, run_column, exp_column, score_column, task="", top_n_models=None, SOTA=None, value_ticks="highest_only", small_fig=False):
        # Filter the relevant columns
        df_filtered = df[[run_column, exp_column, score_column]]
        
        colors = sns.color_palette("viridis", as_cmap=False)

        # delete model prefix and change to normal model name
        if "model" in df_filtered.columns:
            df_filtered["model"] = [model_names[m.split("/")[-1]] for m in df_filtered["model"]]
        
        # Convert Score to numeric values if they are in percentage format
        if df_filtered[score_column].dtype == object:
            df_filtered[score_column] = df_filtered[score_column].str.rstrip('%').astype('float') / 100.0

        # Group by run_column and find the maximum 'Success' value for each group
        group_max = df_filtered.groupby(run_column)[score_column].max()

        # Sort the groups by maximum 'Success' value
        sorted_groups = group_max.sort_values(ascending=False).index.tolist()

        # filter the top n models
        if top_n_models is not None:
            sorted_groups = sorted_groups[:top_n_models]

        # Sort within each group by 'Representation' and concatenate
        if exp_column == "new_representation":
            sorted_df = pd.concat([df_filtered[df_filtered[run_column] == group].sort_values(by=exp_column) for group in sorted_groups])
            sorted_df.reset_index(drop=True, inplace=True)
            keys = list(sorted_df["new_representation"].unique())
            keys.sort()
        else:
            keys = []
            sorted_df = df_filtered.copy()
        

        # Handling for score column not in percentage format
        if score_column == "avg_complexity":
            m = 1
        else:
            m = 1

        # Create a line plot for each model using ax object
        for i, model in enumerate(sorted_groups):
            if i == 0:
                line_type = "-"
            elif i == 5:
                line_type = "--"
            elif i == 10:
                line_type = "-."
            model_data = sorted_df[sorted_df[run_column] == model]
            rows_to_add = []
            for key in keys:
                if not model_data[model_data[exp_column] == key].empty:
                    continue  # Skip to next key if this one already exists
                else:
                    # If the key does not exist, create a new row
                    new_row = model_data.iloc[0].copy()  # Copy any row, here we copy the first row
                    new_row[exp_column] = key  # Set the 'A' column to the missing value
                    new_row[score_column] = 0  # Set the 'success_rate' to 0
                    rows_to_add.append(new_row)
            if rows_to_add:
                new_rows_df = pd.DataFrame(rows_to_add)
                model_data = pd.concat([model_data, new_rows_df])  # Append the new row to the DataFrame
                model_data = model_data.sort_values(by=exp_column)
            line = ax.plot(model_data[exp_column], model_data[score_column] * m, linestyle = line_type, marker='o', label=model.split("/")[-1])
            line_color = line[0].get_color()  # Get the color of the line

            # Highlight highest score for the model
            max_score = model_data[score_column].max()
            max_score_data = model_data[model_data[score_column] == max_score]
            ax.scatter(max_score_data[exp_column], max_score_data[score_column] * m, color=line_color, s=60, edgecolor='black', zorder=5)
            if value_ticks == "highest_only":
                for _, row in max_score_data.iterrows():
                    ax.annotate(f"{row[score_column]*m:.2f}", (row[exp_column], row[score_column]*m), textcoords="offset points", xytext=(0,5), ha='center')
            elif value_ticks == "all":
                # annotate all scores
                for _, row in model_data.iterrows():
                    if i == 0:
                        add = 0
                    elif i == 1:
                        add = 2
                    elif i == 2:
                        add = -4
                    elif i == 3:
                        add = -13
                    elif i == 4:
                        add = -16
                    else: 
                        add = 0
                    # if row[exp_column] == "Naive":
                    #     ax.annotate(f"{row[score_column]*m:.2f}", (row[exp_column], row[score_column]*m), textcoords="offset points", xytext=(-15,5+add), ha='center')
                    # else:
                    #     ax.annotate(f"{row[score_column]*m:.2f}", (row[exp_column], row[score_column]*m), textcoords="offset points", xytext=(13,5+add), ha='center')

            if SOTA is not None and task == "all":
                if i == 0:
                    success_rate = SOTA["arc_1D"]["success_rate"] * m
                    ax.axhline(y=success_rate, color=colors[0], linestyle=':', label='SOTA: 1D-ARC')
                elif i == 1:
                    success_rate = SOTA["arc_h_v"]["success_rate"] * m
                    ax.axhline(y=success_rate, color=colors[1], linestyle=':', label='SOTA: ARC variant:\nobject orientation')
                elif i == 2:    
                    success_rate = SOTA["arc"]["success_rate"] * m
                    ax.axhline(y=success_rate, color=colors[2], linestyle=':', label='SOTA: ARC subset')
            
        # Add SOTA and other lines if applicable, using ax.axhline()
        if SOTA is not None and task in SOTA:
            if task == "all":
                success_rate = SOTA["arc_1D"]["success_rate"] * m
                ax.axhline(y=success_rate, color=colors[0], linestyle=':', label='SOTA: 1D-ARC')
                success_rate = SOTA["arc_h_v"]["success_rate"] * m
                ax.axhline(y=success_rate, color=colors[1], linestyle=':', label='SOTA: ARC variant:\nobject orientation')
                success_rate = SOTA["arc"]["success_rate"] * m
                ax.axhline(y=success_rate, color=colors[2], linestyle=':', label='SOTA: ARC subset')
            else:
                success_rate = SOTA[task]["success_rate"] * m
                ax.axhline(y=success_rate, color='r', linestyle=':', label='SOTA')

        # Adjust the x-axis limits slightly to provide padding
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin - 0.5, xmax + 0.5)

        # set y min value
        if "complexity" in score_column:
            ax.set_ylim(bottom=-2)
        else: 
            ax.set_ylim(bottom=-0.02)
            #ax.set_ylim(bottom=-0.02, top=1.02)

        # Add legend, labels, and title using ax methods
        ax.set_xlabel(exp_column)
        ax.set_ylabel(score_column)
        if task == "arc_1D":
            task_name = "1D-ARC -"
        elif task == "arc_h_v":
            task_name = "ARC Variant: Object Orientation -" 
        elif task == "arc":
            task_name = "ARC Subset -"
        else:
            task_name = ""
        
        if small_fig and task_name != "":
            task_name += "\n"
        else:
            task_name += " "
        
        if score_column == "avg_complexity":
            plt.ylabel('Average Complexity [# of Pixels in test input]')
            ax.set_title(task_name + 'Complexity of Solved Tasks')#  of Models Across Different Pixel Grid Representations')

        else:
            plt.ylabel('Success Rate')
            ax.set_title(task_name + 'Task Performance')#  of Models Across Different Pixel Grid Representations')

        if exp_column == "new_representation":
            plt.xlabel('Pixel Grid Representation')
            if run_column == "dataset":
                title = "Dataset"
            else:
                title = "Model"
            if top_n_models is not None:
                legend = plt.legend(title=title, loc='upper left', bbox_to_anchor=(1, 0.75))
            else:
                legend = plt.legend(title=title, loc='upper left', bbox_to_anchor=(1, 1.05), handlelength=4)
        elif exp_column == "run":
            plt.xlabel('Prompting Strategies')
            if task == "all":
                title = "Model and Dataset"
            else:
                title = "Model"
            if small_fig:
                legend = plt.legend(title=title, loc='upper center', bbox_to_anchor=(0.45, -0.15), ncol=1)
            else:
                legend = plt.legend(title=title, loc='upper left', bbox_to_anchor=(1, 1.05))

    # Check if the input is a list of dataframes
    if isinstance(dataframe, list):
        # Determine the number of rows needed for subplots based on the number of dataframes
        ncols = len(dataframe)
        fig, axs = plt.subplots(ncols=ncols, figsize=(10*ncols,   6))
        
        if ncols > 1:
            for i, df in enumerate(dataframe):
                # Set the current axis to the subplot axis
                ax = axs[i]
                # Generate a subplot title based on the dataframe index or any other criteria
                subplot_title = f"Experiment {i+1}"
                plot_single_dataframe(df, ax, run_column, exp_column, score_column, task, top_n_models, SOTA, value_ticks, small_fig)
    else:
        # If not a list, just plot the original dataframe
        if small_fig:
            fig, ax = plt.subplots(figsize=(4, 6))
        else:
            fig, ax = plt.subplots()
        plot_single_dataframe(dataframe, ax, run_column, exp_column, score_column, task, top_n_models, SOTA, value_ticks, small_fig)

    # Set the style for the plots
    plt.style.use('seaborn-darkgrid')  # This applies globally, so it's okay to keep as plt.
    sns.set_palette("viridis")  # This also applies globally.

    plt.tight_layout()
    plt.show()

def plot_grouped_bar_chart(dataframe, group_by="model", group_col="success_cat", top_n_models=3, asc=True, SOTA=None, highlight_pretrained_models=False):
    # filter on top n models
    if top_n_models is not None:
        # dataframe = dataframe.sort_values(by='success_rate', ascending=False)
        if "success_rate_w/o_str_comparison" in dataframe.columns:
            # dataframe = dataframe.sort_values(by="success_rate_w/o_str_comparison", ascending=False)
            dataframe = dataframe.sort_values(by="success_rate", ascending=False)
        elif group_by == "new_representation":
            pass
        else:
            dataframe = dataframe.sort_values(by='success_rate', ascending=False)
        dataframe = dataframe[:top_n_models]
    if (group_by == "new_representation") | (group_by == "run") | (group_by == "sampled_thoughts") | (group_by == "chosen_thoughts") | ((group_by == "model") and (group_col == "dataset")):
        pass
    else:
        if asc:
            dataframe = dataframe.sort_values(by='success_rate', ascending=True)
        else:
            dataframe = dataframe.sort_values(by='success_rate', ascending=False)
        
    if group_col ==  "success_cat":
        # Filter columns that start with 'cat_success_'
        success_columns = [col for col in dataframe.columns if col.startswith('cat_success_rate_')]
    elif group_col == "str_cmp":
        success_columns = ["success_rate", "success_rate_w/o_str_comparison"]
    elif group_col == "dataset":
        success_columns = [col for col in dataframe.columns if col.startswith('success_rate_')]
    elif group_col == "both_sampled_fraction_of_n_tasks":
        success_columns = [col for col in dataframe.columns if "both_sampled_fraction_of_n_tasks" in col]
    elif group_col == "steps":
        success_columns = [f'step_{i}' for i in range(1, 5)]
    elif group_col == "obj_cmp":
        success_columns = ["success_rate", "object_info_object_representation_success_rate"]
        
    # filter dataframe
    if "model" in dataframe.columns:
        dataframe = dataframe[['model'] + success_columns]
    elif group_by == "new_representation":
        dataframe = dataframe[['new_representation'] + success_columns]
    elif group_by == "run":
        dataframe = dataframe[['run'] + success_columns]
    elif group_by == "sampled_thoughts":
        dataframe = dataframe[['sampled_thoughts'] + success_columns]
    elif group_by == "chosen_thoughts":
        dataframe = dataframe[['chosen_thoughts'] + success_columns]
    else:
        dataframe = dataframe[success_columns]
    # delete model prefix and change to normal model name
    if "model" in dataframe.columns:
        dataframe["model"] = [model_names[m.split("/")[-1]] for m in dataframe["model"]]
        
    # Fort sampled_thoughts and chosen_thoughts save model name and adjust legend labels
    if group_col == "steps":
        if "gpt4" in dataframe[group_by].iloc[0]:
            model_name = "GPT-4 Turbo"
        elif "gpt3" in dataframe[group_by].iloc[0]:
            model_name = "GPT-3.5-Turbo"
        elif "mixtral" in dataframe[group_by].iloc[0]:
            model_name = "Mixtral Instruct"
        else: 
            model_name = "All Models"
        if group_by == "chosen_thoughts":
            dataframe.loc[dataframe['chosen_thoughts'].str.contains('incorrect_when'), 'chosen_thoughts'] = 'Selected incorrect thought\nalthough correct available'
            dataframe.loc[dataframe['chosen_thoughts'].str.contains('correct_when'), 'chosen_thoughts'] = 'Selected correct thought\nalthough incorrect available'
            #Only for chart being same size as sampled_thoughts charts:
            # dataframe.loc[dataframe['chosen_thoughts'].str.contains('incorrect_when'), 'chosen_thoughts'] = 'All incorrect'
            # dataframe.loc[dataframe['chosen_thoughts'].str.contains('correct_when'), 'chosen_thoughts'] = 'Correct and incorrect'
        elif group_by == "sampled_thoughts":
            dataframe.loc[dataframe['sampled_thoughts'].str.contains('_incorrect'), 'sampled_thoughts'] = 'All incorrect'
            dataframe.loc[dataframe['sampled_thoughts'].str.contains('_correct'), 'sampled_thoughts'] = 'All correct'
            dataframe.loc[dataframe['sampled_thoughts'].str.contains('_both'), 'sampled_thoughts'] = 'Correct and incorrect'
    
    # add SOTA
    if SOTA is not None:
        modified_keys_dict = {'cat_success_rate_' + key: value for key, value in SOTA["categories_rate"].items()}
        new_row = pd.DataFrame([modified_keys_dict])
        new_row['model'] = 'SOTA'
        dataframe = pd.concat([dataframe, new_row], ignore_index=True)
    
    if group_by == "model":
        models = dataframe[group_by].unique()
    elif group_by == ["success_rate", "success_rate_w/o_str_comparison"]:
        models = ["success_rate", "success_rate_w/o_str_comparison"]
        labels = ["Substring comparison", "JSON value comparison"]
    elif group_by == ["success_rate", "object_info_object_representation_success_rate"]:
        models = ["success_rate", "object_info_object_representation_success_rate"]
        labels = ["LLM", "Object detection tool"]
    elif (group_by == "new_representation") | (group_by == "run") | (group_by == "sampled_thoughts") | (group_by == "chosen_thoughts"):
        models = dataframe[group_by].unique()
        
    n_models = len(models)
    
    # Number of groups
    if (group_by == ["success_rate", "success_rate_w/o_str_comparison"]) or (group_by == ["success_rate", "object_info_object_representation_success_rate"]):
        n_groups = len(dataframe)
    else:
        n_groups = len(success_columns)

    # Set the style for the plots
    plt.style.use('seaborn-darkgrid')  
    sns.set_palette("viridis")    
    # Create figure and axis
    fig, ax = plt.subplots()
    # plt.axhline(y=1, color='r', linestyle=':', label='100%')
    # Set the positions and width for the bars
    index = np.arange(n_groups)
    bar_width = 1/(n_models+1)
    #spacing = bar_width * 0.1  # Adjust this value to increase or decrease space
    opacity = 0.8

    # Generate a bar for each model in each group
    for i, model in enumerate(models):
        # Select data for the model

        if (group_by == ["success_rate", "success_rate_w/o_str_comparison"]) or (group_by == ["success_rate", "object_info_object_representation_success_rate"]):
            model_data = dataframe[group_by[i]]
        else:
            model_data = dataframe[dataframe[group_by] == model][success_columns].iloc[0]
        # Position of the model's bar in each group
        if len(models) == 1:
            pos = [p for p in index]
        else:
            pos = [-bar_width*(n_models/2) - bar_width/2 + bar_width + p + bar_width * i for p in index]
        
        # Plotting
        if (group_col == "str_cmp") or (group_col == "obj_cmp"):
            plt.bar(pos, model_data, bar_width, alpha=opacity, label=labels[i])
        else:
            plt.bar(pos, model_data, bar_width, alpha=opacity, label=model)
    
    # Adding features to the plot
    if (group_col == "both_sampled_fraction_of_n_tasks") | ( group_col == "steps"):
        plt.ylabel('Fraction of tasks')
    else:
        plt.ylabel('Success Rate')

    if group_col == "success_cat":
        plt.xlabel('Categories')
        plt.title('Success rate by model and category')
        plt.xticks(index, [s.replace("cat_success_rate_", "") for s in success_columns], rotation=35, ha="right")
        legend = plt.legend(title="Model", loc='upper left', bbox_to_anchor=(1, 0.7))
        #legend = plt.legend(title="Model", loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=2)
        plt.ylim(top=1.02)
    if (group_col == "str_cmp"):
        plt.xlabel('Model')
        plt.title('Naive Prompting - Success Rate based on Task Checking Method')
        plt.xticks(index, [m for m in dataframe["model"]], rotation=35, ha="right")
        legend = plt.legend(title="Task checking method", loc='upper left', bbox_to_anchor=(1, 0.7))
    elif (group_col == "obj_cmp"):
        plt.xlabel('Model')
        plt.title('1D-ARC -\nSuccess Rate based on Output Grid Creation Method')
        plt.xticks(index, [m for m in dataframe["model"]], rotation=35, ha="right")
        legend = plt.legend(title="Based on the Transformed\nObjects, Output Grid Created by:", loc='upper left', bbox_to_anchor=(1, 0.7))
        # legend = plt.legend(title="Based on the Transformed Objects, Output Grid Created by:", loc='upper center', bbox_to_anchor=(0.5, -1.0), ncol=2)
    elif group_col == "dataset":
        plt.xlabel('Dataset')
        plt.title('Average Success Rate of all Models')
        # plt.xticks(index + bar_width / 2, [m.split("success_rate_")[-1] for m in success_columns], rotation=35, ha="right")
        plt.xticks(index, [m.split("success_rate_")[-1] for m in success_columns])
        legend = plt.legend(title="Pixel representation", loc='upper left', bbox_to_anchor=(1, 0.7))        
    elif (group_col == "both_sampled_fraction_of_n_tasks") | ( group_col == "steps"):
        plt.xlabel('Step in the reasoning chain')
        if group_by == "chosen_thoughts":
            plt.ylim(top=1.02)
            plt.title(f'{model_name}\nSelection of Correct Thoughts Over Incorrect')
            legend = plt.legend(title="Thought selection", loc='upper left', bbox_to_anchor=(1, 0.7))
            #legend = plt.legend(title="Thought selection", loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
        elif group_by == "sampled_thoughts":
            plt.title(f'{model_name}\nDistribution of Validity of Sampled Thoughts')
            #plt.title(f'{model_name}\nUnsolved Tasks\nDistribution of Validity of Sampled Thoughts')
            legend = plt.legend(title="Sampled thoughts were:", loc='upper left', bbox_to_anchor=(1, 0.7))
            #legend = plt.legend(title="Sampled thoughts were:", loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3)
            plt.ylim(top=1)
        else:
            plt.title('Correctly and Incorrectly Sampled Thoughts in Same Step') 
            legend = plt.legend(title="Run", loc='upper left', bbox_to_anchor=(1, 0.7))
        labels = ["Description", "Pattern", "Instructions", "Transformation"]
        plt.xticks(index, [f"Step {i+1}\n{l}" for i, l in enumerate(labels)], fontsize=9)#, rotation=35, ha="right")
 
    if highlight_pretrained_models:
        # Highlighting specific x-axis ticks
        tick_labels = ax.get_xticklabels()  # Get the current tick labels on the x-axis
        for label in tick_labels:
            text = label.get_text()  # Get the text of the tick label
            # Check if 'chat' or 'instruct' is in the tick label text
            if 'chat' not in text.lower() and 'instruct' not in text.lower() and "gpt-" not in text.lower() and "platypus" not in text.lower():
                # label.set_color('red')  # Change the color to red
                label.set_fontweight('bold')  # Make the label bold
            
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()

##############################################
# Task Analysis
##############################################

def analyze_random_tasks(data, task, exp, n_models, n_tasks, only_unsolved=False, only_solved=False, starting_index=0, only_save_as_txt=False):
    # random seed
    np.random.seed(42)
    # get all tasks
    data_dir = os.path.join("tree-of-thought-llm/src/tot/data", task)
    _, tasks_names, _ = load_arc_tasks(data_dir, task)
    # get experiment data of top n models
    top_n_runs = data.sort_values(by='success_rate_w/o_str_comparison', ascending=False)[:n_models]
    # iterate over the top n runs
    selected_tasks = {}
    for index, row in top_n_runs.iterrows():
        n_tasks_m = n_tasks
        #print(row["solved_tasks"])
        solved_tasks = ast.literal_eval(row["solved_tasks"])
        solved_tasks = [t[0] for t in solved_tasks]
        unsolved_tasks = [t for t in tasks_names if t not in solved_tasks]
        if only_unsolved:
            solved_or_unsolved = "unsolved"
            tasks = unsolved_tasks
        elif only_solved:
            solved_or_unsolved = "solved"
            tasks = solved_tasks
        else:
            solved_or_unsolved = "all"
            tasks = tasks_names
            
        np.random.shuffle(tasks)
        if n_tasks_m > len(tasks):
            n_tasks_m = len(tasks)
        if exp == "baseline_cot":
            run_postfix = "naive_cot_"
        if exp == "tot_objects":
            run_postfix = "object-representation_"
        else:
            run_postfix = ""
        selected_tasks[f'{row["model"].split("/")[-1]}_{run_postfix}{row["date"]}'] = np.random.choice(tasks, n_tasks_m, replace=False)

    txt = ""
    for run, tasks in selected_tasks.items():
        results_path = f"results/{exp}/{task}/{run}/tasks/"
        
        run_txt = '''
############################################################################################################
New Run: {}
############################################################################################################
'''.format(run)
        print(run_txt)
        if only_save_as_txt:
            txt += run_txt+"\n"
        user_input = input("Do you want to continue? (y/n) Or skip to next model? (skip)")
        if user_input == "skip":
            continue
        elif user_input != "y":
            print("Aborted")
            break
        
        for i, t in enumerate(tasks):
            if i < starting_index:
                continue
            task_path = os.path.join(results_path, t[:-5]+"_LLM_answer.txt")
            # check if file exist
            if not os.path.isfile(task_path):
                print(f"Task {t} not found in {run}: {task_path}")
                continue
            with open(task_path) as f:
                content = f.read()
            if only_save_as_txt:
                txt += f'''\n############################################################################################################\nTask: {t}\n\n{content}\n\n'''
            else:
                print(t)
                print(content)
                user_input = input("Do you want to continue? (y/n) Or show again? (again)")
                while user_input == "again":
                    clear_output() 
                    print(t)
                    print(content)
                    user_input = input("Do you want to continue? (y/n) Or show again? (again)")
                if user_input != "y":
                    clear_output() 
                    print("Aborted")
                    break
                clear_output()
    if only_save_as_txt:
        with open(f"results/{exp}/{task}/manual_task_analysis_raw_outputs_{solved_or_unsolved}.txt", "w") as text_file:
            text_file.write(txt)
        print(f"saved to results/{exp}/{task}/manual_task_analysis_raw_outputs_{solved_or_unsolved}.txt")

def get_sankey_source_data(df, only_best_node=False, ATOT=False):
    n_tasks = len(df)
    
    # if tot run, rename columns to use code of baseline_cot run
    if "step_1_incorrect_samples_exist" in df.columns:
        if only_best_node:
            df['description_of_difference_correct_and_complete'] = df["best_node_step_1_correct"]
            df['pattern_correct'] = df["best_node_step_2_correct"]
            df['conditions_correct'] = df["best_node_step_2_correct"]
            df['instructions_correct'] = df["best_node_step_3_correct"]
            df['test_transformation_correct'] = df["best_node_step_4_correct"]
        else:
            df['description_of_difference_correct_and_complete'] = df["step_1_chose_correct"]
            df['pattern_correct'] = df["step_2_chose_correct"]
            df['conditions_correct'] = df["step_2_chose_correct"]
            df['instructions_correct'] = df["step_3_chose_correct"]
            df['test_transformation_correct'] = df["step_4_chose_correct"]
        
    
    sankey_edges = pd.DataFrame(columns=['source', 'target', 'value'])
    sankey_edges['source'] = ["tasks\n"]*2+["description correct\n"]*2+["description wrong\n"]*2+["pattern correct\n"]*2+["pattern wrong\n"]*2+["instructions correct\n"]*3+["instructions wrong\n"]*3
    sankey_edges['target'] = ["description correct\n", "description wrong\n"] + ["pattern correct\n", "pattern wrong\n"]*2 + ["instructions correct\n", "instructions wrong\n"]*2 + ["output correct\n", "output wrong\n","output error\n"]*2 

    sankey_edges["value"].iloc[0] = df['description_of_difference_correct_and_complete'].sum() #/ n_tasks
    sankey_edges["value"].iloc[1] = df[df['description_of_difference_correct_and_complete'] == 0].shape[0] #/ n_tasks

    # remove Step 1 in AToT approach
    if ATOT:
        sankey_edges["target"].iloc[0] = "pattern correct\n"
        sankey_edges["value"].iloc[0] = df["step_2_chose_correct"].sum()
        sankey_edges["target"].iloc[1] = "pattern wrong\n"
        sankey_edges["value"].iloc[1] = df[df['step_2_chose_correct'] == 0].shape[0]
    # source description
    sankey_edges["value"].iloc[2] = df[
        (df['description_of_difference_correct_and_complete'] == 1) 
        & (df['pattern_correct'] == 1)
        & (df['conditions_correct'] == 1)].shape[0] #/ n_tasks
    sankey_edges["value"].iloc[3] = df[
        (df['description_of_difference_correct_and_complete'] == 1) 
        & ((df['pattern_correct'] == 0) | (df['conditions_correct'] == 0))].shape[0] #/ n_tasks
    sankey_edges["value"].iloc[4] = df[
        (df['description_of_difference_correct_and_complete'] == 0) 
        & (df['pattern_correct'] == 1)
        & (df['conditions_correct'] == 1)].shape[0] #/ n_tasks
    sankey_edges["value"].iloc[5] = df[
        (df['description_of_difference_correct_and_complete'] == 0) 
        & ((df['pattern_correct'] == 0) | (df['conditions_correct'] == 0))].shape[0] #/ n_tasks

    # source pattern 
    sankey_edges["value"].iloc[6] = df[
        (df['pattern_correct'] == 1)
        & (df['conditions_correct'] == 1)
        & (df['instructions_correct'] == 1)].shape[0] #/ n_tasks
    sankey_edges["value"].iloc[7] = df[
        (df['pattern_correct'] == 1)
        & (df['conditions_correct'] == 1)
        & (df['instructions_correct'] == 0)].shape[0] #/ n_tasks
    sankey_edges["value"].iloc[8] = df[
        ((df['pattern_correct'] == 0) | (df['conditions_correct'] == 0))
        & (df['instructions_correct'] == 1)].shape[0] #/ n_tasks
    sankey_edges["value"].iloc[9] = df[
        ((df['pattern_correct'] == 0) | (df['conditions_correct'] == 0))
        & (df['instructions_correct'] == 0)].shape[0] #/ n_tasks

    # source instructions 
    sankey_edges["value"].iloc[10] = df[
        (df['instructions_correct'] == 1)
        & (df['test_transformation_correct'] == 1)].shape[0] #/ n_tasks
    sankey_edges["value"].iloc[11] = df[
        (df['instructions_correct'] == 1)
        & (df['test_transformation_correct'] == 0)].shape[0] #/ n_tasks
    sankey_edges["value"].iloc[12] = df[
        (df['instructions_correct'] == 1)
        & (df['test_transformation_correct'].isna())].shape[0] #/ n_tasks
    sankey_edges["value"].iloc[13] = df[
        (df['instructions_correct'] == 0)
        & (df['test_transformation_correct'] == 1)].shape[0] #/ n_tasks
    sankey_edges["value"].iloc[14] = df[
        (df['instructions_correct'] == 0)
        & (df['test_transformation_correct'] == 0)].shape[0] #/ n_tasks
    sankey_edges["value"].iloc[15] = df[
        (df['instructions_correct'] == 0)
        & (df['test_transformation_correct'].isna())].shape[0] #/ n_tasks
    
    if not ATOT and df[df['description_of_difference_correct_and_complete'].isna()].shape[0] > 0:
        new_data = {'source': 'tasks\n', 'target': 'description error\n', 'value': df[df['description_of_difference_correct_and_complete'].isna()].shape[0]}
        sankey_edges.loc[len(sankey_edges)] = new_data
        new_data = {'source': 'description error\n', 'target': 'pattern correct\n', 'value': df[(df['description_of_difference_correct_and_complete'].isna()) & (df['pattern_correct'] == 1) & (df['conditions_correct'] == 1)].shape[0]}
        sankey_edges.loc[len(sankey_edges)] = new_data
        new_data = {'source': 'description error\n', 'target': 'pattern wrong\n', 'value': df[(df['description_of_difference_correct_and_complete'].isna()) & ((df['pattern_correct'] == 0) | (df['conditions_correct'] == 0))].shape[0]}
        sankey_edges.loc[len(sankey_edges)] = new_data

    #sankey_edges["value"] = sankey_edges["value"]*100
    sankey_edges = sankey_edges[sankey_edges["value"] > 0]
    return sankey_edges    
 
cmap_list = process_cmap("viridis")
cmap = {
    'tasks\n': cmap_list[10],
    'description wrong\n': cmap_list[40],
    'description correct\n': cmap_list[70],
    'pattern wrong\n': cmap_list[100],
    'pattern correct\n': cmap_list[130],
    'instructions wrong\n': cmap_list[160],
    'instructions correct\n': cmap_list[190],
    'output wrong\n': cmap_list[210],
    'output correct\n': cmap_list[240],
}
    
def get_task_analysis_kpi(path):
    def calc_metrics(p, df):
        n_tasks = len(df)
        if n_tasks == 0:
            return {
                "step_1_both_sampled_fraction_of_n_tasks": 0,
                "step_2_both_sampled_fraction_of_n_tasks": 0,
                "step_3_both_sampled_fraction_of_n_tasks": 0,
                "step_4_both_sampled_fraction_of_n_tasks": 0,
                "step_1_only_correct_sampled_fraction_of_n_tasks": 0,
                "step_2_only_correct_sampled_fraction_of_n_tasks": 0,
                "step_3_only_correct_sampled_fraction_of_n_tasks": 0,
                "step_4_only_correct_sampled_fraction_of_n_tasks": 0,
                "step_1_only_incorrect_sampled_fraction_of_n_tasks": 0,
                "step_2_only_incorrect_sampled_fraction_of_n_tasks": 0,
                "step_3_only_incorrect_sampled_fraction_of_n_tasks": 0,
                "step_4_only_incorrect_sampled_fraction_of_n_tasks": 0,
                "all_sampled_descriptions_incorrect_fraction_of_n_tasks": 0,
                "all_sampled_descriptions_correct_fraction_of_n_tasks": 0,
                "at_least_one_selected_description_correct_fraction_of_n_tasks": 0,
                "at_least_one_selected_description_incorrect_fraction_of_n_tasks": 0,
                "all_selected_description_correct_fraction_of_n_tasks": 0,
                "all_selected_description_incorrect_fraction_of_n_tasks": 0,
                "one_correct_and_one_incorrect_description_fraction_of_n_tasks": 0,
                "description_error_bc_bad_object_detection_fraction_of_incorrect_descriptions": 0,
                "test_case_description_correct_fraction_of_n_tasks": 0,
                "test_case_description_error_bc_bad_object_detection_fraction_of_incorrect_descriptions": 0,
                "correct_chain_at_least_one_correct_thought_per_step_fraction_of_n_tasks": 0,
                "at_least_one_correct_pattern_fraction_of_n_tasks": 0,
                "at_least_one_correct_instructions_fraction_of_n_tasks": 0,
                "at_least_one_correct_transformation_fraction_of_n_tasks": 0,
                "correct_chain_of_best_node_fraction_of_n_tasks": 0,
                "correct_description_of_best_node_fraction_of_n_tasks": 0,
                "correct_pattern_of_best_node_fraction_of_n_tasks": 0,
                "correct_instructions_of_best_node_fraction_of_n_tasks": 0,
                "correct_transformation_of_best_node_fraction_of_n_tasks": 0,
                "instructions_fit_wrong_pattern_fraction_of_wrong_pattern": 0,
                "transformations_fit_wrong_pattern_fraction_of_wrong_pattern": 0,
                "key_concept_correct_fraction_of_n_tasks": 0,
                "wrong_pattern_but_correct_key_concept_fraction_of_wrong_pattern": 0,
                "arithmetic_interpretation_fraction_of_n_tasks": 0,
                "incorrect_description_chosen_although_correct_available_fraction_of_tasks_w_correct_and_incorrect": 0,
                "correct_description_chosen_although_incorrect_available_fraction_of_tasks_w_correct_and_incorrect": 0,
                "incorrect_pattern_chosen_although_correct_available_fraction_of_tasks_w_correct_and_incorrect": 0,
                "correct_pattern_chosen_although_incorrect_available_fraction_of_tasks_w_correct_and_incorrect": 0,
                "incorrect_instructions_chosen_although_correct_available_fraction_of_tasks_w_correct_and_incorrect": 0,
                "correct_instructions_chosen_although_incorrect_available_fraction_of_tasks_w_correct_and_incorrect": 0,
                "incorrect_transformation_chosen_although_correct_available_fraction_of_tasks_w_correct_and_incorrect": 0,
                "correct_transformation_chosen_although_incorrect_available_fraction_of_tasks_w_correct_and_incorrect": 0,
                "correct_thought_chosen_although_incorrect_available_fraction_of_tasks_w_correct_and_incorrect": 0,
                "incorrect_thought_chosen_although_correct_available_fraction_of_tasks_w_correct_and_incorrect": 0,
            }
        if "baseline_cot" in p:
            # Calculate fraction of tasks where the description of differences between input and output is correct and complete
            description_correct_and_complete_fraction_of_n_tasks = df['description_of_difference_correct_and_complete'].sum() / n_tasks

            # Calculate fraction of tasks where the description of differences between input and output is correct but incomplete
            description_correct_but_missing_relevant_details_fraction_of_n_tasks = df['description_correct_but_missing_relevant_details'].sum() / n_tasks

            # Calculate fraction of tasks where the description of differences between input and output is completely wrong
            description_wrong_fraction_of_n_tasks = 1 - ((df['description_of_difference_correct_and_complete'].sum()+df['description_correct_but_missing_relevant_details'].sum()) / n_tasks)
        
            # calculate fraction of examples where the description error is due to bad object detection
            description_error_bc_bad_object_detection_fraction_of_incorrect_descriptions = df['description_error_bc_bad_object_detection'].sum() / (df['description_of_difference_correct_and_complete'] == 0).sum()

            if "baseline_cot/arc_1D" in p:
                # Calculate fraction of correct test case descriptions
                test_case_description_correct_fraction_of_n_tasks = df['test_description_correct'].sum() / n_tasks
                
                # calculate fraction of examples where the test case description error is due to bad object detection
                test_case_description_error_bc_bad_object_detection_fraction_of_all_where_objects_described_in_detail = (df['test_objects_correct'] == 0).sum() / df['test_object_described_w_all_details'].sum()
                
            
            elif "baseline_cot/" in p:
                # Calculate fraction of correct test case descriptions
                test_case_description_correct_fraction_of_n_tasks = df['test_description_correct_and_complete'].sum() / n_tasks
                
                # calculate fraction of examples where the test case description error is due to bad object detection
                test_case_description_error_bc_bad_object_detection_fraction_of_all_where_objects_described_in_detail = df['test_description_error_bc_bad_object_detection'].sum() / (df['test_description_correct_and_complete'] == 0).sum()
                          
            # Calculate fraction of correct chain
            correct_chain_fraction_of_n_tasks = df[(df['description_of_difference_correct_and_complete'] == 1) & 
                                        (df['pattern_correct'] == 1) & 
                                        (df['conditions_correct'] == 1) & 
                                        (df['instructions_correct'] == 1) & 
                                        (df['test_transformation_correct'] == 1)].shape[0] / n_tasks
                                       
            # Calculate fraction of correct pattern
            correct_pattern_and_condition_fraction_of_n_tasks = df[(df['pattern_correct'] == 1) & 
                                        (df['conditions_correct'] == 1)].shape[0] / n_tasks
            
            # Calculate fraction of correct instructions
            correct_instructions_fraction_of_n_tasks = df['instructions_correct'].sum() / n_tasks
            
            # Calculate fraction of correct transformations
            correct_transformations_fraction_of_n_tasks = df['test_transformation_correct'].sum() / n_tasks
            
            # Calculate fraction of instructions that are correct with respect to wrong pattern
            instructions_fit_wrong_pattern_fraction_of_wrong_pattern = df[(df['pattern_correct'] == 0) & 
                                                        (df['wrong_instructions_but_fit_to_pattern'] == 1)].shape[0] / (df['pattern_correct'] == 0).sum()
            
            # calculate fraction of transformations that are correct with respect to wrong pattern
            transformations_fit_wrong_pattern_fraction_of_wrong_pattern = df[(df['pattern_correct'] == 0) & 
                                                        (df['wrong_test_transformation_but_fit_to_pattern'] == 1)].shape[0] / (df['pattern_correct'] == 0).sum()
            
            # calculate fraction of correct key concept of n tasks
            key_concept_correct_fraction_of_n_tasks = df['key_concept_correct'].sum() / n_tasks
            
            # calculate fraction of wrong pattern but identified key concept correctly
            wrong_pattern_but_correct_key_concept_fraction_of_wrong_pattern = df[(df['pattern_correct'] == 0) & 
                                                        (df['key_concept_correct'] == 1)].shape[0] / (df['pattern_correct'] == 0).sum()
            
            # calculate fraction of tasks where numbers were interpreted arithmetically
            arithmetic_interpretation_fraction_of_n_tasks = df['numbers_interpreted_arithmetical'].sum() / n_tasks
        elif ("tot_normal" in p) or ("tot_objects" in p):
            # calculate fractions per step, where both correct and incorrect thoughts were sampled
            for i in range(1, 5):
                incorrect_exist = f'step_{i}_incorrect_samples_exist'
                correct_exist = f'step_{i}_correct_samples_exist'
                df[f'step_{i}_both_exist'] = df[incorrect_exist] & df[correct_exist]
                df[f'step_{i}_only_correct'] = df[correct_exist] & ~df[incorrect_exist]
                df[f'step_{i}_only_incorrect'] = df[incorrect_exist] & ~df[correct_exist]
            step_1_both_sampled_fraction_of_n_tasks = df['step_1_both_exist'].sum() / n_tasks
            step_2_both_sampled_fraction_of_n_tasks = df['step_2_both_exist'].sum() / n_tasks
            step_3_both_sampled_fraction_of_n_tasks = df['step_3_both_exist'].sum() / n_tasks
            step_4_both_sampled_fraction_of_n_tasks = df['step_4_both_exist'].sum() / n_tasks
            step_1_only_correct_sampled_fraction_of_n_tasks = df['step_1_only_correct'].sum() / n_tasks
            step_2_only_correct_sampled_fraction_of_n_tasks = df['step_2_only_correct'].sum() / n_tasks
            step_3_only_correct_sampled_fraction_of_n_tasks = df['step_3_only_correct'].sum() / n_tasks
            step_4_only_correct_sampled_fraction_of_n_tasks = df['step_4_only_correct'].sum() / n_tasks
            step_1_only_incorrect_sampled_fraction_of_n_tasks = df['step_1_only_incorrect'].sum() / n_tasks
            step_2_only_incorrect_sampled_fraction_of_n_tasks = df['step_2_only_incorrect'].sum() / n_tasks
            step_3_only_incorrect_sampled_fraction_of_n_tasks = df['step_3_only_incorrect'].sum() / n_tasks
            step_4_only_incorrect_sampled_fraction_of_n_tasks = df['step_4_only_incorrect'].sum() / n_tasks
            
            
            # calculate fractions of tasks regarding proportions of selected descriptions beeing correct/incorrect
            all_sampled_descriptions_incorrect_fraction_of_n_tasks = df[(df['step_1_incorrect_samples_exist'] == 1) & (df['step_1_correct_samples_exist'] == 0)].shape[0] / n_tasks
            all_sampled_descriptions_correct_fraction_of_n_tasks = df[(df['step_1_incorrect_samples_exist'] == 0) & (df['step_1_correct_samples_exist'] == 1)].shape[0] / n_tasks
            at_least_one_selected_description_correct_fraction_of_n_tasks = df['step_1_chose_correct'].sum() / n_tasks
            at_least_one_selected_description_incorrect_fraction_of_n_tasks = df['step_1_chose_incorrect'].sum() / n_tasks
            all_selected_description_correct_fraction_of_n_tasks = df[(df['step_1_chose_correct'] == 1) & (df['step_1_chose_incorrect'] == 0)].shape[0] / n_tasks
            all_selected_description_incorrect_fraction_of_n_tasks = df[(df['step_1_chose_correct'] == 0) & (df['step_1_chose_incorrect'] == 1)].shape[0] / n_tasks
            one_correct_and_one_incorrect_description_fraction_of_n_tasks = df[(df['step_1_chose_correct'] == 1) & (df['step_1_chose_incorrect'] == 1)].shape[0] / n_tasks
            
            # calculate fraction of examples where the description error is due to bad object detection
            description_error_bc_bad_object_detection_fraction_of_incorrect_descriptions = df['step_1_chose_incorrect_and_error_bc_bad_object_detection'].sum() / (df['step_1_chose_incorrect'] == 1).sum()

            # Calculate fraction of correct test case descriptions
            test_case_description_correct_fraction_of_n_tasks = df['best_node_test_description_correct'].sum() / n_tasks
            
            # calculate fraction of examples where the test case description error is due to bad object detection
            test_case_description_error_bc_bad_object_detection_fraction_of_incorrect_descriptions = (df['best_node_test_description_error_bc_bad_object_detection'] == 1).sum() / (df['best_node_test_description_correct'] == 0).sum()
            
            # Calculate fraction of correct chain
            correct_chain_at_least_one_correct_thought_per_step_fraction_of_n_tasks = df[(df['step_1_chose_correct'] == 1) & 
                                        (df['step_2_chose_correct'] == 1) & 
                                        (df['step_3_chose_correct'] == 1) & 
                                        (df['step_4_chose_correct'] == 1)].shape[0] / n_tasks
            # Calculate fraction of correct pattern
            at_least_one_correct_pattern_fraction_of_n_tasks = df['step_2_chose_correct'].sum() / n_tasks
            # Calculate fraction of correct instructions
            at_least_one_correct_instructions_fraction_of_n_tasks = df['step_3_chose_correct'].sum() / n_tasks
            # Calculate fraction of correct transformations
            at_least_one_correct_transformation_fraction_of_n_tasks = df['step_4_chose_correct'].sum() / n_tasks

            # Calculate fraction of correct chain - only best node
            correct_chain_of_best_node_fraction_of_n_tasks = df[(df['best_node_step_1_correct'] == 1) & 
                                        (df['best_node_step_2_correct'] == 1) & 
                                        (df['best_node_step_3_correct'] == 1) & 
                                        (df['best_node_step_4_correct'] == 1)].shape[0] / n_tasks
            # Calculate fraction of correct pattern - only best node
            correct_description_of_best_node_fraction_of_n_tasks = df['best_node_step_1_correct'].sum() / n_tasks
            # Calculate fraction of correct pattern - only best node
            correct_pattern_of_best_node_fraction_of_n_tasks = df['best_node_step_2_correct'].sum() / n_tasks
            # Calculate fraction of correct instructions - only best node
            correct_instructions_of_best_node_fraction_of_n_tasks = df['best_node_step_3_correct'].sum() / n_tasks
            # Calculate fraction of correct transformations - only best node
            correct_transformation_of_best_node_fraction_of_n_tasks = df['best_node_step_4_correct'].sum() / n_tasks
            
            # Calculate fraction of instructions that are correct with respect to wrong pattern
            instructions_fit_wrong_pattern_fraction_of_wrong_pattern = df[(df['step_2_chose_correct'] == 0) & 
                                                        (df['wrong_instructions_but_fit_to_pattern'] == 1)].shape[0] / (df['step_2_chose_correct'] == 0).sum()
            
            # calculate fraction of transformations that are correct with respect to wrong pattern
            transformations_fit_wrong_pattern_fraction_of_wrong_pattern = df[(df['step_2_chose_correct'] == 0) & 
                                                        (df['best_node_wrong_test_transformation_but_fit_to_pattern'] == 1)].shape[0] / (df['step_2_chose_correct'] == 0).sum()
            
            # calculate fraction of correct key concept of n tasks
            key_concept_correct_fraction_of_n_tasks = df['key_concept_correct'].sum() / n_tasks
            
            # calculate fraction of wrong pattern but identified key concept correctly
            wrong_pattern_but_correct_key_concept_fraction_of_wrong_pattern = df[(df['step_2_chose_correct'] == 0) & 
                                                        (df['key_concept_correct'] == 1)].shape[0] / (df['step_2_chose_correct'] == 0).sum()
            
            # calculate fraction of tasks where numbers were interpreted arithmetically
            arithmetic_interpretation_fraction_of_n_tasks = df['numbers_interpreted_arithmetical'].sum() / n_tasks
            
            # calculate fraction of incorrect descriptions chosen although correct description was available
            if df[(df['step_1_chose_incorrect_although_correct_exists'] == 0) | (df['step_1_chose_incorrect_although_correct_exists'] == 1)].shape[0] != 0:
                incorrect_description_chosen_although_correct_available_fraction_of_tasks_w_correct_and_incorrect = df['step_1_chose_incorrect_although_correct_exists'].sum() / df[(df['step_1_chose_incorrect_although_correct_exists'] == 0) | 
                                                            (df['step_1_chose_incorrect_although_correct_exists'] == 1)].shape[0]
            else:
                incorrect_description_chosen_although_correct_available_fraction_of_tasks_w_correct_and_incorrect = 0
            
            # calculate fraction of correct descriptions chosen although incorrect description existed
            if df[(df['step_1_chose_correct_although_incorrect_exists'] == 0) | (df['step_1_chose_correct_although_incorrect_exists'] == 1)].shape[0] != 0:
                correct_description_chosen_although_incorrect_available_fraction_of_tasks_w_correct_and_incorrect = df['step_1_chose_correct_although_incorrect_exists'].sum() / df[(df['step_1_chose_correct_although_incorrect_exists'] == 0) | 
                                                        (df['step_1_chose_correct_although_incorrect_exists'] == 1)].shape[0]
            else:
                correct_description_chosen_although_incorrect_available_fraction_of_tasks_w_correct_and_incorrect = 0
                
            # calculate fraction of incorrect pattern chosen although correct description was available
            if df[(df['step_2_chose_incorrect_although_correct_exists'] == 0) | (df['step_2_chose_incorrect_although_correct_exists'] == 1)].shape[0] != 0:
                incorrect_pattern_chosen_although_correct_available_fraction_of_tasks_w_correct_and_incorrect = df['step_2_chose_incorrect_although_correct_exists'].sum() / df[(df['step_2_chose_incorrect_although_correct_exists'] == 0) |
                                                        (df['step_2_chose_incorrect_although_correct_exists'] == 1)].shape[0]
            else:
                incorrect_pattern_chosen_although_correct_available_fraction_of_tasks_w_correct_and_incorrect = 0
                
            # calculate fraction of correct pattern chosen although incorrect description existed
            if df[(df['step_2_chose_correct_although_incorrect_exists'] == 0) | (df['step_2_chose_correct_although_incorrect_exists'] == 1)].shape[0] != 0:
                correct_pattern_chosen_although_incorrect_available_fraction_of_tasks_w_correct_and_incorrect = df['step_2_chose_correct_although_incorrect_exists'].sum() / df[(df['step_2_chose_correct_although_incorrect_exists'] == 0) | 
                                                        (df['step_2_chose_correct_although_incorrect_exists'] == 1)].shape[0]
            else:
                correct_pattern_chosen_although_incorrect_available_fraction_of_tasks_w_correct_and_incorrect = 0
                
            # calculate fraction of incorrect instructions chosen although correct description was available
            if df[(df['step_3_chose_incorrect_although_correct_exists'] == 0) |
                                                        (df['step_3_chose_incorrect_although_correct_exists'] == 1)].shape[0] != 0: 
                incorrect_instructions_chosen_although_correct_available_fraction_of_tasks_w_correct_and_incorrect = df['step_3_chose_incorrect_although_correct_exists'].sum() / df[(df['step_3_chose_incorrect_although_correct_exists'] == 0) |
                                                        (df['step_3_chose_incorrect_although_correct_exists'] == 1)].shape[0]
            else:
                incorrect_instructions_chosen_although_correct_available_fraction_of_tasks_w_correct_and_incorrect = 0
            
            # calculate fraction of correct descriptions chosen although incorrect description existed
            if df[(df['step_3_chose_correct_although_incorrect_exists'] == 0) | (df['step_3_chose_correct_although_incorrect_exists'] == 1)].shape[0] != 0:
                correct_instructions_chosen_although_incorrect_available_fraction_of_tasks_w_correct_and_incorrect = df['step_3_chose_correct_although_incorrect_exists'].sum() / df[(df['step_3_chose_correct_although_incorrect_exists'] == 0) | 
                                                        (df['step_3_chose_correct_although_incorrect_exists'] == 1)].shape[0]
            else:
                correct_instructions_chosen_although_incorrect_available_fraction_of_tasks_w_correct_and_incorrect = 0
                
            # calculate fraction of incorrect transformation chosen although correct description was available
            if df[(df['step_4_chose_incorrect_although_correct_exists'] == 0) | (df['step_4_chose_incorrect_although_correct_exists'] == 1)].shape[0] != 0:
                incorrect_transformation_chosen_although_correct_available_fraction_of_tasks_w_correct_and_incorrect = df['step_4_chose_incorrect_although_correct_exists'].sum() / df[(df['step_4_chose_incorrect_although_correct_exists'] == 0) | 
                                                        (df['step_4_chose_incorrect_although_correct_exists'] == 1)].shape[0]
            else:
                incorrect_transformation_chosen_although_correct_available_fraction_of_tasks_w_correct_and_incorrect = 0
                
            # calculate fraction of correct transformation chosen although incorrect description existed
            if df[(df['step_4_chose_correct_although_incorrect_exists'] == 0) | (df['step_4_chose_correct_although_incorrect_exists'] == 1)].shape[0] != 0:
                correct_transformation_chosen_although_incorrect_available_fraction_of_tasks_w_correct_and_incorrect = df['step_4_chose_correct_although_incorrect_exists'].sum() / df[(df['step_4_chose_correct_although_incorrect_exists'] == 0) | 
                                                        (df['step_4_chose_correct_although_incorrect_exists'] == 1)].shape[0]
            else: 
                correct_transformation_chosen_although_incorrect_available_fraction_of_tasks_w_correct_and_incorrect = 0
                
            # calculate fraction of correct thought chosen although incorrect thought was available, across all steps
            if (df[(df['step_1_chose_correct_although_incorrect_exists'] == 0) | (df['step_1_chose_correct_although_incorrect_exists'] == 1)].shape[0]
                     + df[(df['step_2_chose_correct_although_incorrect_exists'] == 0) | (df['step_2_chose_correct_although_incorrect_exists'] == 1)].shape[0]
                     + df[(df['step_3_chose_correct_although_incorrect_exists'] == 0) | (df['step_3_chose_correct_although_incorrect_exists'] == 1)].shape[0]
                     + df[(df['step_4_chose_correct_although_incorrect_exists'] == 0) | (df['step_4_chose_correct_although_incorrect_exists'] == 1)].shape[0]) != 0:
                correct_thought_chosen_although_incorrect_available_fraction_of_tasks_w_correct_and_incorrect = (df['step_1_chose_correct_although_incorrect_exists'].sum()
                    + df['step_2_chose_correct_although_incorrect_exists'].sum()
                    + df['step_3_chose_correct_although_incorrect_exists'].sum()
                    + df['step_4_chose_correct_although_incorrect_exists'].sum() 
                    ) / (df[(df['step_1_chose_correct_although_incorrect_exists'] == 0) | (df['step_1_chose_correct_although_incorrect_exists'] == 1)].shape[0]
                        + df[(df['step_2_chose_correct_although_incorrect_exists'] == 0) | (df['step_2_chose_correct_although_incorrect_exists'] == 1)].shape[0]
                        + df[(df['step_3_chose_correct_although_incorrect_exists'] == 0) | (df['step_3_chose_correct_although_incorrect_exists'] == 1)].shape[0]
                        + df[(df['step_4_chose_correct_although_incorrect_exists'] == 0) | (df['step_4_chose_correct_although_incorrect_exists'] == 1)].shape[0])
            else:
                correct_thought_chosen_although_incorrect_available_fraction_of_tasks_w_correct_and_incorrect = 0
            
            # calculate fraction of incorrect thought chosen although correct thought was available, across all steps
            if (df[(df['step_1_chose_incorrect_although_correct_exists'] == 0) | (df['step_1_chose_incorrect_although_correct_exists'] == 1)].shape[0]
                     + df[(df['step_2_chose_incorrect_although_correct_exists'] == 0) | (df['step_2_chose_incorrect_although_correct_exists'] == 1)].shape[0]
                     + df[(df['step_3_chose_incorrect_although_correct_exists'] == 0) | (df['step_3_chose_incorrect_although_correct_exists'] == 1)].shape[0]
                     + df[(df['step_4_chose_incorrect_although_correct_exists'] == 0) | (df['step_4_chose_incorrect_although_correct_exists'] == 1)].shape[0]) != 0:
                incorrect_thought_chosen_although_correct_available_fraction_of_tasks_w_correct_and_incorrect = (df['step_1_chose_incorrect_although_correct_exists'].sum()
                    + df['step_2_chose_incorrect_although_correct_exists'].sum()
                    + df['step_3_chose_incorrect_although_correct_exists'].sum()
                    + df['step_4_chose_incorrect_although_correct_exists'].sum() 
                    ) / (df[(df['step_1_chose_incorrect_although_correct_exists'] == 0) | (df['step_1_chose_incorrect_although_correct_exists'] == 1)].shape[0]
                        + df[(df['step_2_chose_incorrect_although_correct_exists'] == 0) | (df['step_2_chose_incorrect_although_correct_exists'] == 1)].shape[0]
                        + df[(df['step_3_chose_incorrect_although_correct_exists'] == 0) | (df['step_3_chose_incorrect_although_correct_exists'] == 1)].shape[0]
                        + df[(df['step_4_chose_incorrect_although_correct_exists'] == 0) | (df['step_4_chose_incorrect_although_correct_exists'] == 1)].shape[0])
            else: 
                incorrect_thought_chosen_although_correct_available_fraction_of_tasks_w_correct_and_incorrect = 0
                
        if "baseline_cot" in p:
            return {
                "n_tasks": n_tasks,
                "description_correct_and_complete_fraction_of_n_tasks": description_correct_and_complete_fraction_of_n_tasks,
                "description_correct_but_missing_relevant_details_fraction_of_n_tasks": description_correct_but_missing_relevant_details_fraction_of_n_tasks,
                "description_wrong_fraction_of_n_tasks": description_wrong_fraction_of_n_tasks,
                "description_error_bc_bad_object_detection_fraction_of_incorrect_descriptions": description_error_bc_bad_object_detection_fraction_of_incorrect_descriptions,
                "test_case_description_correct_fraction_of_n_tasks": test_case_description_correct_fraction_of_n_tasks,
                "test_case_description_error_bc_bad_object_detection_fraction_of_all_where_objects_described_in_detail": test_case_description_error_bc_bad_object_detection_fraction_of_all_where_objects_described_in_detail,
                "correct_chain_fraction_of_n_tasks": correct_chain_fraction_of_n_tasks,
                "correct_pattern_and_condition_fraction_of_n_tasks": correct_pattern_and_condition_fraction_of_n_tasks,
                "correct_instructions_fraction_of_n_tasks": correct_instructions_fraction_of_n_tasks,
                "correct_transformations_fraction_of_n_tasks": correct_transformations_fraction_of_n_tasks,
                "instructions_fit_wrong_pattern_fraction_of_wrong_pattern": instructions_fit_wrong_pattern_fraction_of_wrong_pattern,
                "transformations_fit_wrong_pattern_fraction_of_wrong_pattern": transformations_fit_wrong_pattern_fraction_of_wrong_pattern,
                "key_concept_correct_fraction_of_n_tasks": key_concept_correct_fraction_of_n_tasks,
                "wrong_pattern_but_correct_key_concept_fraction_of_wrong_pattern": wrong_pattern_but_correct_key_concept_fraction_of_wrong_pattern,
                "arithmetic_interpretation_fraction_of_n_tasks": arithmetic_interpretation_fraction_of_n_tasks
            }
        elif ("tot_normal" in p) or ("tot_objects" in p):
            return {
                "step_1_both_sampled_fraction_of_n_tasks": step_1_both_sampled_fraction_of_n_tasks,
                "step_2_both_sampled_fraction_of_n_tasks": step_2_both_sampled_fraction_of_n_tasks,
                "step_3_both_sampled_fraction_of_n_tasks": step_3_both_sampled_fraction_of_n_tasks,
                "step_4_both_sampled_fraction_of_n_tasks": step_4_both_sampled_fraction_of_n_tasks,
                "step_1_only_correct_sampled_fraction_of_n_tasks": step_1_only_correct_sampled_fraction_of_n_tasks,
                "step_2_only_correct_sampled_fraction_of_n_tasks": step_2_only_correct_sampled_fraction_of_n_tasks,
                "step_3_only_correct_sampled_fraction_of_n_tasks": step_3_only_correct_sampled_fraction_of_n_tasks,
                "step_4_only_correct_sampled_fraction_of_n_tasks": step_4_only_correct_sampled_fraction_of_n_tasks,
                "step_1_only_incorrect_sampled_fraction_of_n_tasks": step_1_only_incorrect_sampled_fraction_of_n_tasks,
                "step_2_only_incorrect_sampled_fraction_of_n_tasks": step_2_only_incorrect_sampled_fraction_of_n_tasks,
                "step_3_only_incorrect_sampled_fraction_of_n_tasks": step_3_only_incorrect_sampled_fraction_of_n_tasks,
                "step_4_only_incorrect_sampled_fraction_of_n_tasks": step_4_only_incorrect_sampled_fraction_of_n_tasks,
                "all_sampled_descriptions_incorrect_fraction_of_n_tasks": all_sampled_descriptions_incorrect_fraction_of_n_tasks,
                "all_sampled_descriptions_correct_fraction_of_n_tasks": all_sampled_descriptions_correct_fraction_of_n_tasks,
                "at_least_one_selected_description_correct_fraction_of_n_tasks": at_least_one_selected_description_correct_fraction_of_n_tasks,
                "at_least_one_selected_description_incorrect_fraction_of_n_tasks": at_least_one_selected_description_incorrect_fraction_of_n_tasks,
                "all_selected_description_correct_fraction_of_n_tasks": all_selected_description_correct_fraction_of_n_tasks,
                "all_selected_description_incorrect_fraction_of_n_tasks": all_selected_description_incorrect_fraction_of_n_tasks,
                "one_correct_and_one_incorrect_description_fraction_of_n_tasks": one_correct_and_one_incorrect_description_fraction_of_n_tasks,
                "description_error_bc_bad_object_detection_fraction_of_incorrect_descriptions": description_error_bc_bad_object_detection_fraction_of_incorrect_descriptions,
                "test_case_description_correct_fraction_of_n_tasks": test_case_description_correct_fraction_of_n_tasks,
                "test_case_description_error_bc_bad_object_detection_fraction_of_incorrect_descriptions": test_case_description_error_bc_bad_object_detection_fraction_of_incorrect_descriptions,
                "correct_chain_at_least_one_correct_thought_per_step_fraction_of_n_tasks": correct_chain_at_least_one_correct_thought_per_step_fraction_of_n_tasks,
                "at_least_one_correct_pattern_fraction_of_n_tasks": at_least_one_correct_pattern_fraction_of_n_tasks,
                "at_least_one_correct_instructions_fraction_of_n_tasks": at_least_one_correct_instructions_fraction_of_n_tasks,
                "at_least_one_correct_transformation_fraction_of_n_tasks": at_least_one_correct_transformation_fraction_of_n_tasks,
                "correct_chain_of_best_node_fraction_of_n_tasks": correct_chain_of_best_node_fraction_of_n_tasks,
                "correct_description_of_best_node_fraction_of_n_tasks": correct_description_of_best_node_fraction_of_n_tasks,
                "correct_pattern_of_best_node_fraction_of_n_tasks": correct_pattern_of_best_node_fraction_of_n_tasks,
                "correct_instructions_of_best_node_fraction_of_n_tasks": correct_instructions_of_best_node_fraction_of_n_tasks,
                "correct_transformation_of_best_node_fraction_of_n_tasks": correct_transformation_of_best_node_fraction_of_n_tasks,
                "instructions_fit_wrong_pattern_fraction_of_wrong_pattern": instructions_fit_wrong_pattern_fraction_of_wrong_pattern,
                "transformations_fit_wrong_pattern_fraction_of_wrong_pattern": transformations_fit_wrong_pattern_fraction_of_wrong_pattern,
                "key_concept_correct_fraction_of_n_tasks": key_concept_correct_fraction_of_n_tasks,
                "wrong_pattern_but_correct_key_concept_fraction_of_wrong_pattern": wrong_pattern_but_correct_key_concept_fraction_of_wrong_pattern,
                "arithmetic_interpretation_fraction_of_n_tasks": arithmetic_interpretation_fraction_of_n_tasks,
                "incorrect_description_chosen_although_correct_available_fraction_of_tasks_w_correct_and_incorrect": incorrect_description_chosen_although_correct_available_fraction_of_tasks_w_correct_and_incorrect,
                "correct_description_chosen_although_incorrect_available_fraction_of_tasks_w_correct_and_incorrect": correct_description_chosen_although_incorrect_available_fraction_of_tasks_w_correct_and_incorrect,
                "incorrect_pattern_chosen_although_correct_available_fraction_of_tasks_w_correct_and_incorrect": incorrect_pattern_chosen_although_correct_available_fraction_of_tasks_w_correct_and_incorrect,
                "correct_pattern_chosen_although_incorrect_available_fraction_of_tasks_w_correct_and_incorrect": correct_pattern_chosen_although_incorrect_available_fraction_of_tasks_w_correct_and_incorrect,
                "incorrect_instructions_chosen_although_correct_available_fraction_of_tasks_w_correct_and_incorrect": incorrect_instructions_chosen_although_correct_available_fraction_of_tasks_w_correct_and_incorrect,
                "correct_instructions_chosen_although_incorrect_available_fraction_of_tasks_w_correct_and_incorrect": correct_instructions_chosen_although_incorrect_available_fraction_of_tasks_w_correct_and_incorrect,
                "incorrect_transformation_chosen_although_correct_available_fraction_of_tasks_w_correct_and_incorrect": incorrect_transformation_chosen_although_correct_available_fraction_of_tasks_w_correct_and_incorrect,
                "correct_transformation_chosen_although_incorrect_available_fraction_of_tasks_w_correct_and_incorrect": correct_transformation_chosen_although_incorrect_available_fraction_of_tasks_w_correct_and_incorrect,
                "correct_thought_chosen_although_incorrect_available_fraction_of_tasks_w_correct_and_incorrect": correct_thought_chosen_although_incorrect_available_fraction_of_tasks_w_correct_and_incorrect,
                "incorrect_thought_chosen_although_correct_available_fraction_of_tasks_w_correct_and_incorrect": incorrect_thought_chosen_although_correct_available_fraction_of_tasks_w_correct_and_incorrect,
            }
    
    # Open the Excel file and get sheet names
    xls = pd.ExcelFile(path)
    sheet_names = xls.sheet_names
    sheet_names.remove("template")
    sheet_names.remove("Legend")
    
    kpis = {}
    
    df_unsolved = pd.DataFrame()
    df_solved = pd.DataFrame()
    
    tmp = pd.DataFrame()
    for sheet in sheet_names:
        # Load the sheet and remove rows with NaN in the "task_name" column
        df = pd.read_excel(path, sheet_name=sheet)
        df = df.dropna(subset=['task_name'])
        if "gpt3" in sheet:
            df["model"] = "gpt3"
        elif "gpt4" in sheet:
            df["model"] = "gpt4"
        elif "mixtral" in sheet:
            df["model"] = "mixtral"

        # just take ten in case I have more in the begining
        df = df[:10]
        
        # Store KPIs for the current sheet
        kpis[sheet] = calc_metrics(path, df)
        
        if "unsolved" in sheet:
            tmp = pd.concat([tmp, df])
            df_unsolved = pd.concat([df_unsolved, df])
            df_unsolved = df_unsolved.dropna(how='all')
        else:
            tmp = pd.concat([tmp, df])
            kpis[sheet.split("solved")[0]+"both"] = calc_metrics(path, tmp)
            tmp = pd.DataFrame()
            df_solved = pd.concat([df_solved, df])   
            df_solved = df_solved.dropna(how='all')     
        
    # Store KPIs for the all unsolved
    kpis["all_unsolved"] = calc_metrics(path, df_unsolved)
    # Store KPIs for the all solved
    kpis["all_solved"] = calc_metrics(path, df_solved)
    # Store KPIs for all
    kpis["all_both"] = calc_metrics(path, pd.concat([df_unsolved, df_solved]))
    
    kpi_overview = pd.DataFrame.from_dict(kpis, orient='index').reset_index().rename(columns={'index': 'run'})
    kpi_overview = kpi_overview.fillna(0)
    
    return kpi_overview, df_unsolved, df_solved
