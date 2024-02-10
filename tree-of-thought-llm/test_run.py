import os
import random
import datetime
import json
import argparse
import pandas as pd
from tot.methods import bfs, dfs, search_utils
from tot.tasks import get_task
from tot.methods.tree_nodes import Node
from tot.models import gpt_usage
from tot.methods.arc_config import MODEL_NAMES, REVISIONS
from tot.methods.arc_utils import check_model_selection

########## ARC ##########
args = argparse.Namespace(
    # backend='gpt-3.5-turbo-1106',   # TODO: Set model!
    # backend='gpt-4-1106-preview', 
    # backend='NousResearch/Llama-2-7b-chat-hf',
    # backend='TheBloke/Llama-2-70b-Chat-GPTQ',
    # model_revision='main',
    backend=MODEL_NAMES,
    model_revision=REVISIONS,
    use_api=True,                       # TODO: Use API?!
    temperature=0.7, 
    # task='arc',                       # TODO: Set task!
    task='arc_1D',
    # task = 'arc_h_v',
    input_representation = None,    # TODO: set input representation
    # input_representation = 'objects',
    naive_run=False,                    # TODO: Naive run? TODO: chang in prompts
    search_algo='bfs',                  # TODO: Set search algorithm!
    #search_algo='dfs',
    prompt_sample='cot',                # TODO: Set prompt sample: cot - standard!
    method_generate='sample', 
    method_evaluate='value', 
    method_select='greedy',
    revision=False,                     # TODO: Revision?
    n_generate_sample=2,                # TODO: Set tree search parameters!
    n_evaluate_sample=1, 
    n_select_sample=1)

# get IDs of 50 ARC tasks to be tested # TODO: original ARC???
# data = pd.read_csv('/work/jbriem/repos/master_thesis/ARC_datasets/1D-ARC/LLM4ARC/output-logs/direct-grid/ARC-subset/direct_grid_few_shot_number_3.5.csv')
# tasks = list(data["Task_ID"])
# solved_gpt3 = ["25ff71a9.json", "6150a2bd.json", "74dd1130.json", "9dfd6313.json", "b1948b0a.json", "c8f0f002.json", "d037b0a7.json", "dc433765.json"]
# solved_gpt3 = ['1d_move_1p_2.json', '1d_flip_30.json', '1d_move_1p_33.json', '1d_scale_dp_41.json', '1d_move_1p_27.json', '1d_flip_48.json', '1d_move_1p_12.json', '1d_scale_dp_20.json', '1d_move_1p_4.json', '1d_flip_8.json', '1d_scale_dp_12.json', '1d_flip_10.json', '1d_flip_36.json', '1d_move_1p_30.json', '1d_move_1p_20.json', '1d_scale_dp_11.json', '1d_move_1p_31.json', '1d_flip_33.json', '1d_move_1p_1.json', '1d_move_1p_25.json', '1d_scale_dp_44.json', '1d_flip_29.json', '1d_flip_0.json', '1d_scale_dp_39.json', '1d_move_1p_23.json', '1d_scale_dp_33.json', '1d_scale_dp_47.json', '1d_move_1p_16.json', '1d_scale_dp_22.json', '1d_flip_4.json', '1d_move_1p_39.json', '1d_flip_9.json', '1d_scale_dp_21.json', '1d_scale_dp_28.json', '1d_flip_40.json', '1d_move_1p_41.json', '1d_flip_3.json', '1d_scale_dp_50.json', '1d_move_1p_10.json', '1d_flip_43.json', '1d_flip_46.json', '1d_scale_dp_30.json', '1d_flip_47.json', '1d_move_1p_40.json', '1d_flip_34.json', '1d_scale_dp_10.json', '1d_move_1p_35.json', '1d_move_1p_19.json', '1d_scale_dp_49.json', '1d_move_1p_36.json']
# multi_colour  = ["3c940459.json", "67a3c6ac.json", "88a10436.json", "6150a2bd.json", "74dd1130.json", "b2862040.json"]

def run(args):   
    log, failure_log = [], ""

    current_datetime = datetime.datetime.now()

    # Create directory for results 
    # directory = "results/"                # TODO: set result directory!
    directory = "Testing_none_official_result/"
    if args.naive_run:
        directory += f"{args.task}/{args.backend.split('/')[-1]}_naive_{args.prompt_sample}_" + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    else:
        directory += f"{args.task}/{args.backend.split('/')[-1]}_" + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(directory+"/tasks", exist_ok=True)
       
    # initialize task
    task = get_task(args.task)
    # get further task information for logging, if needed
    task_infos = task.get_task_infos()
    # set representation of inputs
    task.set_input_representation(args.input_representation)

    # solve the task
    indices = list(range(len(task)))
    # random.seed(42)
    # random.shuffle(indices)
    # count = 0 # TODO: delete!!!
    for idx in indices:
        print(f"Task {idx+1} of {len(task)}")
        # if count == 1: # TODO: delete!!!
        #     break
        Node.reset_tree()
        task_name = task.names[idx].split(".json")[0]
        task_category = task.categories[idx]

        # if 'pile_h' in task_name: # TODO: delete!!! 
        # if "scale" in task_name or "move_1p" in task_name or "flip" in task_name: # TODO: delete!!!
        #     count += 1 # TODO: delete!!!
        # else:
        #     continue
        # if not task_name+".json" in solved_gpt3:#+multi_colour: # TODO delete!!!
        #     continue
        
        if args.search_algo == "bfs":
            search_algo = bfs
        elif args.search_algo == "dfs":
            search_algo = dfs 
        if args.naive_run:
            task.update_prompt_modules("naive")
            ys, infos = search_algo.naive_solve(args, task, idx)
        else:
            ys, infos = search_algo.solve(args, task, idx)

        # check best leaf nodes
        if args.naive_run:
            result_infos = task.test_output_naive(idx, ys)
        else:
            result_infos = task.test_output(idx, ys)

        #log 
        infos.update({'idx': idx, 'task': task_name, 'category': task_category, 'ys': [str(y) for y in ys], 'result': result_infos, 'usage_so_far': gpt_usage(args.backend)})
        log.append(infos)
        failure_log = save_log_files(log, task_name, directory, failure_log)
        
        print(f"Solved: {task.full_success} / {idx+1}")    
    
    search_utils.model = None
            
    # save all task log as json file: Add overall information to log
    summary = {'date': current_datetime.strftime("%Y-%m-%d_%H-%M-%S"), 'model': args.backend, 'usage_total': gpt_usage(args.backend), 'dataset': args.task, 'num_tasks': len(task), 'num_tasks_with_too_long_prompts': sum([len(v) for k, v in task.too_long_prompts_no_output.items()])} 
    if task_infos:
        summary.update(task_infos)
    summary.update({'success_cnt': task.full_success, 'success_rate': task.full_success / (len(task)-sum([len(v) for k, v in task.too_long_prompts_no_output.items()])), 'cat_success_cnt': task.cat_success, 'cat_success_rate': {k: v/(v+v2) if v+v2 > 0 else 0 for (k, v), (k2, v2) in zip(task.cat_success.items(), task.cat_failures.items())}, 'solved_tasks': task.solved_tasks, 'solved_tasks_str_comparison': task.solved_tasks_str_comparison, 'tasks_with_too_long_prompts': task.too_long_prompts_no_output , 'args:': vars(args), 'failure_log': failure_log})
    log = [summary] + log
    print(summary)
    try:
        with open(directory+"/all_tasks_log.json", 'w') as f:
            json.dump(log, f, indent=4)
    except Exception as e:
        error = f"Failed to write all tasks log json file for task. Error:\n{e}"
        print(error)
        print(f"\n\n\nRun:{log}")
    

def save_log_files(log, task_name, directory, failure_log=""):
    # save LLM result as txt file
    try:
        log_text = ""
        for step_info in log[-1]['steps']:
            log_text += "\n###########################################################\nNew Step\n###########################################################\n"
            for key, value in step_info.items():
                log_text += f"{key}: {value}\n\n"
        log_text += "\n###########################################################\nResult:\n"
        for key, value in log[-1]['result'].items():
            log_text += f"{key}: {value}\n\n"
        with open(directory+"/tasks/"+task_name+"_LLM_answer.txt", "w") as text_file:
            text_file.write(log_text)
    except Exception as e:
        error = f"Failed to write LLM answer as .txt file for task {task_name}. Error:\n{e}"
        failure_log += error+"\n\n################################################################\n\n"
        print(error)
    
    #save all task log as json file
    try:
        with open(directory+"/all_tasks_log.json", 'w') as f:
            json.dump(log, f, indent=4)
    except Exception as e:
        error = f"Failed to write all tasks log json file for task {task_name}. Error:\n{e}"
        failure_log += error+"\n\n################################################################\n\n"
        print(error)
    
    #save last task log as json file
    try:
        with open(directory+"/tasks/"+task_name+"_log.json", 'w') as f:
            json.dump(log[-1], f, indent=4)
    except Exception as e:
        error = f"Failed to write log as json file for task {task_name}. Error:\n{e}"
        failure_log += error+"\n\n################################################################\n\n"
        print(error)
    
    return failure_log


if __name__ == '__main__':
    print(args)
    if not isinstance(args.backend, list):
        args.backend = [args.backend]
        args.model_revision = [args.model_revision]
    
    print("##################### OVERVIEW ########################")
    check_model_selection(args.backend, args.model_revision)
    
    for model, revision in zip(args.backend, args.model_revision):
        print("##################### NEW MODEL ########################")
        print(model)
        print("########################################################")
        
        args.backend = model
        args.model_revision = revision
        run(args)
