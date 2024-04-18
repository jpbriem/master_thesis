import os
import random
import datetime
import json
import numpy as np
import argparse
import pandas as pd
from tot.methods import bfs, dfs, search_utils
from tot.tasks import get_task
from tot.methods.tree_nodes import Node
from tot.models import gpt_usage, reset_usage
from tot.methods.arc_config import MODEL_NAMES, REVISIONS
from tot.methods.arc_utils import check_model_selection

########## ARC ##########
args = argparse.Namespace(
    continue_run="Mixtral-8x7B-Instruct-v0.1_object-representation_2024-03-30_11-57-57", # TODO: Bisher noch nicht für Object result infos!!!
    backend=MODEL_NAMES,
    model_revision=REVISIONS,
    use_api=False,                       # TODO: Use API?!
    # task='arc',                       # TODO: Set task!
    task='arc_1D', 
    # task = 'arc_h_v',
    # input_representation = None,    # TODO: set input representation
    input_representation = 'objects',
    naive_run=False,                    # TODO: Naive run? TODO: chang in prompts
    search_algo='bfs',                  # TODO: Set search algorithm!
    #search_algo='dfs',
    prompt_sample='cot',                # TODO: Set prompt sample: cot - standard!
    method_generate='sample', 
    method_evaluate='value', 
    method_select='greedy',
    revision=False,                     # TODO: Revision?
    n_generate_sample=4,                # TODO: Set tree search parameters!
    n_evaluate_sample=2, 
    n_select_sample=2)

# get IDs of 50 ARC tasks to be tested # TODO: original ARC???
# data = pd.read_csv('/work/jbriem/repos/master_thesis/ARC_datasets/1D-ARC/LLM4ARC/output-logs/direct-grid/ARC-subset/direct_grid_few_shot_number_3.5.csv')
# tasks = list(data["Task_ID"])
# solved_gpt3 = ["25ff71a9.json", "6150a2bd.json", "74dd1130.json", "9dfd6313.json", "b1948b0a.json", "c8f0f002.json", "d037b0a7.json", "dc433765.json"]
# solved_gpt3 = ['1d_move_1p_2.json', '1d_flip_30.json', '1d_move_1p_33.json', '1d_scale_dp_41.json', '1d_move_1p_27.json', '1d_flip_48.json', '1d_move_1p_12.json', '1d_scale_dp_20.json', '1d_move_1p_4.json', '1d_flip_8.json', '1d_scale_dp_12.json', '1d_flip_10.json', '1d_flip_36.json', '1d_move_1p_30.json', '1d_move_1p_20.json', '1d_scale_dp_11.json', '1d_move_1p_31.json', '1d_flip_33.json', '1d_move_1p_1.json', '1d_move_1p_25.json', '1d_scale_dp_44.json', '1d_flip_29.json', '1d_flip_0.json', '1d_scale_dp_39.json', '1d_move_1p_23.json', '1d_scale_dp_33.json', '1d_scale_dp_47.json', '1d_move_1p_16.json', '1d_scale_dp_22.json', '1d_flip_4.json', '1d_move_1p_39.json', '1d_flip_9.json', '1d_scale_dp_21.json', '1d_scale_dp_28.json', '1d_flip_40.json', '1d_move_1p_41.json', '1d_flip_3.json', '1d_scale_dp_50.json', '1d_move_1p_10.json', '1d_flip_43.json', '1d_flip_46.json', '1d_scale_dp_30.json', '1d_flip_47.json', '1d_move_1p_40.json', '1d_flip_34.json', '1d_scale_dp_10.json', '1d_move_1p_35.json', '1d_move_1p_19.json', '1d_scale_dp_49.json', '1d_move_1p_36.json']
# multi_colour  = ["3c940459.json", "67a3c6ac.json", "88a10436.json", "6150a2bd.json", "74dd1130.json", "b2862040.json"]
# solved_gpt4 = ['d037b0a7.json', '6150a2bd.json', 'a79310a0.json', '74dd1130.json', '25ff71a9.json', 'ce22a75a.json', 'aabf363d.json', 'c8f0f002.json', 'dc433765.json', 'b1948b0a.json']

def run(args):   
    log, failure_log = [], ""

    if hasattr(args, 'continue_run'):
        current_datetime = datetime.datetime.strptime("_".join(args.continue_run.split("_")[-2:]), '%Y-%m-%d_%H-%M-%S')
    else:
        current_datetime = datetime.datetime.now()

    # Create directory for results 
    # directory = "results/"                # TODO: set result directory!
    directory = "Testing_none_official_result/"
    if args.input_representation == "objects":
        object_flag = "_object-representation"
    else: 
        object_flag = ""
    if args.naive_run:
        directory += f"{args.task}/{args.backend.split('/')[-1]}_naive_{args.prompt_sample}{object_flag}_" + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    else:
        directory += f"{args.task}/{args.backend.split('/')[-1]}{object_flag}_" + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(directory+"/tasks", exist_ok=True)
       
    # initialize task
    task = get_task(args.task)
    task.args = args
    
    # get further task information for logging, if needed
    task_infos = task.get_task_infos()
    # set representation of inputs
    task.set_input_representation(args.task, args.input_representation)

    # log: Add overall information to log
    summary = {'date': current_datetime.strftime("%Y-%m-%d_%H-%M-%S"), 'model': args.backend, 'usage_total': gpt_usage(args.backend), 'dataset': args.task, 'num_tasks': len(task), 'num_tasks_with_too_long_prompts': 0, 'num_tasks_error': 0} 
    if task_infos:
        summary.update(task_infos)
    log.append(summary)

    # solve the task
    indices = list(range(0, len(task), 1))      # TODO: check if correct!
    # random.seed(42)
    # random.shuffle(indices)
    # count = 0 # TODO: delete!!!
    if hasattr(args, 'continue_run'):
        intermediate_state = json.load(open(directory+'/all_tasks_log.json'))
        reset_usage(new_completion_tokens=intermediate_state[-1]["usage_so_far"]["completion_tokens"], new_prompt_tokens=intermediate_state[-1]["usage_so_far"]["prompt_tokens"])
        # log = intermediate_state.copy()
        # for t in intermediate_state:
        #     if t["task"] not in task.success: # TODO: works currently only if we have just one try
        #         task.success[t["task"]] = 0
        #     if t["category"] not in task.cat_success:
        #         task.cat_success[t["category"]] = 0
        #         task.cat_failures[t["category"]] = 0
        #     task.success[t["task"]] += int(t["result"]["success"]) 
        #     if task.success[t["task"]] == 1:
        #         task.full_success += 1
        #         task.cat_success[t["category"]] += 1
        #     else:
        #         task.cat_failures[t["category"]] += 1

        #     if task.success[t["task"]] > 0:
        #         task.solved_tasks.append((t["task"], task.success[t["task"]]))
        # idx = intermediate_state[-1]["idx"]
        # indices = indices[idx+1:]
    
    for idx in indices:
        Node.reset_tree()
        task_name = task.names[idx].split(".json")[0]
        task_category = task.categories[idx]

        print(f"Task: {task_name}\nTask {idx+1} of {len(task)}")
        # count += 1 # TODO: delete!!!       
        # if count == 2: # TODO: delete!!!
        #     break
        # if task_category == "move_h": # TODO: delete!!!
        #     continue
        task_already_tried = False
        if hasattr(args, 'continue_run'):
            for old_log in intermediate_state:
                if task_already_tried:
                    break
                if old_log["task"] == task_name:
                    print(f"Task {task_name} already tried!")
                    task_already_tried = True
                    if old_log["task"] not in task.success: # TODO: works currently only if we have just one try
                        task.success[old_log["task"]] = 0
                    if old_log["category"] not in task.cat_success:
                        task.cat_success[old_log["category"]] = 0
                        task.cat_failures[old_log["category"]] = 0
                    if old_log["result"]["success"]:
                        task.success[old_log["task"]] += int(old_log["result"]["success"]) 
                    if task.success[old_log["task"]] == 1:
                        task.full_success += 1
                        task.cat_success[old_log["category"]] += 1
                    else:
                        task.cat_failures[old_log["category"]] += 1
                    if task.success[old_log["task"]] > 0:
                        task.solved_tasks.append((old_log["task"]+".json", task.success[old_log["task"]]))
                    n_tasks_too_long_prompts = sum([len(v) for k, v in task.too_long_prompts_no_output.items()])
                    n_tasks_error = sum([len(v) for k, v in task.tasks_failed_solving.items()])
                    old_log["result"].update({'success_rate': task.full_success / (idx+1-n_tasks_too_long_prompts-n_tasks_error) if (idx+1-n_tasks_too_long_prompts-n_tasks_error) > 0 else 0, 'cat_success_cnt': task.cat_success[task_category], 'cat_success_rate': task.cat_success[task_category] / (task.cat_success[task_category] + task.cat_failures[task_category]) if task.cat_success[task_category] + task.cat_failures[task_category] > 0 else 0})
                    infos = old_log.copy()
        
                # t = task.data[idx]
        # b = False
        # for l in t["train"]:
        #     if np.prod(np.array(l["input"]).shape) > 10:
        #         b = True
        # if b:
        #     continue
        # if np.prod(np.array(t["test"][0]["input"]).shape) < 10:
        #     print(len(t["test"][0]["input"]))
        # if 'pile_h' in task_name: # TODO: delete!!! 
        # if "pcopy" in task_name or "recolor" in task_name: # TODO: delete!!!
        #     count += 1 # TODO: delete!!!
        # else:
        #     continue
        # if not task_name+".json" in solved_gpt4:#+multi_colour: # TODO delete!!!
        #     continue
        if not task_already_tried:
            if args.search_algo == "bfs":
                search_algo = bfs
            elif args.search_algo == "dfs":
                search_algo = dfs 
            try:
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
            except Exception as e:
                error = f"Failed to solve task {task_name}. Error:\n{e}"
                failure_log += error+"\n\n################################################################\n\n"
                print(error)
                # No outputs to test
                if task_category not in task.tasks_failed_solving:
                    task.tasks_failed_solving[task_category] = [task_name]
                else:
                    task.tasks_failed_solving[task_category].append(task_name)
                continue
            
            # log
            infos.update({'idx': idx, 'task': task_name, 'category': task_category, 'ys': [str(y) for y in ys], 'result': result_infos, 'usage_so_far': gpt_usage(args.backend)})
        #log 
        log.append(infos)
        n_tasks_too_long_prompts = sum([len(v) for k, v in task.too_long_prompts_no_output.items()])
        n_tasks_error = sum([len(v) for k, v in task.tasks_failed_solving.items()])
        summary.update({'usage_total': gpt_usage(args.backend), 'num_tasks_with_too_long_prompts': n_tasks_too_long_prompts, 'num_tasks_error': n_tasks_error, 'success_cnt': task.full_success, 'success_rate': task.full_success / (idx+1-n_tasks_too_long_prompts-n_tasks_error) if (idx+1-n_tasks_too_long_prompts-n_tasks_error) > 0 else 0, 'cat_success_cnt': task.cat_success, 'cat_success_rate': {k: v/(v+v2) if v+v2 > 0 else 0 for (k, v), (k2, v2) in zip(task.cat_success.items(), task.cat_failures.items())}, 'solved_tasks': task.solved_tasks, 'solved_tasks_str_comparison': task.solved_tasks_str_comparison, 'tasks_with_too_long_prompts': task.too_long_prompts_no_output, 'too_long_prompts_all': task.too_long_prompts_all, 'error_in_task_solving': task.tasks_failed_solving, 'args:': vars(args), 'failure_log': failure_log})
        if args.input_representation == "objects":
            summary.update({"object_info": {'object_representation_success_cnt': task.object_representation_success_cnt, 'object_representation_success_rate': task.object_representation_success_cnt / (idx+1-n_tasks_too_long_prompts-n_tasks_error) if (idx+1-n_tasks_too_long_prompts-n_tasks_error) > 0 else 0, 'object_representation_cat_success_cnt': task.object_representation_cat_success, 'object_representation_cat_success_rate': {k: v/(v+v2) if v+v2 > 0 else 0 for (k, v), (k2, v2) in zip(task.object_representation_cat_success.items(), task.object_representation_cat_failures.items())}, 'object_representation_solved_tasks': task.solved_tasks_object_representation}})
        log = [summary] + log[1:]
        print(summary)
        failure_log = save_log_files(log, task_name, directory, failure_log, task_already_tried)
        
        print(f"Solved: {task.full_success} / {idx+1}")    
                    
    search_utils.model = None

def save_log_files(log, task_name, directory, failure_log="", task_already_tried=False):
    if not task_already_tried:
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
    
        #save last task log as json file
        try:
            with open(directory+"/tasks/"+task_name+"_log.json", 'w') as f:
                json.dump(log[-1], f, indent=4)
        except Exception as e:
            error = f"Failed to write log as json file for task {task_name}. Error:\n{e}"
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

