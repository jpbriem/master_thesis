import os
import random
import datetime
import json
import argparse
import pandas as pd
from tot.methods import bfs, dfs
from tot.tasks import get_task
from tot.methods.tree_nodes import Node
from tot.models import gpt_usage

########## Game 24 ##########
# args = argparse.Namespace(
#     backend='gpt-3.5-turbo-1106', 
#     temperature=0.7, 
#     task='game24', 
#     naive_run=False, 
#     prompt_sample=None, 
#     method_generate='propose', 
#     method_evaluate='value', 
#     method_select='greedy', 
#     n_generate_sample=1, 
#     n_evaluate_sample=3, 
#     n_select_sample=5)

# task = Game24Task()
# ys, infos = solve(args, task, 900)
# print(ys[0])

########## Text ##########
# args = argparse.Namespace(
#     backend='gpt-3.5-turbo-1106', 
#     temperature=0.7, 
#     task='text', 
#     naive_run=False, 
#     prompt_sample='cot', 
#     method_generate='sample', 
#     method_evaluate='vote', 
#     method_select='greedy', 
#     n_generate_sample=2, 
#     n_evaluate_sample=2, 
#     n_select_sample=1)

# task = TextTask()


########## ARC ##########
args = argparse.Namespace(
    # backend='gpt-3.5-turbo-1106',   # TODO: Set model!
    # backend='gpt-4-1106-preview', 
    # backend='NousResearch/Llama-2-7b-chat-hf',
    backend='TheBloke/Llama-2-70b-Chat-GPTQ',
    model_revision='main',
    use_api=True,                       # TODO: Use API?!
    temperature=0.7, 
    # task='arc',                       # TODO: Set task!
    task='arc_1D',
    # task = 'arc_h_v',
    input_representation = None,    # TODO: set input representation
    # input_representation = 'objects',
    naive_run=True,                    # TODO: Naive run? TODO: chang in prompts
    search_algo='bfs',                  # TODO: Set search algorithm!
    #search_algo='dfs',
    prompt_sample='cot',                # TODO: Set prompt sample: cot - standard!
    method_generate='sample', 
    method_evaluate='value', 
    method_select='greedy',
    revision=False,                     # TODO: Revision?
    n_generate_sample=1,                # TODO: Set tree search parameters!
    n_evaluate_sample=1, 
    n_select_sample=1)

# get IDs of 50 ARC tasks to be tested # TODO: original ARC???
# data = pd.read_csv('/work/jbriem/repos/master_thesis/ARC_datasets/1D-ARC/LLM4ARC/output-logs/direct-grid/ARC-subset/direct_grid_few_shot_number_3.5.csv')
# tasks = list(data["Task_ID"])

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
    random.seed(42)
    random.shuffle(indices)
    count = 0 # TODO: delete!!!
    for idx in indices:
        print(f"Task {idx} of {len(task)}")
        if count == 50: # TODO: delete!!!
            break
        Node.reset_tree()
        task_name = task.names[idx].split(".json")[0]
        task_category = task.categories[idx]
        # if task_name not in tasks: # TODO: original ARC???
        #     continue
        if 'scale' in task_name or "flip" in task_name or "move_1p" in task_name:
            count += 1 # TODO: delete!!!
        else:
            continue
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
                
            
    # save all task log as json file: Add overall information to log
    summary = {'date': current_datetime.strftime("%Y-%m-%d_%H-%M-%S"), 'model': args.backend, 'usage_total': gpt_usage(args.backend), 'dataset': args.task, 'num_tasks': len(task)} 
    if task_infos:
        summary.update(task_infos)
    summary.update({'success_cnt': task.full_success, 'success_rate': task.full_success / len(task), 'cat_success_cnt': task.cat_success, 'cat_success_rate': {k: v/(v+v2) if v+v2 > 0 else 0 for (k, v), (k2, v2) in zip(task.cat_success.items(), task.cat_failures.items())}, 'solved_tasks': task.solved_tasks, 'args:': vars(args), 'failure_log': failure_log})
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
    run(args)