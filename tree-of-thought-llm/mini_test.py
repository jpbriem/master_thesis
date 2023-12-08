import os
OPENAI_KEY = "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"
os.environ['OPENAI_API_KEY'] = OPENAI_KEY

import datetime
import json
import argparse
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

# ys, infos = solve(args, task, 0)
# print(ys[0])

########## ARC ##########
args = argparse.Namespace(
    backend='gpt-3.5-turbo-1106', 
    # backend='gpt-4-1106-preview', 
    use_api=True, 
    temperature=0.7, 
    # task='arc', 
    task='1D-arc',
    naive_run=False, 
    prompt_sample='cot', 
    method_generate='sample', 
    method_evaluate='value', 
    method_select='greedy', 
    n_generate_sample=2, 
    n_evaluate_sample=1, 
    n_select_sample=1)


log, failure_log = [], ""

# Get the current date and time
current_datetime = datetime.datetime.now()
# Format the date and time as a string 
# directory = "results/"+current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
directory = "Testing_none_official_result/"+current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(directory, exist_ok=True)

task = get_task(args.task)
for idx in range(len(task)):
    Node.reset_tree()
    task_name = task.names[idx].split(".json")[0]  
    ys, infos = dfs.solve(args, task, idx)

    # check best leaf nodes
    result_infos = [task.test_output(idx, y.LLM_answer) for y in ys] # TODO: Implement

    #log 
    infos.update({'idx': idx, 'ys': [str(y) for y in ys], 'infos': result_infos, 'usage_so_far': gpt_usage(args.backend)})
    log.append(infos)
    
    # save LLM result as txt file
    try:
        log_text = ""
        for step_info in infos['steps']:
            log_text += "\n###########################################################\nNew Step\n###########################################################\n"
            for key, value in step_info.items():
                log_text += f"{key}: {value}\n\n"
            
        task_name = task.names[idx]
        with open(directory+"/"+task_name+"_LLM_answer.txt", "w") as text_file:
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
        with open(directory+"/"+task_name+"_log.json", 'w') as f:
            json.dump(log[-1], f, indent=4)
    except Exception as e:
        error = f"Failed to write log as json file for task {task_name}. Error:\n{e}"
        failure_log += error+"\n\n################################################################\n\n"
        print(error)