import os
import datetime
import json
import argparse
from tot.methods import bfs, dfs, search_utils
from tot.tasks import get_task
from tot.methods.tree_nodes import Node
from tot.models import gpt_usage, reset_usage
from tot.methods.arc_config import MODEL_NAMES, REVISIONS
from tot.methods.arc_utils import check_model_selection

########## ARC ##########
args = argparse.Namespace(
    # continue_run="", 
    backend=MODEL_NAMES,
    model_revision=REVISIONS,
    use_api=True,                       # TODO: Use API?!
    # task='arc',                       # TODO: Set task!
    # task='arc_1D', 
    task = 'arc_h_v',
    # input_representation = None,    # TODO: set object tool for ARC tasks
    input_representation = 'objects',
    naive_run=False,                    # TODO: Naive run or AToT? TODO: chang in prompts
    search_algo='bfs',                  # TODO: Set search algorithm!
    #search_algo='dfs',
    prompt_sample='cot',                # TODO: Set prompt sample: cot - standard?
    method_generate='sample', 
    method_evaluate='value', 
    method_select='greedy',
    revision=False,                     # TODO: Revision?
    n_generate_sample=4,                # TODO: Set tree search parameters!
    n_evaluate_sample=2, 
    n_select_sample=2)

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
    if hasattr(args, 'continue_run'):
        intermediate_state = json.load(open(directory+'/all_tasks_log.json'))
        reset_usage(new_completion_tokens=intermediate_state[-1]["usage_so_far"]["completion_tokens"], new_prompt_tokens=intermediate_state[-1]["usage_so_far"]["prompt_tokens"])
    
    for idx in indices:
        Node.reset_tree()
        task_name = task.names[idx].split(".json")[0]
        task_category = task.categories[idx]

        print(f"Task: {task_name}\nTask {idx+1} of {len(task)}")
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

