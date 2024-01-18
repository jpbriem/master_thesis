import numpy as np
import json
import re
import itertools

import matplotlib.pyplot as plt
import datetime
from utils import *
from credentials import *
import sys
from IPython.display import clear_output
import time
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

import torch

from datasets import load_dataset, Dataset
from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from auto_gptq import exllama_set_max_input_length

from tot.methods.arc_config import * 
from tot.methods.arc_utils import *
from tot.models import gpt_usage

# ### Main
if __name__ == "__main__":
    print("##################### OVERVIEW ########################")
    check_model_selection(MODEL_NAMES, REVISIONS)
        
    for model_name, revision in zip(MODEL_NAMES, REVISIONS):
        print("##################### NEW MODEL ########################")
        print(model_name)
        print("########################################################")
        
        ###### TODO: Change FOLDER ######
        # Get the current date and time
        current_datetime = datetime.datetime.now()
        # Format the date and time as a string 
        # directory = "results/"+current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        directory = f"Testing_none_official_result/{TASK}/{model_name.split('/')[-1]}_naive_standard_" + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        # directory = f"Testing_none_official_result/{TASK}/{model_name}_naive_cot_" + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(directory+"/tasks", exist_ok=True)
        
        # Load Model and Tokenizer
        try:
            # Free up GPU memory
            torch.cuda.empty_cache()
            if model_name != MODEL_NAMES[0]:
                llm = None
                tokenizer = None
                time.sleep(10) # wait 10 seconds to avoid CUDA Memory issues
                torch.cuda.empty_cache()
                time.sleep(10) # wait 10 seconds to avoid CUDA Memory issues
            if "gpt" in model_name:
                llm = load_gpt
                tokenizer = None
            elif model_name in ["TheBloke/Falcon-7B-Instruct-GPTQ", "TheBloke/Falcon-40B-Instruct-GPTQ"]:
                falcon_model, tokenizer = load_falcon(model_name, revision)
                llm = run_falcon
            else:
                tokenizer, _, llm = load_llama(model_name, revision, MAX_TOKEN, MODEL_CONFIG_LLAMA)
        except Exception as e:
            error = f"Failed to load LLM: {model_name}. Error:\n{e}"
            print(error)
            with open(directory+"/log.txt", "w") as text_file:
                text_file.write(error)
            continue 
                                
        # create data generator
        ds = Dataset.from_generator(data_generator, gen_kwargs={"model_name": MODEL_NAMES[0], "dataset": DATASET, "directories": DIR, "delimiter": DELIMITER[DATASET], "prompt_template": TEMPLATE, "sys": SYSTEM_MESSAGE, "output_format": OUTPUT_FORMAT, "pre_test_case": PRE_TEST_CASE,"post_test_case": POST_TEST_CASE, "instruction_end": INSTRUCTION_END, "tokenizer": tokenizer, "change_representation": CHANGE_REPRESENTATION, "new_representation": NEW_REPRESENTATION})
        ########### TODO: Filter for tests ###########
        # ds = ds.filter(lambda x: x["task_name"] == "29c11459.json")
        # ds = ds.filter(lambda x: x["task_name"] == "1.json" or x["task_name"] == "2.json")
        # ds = ds.filter(lambda x: x["task_name"] in [
        #     "1d_fill_0.json", "1d_fill_1.json", "1d_fill_2.json", "1d_move_1p_0.json", "1d_move_1p_1.json"
        #     "arc2smr_0.json", "arc2smr_1.json", "arc2smr_2.json", "arc2smr_v_0.json", "arc2smr_v_1.json",
        #     "05f2a901.json", "a79310a0.json", "d2abd087.json", "a699fb00.json"])
        #############################################
        num_tasks = len(ds.filter(lambda x: x["test_case_index"] == 0 and x["descriptions_index"] == 0))

        # My Approach: 
        log = []    
        task_counter = 1
        success, infos, result_infos = {}, {}, {}
        cat_success = {c: 0 for c in list(set(ds["category"]))}
        cat_failures = cat_success.copy()
        failure_log = "\n"
        total_input_tokens, total_output_tokens = 0, 0
        for i, row in enumerate(ds):
            # print progress in terms of task counter
            if row["test_case_index"] == 0 and row["descriptions_index"] == 0:
                print(task_counter, "/", num_tasks)
                task_counter += 1
                task_is_solved = False
            
            # check if task has been solved already, in case of multiple descriptions with LARC
            if TASK == "LARC" and row["test_case_index"] == 0 and row["descriptions_index"] > 0: 
                if success[row["task_name"]+"-"+str(row["descriptions_index"]-1)] == 1:
                    task_is_solved = True
            if task_is_solved:
                continue
            
            # call LLM 
            try:
                if "gpt" in model_name:
                    if MANUAL_GPT:
                        print(row["prompt_gpt"])
                        output = input("Enter GPT's answer: ")
                        clear_output()
                    else:
                        response = llm(row["prompt_gpt"], **MODEL_CONFIG_GPT)
                        output = response['choices'][0]['message']['content']
                        input_tokens = response["usage"]["prompt_tokens"]
                        output_tokens = response["usage"]["completion_tokens"]
                elif model_name in ["TheBloke/Falcon-7B-Instruct-GPTQ", "TheBloke/Falcon-40B-Instruct-GPTQ"]:
                    output = llm(tokenizer, falcon_model, row["prompt_llama"], **MODEL_CONFIG_FALCON)
                    input_tokens = row["prompt_llama_tokens"]
                    output_tokens = count_tokens(output, model_name, tokenizer)[0]
                else:
                    output = llm(row["prompt_llama"])
                    input_tokens = row["prompt_llama_tokens"]
                    output_tokens = count_tokens(output, model_name, tokenizer)[0]
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
            except Exception as e:
                error = f"Failed to run LLM for task {row['task_name']}. Error:\n{e}"
                failure_log += error+"\n\n################################################################\n\n"
                print(error)
                continue    
            
            # add description id to task name if LARC
            if TASK == "LARC": 
                description_id = "-"+str(row["descriptions_index"])
            else:
                description_id = ""

            # Check answers and save success rates. 
            if row["task_name"]+description_id not in success:
                success[row["task_name"]+description_id] = 0
            is_success = re.sub(r'\s+', ' ', row["solution"]).strip() in re.sub(r'\s+', ' ', output).strip()
            success[row["task_name"]+description_id] += is_success / row["total_test_cases"]
            if success[row["task_name"]+description_id] == 1:
                cat_success[row["category"]] += success[row["task_name"]+description_id]
            else:
                cat_failures[row["category"]] += 1
            result_infos.update({
                'success': success[row["task_name"]+description_id], 
                'tries': 1, # currently LLM has just 1 try
                'success_rate': sum([1 if v == 1 else 0 for k, v in success.items()]) / (task_counter-1), # success rate of tasks so far
                'cat_success_cnt': cat_success[row["category"]],
                'cat_success_rate': cat_success[row["category"]] / (cat_success[row["category"]] + cat_failures[row["category"]])})
            
            # save LLM task output as json file
            try:
                LLM_result_json = get_LLM_result_as_json([row["test_case"]], [output]) 
                with open(directory+"/tasks"+"/"+row["task_name"]+description_id+"_"+str(row["test_case_index"])+"_LLM_result.json", "w") as json_file:
                    json.dump(LLM_result_json, json_file)
            except Exception as e:
                error = f"Failed to write LLM result as .json file for task {row['task_name']+description_id}. Error:\n{e}"
                failure_log += error+"\n\n################################################################\n\n"
                print(error)
                continue
            
            # save LLM result as txt file
            try:
                if len(row['prompt_gpt']) > 0:
                    prompt_gpt = ""
                    for message in row['prompt_gpt']:
                        prompt_gpt += message['content']+"\n"
                else:
                    prompt_gpt = ""
                LLM_answer = f"Input token: {input_tokens}\nOutput token: {output_tokens}\n################################################################\n\n"
                LLM_answer += f"LLM prompt:\n{row['prompt_llama']}{prompt_gpt}\n################################################################\n\n"
                LLM_answer += f"LLM answer:\n{output}\n################################################################\n\n"
                LLM_answer += f"Solution:\n{row['solution']}\n"
                with open(directory+"/tasks"+"/"+row["task_name"]+description_id+"_"+str(row["test_case_index"])+"_LLM_answer.txt", "w") as text_file:
                    text_file.write(LLM_answer)
            except Exception as e:
                error = f"Failed to write LLM answer as .txt file for task {row['task_name']+description_id}. Error:\n{e}"
                failure_log += error+"\n\n################################################################\n\n"
                print(error)
                continue
            
            # add logs
            if ("idx" in infos and infos["idx"] != task_counter-2):
                # if new task, add infos of previous task before updating "infos"
                log.append(infos.copy())
            infos.update({
                'idx': task_counter-2, 
                'task': row["task_name"],
                'category': row["category"],
                'x': row["task_json"], 
                'y_gt': row["solution"], 
                'y': output, 
                'result': result_infos.copy(), 
                'usage_so_far': gpt_usage(model_name)})
                
            # print status, only count tasks with success rate of 1
            success_count = sum(1 for value in success.values() if value == 1)
            print(row["task_name"]+description_id, "Success:", success[row["task_name"]+description_id], "Total:", f"{success_count} / {len(success)}")

        
        # add last log infos
        log.append(infos.copy())
        
        # get prompt_oversize_counter: counts how many tasks have been skipped because prompt was too long; For LARC only counts + 1 if all descriptions are too long
        promp_oversize_counter = ds["prompt_oversize_counter"][-1]
        
        # Save (task_name, success) of all tasks, where at least 1 test case was solved
        success_log = []
        for key, value in success.items():
            if value > 0:
                success_log.append((key, value))
        
        # track time
        end_time = datetime.datetime.now()
        duration = end_time - current_datetime
        
        # save all task log as json file: Add overall information to log
        summary = {'date': current_datetime.strftime("%Y-%m-%d_%H-%M-%S"), 'model': model_name, 'usage_total': gpt_usage(model_name, input_tokens, output_tokens), 'dataset': DATASET, "change_representation": CHANGE_REPRESENTATION, "new_representation": NEW_REPRESENTATION if CHANGE_REPRESENTATION else None, 'num_tasks': num_tasks, 'success_cnt': success_count, 'success_rate': success_count / num_tasks,  'cat_success_cnt': cat_success, 'cat_success_rate': {k: v/(v+v2) for (k, v), (k2, v2) in zip(cat_success.items(), cat_failures.items())}, 'solved_tasks': success_log, 'failure_log': failure_log}
        log = [summary] + log
        print(summary)
        try:
            with open(directory+"/all_tasks_log.json", 'w') as f:
                json.dump(log, f, indent=4)
        except Exception as e:
            error = f"Failed to write all tasks log json file for task. Error:\n{e}"
            print(error)
            print(f"\n\n\nRun:{log}")
            failure_log += error+"\n\n################################################################\n\n"

        # save log result as txt file
        if "gpt" in model_name:
            revision = ""
        else:
            revision =  ':'+revision
        try:
            log =  f"{model_name+revision}\nDuration: {duration}\nTotal: {success_count} / {num_tasks}\nToo long prompts: {promp_oversize_counter}\nTotal input token: {total_input_tokens}\nTotal output token: {total_output_tokens}\nSuccess log: {success_log}\nFailure log: {failure_log}"
            with open(directory+"/log.txt", "w") as text_file:
                text_file.write(log)
        except Exception as e:
            print("log", log)
            print()
            print("Failed to write log as .txt file", f"Error: {e}")
        
        print("Done.")
        print("Duration:", duration)
        print("Too long prompts:", promp_oversize_counter)
        print("Success log:", success_log)
        print("Failure log:", failure_log)


