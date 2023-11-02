import numpy as np
import json
import re
import matplotlib.pyplot as plt
import datetime
from utils import *
from config import *
from credentials import *
import sys
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


# ### Main
if __name__ == "__main__":
    # print overview of planned runs and ask user to confirm continuation
    print("##################### OVERVIEW ########################")
    for model_name, revision in zip(MODEL_NAMES, REVISIONS):
        print(model_name, ":", revision)
    user_input = input("Do you want to continue running the script? (yes/no): ").lower().strip()
    if  user_input == 'yes':
        # Your script logic here
        print("Continuing the script...")
    else:
        print("Terminating script.")
        sys.exit()
        
    for model_name, revision in zip(MODEL_NAMES, REVISIONS):
        print("##################### NEW MODEL ########################")
        print(model_name)
        print("##################### NEW MODEL ########################")
        
        # Load Model and Tokenizer
        if "gpt" in model_name:
            llm = load_gpt
            tokenizer = None
        else:
            tokenizer, model, llm = load_llama(model_name, revision, MAX_TOKEN, MODEL_CONFIG_LLAMA)
              
        # create data generator
        ds = Dataset.from_generator(data_generator, gen_kwargs={"model_name": model_name, "directory_train": TASK_DIR_TRAIN, "directory_eval": TASK_DIR_EVAL, "pre_context": PRE_CONTEXT, "post_context": POST_CONTEXT, "tokenizer": tokenizer, "delimiter": DELIMITER, "prompt_template": TEMPLATE, "sys": SYSTEM_MESSAGE, "output_format": OUTPUT_FORMAT, "instruction_end": INSTRUCTION_END, "change_representation": CHANGE_REPRESENTATION, "new_representation": NEW_REPRESENTATION})

        ###### TODO: Change FOLDER ######
        # Get the current date and time
        current_datetime = datetime.datetime.now()
        # Format the date and time as a string 
        # directory = "results/"+current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        directory = "Testing_none_official_result/"+current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

        os.makedirs(directory, exist_ok=True)

        # My Approach: 
        task_counter = 1
        success = {}
        failure_log = "\n"
        total_input_tokens = 0
        total_output_tokens = 0
        for row in ds:
            # print progress in terms of task counter
            if row["test_case_index"] == 0:
                print(task_counter, "/", len(ds), "prompts")
                task_counter += 1
                
            # call LLM 
            try:
                if "gpt" in model_name:
                    response = llm(row["prompt_gpt"], **MODEL_CONFIG_GPT)
                    output = response['choices'][0]['message']['content']
                    input_tokens = response["usage"]["prompt_tokens"]
                    output_tokens = response["usage"]["completion_tokens"]

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

            # Check answers and save success rates. 
            if row["task_name"] not in success:
                success[row["task_name"]] = 0
            is_success = row["solution"].strip() in output
            success[row["task_name"]] += is_success / row["total_test_cases"]

            # save LLM task output as json file
            try:
                LLM_result_json = get_LLM_result_as_json([row["test_case"]], [output]) 
                with open(directory+"/"+row["task_name"]+"_"+str(row["test_case_index"])+"_LLM_result.json", "w") as json_file:
                    json.dump(LLM_result_json, json_file)
            except Exception as e:
                error = f"Failed to write LLM result as .json file for task {row['task_name']}. Error:\n{e}"
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
                with open(directory+"/"+row["task_name"]+"_"+str(row["test_case_index"])+"_LLM_answer.txt", "w") as text_file:
                    text_file.write(LLM_answer)
            except Exception as e:
                error = f"Failed to write LLM answer as .txt file for task {row['task_name']}. Error:\n{e}"
                failure_log += error+"\n\n################################################################\n\n"
                print(error)
                continue
                
            # print status, only count tasks with success rate of 1
            success_count = sum(1 for value in success.values() if value == 1)
            print(row["task_name"], "Success:", success[row["task_name"]], "Total:", f"{success_count} / {len(success)}")

        # get prompt_oversize_counter
        promp_oversize_counter = ds["prompt_oversize_counter"][-1]
        
        # All test cases need to be correct to pass
        success_log = []
        for key, value in success.items():
            if value > 0:
                success_log.append((key, value))
            if value < 0.999:
                success[key] = 0
            else:
                success[key] = 1
        
        # track time
        end_time = datetime.datetime.now()
        duration = end_time - current_datetime

        # save log result as txt file
        if "gpt" in model_name:
            revision = ""
        else:
            revision =  ':'+revision
        try:
            log =  f"{model_name+revision}\nDuration: {duration}\nTotal: {success_count} / {len(success)}\nToo long prompts: {promp_oversize_counter}\nTotal input token: {total_input_tokens}\nTotal output token: {total_output_tokens}\nSuccess log: {success_log}\nFailure log: {failure_log}"
            with open(directory+"/log.txt", "w") as text_file:
                text_file.write(log)
        except Exception as e:
            print("Failed to write log as .txt file", f"Error: {e}")
        
        print("Done.")
        print("Duration:", duration)
        print("Too long prompts:", promp_oversize_counter)
        print("Success log:", success_log)
        print("Failure log:", failure_log)


