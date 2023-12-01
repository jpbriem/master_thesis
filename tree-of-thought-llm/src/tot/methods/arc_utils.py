
from tot.methods.arc_config import * 
from tot.methods.credentials import *
import numpy as np
import json
import re
import matplotlib.pyplot as plt
import shutil
import datetime
import sys
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
from copy import deepcopy
import torch
import tiktoken
from datasets import load_dataset, Dataset
from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline, logging
from langchain.llms import HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate 
from auto_gptq import exllama_set_max_input_length, AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse
import openai



##################### Load Model and Tokenizer + count Tokens #####################

def load_llama(model_name, revision, max_token, model_config):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.model_max_length is None or tokenizer.model_max_length > 9999999999:
        tokenizer.model_max_length = max_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16, revision=revision
    )

    # fix bug for certain models 
    if model_name in ["TheBloke/Camel-Platypus2-70B-GPTQ", "TheBloke/Platypus2-70B-GPTQ", "TheBloke/Llama-2-70b-Chat-GPTQ", "TheBloke/Mistral-7B-v0.1-GPTQ", "TheBloke/Llama-2-70B-GPTQ"]:
        model = exllama_set_max_input_length(model, 4096)


    # make pipeline
    # Docs for config: https://huggingface.co/docs/transformers/v4.33.3/en/main_classes/configuration#transformers.PretrainedConfig
    # https://www.promptingguide.ai/introduction/settings
    generation_config = GenerationConfig.from_pretrained(model_name)
    generation_config.max_new_tokens = model_config["max_new_tokens"]
    generation_config.temperature = model_config["temperature"]
    #generation_config.top_p = 0.9 #  If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    generation_config.do_sample = True # Whether or not to use sampling ; use greedy decoding otherwise.
    generation_config.repetition_penalty = model_config["repetition_penalty"] # 1.0 means no penalty.

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        generation_config=generation_config,
        # num_workers = 2, # Default=8, When the pipeline will use DataLoader [..] the number of workers to be used.
        # batch_size=2, # Default=1, When the pipeline will use DataLoader [..] the size of the batch to use.
    )

    # make pipeline compatbile with langchain and return
    hf_pipeline = HuggingFacePipeline(pipeline=text_pipeline) #, model_kwargs={"temperature": 0})
    return tokenizer, model, hf_pipeline

def load_gpt(messages, model_name, temperature):
    response = openai.ChatCompletion.create(
        temperature = temperature,
        model=model_name,
        messages=messages,
        response_format={ "type": "json_object" } # forces gpt to output JSON
    )
    return response    

def load_falcon(model_name, revision):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(model_name,
            model_basename=revision,
            use_safetensors=True,
            trust_remote_code=True,
            #device="cuda:0",
            use_triton=False,
            quantize_config=None)
    # fix bug for certain models 
    if model_name in ["TheBloke/Falcon-40B-Instruct-GPTQ"]:
        model = exllama_set_max_input_length(model, 4096)
    return model, tokenizer

def run_falcon(tokenizer, model, prompt, max_new_tokens, temperature):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=temperature, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0])

def count_tokens(prompt, model_name, tokenizer):
    if "gpt" in model_name:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = 0
        tokens_per_message = 3 # for model gpt-3.5-turbo-0613 & gpt-4-0613
        tokens_per_name = 1
        for message in prompt:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        if "gpt-3.5" in model_name:
            token_limit = 4096
        elif "gpt-4" in model_name:
            token_limit = 8192
    else: 
        num_tokens = len(tokenizer.encode(prompt, add_special_tokens=True))
        token_limit = tokenizer.model_max_length
    return num_tokens, token_limit

def replace_quotes_in_text(res, json_format):
    # do some regex to remove unwanted single aprostrophes
    res = res.replace("'", '"')
    res = res.replace("\n", " ")
    # replace any color name or other word enclosed in double quotation marks to single quotation marks, in case it is inside a string field
    pattern = r'"([^\s"]+)"'
    res = re.sub(pattern, r"'\1'", res)
    pattern = r'(\': \s*)\'(\w+)\'(, \s*\')'
    res = re.sub(pattern, r'\1"\2"\3', res)

    # replace only single aprostrophe at the end of a word
    # pattern = r'\b(?<!")(\w+)"\s'
    # res = re.sub(pattern, r'\1 ', res)
    # print(res)

    # add back double quotes to header names
    for key in list(json_format.keys())+["Choice"]:
        pattern = fr"'({key}(?:_\d+)?)'"
        res = re.sub(pattern, r'"\1"', res)

    # ensure that we don't replace away aprostophes in text 
    res = re.sub(r"(\w)\"(\w)", r"\1'\2", res)

    # add double quotes when we have a single number als field value
    pattern = r'(": )\'(\d+)\'(,|})'
    res = re.sub(pattern, r'\1"\2"\3', res)
    
    # replace any characters with a backslash away, except \n and \t
    pattern = r"(\\[^nt])"
    res = re.sub(pattern, "", res)

    # In case the test output is an array but with double quotes
    pattern = r'(":\s*)(\[\[.*?\]\])'
    res = re.sub(pattern, r'\1"\2"', res)
    
    # # replace newline and tabs
    # res = res.replace("\n", "\\n").replace("\t", "\\t")
    return res

def get_json_from_text(string, json_format):
    try:
        return json.loads(string)
    except:
        print("Wrong json format, trying to fix...")
    input_string = string
    try:
        list_of_jsons = []
        indices = []
        # search for json-like segment in string, including nested jsons
        while True:
            # Find the start and end of the JSON segment in the string
            json_start = string.find("{")
            json_end = string.rfind("}") + 1
            if any([json_start == -1, json_end == 0]):
                break
            
            # Extract the JSON-like segment           
            list_of_jsons.append(string[json_start:json_end])
            indices.append((json_start, json_end))
            try:
                string = string[json_start+1:json_end-1]
            except:
                break
        
        previous_segment = None
        for i, json_segment in reversed(list(enumerate(list_of_jsons))):
            if previous_segment:
                json_segment = json_segment[:indices[i+1][0]+1] + previous_segment + json_segment[indices[i+1][1]+1:]
            try:
                x = json.loads(json_segment)
            except:
                json_segment = replace_quotes_in_text(json_segment, json_format)
            previous_segment = json_segment
        json_data = json.loads(json_segment)
        print("JSON parsing successful.")
        return json_data
    except json.JSONDecodeError as e:
        error_msg = f"JSON Parsing Error: {e}\n"
    except Exception as e:
        error_msg = f"General Error: {e}"
    print(error_msg)
    log = f'Output format:\n{json_format}\n\n\n'
    log += f'Input string: {input_string}\n\n\n'
    log += f'JSON parsing error: {error_msg}\n\n\n'
    current_datetime = datetime.datetime.now()
    path = "json_parsing_errors/"+current_datetime.strftime("%Y-%m-%d_%H-%M-%S")+".txt"
    with open(path, "w") as text_file:
        text_file.write(log)
    return path+"\n\n"+log+error_msg

def extract_json_value(string, json_format, key):
    data = get_json_from_text(string, json_format)
    if isinstance(data, str): # error in json parsing
        # get path
        path = data.split(".txt")[0]+".txt"
        data = data.split(".txt")[-1]
        data += f'Key to extract:\n{key}'
        with open(path, "w") as text_file:
            text_file.write(data)
        return data
    # Return the value for the given key
    return data.get(key)

   
##################### Prompt Helper #####################

# load tasks
def load_arc_tasks(path):
    # load data 
    tasks_jsons = []
    tasks_names = []
    tasks_len = []

    # train and test path
    train_path = os.path.join(path, "training")
    test_path = os.path.join(path, "evaluation")
    
    for task_file in sorted(os.listdir(train_path)):
        with open(os.path.join(train_path, task_file)) as fid:
            task_json = json.load(fid)
        tasks_jsons.append(task_json)
        tasks_names.append(task_file)

    for task_file in sorted(os.listdir(test_path)):
        with open(os.path.join(test_path, task_file)) as fid:
            task_json = json.load(fid)
        tasks_jsons.append(task_json)
        tasks_names.append(task_file)

    print("Total number of tasks:", len(tasks_jsons))
    return tasks_jsons, tasks_names

# get context out of json
def get_context(task_json, delimiter):
    text = ""
    for sample in task_json["train"]:
        text += delimiter["example_start"]
        text += delimiter["input_train"]
        text += delimiter["grid_start"]
        for i, row in enumerate(sample["input"]):
            text += delimiter["row_start"]
            for j, value in enumerate(row):
                text += str(value)
                if j < len(row) - 1:
                    text += delimiter["item"]
            if i < len(sample["input"]) - 1:
                text += delimiter["row_end"]
            #text += delimiter["row_end"]
        text += delimiter["grid_end"]
        text += delimiter["output_train"]
        text += delimiter["grid_start"]
        for i, row in enumerate(sample["output"]):
            text += delimiter["row_start"]
            for j, value in enumerate(row):
                text += str(value)
                if j < len(row) - 1:
                    text += delimiter["item"]
            if i < len(sample["output"]) - 1:
                text += delimiter["row_end"]
        text += delimiter["grid_end"]
        text += delimiter["example_end"]
    return text

# get tasks out of json
def get_tasks(task_json, delimiter):
    tasks = []
    solutions = []
    
    for sample in task_json["test"]:
        task = delimiter["task_start"]
        task += delimiter["input_test"]
        task += delimiter["grid_start"]
        for i, row in enumerate(sample["input"]):
            task += delimiter["row_start"]
            for j, value in enumerate(row):
                task += str(value)
                if j < len(row) - 1:
                    task += delimiter["item"]
            if i < len(sample["input"]) - 1:
                task += delimiter["row_end"]
        task += delimiter["grid_end"]
        task += delimiter["output_test"]
        task += delimiter["task_end"]

        solution = delimiter["grid_start"]
        for i, row in enumerate(sample["output"]):
            solution += delimiter["row_start"]
            for j, value in enumerate(row):
                solution += str(value)
                if j < len(row) - 1:
                    solution += delimiter["item"]
            if i < len(sample["output"]) - 1:
                solution += delimiter["row_end"]
        solution += delimiter["grid_end"]
        tasks.append(task)
        solutions.append(solution)
    return tasks, solutions

# get LARC descriptions and tasks
def get_successful_descriptions(task_json):
    descriptions = []
    task = {
        'train': task_json["train"],
        'test': task_json["test"]
    }
    for _, description in task_json["descriptions"].items():
        for _, build in description["builds"].items():
            if build["success"]:
                descriptions.append(f'{description["see_description"].replace("...", " ")}\n{description["do_description"].replace("...", " ")}\n{description["grid_description"].replace("...", " ")}')
    return descriptions, task

# transform string to integer array
def string_to_integer_array(input_string):
    try:
        integer_array = []
        # split the input string by "\n"
        input_string = [row for row in input_string.split('\n')]
        # Split the input string by commas and convert each substring to an integer
        for row in input_string:
            integer_array.append([int(num) for num in row.split(',')])
        return integer_array
    except ValueError:
        # Handle the case where some elements are not valid integers
        return None

# extract lines with numbers out of string
def extract_lines_with_numbers(input_string, ignore_input= False):
    output_found= False
    
    # Define a regular expression pattern to match lines with arbitrary numbers separated by commas
    pattern = r'\d+(?:,\s*\d+)*'  # This pattern matches one or more digits, possibly separated by commas

    # Split the input_string into lines
    lines = input_string.split('\n')

    # Initialize an empty list to store the matched lines
    matched_lines = []

    # Initialize a flag to determine whether to ignore lines
    ignore_lines = False

    # Iterate through the lines
    for line in lines:
        if ignore_input and ignore_lines:
            # If we're in ignore mode, continue until a line with text occurs
            if len(re.findall(pattern, line)) == 0: # Check if the line contains text (ignoring leading/trailing whitespace)
                ignore_lines = False
            else:
                continue

        # Check if the line contains "Input" or "input"
        if ignore_input and ("Input" in line or "input" in line or "train" in line):
            ignore_lines = True
            continue

        # Check if "End of example" is encountered
        if "End of example" in line:
            break

        # Find matches in the current line and add them to the list
        matches = re.findall(pattern, line)
        #print(line)
        if len(matches) > 0:
            matched_lines.extend(matches)
            output_found = True
        elif output_found:
            break

    # Join the matched lines into a single string with line breaks
    result_string = '\n'.join(matched_lines)

    return result_string

# transform LLM result to task json
def get_LLM_result_as_json(tasks, results):
    llm_task_results = []
    for task, result in zip(tasks, results):
        clean_task = extract_lines_with_numbers(task)
        input = string_to_integer_array(clean_task)
        clean_result = extract_lines_with_numbers(result, True)
        output = string_to_integer_array(clean_result) 
        d = {"input": input, "output": output}
        llm_task_results.append(d)
    llm_task_results = dict({
        "test": llm_task_results,
    })
    return llm_task_results

# create data generator for efficient loading of data
def data_generator(model_name, directory_train, directory_eval, delimiter, prompt_template, sys, output_format, pre_test_case, post_test_case, instruction_end, tokenizer, change_representation=False, new_representation=None, LARC=False):
    # get list of files and respective directories
    directories = [directory_train]*len(os.listdir(directory_train)) + [directory_eval]*len(os.listdir(directory_eval))
    task_files =  sorted(os.listdir(directory_train))+sorted(os.listdir(directory_eval))
    # initialize counter for too long prompts
    promp_oversize_counter = 0
    # iterate over files
    for directory, task_file in zip(directories, task_files):
        with open(os.path.join(directory, task_file)) as fid:
            task_json = json.load(fid)
        
        # if we load LARC data, we need to check if the task has been solved by humans
        if LARC:
            descriptions, task_json = get_successful_descriptions(task_json)
            if len(descriptions) == 0:
                continue
            
        else:
            descriptions = [""]       
    
        # change numbers to other representation if wanted
        if change_representation:
            
            task_json = change_color_representation(task_json, new_representation)

        # create context
        if LARC:
            context = ""
        else:
            context = get_context(task_json, delimiter)
        
        # get test cases + solutions
        test_cases, solutions = get_tasks(task_json, delimiter)
        
        # get index of longest test case to check if prompt is too long
        index_of_longest_prompt = max(enumerate(test_cases), key=lambda x: len(x[1]))[0]
        index_of_shortest_description = min(enumerate(descriptions), key=lambda x: len(x[1]))[0]
        
        for i, LARC_description in enumerate(descriptions):
            # check if prompt of longest task is too long
            if "gpt" in model_name:
                prompt = [
                        {"role": "system", "content": prompt_template[0].format(sys=sys, output_format=output_format)},
                        {"role": "user", "content": prompt_template[1].format(pre_task=pre_test_case, task=context+test_cases[index_of_longest_prompt], post_task=post_test_case+LARC_description)}
                    ]
            else:
                prompt = prompt_template.format(sys=sys, output_format=output_format, pre_task=pre_test_case, task=context+test_cases[index_of_longest_prompt], post_task=post_test_case+LARC_description, instruction_end=instruction_end)
            num_tokens, token_limit = count_tokens(prompt, model_name, tokenizer)
            if  num_tokens > token_limit:
                if i == index_of_shortest_description: # only count, if all descriptions for this task are too long! (for non-LARC this is always True: 0 == 0)
                    promp_oversize_counter += 1
                if "LARC" in directory:
                    description_id = "-"+str(i)
                else:
                    description_id = ""
                print(task_file+description_id, "Prompt too long.")
                
                continue
          
            # yield prompts
            for (j, test_case), solution in zip(enumerate(test_cases), solutions):
                # distinguish between llama and gpt model prompt
                if "gpt" in model_name:
                    prompt_llama = ""
                    prompt_gpt = [
                        {"role": "system", "content": prompt_template[0].format(sys=sys, output_format=output_format).strip()},
                        {"role": "user", "content": prompt_template[1].format(pre_task=pre_test_case, task=context+test_case, post_task=post_test_case+LARC_description).strip()}
                    ]
                else:
                    prompt_llama = prompt_template.format(sys=sys, output_format=output_format, pre_task=pre_test_case, task=context+test_case, post_task=post_test_case+LARC_description, instruction_end=instruction_end)
                    prompt_gpt = ""      
                yield {
                    "task_name": task_file,
                    "descriptions_index": i,
                    "test_case_index": j,
                    "total_test_cases": len(test_cases),
                    "test_case": test_case,
                    "context": context,
                    "prompt_llama": prompt_llama.strip(),
                    "prompt_llama_tokens": count_tokens(prompt_llama, model_name, tokenizer)[0],
                    "prompt_gpt": prompt_gpt,
                    "solution": solution,
                    "directory": directory,
                    "prompt_oversize_counter": promp_oversize_counter}
        
def change_color_representation(task_original, new_representation):
    task = deepcopy(task_original)
    for test_train in task:
        for sample in task[test_train]:
            for i, row in enumerate(sample["input"]):
                for j, value in enumerate(row):
                    sample["input"][i][j] = new_representation[value]
            for i, row in enumerate(sample["output"]):
                for j, value in enumerate(row):
                    sample["output"][i][j] = new_representation[value]
    
    return task

def grid_to_nparray(grid):
    if isinstance(grid, str):
        # array in string
        array_start = grid.find("[[")
        array_end = grid.rfind("]]") + 2
        if array_start == -1 or array_end == 1:
            error = "No array found in final output string: " + grid
            return error
        try:
            grid = grid[array_start:array_end]          
            return np.array(eval(grid))
        except:
            error = "Array found in string but error while converting string to array: " + str(grid)
            return error
    else:
        try: 
            return np.array(grid)
        except:
            error = f"Error while converting grid of type {type(grid)} to nparray: " + str(grid)
            return error
        
##################### Evaluation Helper #####################

def grid_to_img(grid):
    colors = [(0, 0, 0),
            (0, 116, 217),
            (255, 65, 54),
            (46, 204, 6),
            (255, 220, 0),
            (170, 170, 170),
            (240, 18, 190),
            (255, 133, 27),
            (127, 219, 255),
            (135, 12, 37)]

    grid = np.int32(grid)
    scale = 10
    img = np.zeros((grid.shape[0] * scale + 1, grid.shape[1] * scale + 1, 3), dtype=np.uint8)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            img[r*scale+1:(r+1)*scale, c*scale+1:(c+1)*scale, :] = colors[grid[r, c]]
    new_img = img.copy()
    new_img[0::10, :, :] = np.uint8(np.round((0.7 * np.float32(img[0::10, :, :]) + 0.3 * 255)))
    new_img[:, 0::10, :] = np.uint8(np.round((0.7 * np.float32(img[:, 0::10, :]) + 0.3 * 255)))
    return new_img


##################### Other #####################

def copy_solved_tasks(result_dir, task_training_dir, task_evaluation_dir, target_dir):
    # check if directories exist
    if not os.path.isdir(result_dir):
        return "Error, result_dir is not a directory"
    elif not os.path.isdir(task_training_dir):
        return "Error, task_training_dir is not a directory"
    elif not os.path.isdir(task_evaluation_dir):
        return "Error, task_evaluation_dir is not a directory"
       
    solved_tasks = set()

    # This regex matches date format in the format YYYY-MM-DD_HH-MM-SS
    date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$')
    # This regex matches tuples in the format ('string', float)
    tuple_pattern = re.compile(r"\('([\w.]+)',\s*([\d.]+)\)")

    # get all solved tasks so far from the log files of result_dir
    for directory in os.listdir(result_dir):
        if not date_pattern.match(directory):
            continue
        log_file_path = os.path.join(result_dir, directory)
        with open(os.path.join(log_file_path, "log.txt")) as fid:
            content = fid.read()
        # Find all matches of the tuple pattern
        matches = tuple_pattern.findall(content)
        # Convert string matches to actual tuples (string, float)
        parsed_list = [(str(filename), float(score)) for filename, score in matches]
        for filename, score in parsed_list:
            solved_tasks.add(filename)
   
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True) 
    os.makedirs(os.path.join(target_dir, "training/"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "evaluation/"), exist_ok=True)
    
    # Iterate over the file names
    counter = 0
    for task in solved_tasks:
        # Check if the file is in the training directory
        training_path = os.path.join(task_training_dir, task)
        evaluation_path = os.path.join(task_evaluation_dir, task)
        
        if os.path.isfile(training_path):
            # File found in training, copy it to the target directory
            shutil.copy(training_path, os.path.join(target_dir, "training/", task))
            counter += 1
        elif os.path.isfile(evaluation_path):
            # File found in evaluation, copy it to the target directory
            shutil.copy(evaluation_path, os.path.join(target_dir, "evaluation/", task))
            counter += 1
        else:
            # File is not found in either of the source directories
            print(f"File '{task}' not found in both training and evaluation directories.")
    
    print(f"Copied {counter} / {len(solved_tasks)} files to '{target_dir}'.")


def check_model_selection(MODEL_NAMES, REVISIONS):
    for model_name, revision in zip(MODEL_NAMES, REVISIONS):
        print(model_name + ":" + revision)
    user_input = input("Do you want to continue running the script? (yes/no): ").lower().strip()
    if  user_input == 'yes':
        # Your script logic here
        print("Continuing the script...")
    else:
        print("Terminating script.")
        sys.exit(0)
