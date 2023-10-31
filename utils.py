
from config import *
import numpy as np
import json
import re
import matplotlib.pyplot as plt
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
from copy import deepcopy
import torch
from datasets import load_dataset, Dataset
from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate 
from auto_gptq import exllama_set_max_input_length
import openai



##################### Load Model and Tokenizer #####################

def load_llama(model_name, revision, max_token, model_config):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.model_max_length is None or tokenizer.model_max_length > 9999999999:
        tokenizer.model_max_length = max_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16, revision=revision
    )

    # fix bug for 70b model
    if model_name in ["TheBloke/Llama-2-70b-Chat-GPTQ", "TheBloke/Mistral-7B-v0.1-GPTQ"]:
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
        messages=messages
    )
    return response['choices'][0]['message']['content']

    

##################### Prompt Helper #####################

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

        solution = ""
        for i, row in enumerate(sample["output"]):
            solution += delimiter["grid_start"]
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
def data_generator(model_name, directory_train, directory_eval, pre_context, post_context, delimiter, prompt_template, sys, output_format, instruction_end, tokenizer, change_representation=False, new_representation= None):
    # get list of files and respective directories
    directories = [directory_train]*len(os.listdir(directory_train)) + [directory_eval]*len(os.listdir(directory_eval))
    task_files =  sorted(os.listdir(directory_train))+sorted(os.listdir(directory_eval))
    # initialize counter for too long prompts
    promp_oversize_counter = 0
    # iterate over files
    for directory, task_file in zip(directories, task_files):
        with open(os.path.join(directory, task_file)) as fid:
            task_json = json.load(fid)
            
        # change numbers to other representation if wanted
        if change_representation:
            task_json = change_color_representation(task_json, new_representation)

        # create context
        context = pre_context
        context += get_context(task_json, delimiter)
        context += post_context
        # get test cases + solutions
        test_cases, solutions = get_tasks(task_json, delimiter)
        # check if prompt of longest task is too long
        index_of_longest_prompt = max(enumerate(test_cases), key=lambda x: len(x[1]))[0]
        token_limit = tokenizer.model_max_length
        if  len(tokenizer.encode(sys+str(output_format)+context+test_cases[index_of_longest_prompt]+instruction_end, add_special_tokens=True)) > token_limit:
            print(task_file, "Prompt too long.")
            promp_oversize_counter += 1
            continue
          
        # yield prompts
        for (i, test_case), solution in zip(enumerate(test_cases), solutions):
            # distinguish between llama and gpt model prompt
            if "gpt" in model_name:
                prompt_llama = ""
                prompt_gpt = [
                    {"role": "system", "content": prompt_template[0].format(sys=sys, output_format=output_format)},
                    {"role": "user", "content": prompt_template[1].format(task=context+test_case)}
                ]
            else:
                prompt_llama = prompt_template.format(sys=sys, output_format=output_format, task=context+test_case, instruction_end=instruction_end)
                prompt_gpt = ""      
            yield {
                "task_name": task_file,
                "test_case_index": i,
                "total_test_cases": len(test_cases),
                "test_case": test_case,
                "context": context,
                "prompt_llama": prompt_llama,
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
