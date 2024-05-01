from tot.methods.arc_config import * 
from tot.methods.credentials import *
import numpy as np
import itertools
import json
import re
import ast
import shutil
import datetime
from collections import deque
import sys
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
from copy import deepcopy
import torch
import tiktoken
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline
from auto_gptq import exllama_set_max_input_length, AutoGPTQForCausalLM
import openai



##################### Load Model and Tokenizer + count Tokens #####################

def load_llama(model_name, revision, max_token, model_config):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.model_max_length is None or tokenizer.model_max_length > 9999999999:
        tokenizer.model_max_length = max_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16, revision=revision
    )

    # # fix bug for certain models - fixed in new Optimum version
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

    # make pipeline compatible with langchain and return
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
        elif "gpt-4-1106-preview" in model_name:
            token_limit = 128000
        elif "gpt-3.5-turbo-1106" in model_name:
            token_limit = 8192
    else: 
        num_tokens = len(tokenizer.encode(prompt, add_special_tokens=True))
        token_limit = tokenizer.model_max_length
    return num_tokens, token_limit

def replace_quotes_in_text(res, json_format):  
    # preprocess json format
    output_format = {}
    for key, value in json_format.items():
        if "example_1" in json_format:
            for i in range(2, 11): # do for 10 examples
                k = "example_" + str(i)
                output_format.update({k: ""})
        if "Example_1" in json_format:
            for i in range(2, 11): # do for 10 examples
                k = "Example_" + str(i)
                output_format.update({k: ""})
        if "example_1_description" in json_format:
            for i in range(2, 11): # do for 10 examples
                k = "example_" + str(i) + "_description"
                output_format.update({k: ""})
        output_format.update({key: ""})
        if isinstance(value, dict):
            if "example_1" in value:
                for i in range(2, 11): # do for 10 examples
                    k = "example_" + str(i)
                    output_format.update({k: ""})
            if "Example_1" in value:
                for i in range(2, 11): # do for 10 examples
                    k = "Example_" + str(i)
                    output_format.update({k: ""})
            if "example_1_description" in value:
                for i in range(2, 11): # do for 10 examples
                    k = "example_" + str(i) + "_description"
                    output_format.update({k: ""})
            for key2, value2 in value.items():
                output_format.update({key2: ""})
                if isinstance(value2, dict):
                    if "example_1" in value2:
                        for i in range(2, 11): # do for 10 examples
                            k = "example_" + str(i)
                            output_format.update({k: ""})
                    if "Example_1" in value:
                        for i in range(2, 11): # do for 10 examples
                            k = "Example_" + str(i)
                            output_format.update({k: ""})
                    if "example_1_description" in value2:
                        for i in range(2, 11): # do for 10 examples
                            k = "example_" + str(i) + "_description"
                            output_format.update({k: ""})
                    for key3, value3 in value2.items():
                        output_format.update({key3: ""})   
        keys = list(output_format.keys())
    # add some potential artificially created keys from the model
    keys += ["choice", "test_case", "test case", "test_output", "test output", "test input", "test_input"]

    # correct backslashes "\_"
    res = res.replace("\_", "_")

    # do some regex to remove unwanted line breakes
    res = res.replace("\n", " ")
    
    # check if this is already enough procesing:
    try: 
        json.loads(res)
        return res
    except:
        pass
    
    # do some regex to remove unwanted single aprostrophes
    res = res.replace("'", '"')

    # replace any color name enclosed in double quotation marks to single quotation marks
    # pattern = r'"([^\s"]+)"'
    
    pattern = r'"((?:(?!np\.array|numpy\.array)[^"\s])+)"'
    res = re.sub(pattern, r"'\1'", res)
    pattern = r'(\': \s*)\'(\w+)\'(, \s*\')'
    res = re.sub(pattern, r'\1"\2"\3', res)

    # replace only single aprostrophe at the end of a word
    pattern = r'\b(?<!")(\w+)"\s*(?!\s*(,|}))'
    res = re.sub(pattern, r'\1 ', res)

    # add back double quotes to header names
    def replace_match(match):
        # Check the preceding character
        preceding_char = match.group(1)
        if preceding_char in ['{', ',']:
            # If it's '{' or ',', return the match without replacement
            return match.group(0)
        else:
            # Otherwise, replace 'key' # ], ] "objec
            new_string = match.group(1)+","+str(match.group(0))[1:]
            return new_string
    for key in keys:
        pattern = fr"'({key}(?:_\d+)?)'"
        res = re.sub(pattern, r'"\1"', res)
        pattern = r'(.)\s*"' + re.escape(key)
        res = re.sub(pattern, replace_match, res)

    # check for wrong array '"output": '['.", '.'...
    def replace_apostrophes(match):
        before = match.group(2)  # The text between the single apostrophe and the next key or ending sequence
        after = before.replace('"', "'")  # Replace single apostrophes with double
        return f'"{match.group(1)}": "{after}{match.group(3)}'    
    keys_pattern = '|'.join([re.escape(key) for key in keys])  # Escape each key and join with '|'
    pattern = rf'"({keys_pattern})":\s*\'(.*?)("(?:, "({keys_pattern})")|\s*"\s*\}})'
    res = re.sub(pattern, replace_apostrophes, res)   

    # ensure that we don't replace away aprostophes in text 
    res = re.sub(r"(\w)\"(\w)", r"\1'\2", res)

    # add double quotes when we have a single number als field value
    pattern = r'(": )\'(\d+)\'(,|})'
    res = re.sub(pattern, r'\1"\2"\3', res)
    
    # replace any characters with a backslash away, except \n and \t
    pattern = r"(\\[^nt])"
    res = re.sub(pattern, "", res)

    # in case the model outputs the string "np.array" to indicate such an object
    res = res.replace("import numpy as np", "")
    pattern = r'"(?:np|numpy)\.array\(([^)]*?)\)"'
    res = re.sub(pattern, r'\1', res)
    pattern = r'(?:np|numpy)\.array\(([^)]*?)\)'
    res = re.sub(pattern, r'"\1"', res)

    # In case any output is an array but with letters w/o double quotes
    for k in keys:
        pattern = r'('+k+'":\s*)(\[.*?)(],\s*")('+'|'.join(keys)+r'")'
        res = re.sub(pattern, lambda m: m.group(1) +'"'+ str(m.group(2)).replace('"', "'") +']", "'+ m.group(4), res)
        pattern = r'('+k+'":\s*)(\[.*?)(})'
        res = re.sub(pattern, lambda m: m.group(1) +'"'+ str(m.group(2)).replace('"', "'") +'"'+ m.group(3), res)

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
        # Find the start and end of the JSON segment in the string
        json_start = string.find("{")
        json_end = string.rfind("}") + 1
        
        # Extract the JSON-like segment           
        list_of_jsons.append(string[json_start:json_end])
        indices.append((json_start, json_end))
        try:
            string = string[json_start:json_end]
            return json.loads(string)
        except:
            pass
        previous_segment = None
        for i, json_segment in reversed(list(enumerate(list_of_jsons))):
            if previous_segment:
                json_segment = json_segment[:indices[i+1][0]+1] + previous_segment + json_segment[indices[i+1][1]+1:]
            try:
                json_segment = json.loads(json_segment)
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
    path = "error_log/json_parsing_errors/"+current_datetime.strftime("%Y-%m-%d_%H-%M-%S")+".txt"
    with open(path, "w") as text_file:
        text_file.write(log)
    return path+"\n\n"+log+error_msg

def find_key(dictionary, target_key):
    for key, value in dictionary.items():
        if key == target_key:
            return True, [target_key]
        elif isinstance(value, dict):
            result, keys = find_key(value, target_key)
            keys.insert(0, key)
            if result:
                return result, keys
    return False, []

def extract_json_value(string, json_format, keys):
    data = get_json_from_text(string, json_format)
    if isinstance(data, str): # error in json parsing
        # get path from beginning of string
        path = data.split(".txt")[0]+".txt"
        # get error from end of string
        data = data.split(".txt")[-1]
        data += f'Key to extract:\n{keys}'
        with open(path, "w") as text_file:
            text_file.write(data)
        return None
    
    # if only one key is given, make it a list
    # if a list of keys is given, use this list to find data, starting from first potential key
    if isinstance(keys, str):
        keys = [keys]    

    # Check if the key exists in the JSON, also check nested dictionaries
    for key in keys:
        key_exists, key_path = find_key(data, key)
        if key_exists:
            for next_key in key_path:
                data = data[next_key]
            break
    
    # Return the value for the given key or entire dictionar if not found
    if isinstance(data, str):
        # in case the model outputs the string "np.array" to indicate such an object
        data = data.replace("import numpy as np", "")
        pattern = r'"(?:np|numpy)\.array\(([^)]*?)\)"'
        data = re.sub(pattern, r'\1', data)
        pattern = r'(?:np|numpy)\.array\(([^)]*?)\)'
        data = re.sub(pattern, r'\1', data)
    return data

def extract_dict_keys(d, target, keys=set(), found=False):
    for key, value in d.items():
        if found:
            keys.add(key)
        if key == target:
            found = True
        if isinstance(value, dict):
            extract_dict_keys(value, target, keys, found)
    return list(keys)

def get_int_from_dict_value(d, key):
    try:
        value = d[key]
        if isinstance(value, str):
            pattern = r"\d+"
            return int(re.findall(pattern, value)[0])
        elif isinstance(value, int):
            return value
        elif isinstance(value, float):
            return int(value) 
    except:
        print(f'Key ({key}) not found in dict: {d}')
    return None

def get_thought(LLM_answer, prompt_modules, current_step, isRevision=False):
    if isRevision:
        output_format = prompt_modules[str(current_step)]["revision"]["analysis"]["output_format"]
    else:
        output_format = prompt_modules[str(current_step)]["generation"]["output_format"]
    thought_key = list(output_format.keys())[-1] # new thought is always last item in dict
    thought_data = extract_json_value(LLM_answer, output_format, thought_key)
    if isinstance(thought_data, dict):
        thought = " ".join(thought_key.split("_")) + ":"
        for key, value in thought_data.items():
            thought += f'\n{" ".join(key.split("_"))}: {value}'
        thought += "\n"
    else:
        thought = "\n" + " ".join(thought_key.split("_")) + ": "
        thought += f'{thought_data}'
    return thought

def get_previous_thoughts(node, climbing_layers=-1):
    thoughts = ""
    while True:
        if climbing_layers == 0:
            break
        if node.thought != "":
            thoughts = f'{node.thought}\n' + thoughts
        node = node.parent
        if node is None:
            break
        climbing_layers -= 1
    return thoughts

# function to ensure correct Example numbering, when using abstraction revision on Examples
current_number = 1  # Starting number for incrimination
def incremental_replace(match):
    global current_number
    result = f'Example {current_number}'
    current_number += 1
    return result

##################### Prompt Helper #####################
# read multi line user inputs 
def read_multiline_input(query):
    lines = []
    eos = "<end>"
    print(query)
    print(f'type {eos} after answer.')

    while True:
        line = input()
        if line == eos:
            break
        lines.append(line)

    text = "\n".join(lines)
    return text

# get prompt templates/modules
def get_prompts(dataset="arc"):
    if dataset=="arc":
        import tot.prompts.arc as prompts
    elif dataset=="arc_1D":
        import tot.prompts.arc_1D as prompts # TODO: use ARC prompts
    arc_prompts = {
        "standard_prompt": prompts.standard_prompt,
        "cot_prompt": prompts.cot_prompt,
        "vote_prompt": prompts.vote_prompt,
        "value_prompt": prompts.value_prompt,
        "prompt_modules": prompts.prompt_modules
        }
    return arc_prompts

# load tasks
def load_arc_tasks(path, dataset="arc"):
    # load data 
    tasks_jsons = []
    tasks_names = []
    paths = []

    if dataset == "arc":
        # train and test path
        paths.append(os.path.join(path, "training"))
        paths.append(os.path.join(path, "evaluation"))
    elif dataset in ["arc_1D", "arc_h_v"]:
        paths = [os.path.join(path, f.name) for f in os.scandir(path) if f.is_dir()]
    
    subdirecotries = []
    for path in paths:
        subdirecotry = path.split("/")[-1]
        for task_file in sorted(os.listdir(path)):
            with open(os.path.join(path, task_file)) as fid:
                task_json = json.load(fid)
            tasks_jsons.append(task_json)
            tasks_names.append(task_file)
            subdirecotries.append(subdirecotry)

    print("Total number of tasks:", len(tasks_jsons))
    return tasks_jsons, tasks_names, subdirecotries

# Find objects in pixel grids
def find_objects(task, name, grid, bg_color):
    def bfs(start_row, start_col, color):
        # Start BFS from the given cell
        queue = deque([(start_row, start_col)])
        components = [[start_row, start_col]]
        size = 1
        while queue:
            row, col = queue.popleft()
            for dr, dc in directions:
                r, c = row + dr, col + dc
                if 0 <= r < rows and 0 <= c < cols and not visited[r][c] and grid[r][c] == color:
                    visited[r][c] = True
                    queue.append((r, c))
                    components.append([r, c])
                    size += 1
        return {'color': color, 'coordinates': components, 'size': size}
    def find_inner_blank_areas(objects, bg_color):
        def flood_fill(r, c, fill_val, bg_color):
            if r < 0 or r > height-1 or c < 0 or c > width-1 or marked_grid[r][c] != 0:
                return
            # Mark the cell with fill_val
            marked_grid[r][c] = fill_val
            # Recursively fill the neighboring cells
            flood_fill(r - 1, c, fill_val, bg_color)
            flood_fill(r + 1, c, fill_val, bg_color)
            flood_fill(r, c - 1, fill_val, bg_color)
            flood_fill(r, c + 1, fill_val, bg_color)
        new_objects = []
        for obj in objects:
            # Determine the bounding box for the coordinates
            min_row = min(coord[0] for coord in obj['coordinates'])
            max_row = max(coord[0] for coord in obj['coordinates'])
            min_col = min(coord[1] for coord in obj['coordinates'])
            max_col = max(coord[1] for coord in obj['coordinates'])

            # Create a grid for the bounding box
            height = max_row - min_row + 1
            width = max_col - min_col + 1
            marked_grid  = [[0] * width for _ in range(height)]

            # Mark the object's coordinates in the grid
            for r, c in obj['coordinates']:
                marked_grid [r - min_row][c - min_col] = -1

            inner_coordinates = []
            found_pixel = True
            for r in range(1, height - 1):
                if not found_pixel:
                    break
                for c in range(1, width - 1):
                    if not found_pixel:
                        break
                    if marked_grid[r][c] == 0:  # This is part of the inner blank area
                        flood_fill(r, c, 2, bg_color)  # Fill with a new unique number (2)
                        # Collect all coordinates filled with 2
                        for i in range(height):
                            if not found_pixel:
                                break
                            found_pixel = False
                            for j in range(width):
                                if marked_grid[i][j] == 2:
                                    inner_coordinates.append([i + min_row, j + min_col])
                                    found_pixel = True    
                            if len(inner_coordinates) == 0 or ("a5313dff" in name):
                                found_pixel = True
                        break
            
            # check if other object inside
            for o in objects:
                for coords_other in o["coordinates"]:
                    for i, coords in enumerate(inner_coordinates):
                        if coords == coords_other:
                            del inner_coordinates[i]
            # check for duplicates
            if "a5313dff" in name:
                inner_coordinates = list(set([tuple(coord) for coord in inner_coordinates]))
                inner_coordinates = [list(coords) for coords in inner_coordinates]
                inner_coordinates.sort()
                if len(inner_coordinates) == 0:
                    break
                # delete outside pixels
                outside_pixels = []
                for r, c in inner_coordinates:
                    if r == 11 and (c==8 or c==10):
                        continue
                    if r-min_row == height-1 or c-min_col == width-1 or r-min_row == 0 or c-min_col == 0:
                        outside_pixels.append([r,c])
                indices_to_delete = []
                for k in range(3):
                    for i, coords in enumerate(inner_coordinates):
                        for coords_outer in outside_pixels:
                            if coords == coords_outer or abs(coords_outer[0] - coords[0]) + abs(coords_outer[1] - coords[1]) == 1:
                                indices_to_delete.append(i)
                                outside_pixels.append(coords)
                                break
                indices_to_delete = list(set(indices_to_delete))
                indices_to_delete.reverse()
                for index in indices_to_delete:
                    del inner_coordinates[index]
                # check if other object inside
                for o in objects:
                    for coords_other in o["coordinates"]:
                        for i, coords in enumerate(inner_coordinates):
                            if coords == coords_other:
                                del inner_coordinates[i]
                # segment into multiple objects and keep last
                if len(inner_coordinates) == 0: 
                    return []
                new_objects = [[inner_coordinates[0]]]
                # new_object = [inner_coordinates[0]]
                for r, c in inner_coordinates[1:]:
                    is_neighbor = False
                    for new_object in new_objects:
                        for ro, co in new_object:
                            if abs(r - ro) + abs(c - co) == 1:
                                new_object.append([r, c])
                                is_neighbor = True
                                break
                    if not is_neighbor:
                        new_objects.append([[r,c]])
                new = []
                for o in new_objects:
                    new.append({
                        'coordinates': o,
                        'color': bg_color,
                        'size': len(o)
                    })
                return new
            else:
                coord_set = set([tuple(coord) for coord in inner_coordinates])
                if len(coord_set) != len(inner_coordinates):
                    continue
            # check if actually surrounded by colored pixels
            surrounded = True
            for r, c in inner_coordinates:
                r = r - min_row
                c = c - min_col
                if r < 1 or r > height-2 or c < 1 or c > width-2:
                    surrounded = False
                    break
            if inner_coordinates and surrounded:
                new_objects.append({
                    'coordinates': inner_coordinates,
                    'color': bg_color,
                    'size': len(inner_coordinates)
                })

        return new_objects
    def find_extra_pixel_and_fix(obj):
        # Extract the coordinates
        coordinates = obj['coordinates']
        # Determine the bounding box
        min_row = min(coord[0] for coord in coordinates)
        max_row = max(coord[0] for coord in coordinates)
        min_col = min(coord[1] for coord in coordinates)
        max_col = max(coord[1] for coord in coordinates)
        
        # Create a grid that represents the bounding box
        expected_grid = set(
            (r, c) for r in range(min_row, max_row + 1) for c in range(min_col, max_col + 1)
        )
        
        # Convert the original coordinates into a set
        original_coordinates = set(tuple(coord) for coord in coordinates)
        
        
        # Find the pixel that's not in the expected grid (the extra pixel)
        extra_pixels = list(expected_grid - original_coordinates)
        
        if len(extra_pixels) == 0:
            return [obj]
        
        if extra_pixels[0][0] == extra_pixels[1][0]:
            # extra pixel in respective row
            for i, coord in enumerate(obj["coordinates"]):
                if coord[0] == extra_pixels[0][0]:
                    c = obj["coordinates"].pop(i)
        else:
            # extra pixel in respective col
            for i, coord in enumerate(obj["coordinates"]):
                if coord[1] == extra_pixels[0][1]:
                    c = obj["coordinates"].pop(i) 
        obj["size"] -= 1
                            
        # Create the new object for the extra pixel
        new_object = {
            'color': obj['color'],  # Use the same color
            'coordinates': [c],  # This will have only one coordinate
            'size': 1  # Since it's a single-pixel object
        }
               
        return [obj, new_object]

    objects = []
    current_object = None
    if task == "arc_1D":
        for i, pixel in enumerate(grid[0]):
            if "denoising_mc" in name or "flip" in name:
            # use multi color objects
                if pixel != bg_color:
                    if current_object is None:
                        # Start a new object
                        current_object = {'color': [pixel], 'coordinates': [[0, i]], 'size': 1}
                    else:
                        # Continue the current object
                        current_object['color'].append(pixel)
                        current_object['coordinates'].append([0, i])
                        current_object['size'] += 1
            else:
            # all other tasks
                if pixel != bg_color:
                    if current_object is None:
                        # Start a new object
                        current_object = {'color': pixel, 'start_index': i, 'end_index': i, 'size': 1}
                    elif pixel == current_object['color']:
                        # Continue the current object
                        current_object['end_index'] = i
                        current_object['size'] += 1
                    else:
                        # Finish the current object and start a new one
                        objects.append(current_object)
                        current_object = {'color': pixel, 'start_index': i, 'end_index': i, 'size': 1}
                else:
                    if current_object is not None:
                        # Finish the current object
                        objects.append(current_object)
                        current_object = None
    elif task == "arc_h_v":
        if "arc2smr_v" in name:
            # fill vertical
            # Transpose grid to work column by column
            transposed_grid = [list(col) for col in zip(*grid)]
            
            outside_color = None
            # get outside color
            for col in transposed_grid:
                if outside_color is not None:
                    break
                for i, pixel in enumerate(col):
                    if pixel != bg_color:
                        outside_color = pixel
                        inside_color = col[i+1]
                        for j in range(i+1, len(col)):
                            if col[j] == outside_color:
                                step = j - i 
                                break
                        break
            # get objects per column
            for i, col in enumerate(transposed_grid):
                if outside_color not in col:
                    continue
                # Find index of first occurrence of outside color
                index = col.index(outside_color)
                # Find index of last occurrence of outside color
                last_index = len(col) - 1 - col[::-1].index(outside_color)
                while index < last_index:
                    objects.append({'color': outside_color, 'coordinates': [[index, i]], 'size': 1})
                    if inside_color is not None:
                        objects.append({'color': inside_color, 'coordinates': [], 'size': 1})
                        for j in range(step-1):
                            objects[-1]['coordinates'].append([index+j+1, i])
                    index = index + step
                objects.append({'color': outside_color, 'coordinates': [[last_index, i]], 'size': 1})
        elif "arc2smr_" in name and "arc2smr_v" not in name:
            # fill horizontal
            outside_color = None
            # get outside color
            for row in grid:
                if outside_color is not None:
                    break
                for i, pixel in enumerate(row):
                    if pixel != bg_color:
                        outside_color = pixel
                        inside_color = row[i+1]
                        for j in range(i+1, len(row)):
                            if row[j] == outside_color:
                                step = j - i 
                                break
                        break
                    
            # get objects per row
            for i, row in enumerate(grid):
                if outside_color not in row:
                    continue
                # Find index of first occurrence of outside color
                index = row.index(outside_color)
                # Find index of last occurrence of outside color
                last_index = len(row) - 1 - row[::-1].index(outside_color)
                while index < last_index:
                    objects.append({'color': outside_color, 'coordinates': [[i, index]], 'size': 1})
                    objects.append({'color': inside_color, 'coordinates': [], 'size': 1})
                    for j in range(step-1):
                        objects[-1]['coordinates'].append([i, index+j+1])
                    index = index + step
                objects.append({'color': outside_color, 'coordinates': [[i, last_index]], 'size': 1})
        elif "arc_3906de3d_v" in name:
            # move vertical
            # Transpose grid to work column by column
            transposed_grid = [list(col) for col in zip(*grid)]
            for i, col in enumerate(transposed_grid):
                coordinates_left = []
                coordinates_right = []
                if col[0] != bg_color:
                    coordinates_left.append([0,i])
                    pixel_left = col[0]
                    for j in range(1,len(col)):
                        if col[j] == col[0]:
                            coordinates_left.append([j,i])
                            pixel_left = col[j]
                        elif col[j] != bg_color:
                            coordinates_right.append([j,i])
                            pixel_right = col[j]
                    if len(coordinates_left) > 0:
                        objects.append({'color': pixel_left, 'coordinates': coordinates_left, 'size': len(coordinates_left)})
                    if len(coordinates_right) > 0:
                        objects.append({'color': pixel_right, 'coordinates': coordinates_right, 'size': len(coordinates_right)})
        elif "arc_3906de3d_h" in name:
            # move horizontal
            for i, row in enumerate(grid):
                coordinates_left = []
                coordinates_right = []
                if row[0] != bg_color:
                    coordinates_left.append([i,0])
                    pixel_left = row[0]
                    for j in range(1,len(row)):
                        if row[j] == row[0]:
                            coordinates_left.append([i,j])
                            pixel_left = row[j]
                        elif row[j] != bg_color:
                            coordinates_right.append([i,j])
                            pixel_right = row[j]
                    if len(coordinates_left) > 0:
                        objects.append({'color': pixel_left, 'coordinates': coordinates_left, 'size': len(coordinates_left)})
                    if len(coordinates_right) > 0:
                        objects.append({'color': pixel_right, 'coordinates': coordinates_right, 'size': len(coordinates_right)})
                left_object = False
                for i, pixel in enumerate(row):
                    if not left_object and pixel != bg_color:
                        left_object = True
                        right_color = pixel
                        break
        elif "pile_h" in name:
            for i, row in enumerate(grid):
                current_object = None
                for j, pixel in enumerate(row):
                    if pixel != bg_color:
                        if current_object is None:
                            # Start a new object
                            current_object = {'color': pixel, 'coordinates': [[i, j]], 'size': 1}
                        elif pixel == current_object['color']:
                            # Continue the current object
                            current_object['coordinates'].append([i, j])
                            current_object['size'] += 1
                        else:
                            # Finish the current object and start a new one
                            objects.append(current_object)
                            current_object = {'color': pixel, 'coordinates': [[i, j]], 'size': 1}
                    else:
                        if current_object is not None:
                            # Finish the current object
                            objects.append(current_object)
                            current_object = None
                # Add the last object if it exists
                if current_object is not None:
                    objects.append(current_object)
                    current_object = None
        elif "pile_v" in name:
            # Transpose grid to work column by column
            transposed_grid = [list(col) for col in zip(*grid)]
            for i, col in enumerate(transposed_grid):
                current_object = None
                for j, pixel in enumerate(col):
                    if pixel != bg_color:
                        if current_object is None:
                            # Start a new object
                            current_object = {'color': pixel, 'coordinates': [[j, i]], 'size': 1}
                        elif pixel == current_object['color']:
                            # Continue the current object
                            current_object['coordinates'].append([j, i])
                            current_object['size'] += 1
                        else:
                            # Finish the current object and start a new one
                            objects.append(current_object)
                            current_object = {'color': pixel, 'coordinates': [[j, i]], 'size': 1}
                    else:
                        if current_object is not None:
                            # Finish the current object
                            objects.append(current_object)
                            current_object = None
                # Add the last object if it exists
                if current_object is not None:
                    objects.append(current_object)
                    current_object = None
    elif task == "arc":
        if "3906de3d" in name: 
            return find_objects("arc_h_v", "arc_3906de3d_v", grid, bg_color)
        if "a699fb00" in name:
            return find_objects("arc_h_v", "arc2smr_h", grid, bg_color)
        grid = np.array(grid)
        rows, cols = grid.shape
        visited = np.zeros((rows, cols), dtype=bool)
        objects = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if ("3c9b0459" in name) or ("74dd1130" in name) or ("ed36ccf7" in name) or ("6150a2bd" in name) or ("67a3c6ac" in name) or ("9dfd6313" in name):
            # single object of multiple colors
            current_object = {'color': [], 'coordinates': [], 'size': 0}
            for row in range(rows):
                for col in range(cols):
                    current_object["color"].append(grid[row][col])
                    current_object["coordinates"].append([row,col])
                    current_object["size"] += 1
            return [current_object]
        if "b2862040" in name:
            bg_color = 9
        if "88a10436" in name:
            # multicolor objects, separation by >=1 empty rows
            current_object = {'color': [], 'coordinates': [], 'size': 0}
            for row in range(rows):
                new_object = True
                for col in range(cols):
                    if grid[row][col] != bg_color:
                        new_object = False
                        current_object["color"].append(grid[row][col])
                        current_object["coordinates"].append([row,col])
                        current_object["size"] += 1 
                if new_object and current_object["size"] > 0: 
                    objects.append(current_object)
                    current_object = {'color': [], 'coordinates': [], 'size': 0}
            return objects
                         
        # Explore each cell in the grid
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] != bg_color and not visited[row][col]:
                    visited[row][col] = True
                    object_info = bfs(row, col, grid[row][col])
                    objects.append(object_info)
        if ("d5d6de2d" in name):
            if objects[0]["color"] == 2:
                objects += find_inner_blank_areas(objects, bg_color)
        if ("b2862040" in name) or ("810b9b61" in name) or ("a5313dff" in name):
            objects += find_inner_blank_areas(objects, bg_color)
        if "7f4411dc" in name: 
            new_objects = []
            for o in objects:
                if o["size"] > 1:
                    new_objects = new_objects+find_extra_pixel_and_fix(o)
                else:
                    new_objects.append(o)
            return new_objects    
    # Add the last object if it exists
    if current_object is not None:
        objects.append(current_object)
    return objects

# extract objects from the LLM output 
def extract_dicts_from_string(input_string):
    if isinstance(input_string, dict):
        if "object_1" in [k.lower() for k in list(input_string.keys())]:
            input_string = str(input_string)[1:-1] # without curly brackets
        input_string = str(input_string)
    elif isinstance(input_string, list):
        for i, element in enumerate(input_string):
            if isinstance(element, dict):
                if (f"object_{i}" in [k.lower() for k in list(element.keys())]) | (f"object_{i+1}" in [k.lower() for k in list(element.keys())]):
                    input_string[i] = str(element)[1:-1]
        input_string = str(input_string)
    else:
        input_string = str(input_string)
    
    # check if brackets are around objects
    if "{" not in input_string:
        # Insert "{" before "color:"
        input_string = re.sub(r'(?=color:)', '{', input_string)
        # Insert "}" after "size: [number]"
        input_string = re.sub(r'(size: \d{1,4})', r'\1}', input_string)
    
    # Define the pattern to match dictionaries within the string
    pattern = r'\{([^}]+)\}'
    
    # Find all matches of dictionaries in the input string
    matches = re.findall(pattern, input_string)
    
    # Initialize an empty list to store the extracted dictionaries
    extracted_dicts = []
    
    # check that all keys of dict have quotes
    def format_keys_with_quotes(match):
        # This regular expression pattern identifies keys that may or may not be wrapped in quotes.
        # It captures:
        # - Optional existing quote (' or ")
        # - The key (sequence of word characters)
        # - Another optional existing quote (matching the first)
        # This is modified to ensure that any quotes around the keys are consistent single quotes.
        formatted_match = re.sub(r'(?<!\S)(["\']?)(\w+)\1(?=\s*:)', r"'\2'", match)
        return formatted_match
    
    # Iterate over the matches and convert them to dictionaries
    for match in matches:
        # check dict format
        formatted_match = format_keys_with_quotes(match)
        # Use eval to convert the matched string to a dictionary
        extracted_dict = eval('{' + formatted_match + '}')
        if "color" in extracted_dict:
            if isinstance(extracted_dict["color"], str):
                pattern = r'^\d$'
                if re.match(pattern, extracted_dict["color"]):
                    extracted_dict["color"] = int(extracted_dict["color"])
        extracted_dicts.append(extracted_dict)
    
    return extracted_dicts

# compare two lists of objects
def compare_object_lists(l1, l2):
    list1 = l1.copy()
    list2 = l2.copy()
    # Check if the number of objects is the same
    if len(list1) != len(list2) or len(list1) == 0 or len(list2) == 0:
        return False
    
    # Define a function to sort coordinates lists
    def sort_coordinates(coords_list):
        return sorted(coords_list)
    
    # Iterate over each object in the first list
    for obj1 in list1:
        found_match = False
        # Iterate over each object in the second list
        for obj2 in list2:
            # Check if the dictionary keys match
            if set(obj1.keys()) != set(obj2.keys()):
                continue
            # Compare all dictionary values
            if all(obj1[key] == obj2[key] for key in obj1):
                # If a match is found, remove the object from list2
                list2.remove(obj2)
                found_match = True
                break
            
            # if no match, check if list within values are the same but diffrently ordered
            if 'coordinates' in obj1 and 'coordinates' in obj2:
                indices_obj1, obj1["coordinates"] = zip(*sorted(enumerate(obj1["coordinates"]), key=lambda x: x[1]))
                indices_obj2, obj2["coordinates"] = zip(*sorted(enumerate(obj2["coordinates"]), key=lambda x: x[1]))
                # if color is a list: multicolor object -> sort color accordingly
                if isinstance(obj1['color'],list):
                    obj1["color"] = [obj1["color"][i] for i in indices_obj1]
                if isinstance(obj2['color'],list):
                    obj2["color"] = [obj2["color"][i] for i in indices_obj2]
                # Compare all dictionary values again 
                if all(obj1[key] == obj2[key] for key in obj1):
                    # If a match is found, remove the object from list2
                    list2.remove(obj2)
                    found_match = True
                    break
                
        # If no match is found for any object, return False
        if not found_match:
            return False
    
    return True

# compare the dimensions of two grids, given as lists [rows, columns]
def compare_dimensions(test_output_dimension, gt_dimension):
    if isinstance(test_output_dimension, list) and isinstance(gt_dimension, list):
        if test_output_dimension == gt_dimension:
            return True 
    elif isinstance(test_output_dimension, str):
        # Find all matches of digits within square brackets
        matches = re.findall(r'\[(\d+),\s?(\d+)\]', test_output_dimension)
        try:
            # Convert matches to list of tuples of integers
            test_output_dimension = list(map(int, matches[0])) 
        except:
            test_output_dimension = None
        if test_output_dimension == gt_dimension:
            return True 
    return False

# create grid from object description
def create_grid_from_objects(output_object_description, CHANGE_REPRESENTATION, NEW_REPRESENTATION, task):
    if ("test_case_output_dimension" not in output_object_description) | ("transformed_objects" not in output_object_description):
        return None
    try:
        test_output_dimension = output_object_description["test_case_output_dimension"]
        test_output_objects = output_object_description["transformed_objects"]
        output_objects = extract_dicts_from_string(test_output_objects)
    
        if isinstance(test_output_dimension, str):
            array_start = test_output_dimension.find("[")
            array_end = test_output_dimension.rfind("]")
            test_output_dimension = test_output_dimension[array_start:array_end+1] 
            test_output_dimension = ast.literal_eval(test_output_dimension)

                
        # Initialize the grid
        rows = int(test_output_dimension[0])
        cols = int(test_output_dimension[1])
        if CHANGE_REPRESENTATION:
            grid = [[NEW_REPRESENTATION[0] for _ in range(cols)] for _ in range(rows)]
        else:
            grid = [[0 for _ in range(cols)] for _ in range(rows)]

        if task == "arc_1D":
            # Place objects on the grid
            for obj in output_objects:
                color = None
                start = None
                end = None
                coordinates = None
                if "color" in obj:
                    color = obj["color"]
                if "start_index" in obj:
                    start = obj["start_index"]
                if "end_index" in obj:
                    end = obj["end_index"]
                if color is not None and start is not None and end is not None:
                    for i in range(start, end + 1):
                        grid[0][i] = color
                if "coordinates" in obj:
                    coordinates = obj["coordinates"]
                if color is not None and coordinates is not None:                
                    for i, coord in enumerate(coordinates):
                        row = coord[0]
                        col = coord[1]
                        if isinstance(color, list):
                            grid[row][col] = color[i]
                        else:
                            grid[row][col] = color
            return grid
        else:
            # Place objects on the grid
            for obj in output_objects:
                color = None
                coordinates = None
                if "color" in obj:
                    color = obj["color"]
                if "coordinates" in obj:
                    coordinates = obj["coordinates"]
                if color is not None and coordinates is not None:
                    for i, coord in enumerate(coordinates):
                        row = coord[0]
                        col = coord[1]
                        if isinstance(color, list):
                            grid[row][col] = color[i]
                        else:
                            grid[row][col] = color
            return grid      
    except:
        return None 

# get context out of json
def get_context(task_name, task_json, delimiter, with_intro=True, use_object_representation=None):
    if with_intro:
        text = "The following input-output pairs are examples and share the same underlying transformation pattern.\n"
    else:
        text = ""
        
    for i, sample in enumerate(task_json["train"], 1):
        if delimiter["example_start"] == "Example_X":
            text += f"Example_{i}:\n"
        else:
            text += delimiter["example_start"]
        text += delimiter["input_train"]
        if use_object_representation:
            if CHANGE_REPRESENTATION:
                bg_color = NEW_REPRESENTATION[0]
            else:
                bg_color = 0
            text+= f"Dimension: {[len(sample['input']),len(sample['input'][0])]}, Objects: "
            objects = find_objects(task=use_object_representation, name=task_name, grid=sample["input"], bg_color=bg_color)
            for i, o in enumerate(objects, 1):
                text += "Object_" + str(i) + ": " + str(o) + ", "
            text = text[:-2] + "\n"
        else:
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
        if use_object_representation:
            text+= f"Dimension: {[len(sample['output']),len(sample['output'][0])]}, Objects: "
            objects = find_objects(task=use_object_representation, name=task_name, grid=sample["output"], bg_color=bg_color)
            for i, o in enumerate(objects, 1):
                text += "Object_" + str(i) + ": " + str(o) + ", "
            text = text[:-2] + "\n"
        else:
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
def get_tasks(task_name, task_json, delimiter, use_object_representation=False):
    tasks = []
    solutions = []
    
    for sample in task_json["test"]:
        task = delimiter["task_start"]
        task += delimiter["input_test"]
        if use_object_representation:
            if CHANGE_REPRESENTATION:
                bg_color = NEW_REPRESENTATION[0]
            else:
                bg_color = 0
            task += f"Dimension: {[len(sample['input']),len(sample['input'][0])]},  Objects: "
            objects = find_objects(task=use_object_representation, name=task_name, grid=sample["input"], bg_color=bg_color)
            for i, o in enumerate(objects, 1):
                task += "Object_" + str(i) + ": " + str(o) + ", "
            task = task[:-2] + "\n"
        else:
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
def data_generator(model_name, dataset, directories, delimiter, prompt_template, sys, output_format, pre_test_case, post_test_case, instruction_end, tokenizer, change_representation=False, new_representation=None):
    task_files = [sorted(os.listdir(os.path.join(dir,d))) for dir in directories for d in os.listdir(dir)]
    task_files = list(itertools.chain(*task_files))
    directories = [[os.path.join(dir,d)]*len(os.listdir(os.path.join(dir,d))) for dir in directories for d in os.listdir(dir)]
    directories = list(itertools.chain(*directories)) 
    categories = [d.split("/")[-1] for d in directories]

    # initialize counter for too long prompts
    promp_oversize_counter = 0
    # iterate over files
    for directory, task_file, category in zip(directories, task_files, categories):
        with open(os.path.join(directory, task_file)) as fid:
            task_json = json.load(fid)
        
        # if we load LARC data, we need to check if the task has been solved by humans
        if dataset == "LARC":
            descriptions, task_json = get_successful_descriptions(task_json)
            if len(descriptions) == 0:
                continue
        else:
            descriptions = [""]       
    
        # change numbers to other representation if wanted
        if change_representation:
            task_json = change_color_representation(task_json, new_representation)

        # create context
        if dataset == "LARC":
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
                    "task_json": task_json,
                    "category": category,
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
        if test_train  in ["test", "train"]:
            for sample in task[test_train]:
                for i, row in enumerate(sample["input"]):
                    for j, value in enumerate(row):
                        sample["input"][i][j] = new_representation[value]
                for i, row in enumerate(sample["output"]):
                    for j, value in enumerate(row):
                        sample["output"][i][j] = new_representation[value]
    
    return task

def grid_to_2D_nparray(grid):
    if isinstance(grid, str):
        # array in string
        for i in range(2):
            array_start = grid.find("[[")
            array_end = grid.rfind("]]") + 2
            if array_start == -1 or array_end == 1:
                if i == 1:
                    error = "No 2D-array found in final output string: " + grid
                    return error
                print("No 2D-array found, trying to add extra bracket: [..]")
                array_start = grid.find("[")
                array_end = grid.rfind("]")
                grid = grid[array_start:array_end+1]
                grid = "[" + grid.strip() + "]"
            else:
                break
        grid = grid[array_start:array_end] 
                 
        # Replace single letters with quotes around them
        pattern = re.compile(r'(?<![\'"])([a-zA-Z\.])(?![\'"])')
        grid = pattern.sub(r"'\1'", grid)  
        
        try:
            grid = ast.literal_eval(grid)
            return np.array(grid).astype(str)
        except:
            error = "Array found in string but error while converting string to array: " + str(grid)
            return error
    else:
        try: 
            arr = np.array(grid)
            # check if 1D, then add extra dimension
            if len(arr.shape) == 1:
                arr = np.expand_dims(arr, axis=0)            
            return arr.astype(str)
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
