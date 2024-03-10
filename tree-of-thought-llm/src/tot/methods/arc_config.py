from langchain.prompts import PromptTemplate 

#################### General ####################
GPU = '2,3,6'

#################### Prompt ####################
CHANGE_REPRESENTATION = False
NEW_REPRESENTATION = [".", "a", "b", "c", "d", "e", "f", "g", "h", "i"]

#################### Model ####################
MODEL_NAMES = []
REVISIONS = []
##### Open-Source #####
# Llama Chat
# MODEL_NAMES.append("meta-llama/Llama-2-7b")
# fine-tuned by meta 
# MODEL_NAMES.append("TheBloke/Llama-2-70b-Chat-GPTQ")
# REVISIONS.append("main")
# MODEL_NAMES.append("TheBloke/Llama-2-13B-chat-GPTQ") # TODO: Run all tests)
# REVISIONS.append("main")
# MODEL_NAMES.append("NousResearch/Llama-2-7b-chat-hf") # TODO: TODO: Replace with Bloke's model & see if differences?!)
# REVISIONS.append("main")
# MODEL_NAMES.append("TheBloke/Llama-2-7B-chat-GPTQ") # TODO: Run all tests)
# fine-tuned by others
# MODEL_NAMES.append("TheBloke/Llama-2-7B-32K-Instruct-GPTQ") # TODO: Run all tests)

# Llama pre-trained
# MODEL_NAMES.append("TheBloke/Llama-2-70B-GPTQ") 
# REVISIONS.append("main")
# MODEL_NAMES.append("TheBloke/Llama-2-13B-GPTQ") 
# REVISIONS.append("main")
# MODEL_NAMES.append("TheBloke/Llama-2-7B-GPTQ") 
# REVISIONS.append("main")

# Platypus2 
# MODEL_NAMES.append("garage-bAInd/Platypus2-70B") --> dauert lange und braucht tausend GPUs?! liegt vielleicht an dem 16float oder so)
# MODEL_NAMES.append("TheBloke/Platypus2-70B-GPTQ") 
# REVISIONS.append("main")
# MODEL_NAMES.append("TheBloke/Platypus2-13B-GPTQ") 
# REVISIONS.append("main")
# MODEL_NAMES.append("TheBloke/Camel-Platypus2-70B-GPTQ") 
# REVISIONS.append("main")

# Mistral
# MODEL_NAMES.append("mistralai/Mistral-7B-Instruct-v0.1")
# REVISIONS.append("main")
# MODEL_NAMES.append("mistralai/Mistral-7B-v0.1")
# REVISIONS.append("main")
# MODEL_NAMES.append("TheBloke/Mistral-7B-v0.1-GPTQ") # TODO: TODO: Replace with Bloke's model & see if differences?!)
# REVISIONS.append("main")
# MODEL_NAMES.append("TheBloke/Mistral-7B-v0.1-GPTQ") # TODO: TODO: Replace with Bloke's model & see if differences?!)
# REVISIONS.append("gptq-4bit-32g-actorder_True")
# MODEL_NAMES.append("TheBloke/Mistral-7B-Instruct-v0.1-GPTQ") # TODO: TODO: Replace with Bloke's model & see if differences?!)
# REVISIONS.append("main")
# MODEL_NAMES.append("mistralai/Mixtral-8x7B-v0.1")
# REVISIONS.append("main")
MODEL_NAMES.append("mistralai/Mixtral-8x7B-Instruct-v0.1")
REVISIONS.append("main")
# MODEL_NAMES.append("TheBloke/Mixtral-8x7B-v0.1-GPTQ")
# REVISIONS.append("main") 
# MODEL_NAMES.append("TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ") 
# REVISIONS.append("main")
# MODEL_NAMES.append("TheBloke/Nous-Hermes-2-Mixtral-8x7B-DPO-GPTQ")
# REVISIONS.append("main")

# Vicuna
# MODEL_NAMES.append("TheBloke/vicuna-7B-v1.5-GPTQ")
# REVISIONS.append("main")
# MODEL_NAMES.append("TheBloke/vicuna-13B-v1.5-GPTQ")
# REVISIONS.append("main")

# Qwen 
# MODEL_NAMES.append("Qwen/Qwen-7B-Chat-Int4")
# REVISIONS.append("main") 
# MODEL_NAMES.append("Qwen/Qwen-14B-Chat-Int4") # Too small context length
# REVISIONS.append("main")
# MODEL_NAMES.append("Qwen/Qwen-72B-Chat-Int4")
# REVISIONS.append("main") 

# Falcon 
# MODEL_NAMES.append("TheBloke/Falcon-7B-Instruct-GPTQ") # TODO: Run all tests )
# REVISIONS.append("model")
# MODEL_NAMES.append("TheBloke/Falcon-40B-Instruct-GPTQ") # TODO: Run all tests )
# REVISIONS.append("model")
# MODEL_NAMES.append("TheBloke/Falcon-180B-Chat-GPTQ") # TODO: Run all tests )
# REVISIONS.append("main")

##### Proprietary #####
# MODEL_NAMES.append('gpt-4-1106-preview')
# REVISIONS.append('')
# MODEL_NAMES.append('gpt-3.5-turbo-1106')
# REVISIONS.append('')


#################### automatic completion of parameters ####################
MODEL_CONFIGS = {}
for model in MODEL_NAMES:
    config = {
        'model_config': {
                'max_new_tokens': 2048,
                'temperature': 0.001,   
        },
        'max_token': 4096,
    }
    if not "falcon" in model.lower():
        config["model_config"]["repetition_penalty"] = 1.15
    # change max_token of context
    if "gpt-4-1106-preview" in model.lower():
        config["max_token"] = 128000
    elif "gpt-3.5-turbo-1106" in model.lower():
        config["max_token"] = 16385
    elif "Qwen".lower() in model.lower():
        config["model_config"]["temperature"] = 0.7
        if "Qwen-7B".lower() in model.lower():
            config["max_token"] = 8192
        elif "Qwen-14B".lower() in model.lower():
            config["max_token"] = 2048
        elif "Qwen-72B".lower() in model.lower():
            config["max_token"] = 32768
    elif "Falcon".lower() in model.lower():
        config["max_token"] = 2048
    elif "Mixtral".lower() in model.lower():
        config["max_token"] = 32768
        # TODO: für TOT Runs! 
        # config["model_config"]["temperature"] = 0.7
    MODEL_CONFIGS[model] = config

DELIMITER = {
    "arc": {
        "item": ", ", # TODO: add apostroph if needed
        "grid_start": "[",
        "grid_end": "]]\n", # include end of last row # TODO: add apostroph if needed
        "row_start": "[", # TODO: add apostroph if needed
        "row_end": "], ", # except for last row # TODO: add apostroph if needed
        "example_start": "Example_X", # If "Example_X" -> automatically adds example number and \n: 'Example_1\n'
        "example_end": "\n",
        "task_start": "Test case:\n",
        "task_end": "",
        "input_train": "input: ",
        "output_train": "output: ",    
        "input_test": "input: ",
        "output_test": "",
    },
    "arc_1D": {
        "item": ", ", # TODO: add apostroph if needed
        "grid_start": "[",
        "grid_end": "]\n", # include end of last row # TODO: add apostroph if needed
        "row_start": "", # TODO: add apostroph if needed
        "row_end": "", # except for last row
        "example_start": "Example_X", # If "Example_X" -> automatically adds example number and \n: 'Example_1\n'
        "example_end": "\n",
        "task_start": "Test case:\n",
        "task_end": "",
        "input_train": "input: ",
        "output_train": "output: ",    
        "input_test": "input: ",
        "output_test": "", 
    }    
}

