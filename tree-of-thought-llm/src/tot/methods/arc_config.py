from langchain.prompts import PromptTemplate 

#################### General ####################
GPU = '6,7'

#################### OPEN SOURCE ###############
MAX_TOKEN = 4096
MODEL_NAMES = []
REVISIONS = []
#### Llama Chat ####
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

#### Llama pre-trained ####
# MODEL_NAMES.append("TheBloke/Llama-2-70B-GPTQ") # TODO: Run all tests )
# REVISIONS.append("main")
# # MODEL_NAMES.append("TheBloke/Llama-2-13B-GPTQ") # TODO: Run all tests )
# REVISIONS.append("main")
# MODEL_NAMES.append("TheBloke/Llama-2-7B-GPTQ") # TODO: Run all tests )
# REVISIONS.append("main")

#### Platypus2 ####
# MODEL_NAMES.append("garage-bAInd/Platypus2-70B") --> dauert lange und braucht tausend GPUs?! liegt vielleicht an dem 16float oder so)
# MODEL_NAMES.append("TheBloke/Platypus2-70B-GPTQ") 
# REVISIONS.append("main")
# MODEL_NAMES.append("TheBloke/Platypus2-13B-GPTQ") 
# REVISIONS.append("main")
# MODEL_NAMES.append("TheBloke/Camel-Platypus2-70B-GPTQ") 
# REVISIONS.append("main")

#### Mistral ####
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

#### LLAMA CONFIG ####
MODEL_CONFIG_LLAMA = {
    'max_new_tokens': 1024,
    'temperature': 0.001,
    'repetition_penalty': 1.15,
}

#### Falcon ####
# MODEL_NAMES.append("TheBloke/Falcon-7B-Instruct-GPTQ") # TODO: Run all tests )
# REVISIONS.append("model")
# MODEL_NAMES.append("TheBloke/Falcon-40B-Instruct-GPTQ") # TODO: Run all tests )
# REVISIONS.append("model")
# MODEL_NAMES.append("TheBloke/Falcon-180B-Chat-GPTQ") # TODO: Run all tests )
# REVISIONS.append("main")

##### Falcon CONFIG ####
MODEL_CONFIG_FALCON = {
    'max_new_tokens': 1024,
    'temperature': 0.001,
}

#################### CLOSED SOURCE #############
MODEL_NAMES.append('gpt-3.5-turbo')
# MODEL_NAMES.append('gpt-4')
REVISIONS.append("")
MANUAL_GPT = False
#################### CONFIG ####################
MODEL_CONFIG_GPT = {
    'model_name': MODEL_NAMES[0],
    'temperature': 0.001,
}

#################### Prompt ####################
CHANGE_REPRESENTATION = True
NEW_REPRESENTATION = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]

POST_TEST_CASE = ""
DELIMITER = {
    "item": ", ",
    "grid_start": "[",
    "grid_end": "]]\n", # include end of last row
    "row_start": "[",
    "row_end": "], ", # except for last row
    "example_start": "Example:\n",
    "example_end": "\n",
    "task_start": "",
    "task_end": "",
    "input_train": "train input:\n",
    "output_train": "train output:\n",    
    "input_test": "test input:\n",
    "output_test": "", 
}


#################### Directories ####################

# TASK_DIR_TRAIN = "../ARC/ARC/data/training"
# TASK_DIR_EVAL = "../ARC/ARC/data/evaluation"

# TASK_DIR_TRAIN = "ARC_datasets/ARC_solved_tasks/training/"
# TASK_DIR_EVAL = "ARC_datasets/ARC_solved_tasks/evaluation/"

# TASK_DIR_TRAIN = "ARC_datasets/ARC_only_two_tasks/training/"
# TASK_DIR_EVAL = "ARC_datasets/ARC_only_two_tasks/evaluation/"

TASK_DIR_TRAIN = "ARC_datasets/LARC/training/"
TASK_DIR_EVAL = "ARC_datasets/LARC/evaluation/"

######## TODO: DELETE ########
# TASK_DIR_TRAIN = "ARC_datasets/test_mistral_gptq/training/"
# TASK_DIR_EVAL = "ARC_datasets/test_mistral_gptq/evaluation/"
