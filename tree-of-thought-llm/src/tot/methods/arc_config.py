from langchain.prompts import PromptTemplate 

#################### General ####################
GPU = '0,4'

#################### OPEN SOURCE ###############
MAX_TOKEN = 4096
MODEL_NAMES = []
REVISIONS = []
#### Llama Chat ####
# MODEL_NAMES.append("meta-llama/Llama-2-7b")
# fine-tuned by meta 
MODEL_NAMES.append("TheBloke/Llama-2-70b-Chat-GPTQ")
REVISIONS.append("main")
MODEL_NAMES.append("TheBloke/Llama-2-13B-chat-GPTQ") # TODO: Run all tests)
REVISIONS.append("main")
MODEL_NAMES.append("NousResearch/Llama-2-7b-chat-hf") # TODO: TODO: Replace with Bloke's model & see if differences?!)
REVISIONS.append("main")
# MODEL_NAMES.append("TheBloke/Llama-2-7B-chat-GPTQ") # TODO: Run all tests)
# fine-tuned by others
# MODEL_NAMES.append("TheBloke/Llama-2-7B-32K-Instruct-GPTQ") # TODO: Run all tests)

#### Llama pre-trained ####
MODEL_NAMES.append("TheBloke/Llama-2-70B-GPTQ") 
REVISIONS.append("main")
MODEL_NAMES.append("TheBloke/Llama-2-13B-GPTQ") 
REVISIONS.append("main")
MODEL_NAMES.append("TheBloke/Llama-2-7B-GPTQ") 
REVISIONS.append("main")

#### Platypus2 ####
# MODEL_NAMES.append("garage-bAInd/Platypus2-70B") --> dauert lange und braucht tausend GPUs?! liegt vielleicht an dem 16float oder so)
MODEL_NAMES.append("TheBloke/Platypus2-70B-GPTQ") 
REVISIONS.append("main")
MODEL_NAMES.append("TheBloke/Platypus2-13B-GPTQ") 
REVISIONS.append("main")
# MODEL_NAMES.append("TheBloke/Camel-Platypus2-70B-GPTQ") 
# REVISIONS.append("main")

#### Mistral ####
MODEL_NAMES.append("mistralai/Mistral-7B-Instruct-v0.1")
REVISIONS.append("main")
MODEL_NAMES.append("mistralai/Mistral-7B-v0.1")
REVISIONS.append("main")
# MODEL_NAMES.append("TheBloke/Mistral-7B-v0.1-GPTQ") # TODO: TODO: Replace with Bloke's model & see if differences?!)
# REVISIONS.append("main")
# MODEL_NAMES.append("TheBloke/Mistral-7B-v0.1-GPTQ") # TODO: TODO: Replace with Bloke's model & see if differences?!)
# REVISIONS.append("gptq-4bit-32g-actorder_True")
# MODEL_NAMES.append("TheBloke/Mistral-7B-Instruct-v0.1-GPTQ") # TODO: TODO: Replace with Bloke's model & see if differences?!)
# REVISIONS.append("main")
# MODEL_NAMES.append("mistralai/Mixtral-8x7B-v0.1")
# REVISIONS.append("main")
# MODEL_NAMES.append("mistralai/Mixtral-8x7B-Instruct-v0.1")
# REVISIONS.append("main")
# MODEL_NAMES.append("TheBloke/Mixtral-8x7B-v0.1-GPTQ")
# REVISIONS.append("main") # TODO: Mixtral dann am Ende ?!

#### LLAMA CONFIG ####
MODEL_CONFIG_LLAMA = {
    'max_new_tokens': 2048,
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
    'max_new_tokens': 2048,
    'temperature': 0.001,
}

#################### CLOSED SOURCE #############
# MODEL_NAMES.append('gpt-3.5-turbo')
# MODEL_NAMES.append('gpt-4')
# REVISIONS.append("")
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
    "arc": {
        "item": "', '", # TODO: add apostroph if needed
        "grid_start": "[",
        "grid_end": "']]\n", # include end of last row # TODO: add apostroph if needed
        "row_start": "['", # TODO: add apostroph if needed
        "row_end": "'], ", # except for last row # TODO: add apostroph if needed
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

#initialize template
template = """{sys}{output_format}{pre_task}{task}{post_task}{instruction_end}"""
TEMPLATE = PromptTemplate(
    input_variables=["sys", "output_format", "pre_task", "task", "post_task", "instruction_end"],
    template=template,
)
SYSTEM_MESSAGE = ""
OUTPUT_FORMAT = ""
PRE_TEST_CASE = ""
POST_TEST_CASE = ""
INSTRUCTION_END = ""

# SYSTEM_MESSAGE = "[INST] <<SYS>>\nYou are given a puzzle with a series of train input and train output pairs as examples. Your task is to identify the step-by-step pattern to get the output from its input. Then, apply the pattern to the final test input to get the test output. The inputs and outputs are all in the form of rows of letters, representing a 2D grid.\n<</SYS>>\n\n"
# SYSTEM_MESSAGE = """[INST] <<SYS>>\nYou are to output only the following in json format: {'reflection': 'reflect on the answer', 'grid_changes': 'describe if the dimension of the input grid is different to its output grid', 'pixel_changes': 'describe the changes between the input and output pixels, focusing on movement or pattern changes', 'object_changes': 'describe the changes between the input and output objects, focusing on movement, object number, size, shape, position, value, cell count', 'overall_pattern': 'describe the simplest input-output relationship for all input-output pairs', 'instructions': 'describe the transformation actions in detail step by step', 'test_output': "Use the instructions to transform the test input grid and return only the resulting output grid"}.
# Do not use quotation marks ' or " within the fields unless it is required for the python code.\n<</SYS>>\n\nYou are given a series of inputs and output pairs that share the same logic of getting the output from its input. Each input and output is a 2-dimensional grid of pixels. The values from '0' to '9' represent different colors, where '0' represents the background. No calculations! For example, [['0','2','0'],['0','0','5']] represents a 2 row x 3 column grid with color '2' at position (1,0) and color '5' at position (2,1). The coordinates are 2D coordinates (row, column), row representing row number, column representing col number, with zero-indexing.
# You are to infer the simplest possible relation beetween input and output. The given sample pairs may not reflect all possibilities.

# You can refer to concepts as follows:
# - Goal-directedness: input is start and output is end state of process 
# - Geometry & topology:
# 	- Lines, rectangular shapes.
# 	- Symmetries, mirroring, rotations, translations.
# 	- Shape upscaling or downscaling, elastic distortions.
# 	- Containing / being contained / being inside or outside of a perimeter.
# 	- Drawing lines, connecting points, orthogonal projections.
# 	- Copying, repeating.
# 	- Patterns or mosaic based on sections.
# - Objects:
# 	- Objects are shapes based on similar colors or based on surroundings.
# 	- Object transformations based on geometry and topology.
# 	- Touching objects have contact with each other.
# 	- Noise pixels.
# -  Arithmetics based on objects or shapes pixels:
# 	- Counting.
# 	- Sorting.

# The list is not exhaustive. Transformations can be conditional.\n
# """
# SYSTEM_MESSAGE = """You are given a 2D input grid of pixels. The values from 'a' to 'j' represent different colors, where 'a' represents the background. The color mapping is as follows: {'a': 'black', 'b': 'blue', 'c': 'red', 'd': 'green', 'e': 'yellow', 'f': 'gray', 'g': 'magenta', 'h': 'orange', 'i': 'cyan', 'j': 'brown'}.
# For example, [['a','b','a'],['a','a','c']] represents a 2 row x 3 column grid with color 'b' at position (1,0) and color 'c' at position (2,1). The coordinates are 2D coordinates (row, column), row representing row number, column representing col number, with zero-indexing.

# Furthermore, you are given a description to transform the input grid into its output grid.

# You are to output only the following in json format: 
# """
# OUTPUT_FORMAT = {
#     'input_grid': 'describe the input grid and check if it matches the given description', 
#     'instructions': 'describe the transformation actions step by step provided by the description', 
#     'output_dimension': 'describe the output grid dimension provided by the description',
#     'test_output': 'transform the test input grid and return only the resulting output grid'
#     }

# PRE_TEST_CASE = """\nDo not use quotation marks ' or " within the fields.\n
# Test input grid:\n"""
# POST_TEST_CASE = """Please fill the json fields with content and create the corresponding output grid based on the following description:\n"""
# INSTRUCTION_END = "[/INST]"

#################### Directories ####################
# DIR = ["ARC_datasets/ARC"] # complete ARC
# DATASET = "arc"
# TASK = "arc"

# DIR = ["ARC_datasets/arc_subset"] # 50 ARC tasks 
# DATASET = "arc"
# TASK = "arc"

# DIR = ["ARC_datasets/1D-ARC/dataset"]
# DATASET = "arc_1D" 
# TASK = "arc_1D"

DIR = ["ARC_datasets/arc_new"]
DATASET = "arc" # tasks are the same as for 2D ARC
TASK = "arc_h_v"

# X = "ARC_datasets/ARC_solved_tasks/training/"
# X = "ARC_datasets/ARC_solved_tasks/evaluation/"

# X = "ARC_datasets/ARC_only_two_tasks/training/"
# X = "ARC_datasets/ARC_only_two_tasks/evaluation/"

# X = "ARC_datasets/LARC/training/"
# X = "ARC_datasets/LARC/evaluation/"
# TASK ="LARC"

