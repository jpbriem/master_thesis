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
MODEL_NAMES.append("TheBloke/Llama-2-70b-Chat-GPTQ")
REVISIONS.append("main")
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
# MODEL_NAMES.append("TheBloke/Llama-2-13B-GPTQ") # TODO: Run all tests )
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
# MODEL_NAMES.append('gpt-3.5-turbo')
# MODEL_NAMES.append('gpt-4')
# REVISIONS.append("")
#################### CONFIG ####################
MODEL_CONFIG_GPT = {
    'model_name': MODEL_NAMES[0],
    'temperature': 0.001,
}

#################### Prompt ####################
CHANGE_REPRESENTATION = False
NEW_REPRESENTATION = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]

PRE_CONTEXT = ""
POST_CONTEXT = ""
DELIMITER = {
    "item": ", ",
    "grid_start": "",
    "grid_end": "\n", # include end of last row
    "row_start": "",
    "row_end": "\n", # except for last row
    "example_start": "",
    "example_end": "End of example.\n",
    "task_start": "",
    "task_end": "",
    "input_train": "train input:\n",
    "output_train": "train output:\n",    
    "input_test": "test input:\n",
    "output_test": "test output:", 
}

template_string = """{sys}{output_format}{task}{instruction_end}"""
TEMPLATE = PromptTemplate(
    input_variables=["sys", "output_format", "task", "instruction_end"],
    template=template_string,
)

SYSTEM_MESSAGE = "[INST] <<SYS>>\nYou are given a puzzle with a series of train input and train output pairs as examples. Your task is to identify the step-by-step pattern to get the output from its input. Then, apply the pattern to the final test input to get the test output. The inputs and outputs are all in the form of rows of letters, representing a 2D grid.\n<</SYS>>\n\n"
SYSTEM_MESSAGE = """[INST] <<SYS>>\nYou are given a series of inputs and output pairs that share the same logic of getting the output from its input. Each input and output is a 2-dimensional grid of pixels. The values from '0' to '9' represent different colors, where '0' represents the background. No calculations! For example, [['0','2','0'],['0','0','5']] represents a 2 row x 3 column grid with color '2' at position (1,0) and color '5' at position (2,1). The coordinates are 2D coordinates (row, column), row representing row number, column representing col number, with zero-indexing.
You are to infer the simplest possible relation beetween input and output. The given sample pairs may not reflect all possibilities.

You can refer to concepts as follows:
- Goal-directedness: input is start and output is end state of process 
- Geometry & topology:
	- Lines, rectangular shapes.
	- Symmetries, mirroring, rotations, translations.
	- Shape upscaling or downscaling, elastic distortions.
	- Containing / being contained / being inside or outside of a perimeter.
	- Drawing lines, connecting points, orthogonal projections.
	- Copying, repeating.
	- Patterns or mosaic based on sections.
- Objects:
	- Objects are shapes based on similar colors or based on surroundings.
	- Object transformations based on geometry and topology.
	- Touching objects have contact with each other.
	- Noise pixels.
-  Arithmetics based on objects or shapes pixels:
	- Counting.
	- Sorting.

The list is not exhaustive. Transformations can be conditional.

You are to output only the following in json format: {'reflection': 'reflect on the answer', 'grid_changes': 'describe if the dimension of the input grid is different to its output grid', 'pixel_changes': 'describe the changes between the input and output pixels, focusing on movement or pattern changes', 'object_changes': 'describe the changes between the input and output objects, focusing on movement, object number, size, shape, position, value, cell count', 'overall_pattern': 'describe the simplest input-output relationship for all input-output pairs', 'instructions': 'describe the transformation actions in detail step by step', 'test_output': "Use the instructions to transform the test input grid and return only the resulting output grid"}.
Do not use quotation marks ' or " within the fields unless it is required for the python code.\n<</SYS>>\n\n"""
# SYSTEM_MESSAGE = ""
OUTPUT_FORMAT = ""
INSTRUCTION_END = "[/INST]"
# INSTRUCTION_END = ""

#################### Directories ####################

# TASK_DIR_TRAIN = "../ARC/ARC/data/training"
# TASK_DIR_EVAL = "../ARC/ARC/data/evaluation"

TASK_DIR_TRAIN = "ARC_datasets/ARC_solved_tasks/training/"
TASK_DIR_EVAL = "ARC_datasets/ARC_solved_tasks/evaluation/"

# TASK_DIR_TRAIN = "ARC_datasets/ARC_only_two_tasks/training/"
# TASK_DIR_EVAL = "ARC_datasets/ARC_only_two_tasks/evaluation/"


######## TODO: DELETE ########
# TASK_DIR_TRAIN = "ARC_datasets/test_mistral_gptq/training/"
# TASK_DIR_EVAL = "ARC_datasets/test_mistral_gptq/evaluation/"
