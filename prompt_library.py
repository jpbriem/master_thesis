from langchain.prompts import PromptTemplate 

################################## Template ##################################

TEMPLATE = """{sys}{output_format}{task}{instruction_end}"""

prompt = PromptTemplate(
    input_variables=["sys", "output_format", "task"],
    template=TEMPLATE,
)


################################## System Message ##################################
SYSTEM_MESSAGE = "[INST] <<SYS>>\nYou are given a puzzle with a series of train input and train output pairs as examples. Your task is to identify the step-by-step pattern to get the output from its input. Then, apply the pattern to the final test input to get the test output. The inputs and outputs are all in the form of rows of numbers, representing a 2D grid, with values from 0-9. The values are not representative of any ordinal ranking. Do not sum them or modulo them or calculate with them.\n<</SYS>>\n\n"
SYSTEM_MESSAGE = "[INST] <<SYS>>\nYou are given a puzzle with a series of train input and train output pairs as examples. Your task is to identify the step-by-step pattern to get the output from its input. Then, apply the pattern to the final test input to get the test output. The inputs and outputs are all in the form of rows of letters, representing a 2D grid.\n<</SYS>>\n\n"
SYSTEM_MESSAGE = ""
SYSTEM_MESSAGE = """You are to output the your answer in json format, following the format below: 
"""
SYSTEM_MESSAGE = """You are given a series of input and output pairs. 
These are all in the form of a 2D array, representing a 2D grid, with values from 0-9. 
The values are not representative of any ordinal ranking. Do not sum them or modulo them.
Input/output pairs may not reflect all possibilities, you are to infer the simplest possible relation making use of symmetry and invariance as much as possible.

The input can be something like:
> entire grid being the sandbox to manipulate
> using a part of the grid (individual squares or portions of the grid) to depict instructions of how to do the task. Position and symmetry is very important.
> using regions of similar value to depict area for answer of the task

The output grid can be something like:
> same output size as input after performing action
> output one of the fixed predetermined patterns used to classify the input image
> using output to show the ordering of objects, such as by size, height, width, position, value

You are to output the following in json format: 
"""
SYSTEM_MESSAGE = "You are to output your answer in json format, following this format: "


################################## Output Format ##################################
OUTPUT_FORMAT = ""
OUTPUT_FORMAT = {"Explanation": "describe the simplest input-output relationship for all input-output pairs",
"test_input": "copy the test input grid",     
"test_output": "create the test output grid"}


################################## Instruction End ##################################
INSTRUCTION_END = "[/INST]"
INSTRUCTION_END = ""