import os
import re
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.arc import * # TODO: use ARC prompts
from tot.models import gpt
from tot.methods.arc_utils import *
from tot.methods.arc_config import * 
from tot.methods.tree_nodes import Node

class ARCTask(Task):
    """
    Input (x)   : 2D grid of pixels
    Output (y)  : 2D grid of pixels 
    Input Example:  [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    Output Example: [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    """
    def __init__(self):
        """
        2 subfolders: training, evaluation
        """
        super().__init__()
        path = os.path.join(DATA_PATH, 'arc')
        self.data, self.names = load_arc_tasks(path)
        self.steps = 4
        self.stops = [None]*self.steps # TODO: adjust to prompt! 
        self.success = {} # saves success rates for each task
        self.full_success = 0 # counts completely solved tasks
        self.value_cache = {}

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        task_json = self.data[idx]
        
        # transform all grids into desired representation, e.g. numbers or letters
        if CHANGE_REPRESENTATION:
            task_json = change_color_representation(task_json, NEW_REPRESENTATION)
            self.data[idx] = task_json
            
        return task_json
    
    def test_output(self, idx: int, output: str):      
        output_format = {'reflection': "", 'grid_view': "", 'pixel_view': "",  'object_view': "", 'description': "", 'grid_changes': "", 'pixel_changes': "",  'object_changes': "", 'overall_pattern': "", 'part_of_interest': "", 'conditions': "", 'instructions': "",  'intermediate_results': "", 'test_output': ""}
        # get test case
        task_json = self.data[idx]
        task_name = self.names[idx]
        _, solutions = get_tasks(task_json, DELIMITER)
        if task_name not in self.success: # TODO: works currently only if we have just one try
            self.success[task_name] = 0
        for solution in solutions[:1]: # TODO: currently just check first test case in case more exist
            test_output_grid = extract_json_value(output, output_format, "test_output") 
            test_output_grid = grid_to_nparray(test_output_grid)
            solution = grid_to_nparray(solution)
            is_success = np.array_equal(test_output_grid, solution)
            self.success[task_name] += is_success / len(solutions)
        
        if self.success[task_name] == 1:
            self.full_success += 1
            
        # print('------------')
        info = {'rs': self.success[task_name], 'r': self.full_success / len(self)}
        return info
    
    @staticmethod
    def standard_prompt_wrap(node) -> str:
        task_context = get_context(node.x, DELIMITER)
        task_input, _ = get_tasks(node.x, DELIMITER) # TODO: currently just check first test case in case more exist        
        prompt = standard_prompt.copy()
        prompt["user"] = prompt["user"].format(context=task_context, test_input=task_input[0])
        return prompt

    @staticmethod
    def cot_prompt_wrap(node, total_steps: int=1) -> str:
        current_step = node.level
        
        # get arc examples
        task_context = get_context(node.x, DELIMITER)
        # get test case
        if current_step == total_steps-1:
            task_input, _ = get_tasks(node.x, DELIMITER) # TODO: currently just check first test case in case more exist
            task_input[0] = "\n\n" + task_input[0]
        else:
            task_input = [""]        
        # get output format for current step
        output_format = prompt_modules[str(current_step)]["generation"]["output_format"]
        # get instructions for current step
        instruct = ""
        for i in range(current_step):
            instruct += prompt_modules[str(i)]["evaluation"]["instruct_previous_thoughts"]
        instruct += prompt_modules[str(current_step)]["generation"]["instruct_task"]
        # get previous thoughts
        previous_thoughts = f'{get_previous_thoughts(node)}' # add own thought to previous thoughts to generate children

        # get prompt template and fill
        prompt = cot_prompt.copy()
        prompt["system"] = cot_prompt["system"].format(output=output_format, special_instructions=instruct)
        prompt["user"] = cot_prompt["user"].format(context=task_context, test_input=task_input[0], previous_thoughts=previous_thoughts)


        # old
        # if total_steps == 1:
        #     """
        #     No ToT, just CoT 
        #     """
        #     output_format = {
        #         'reflection': 'reflect on the answer', 
        #         'grid_changes': 'describe if the dimension of the input grid is different to its output grid', 
        #         'pixel_changes': 'describe the changes between the input and output pixels, focusing on movement or pattern changes', 
        #         'object_changes': 'describe the changes between the input and output objects, focusing on movement, object number, size, shape, position, value, cell count', 
        #         'overall_pattern': 'describe the simplest input-output relationship for all input-output pairs', 
        #         'instructions': 'describe the transformation actions step by step', 
        #         'test_output': 'Use the instructions to transform the test input grid and return only the resulting output grid in numpy array format.'
        #         }
        #     instruct = 'You are to infer the simplest possible relation beetween input and output.'
        # elif node.level == 0: 
        #     """
        #     ToT first step: get description # TODO: Add object information!?   
        #     """
        #     instruct = '''\nYour task is to give only an abstract description about how an input grid and how an output grid typically look like based on the examples.\n'''
        #     output_format = {
        #         'grid_view': 'describe the dimensions of all input grids and of all output grids one after another.', 
        #         'pixel_view': 'describe the pixels of all input grids and of all output grids one after another, focusing on positions or patterns', 
        #         'object_view': 'describe the objects in all input grids and in all output grids one after another, focusing on shape, amount, size, position, values, cell count', 
		# 		'description': {	
     	# 			'input_description': 'Regarding all input grids, summarize your findings about the dimensions, pixel view and object view in an abstract description by completing the sentence: "A typical input grid has a dimension of ... and looks like...',
		# 			'output_description': 'Regarding all output grids, summarize your findings about the dimensions, pixel view and object view in an abstract description by completing the sentence: "A typical output grid has a dimension of ... and looks like...',
		# 			},
        #         }
            # instruct = '''\nYour task is to give an abstract description (valid for all example pairs) about similarities and differences between an input and its respective output.\n'''
            # output_format = {
            #     'grid_changes': 'describe if the dimension of the input grid is different to its output grid', 
            #     'pixel_changes': 'describe the changes between the input and output pixels, focusing on movement or pattern changes', 
            #     'object_changes': 'describe the changes between the input and output objects, focusing on movement, object number, size, shape, position, value, cell count', 
            #     'description': 'summarize your findings in an abstract description that is valid for all example pairs',
            #     }
#             task_input = [""] # TODO: when changing to multiple test cases, change this!
#         elif node.level == 1:  
#             """
#             ToT second step: get overall pattern
#             """
#             instruct = '''\nMoreover, you are given an abstract description about how an input grid and an output grid typically look like.
# Your task is to infer an overall pattern that describes the simplest relation between all input and output pairs.\n'''
#             output_format = {
#                 'grid_changes': 'For each example: describe if and how the dimension of the input grids is different from its output grid', 
#                 'pixel_changes': 'For each example: describe the changes between the input and output pixels, focusing on movement or pattern changes', 
#                 'object_changes': 'For each example: describe the changes between the input and output objects, focusing on movement, object number, size, shape, position, value, cell count', 
#                 'overall_pattern': 'summarize your findings and describe the simplest input-output relationship valid for all examples', 
#                 }
#             instruct = '''\nMoreover, you are given an abstract description for all examples about similarities and differences between the input and its respective output.
# Your task is to give an overall pattern that describes the relation between all input and output pairs.\n'''
#             output_format = {
#                 'grid_changes': 'describe if the dimension of the input grid is different to its output grid', 
#                 'pixel_changes': 'describe the changes between the input and output pixels, focusing on movement or pattern changes', 
#                 'object_changes': 'describe the changes between the input and output objects, focusing on movement, object number, size, shape, position, value, cell count', 
#                 'description': 'summarize your findings in an abstract description that is valid for all example pairs',
#                 'overall_pattern': 'describe the simplest input-output relationship for all input-output pairs', 
#                 }
        #     previous_thoughts = '''\nDescription: ''' + str(extract_json_value(node.LLM_answer, output_format, "input_description")) + " " + str(extract_json_value(node.LLM_answer, output_format, "output_description")) + '''\n'''
        #     task_input = [""] # TODO: when changing to multiple test cases, change this!
        # elif node.level == 2:  
        #     """
        #     ToT third step: get instructions
        #     """
#             instruct = '''\nMoreover, you are given an abstract description about how an input grid and an output grid typically look like.
# Moreover, you are given an overall pattern that might describe the relation between the input and output grids of all examples.
# Your task is to give step-by-step instructions that are general applicable to all examples to get from the input grid to its output grid.\n'''
            # output_format = {
            #     'part_of_interest': 'regarding the transformation, describe the parts of interest of the input grid, e.g. the grid dimension, pixel pattern, or objects',
            #     'conditions': 'describe if and how the transformation process is based on conditions, e.g. object characteristics (number, shape, symmetry, color, size, position) or pixel characteristics (color, position)',
            #     'instructions': 'describe all transformation steps with potential conditions and provide step-by-step instructions that are general applicable to transform an input grid into its output grid', 
            #     }
#             instruct = '''\nMoreover, you are given an abstract description for all examples about similarities and differences between the input and its respective output.
# Moreover, you are given an overall pattern that might describe the relation between all input and output pairs.
# Your task is to give step-by-step instructions that are general applicable to all input examples to create their outputs.\n'''
#             output_format = {
#                 'grid_changes': 'describe if the dimension of the input grid is different to its output grid', 
#                 'pixel_changes': 'describe the changes between the input and output pixels, focusing on movement or pattern changes', 
#                 'object_changes': 'describe the changes between the input and output objects, focusing on movement, object number, size, shape, position, value, cell count', 
#                 'description': 'summarize your findings in an abstract description that is valid for all example pairs',
#                 'overall_pattern': 'describe the simplest input-output relationship for all input-output pairs',
#                 'instructions': 'describe the transformation actions step by step', 
#                 }
#             previous_thoughts = '''\nDescription: ''' + str(extract_json_value(node.parent.LLM_answer, output_format, "description")) + '''\n'''
#             previous_thoughts += '''\nOverall pattern: ''' + str(extract_json_value(node.LLM_answer, output_format, "overall_pattern")) + '''\n'''
#             task_input = [""] # TODO: when changing to multiple test cases, change this!
#         elif node.level == 3:   
#             """
#             ToT fourth step: get test output
#             """
#             instruct = '''\nMoreover, you are given an abstract description about how an input grid and an output grid typically look like.
# Moreover, you are given an overall pattern that might describe the relation between the input and output grids of all examples.
# Moreover, you are given step-by-step instructions that are general applicable to transform an input grid into its output grid.
# Based on the provided information, your task is to apply the general instructions to a new test case and you are to transform the test input grid into its test output grid.\n'''
#             output_format = {
#                 'description': 'describe the test input and check if it fits to the given abstract description',
#                 'intermediate_results': 'apply the instructions step-by-step to the test input grid; focus on potential transformation conditions and provide all intermediate grids',
#                 'test_output': 'return only the resulting test output grid as numpy array' 
#                 }
#             instruct = '''\nMoreover, you are given an abstract description for all examples about similarities and differences between the input and its respective output.
# Moreover, you are given an overall pattern that might describe the relation between all input and output pairs.
# Moreover, you are given step-by-step instructions that are general applicable to all input examples to create their outputs.
# Based on the provided information, your task is to apply the general instructions to the test input to create the test output.\n'''
#             output_format = {
#                 'grid_changes': 'describe if the dimension of the input grid is different to its output grid', 
#                 'pixel_changes': 'describe the changes between the input and output pixels, focusing on movement or pattern changes', 
#                 'object_changes': 'describe the changes between the input and output objects, focusing on movement, object number, size, shape, position, value, cell count', 
#                 'description': 'summarize your findings in an abstract description that is valid for all example pairs',
#                 'overall_pattern': 'describe the simplest input-output relationship for all input-output pairs',
#                 'instructions': 'describe the transformation actions step by step',
#                 'test_output': 'Use the instructions to transform the test input grid and return only the resulting output as numpy array.' 
#                 }
        #     previous_thoughts = f'\nDescription: {extract_json_value(node.parent.LLM_answer, json_keys, "input_description")}  {extract_json_value(node.LLM_answer, output_format, "output_description")}' 
        #     previous_thoughts += '''\nOverall pattern: ''' + str(extract_json_value(node.parent.LLM_answer, output_format, "overall_pattern")) + '''\n'''
        #     previous_thoughts += '''\nInstructions: ''' + str(extract_json_value(node.LLM_answer, output_format, "instructions")) + '''\n'''
        
        # prompt["system"] = cot_prompt["system"].format(output=output_format, special_instructions=instruct)
        # prompt["user"] = cot_prompt["user"].format(context=task_context, test_input=task_input[0], previous_thoughts=previous_thoughts)
        return prompt 

    @staticmethod
    def vote_prompt_wrap(node, total_steps: int=1) -> str:
        task_context = get_context(node.x, DELIMITER)
        task_input, _ = get_tasks(node.x, DELIMITER) # TODO: currently just check first test case in case more exist
        instruct, previous_thoughts = "", ""  
        prompt = vote_prompt.copy()
        json_keys = {'reflection': "", 'grid_view': "", 'pixel_view': "",  'object_view': "", 'description': "", 'grid_changes': "", 'pixel_changes': "",  'object_changes': "", 'overall_pattern': "", 'part_of_interest': "", 'conditions': "", 'instructions': "",  'intermediate_results': "", 'test_output': ""}

        if total_steps == 1:
            """
            No ToT, just CoT 
            """
            output_format = {
                'test_output_analysis': {
                    'Choice_1': 'analyze if the first given test output fit to the given description, overall pattern, and instructions.',
                    'Choice_2': '...'
                    },
                'vote': 'vote for the best choice by entering the number of the choice as integer'
                }
            instruct = '''Moreover, you are given a test input and multiple potential test outputs.
Evaluate the given test outputs and analyze if they share the same input to output pattern as the examples.\n'''
            voting_object = "test_output"
        elif node.level == 0: 
            """
            ToT first step: vote for description   
            """
            output_format = {
                'description_analysis': {
                    #'Choice_1': 'analyze if the first given description correctly describes similarities and differences between all inputs and respective outputs.',
                    'Choice_1': 'analyze if the first given description correctly describes the inputs and outputs of all examples.',
                    'Choice_2': '...'
                    },
                'vote': 'vote for the best choice by entering the number of the choice as integer'
                }
            instruct = '''\nMoreover, you are given multiple abstract descriptions about how an input grid and an output grid typically look like.
Evaluate the given descriptions and analyze if they correctly describe the provided example input and output grids.\n'''
#             instruct = '''\nMoreover, you are given multiple abstract descriptions about similarities and differences between the input and its respective output regarding all examples.
# Evaluate the given descriptions and analyze if they correctly describe the provided training input and output pairs.\n'''   
            task_input = [""] # TODO: when changing to multiple test cases, change this!         
            
            voting_object = "description"
        elif node.level == 1:  
            """
            ToT second step: vote for overall pattern
            """
            instruct = '''\nMoreover, you are given an abstract description about how an input grid and an output grid typically look like.
Moreover, you are given multiple overall patterns that might describe the relation between all input and output pairs.
Evaluate the given patterns and analyze if they correctly describe the relation between all input and output pairs.\n''' 
            output_format = {
                'overall_pattern_analysis': {
                    'Choice_1': 'analyze if the first given overall pattern correctly describes the relation between all input and output pairs.',
                    'Choice_2': '...'
                    },
                'vote': 'vote for the best choice by entering the number of the choice as integer'
                }
            instruct = '''\nMoreover, you are given an abstract description about how an input grid and an output grid typically look like.
Moreover, you are given multiple overall patterns that might describe the relation between all input and output pairs.
Evaluate the given patterns and analyze if they correctly describe the relation between all input and output pairs.\n''' 
#             instruct = '''\nMoreover, you are given an abstract description for all examples about similarities and differences between the input and its respective output.
# Moreover, you are given multiple overall patterns describing the relation between all input and output pairs.
# Evaluate the given patterns and analyze if they correctly describe the relation between all input and output pairs.\n'''            
            previous_thoughts = '''\nDescription: ''' + str(extract_json_value(node.LLM_answer, json_keys, "description")) + '''\n'''
            task_input = [""] # TODO: when changing to multiple test cases, change this!   
            voting_object = "overall_pattern"
        elif node.level == 2:  
            """
            ToT third step: vote for instructions
            """
            output_format = {
                'instruction_analysis': {
                    #'Choice_1': 'analyze if the first given instruction correctly describes the transformation for all input and output pairs.',
                    'Choice_1': 'analyze if the first given instruction correctly describes the transformation from input to output for all examples.',
                    'Choice_2': '...'
                    },
                'vote': 'vote for the best choice by entering the number of the choice as integer'
                }
            instruct = '''\nMoreover, you are given an abstract description about how an input grid and an output grid typically look like.
Moreover, you are given an overall pattern that might describe the relation between all input and output pairs.
Moreover, you are given multiple sets of instructions that might be general applicable to transform an input grid into its output grid.
Evaluate the given sets of instructions and analyze if they correctly describe the transformation for all input and output examples.\n''' 
            
#             instruct = '''\nMoreover, you are given an abstract description for all examples about similarities and differences between the input and its respective output.
# Moreover, you are given an overall pattern describing the relation between all input and output pairs.
# Moreover, you are given multiple sets of instructions describing the transformation for all input and output pairs.
# Evaluate the given sets of instructions and analyze if they correctly describe the transformation for all input and output pairs.\n'''            
            previous_thoughts = '''\nDescription: ''' + str(extract_json_value(node.parent.LLM_answer, json_keys, "description")) + '''\n'''
            previous_thoughts += '''\nOverall pattern: ''' + str(extract_json_value(node.LLM_answer, json_keys, "overall_pattern")) + '''\n'''
            task_input = [""] # TODO: when changing to multiple test cases, change this!   
            voting_object = "instructions"
        elif node.level == 3:  
            """
            ToT fourth step: vote for test output
            """
            output_format = {
                'test_output_analysis': {
                    'Choice_1': 'analyze if the first given test output fit to the given description, overall pattern, and instructions.',
                    'Choice_2': '...'
                    },
                'vote': 'vote for the best choice by entering the number of the choice as integer'
                }
            instruct = '''\nMoreover, you are given an abstract description about how an input grid and an output grid typically look like.
Moreover, you are given an overall pattern that might describe the relation between all input and output pairs.
Moreover, you are given step by step instructions that are general applicable to transform an input grid into its output grid.
Moreover, you are given a test input grid and multiple potential test output grids.
Evaluate the given test output grids and analyze if they fit to the given description, overall pattern, and instructions.\n'''   
#             instruct = '''\nMoreover, you are given an abstract description for all examples about similarities and differences between the input and its respective output.
# Moreover, you are given an overall pattern describing the relation between all input and output pairs.
# Moreover, you are given step-by-step instructions that are general applicable to all input examples to create their outputs.
# Moreover, you are given a test input and multiple potential test outputs.
# Evaluate the given test outputs and analyze if they fit to the given description, overall pattern, and instructions.\n'''            
            previous_thoughts = '''\nDescription: ''' + str(extract_json_value(node.parent.parent.LLM_answer, json_keys, "description")) + '''\n'''
            previous_thoughts += '''\nOverall pattern: ''' + str(extract_json_value(node.parent.LLM_answer, json_keys, "overall_pattern")) + '''\n'''
            previous_thoughts += '''\nInstructions: ''' + str(extract_json_value(node.LLM_answer, json_keys, "instructions")) + '''\n'''
            voting_object = "test_output"
                     
        prompt["system"] = vote_prompt["system"].format(output=output_format, special_instructions=instruct)
        prompt["user"] = vote_prompt["user"].format(context=task_context, test_input=task_input[0], previous_thoughts=previous_thoughts)

        for i, child in enumerate(node.children, 1):
            # get the needed information from LLM answer
            choice = extract_json_value(child.LLM_answer, json_keys, voting_object) 
            prompt["user"] += f'\nChoice {i}: {choice}'   
        return prompt
    
    @staticmethod
    def vote_outputs_unwrap(node, vote_outputs: list) -> list:
        values = [0]*len(node.children)
        output_keys = {"test_output_analysis": "", "instruction_analysis": "", "overall_pattern_analysis": "", "description_analysis": "", "vote": ""}
        for vote_output in vote_outputs:
            try:
                vote = int(extract_json_value(vote_output, output_keys, "vote"))
            except:
                vote = -1
                log = "'vote' from LLM not a single integer: " + str(vote_output)
                print(log)
                path = "json_parsing_errors/"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+".txt"
                with open(path, "w") as text_file:
                    text_file.write(log)
            
            if (vote-1) in range(len(node.children)): # vote -1 bc. indexing starts at 0
                values[vote-1] += 1
        return values
    
    @staticmethod
    def value_prompt_wrap(node, total_steps: int=1) -> str:
        current_step = node.level-1 # -1 bc. node is the child to be evaluated
        
        # get arc examples
        task_context = get_context(node.x, DELIMITER)
        # get test case
        if current_step == total_steps-1:
            task_input, _ = get_tasks(node.x, DELIMITER) # TODO: currently just check first test case in case more exist
            task_input[0] = "\n\n" + task_input[0]
        else:
            task_input = ["\n"]
        # get output format for current step
        output_format = prompt_modules[str(current_step)]["evaluation"]["output_format"]
        # get instructions for current step
        instruct = ""
        for i in range(current_step+1):
            instruct += prompt_modules[str(i)]["evaluation"]["instruct_previous_thoughts"]
        instruct += prompt_modules[str(current_step)]["evaluation"]["instruct_task"]
        # get previous thoughts
        previous_thoughts = get_previous_thoughts(node)

        # add thought to be valued 
        thought = get_thought(node.LLM_answer, prompt_modules, current_step)
        node.thought = thought
        task_input[0] += f'{thought}'
        
        # get prompt template and fill 
        prompt = value_prompt.copy()
        prompt["system"] = value_prompt["system"].format(output=output_format, special_instructions=instruct)
        prompt["user"] = value_prompt["user"].format(context=task_context, test_input=task_input[0], previous_thoughts=previous_thoughts)

#         # Old:
#         if total_steps == 1:
#             """
#             No ToT, just CoT 
#             """
#             output_format = {
#                 'test_output_analysis': 'Analyze whether the given test output fit to the given description, overall pattern, and instructions.',
#                 'value': 'Based on your analysis, give a rating between 0 and 10 for the test output as integer.'
#                 }
#             instruct = '''Moreover, you are given a test input and a potential test output.
# Evaluate the given test output and analyze if it shares the same input to output pattern as the examples.\n'''
#             task_input[0] += f'\ntest output: {extract_json_value(node.LLM_answer, json_keys, "test_output")}'
#         elif current_step == 0: 
#             """
#             ToT first step: value for description   
#             """
#             # output_format = {
#             #     'Example_1': {
#             #         'description_analysis': 'Regarding the first example, analyze if both the input and output grid fit to the given description.',
#             #         'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the description as integer.'
#             #         },
#             #     'Example_2': {
#             #         'description_analysis': '...',
#             #         'value': '...'
#             #         },
#             #     }
#             # instruct = '''\nMoreover, you are given an abstract description about how an input grid and an output grid typically look like.
# # Evaluate the given description and analyze if it correctly describes the provided input and output grids of all examples.\n'''
#             task_input = [""] # TODO: when changing to multiple test cases, change this!
#             task_input[0] += f'\nDescription: {extract_json_value(node.LLM_answer, json_keys, "input_description")}  {extract_json_value(node.LLM_answer, output_format, "output_description")}'  
#         elif current_step == 1:  
#             """
#             ToT second step: value for overall pattern
#             """
#             # output_format = {
#             #     'Example_1': {
#             #         'overall_pattern_analysis': 'Regarding the first example, analyze if the given overall pattern correctly describes the relation between the input and output.',
#             #         'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the overall pattern as integer.'
#             #         },
#             #     'Example_2': {
#             #         'overall_pattern_analysis': '...',
#             #         'value': '...'
#             #         },
#             #     }
# #             instruct = '''\nMoreover, you are given an abstract description about how an input grid and an output grid typically look like.
# # Moreover, you are given an overall pattern that might describe the relation between all input and output pairs.
# # Evaluate the given pattern and analyze if it correctly describes the relation between the inputs and outputs of all examples.\n''' 
#             previous_thoughts = f'\nDescription: {extract_json_value(node.parent.LLM_answer, json_keys, "input_description")}  {extract_json_value(node.parent.LLM_answer, output_format, "output_description")}' 
#             task_input = [""] # TODO: when changing to multiple test cases, change this!
#             task_input[0] += f'\nOverall pattern: {extract_json_value(node.LLM_answer, json_keys, "overall_pattern")}'           
#         elif current_step == 2:
#             """
#             ToT third step: vote for instructions
#             """
#             # output_format = {
#             #     'Example_1': {
#             #         'instruction_analysis': 'Regarding the first example, analyze if the given instructions correctly transform the input grid into its output grid. ',
#             #         'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the instructions as integer.'
#             #         },
#             #     'Example_2': {
#             #         'instruction_analysis': '...',
#             #         'value': '...'
#             #         },
#             #     }
# #             instruct = '''\nMoreover, you are given an abstract description about how an input grid and an output grid typically look like.
# # Moreover, you are given an overall pattern that might describe the relation between all input and output pairs.
# # Moreover, you are given a set of instructions that might be generally applicable to transform an input grid into its output grid.
# # Evaluate the given set of instructions and analyze if it correctly describes the transformation for all examples.\n'''            
#             previous_thoughts = f'\nDescription: {extract_json_value(node.parent.LLM_answer, json_keys, "input_description")}  {extract_json_value(node.LLM_answer, output_format, "output_description")}' 
#             previous_thoughts += f'\nOverall pattern: {extract_json_value(node.parent.LLM_answer, json_keys, "overall_pattern")}'
#             task_input = [""] # TODO: when changing to multiple test cases, change this!
#             task_input[0] += f'\nInstruction: {extract_json_value(node.LLM_answer, json_keys, "instructions")}'     
#         elif current_step == 3:
#            """
#            ToT fourth step: value for test output
#            """
#             output_format = {
#                 'test_output_analysis': 'analyze if the given test output fits to the given description, overall pattern, and instructions.',
#                 'value': 'Based on your analysis, give a rating between 0 and 10 for the test output as integer.'
#                 }
#             instruct = '''\nMoreover, you are given an abstract description about how an input grid and an output grid typically look like.
# Moreover, you are given an overall pattern that might describe the relation between all input and output pairs.
# Moreover, you are given a set of instructions that might be generally applicable to transform an input grid into its output grid.
# Moreover, you are given a test input grid and a potential test output grid.
# Evaluate the given test output grid and analyze if it fits to the given description, overall pattern, and instructions.\n'''   
            # previous_thoughts = f'\nDescription: {extract_json_value(node.parent.LLM_answer, json_keys, "input_description")}  {extract_json_value(node.LLM_answer, output_format, "output_description")}' 
            # previous_thoughts += f'\nOverall pattern: {extract_json_value(node.parent.parent.LLM_answer, json_keys, "overall_pattern")}\n'
            # previous_thoughts += f'\nInstruction: {extract_json_value(node.parent.LLM_answer, json_keys, "instructions")}\n\n' 
            # task_input[0] += f'test output: {extract_json_value(node.LLM_answer, json_keys, "test_output")}'
        # prompt["system"] = value_prompt["system"].format(output=output_format, special_instructions=instruct)
        # prompt["user"] = value_prompt["user"].format(context=task_context, test_input=task_input[0], previous_thoughts=previous_thoughts)
        return prompt

    @staticmethod
    def value_outputs_unwrap(value_outputs: list, current_step: int=0) -> float:
        final_value = 0
        cnt_outputs = 0 # counter for number of outputs w valid value
        output_keys = extract_dict_keys(prompt_modules[str(current_step)]["evaluation"], "output_format")
        for value_output in value_outputs:
            value_output = get_json_from_text(value_output, output_keys)
            if isinstance(value_output, str):
                continue
            cnt_examples = 0 # counter for number of examples w valid value
            value = 0 # sum up values over all examples
            example_id = 1
            if "Example_1" in value_output:
                while "Example_"+str(example_id) in value_output:
                    try:
                        value += int(value_output["Example_"+str(example_id)]["value"])
                        cnt_examples += 1
                    except:
                        log = "'value' from LLM not a single integer: " + str(value_output["Example_"+str(example_id)])
                        print(log)
                        path = "json_parsing_errors/"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+".txt"
                        with open(path, "w") as text_file:
                            text_file.write(log)
                    example_id += 1
            elif "value" in value_output:
                value += int(value_output["value"])
                cnt_examples += 1

            if cnt_examples > 0:
                value /= cnt_examples
                final_value += value
                cnt_outputs += 1
        if cnt_outputs > 0:
            final_value /= cnt_outputs
        else:
            final_value = 0
        return final_value








    # TODO: NEEDED?!
    @staticmethod
    def compare_prompt_wrap(x: str, ys: list) -> str:
        assert len(ys) == 2, 'compare prompt only supports 2 candidates'
        ys = [y.split('Passage:\n')[-1] for y in ys]
        prompt = compare_prompt + f'Passage 1:\n{ys[0]}\n\nPassage 2:\n{ys[1]}\n'
        return prompt
    
    @staticmethod
    def compare_output_unwrap(compare_output: str):
        if 'more coherent passage is 1' in compare_output:
            return 0
        elif 'more coherent passage is 2' in compare_output:
            return 1
        elif 'two passages are similarly coherent' in compare_output:
            return 0.5
        else:
            print(f'-----------------compare no match: {[compare_output]}')
            return -1