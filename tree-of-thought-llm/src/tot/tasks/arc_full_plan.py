import os
import re
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.arc import * # TODO: use ARC prompts
from tot.models import gpt
from utils import *
from arc_config import * 

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
        self.data, names = load_arc_tasks(path)
        self.steps = 2
        self.stops = ['\Test output:\n', None] # TODO: adjust to prompt! 
        self.success = {} # saves success rates for each task
        self.full_success = 0 # counts completely solved tasks

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        task_json = self.data[idx]
        
        # transform all grids into desired representation, e.g. numbers or letters
        if CHANGE_REPRESENTATION:
            task_json = change_color_representation(task_json, NEW_REPRESENTATION)
        
        return task_json
    
    def test_output(self, idx: int, output: str):      
        # get test case
        task_json = self.data[idx]
        task_name = self.names[idx]
        test_cases, solutions = get_tasks(task_json, DELIMITER)
        if task_name not in self.success: # TODO: works currently only if we have just one try
            self.success[task_name] = 0
        for test_case, solution in zip(test_cases[:1], solutions[:1]): # TODO: currently just check first test case in case more exist

            test_output_grid = get_content_from_llm_answer(output, "test_output") 
            # Check answers and save success rates. # TODO: Maybe change check to np.array comparison if possible
            is_success = str(test_output_grid).strip() in solution
            self.success[task_name] += is_success / len(test_cases)
        
        if self.success[task_name] == 1:
            self.full_success += 1
            
        # print('------------')
        info = {'rs': self.success[task_name], 'r': self.full_success / len(self)}
        return info
    
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        task_context = get_context(x, DELIMITER)
        task_input, _ = get_tasks(x, DELIMITER) # TODO: currently just check first test case in case more exist
        return standard_prompt.format(context=task_context, input=task_input[0]) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='', current_step: int=0, total_steps: int=1) -> str:
        task_context = get_context(x, DELIMITER)
        task_input, _ = get_tasks(x, DELIMITER) # TODO: currently just check first test case in case more exist
        
        if total_steps == 1:
            """
            baseline experiment - no ToT -> no vote
            """
            output_format = {
                'reflection': 'reflect on the answer', 
                'grid_changes': 'describe if the dimension of the input grid is different to its output grid', 
                'pixel_changes': 'describe the changes between the input and output pixels, focusing on movement or pattern changes', 
                'object_changes': 'describe the changes between the input and output objects, focusing on movement, object number, size, shape, position, value, cell count', 
                'overall_pattern': 'describe the simplest input-output relationship for all input-output pairs', 
                'instructions': 'describe the transformation actions step by step', 
                'test_output': 'Use the instructions to transform the test input grid and return only the resulting output grid in numpy array format.'
                }
            prompt = cot_prompt.format(context=task_context, input=task_input[0], output=output_format, special_instructions="") + y 
        elif current_step == 0: 
            """
            ARC-Plan - single_level -> first step: get plans        
            """
            output_format = {
                'grid_changes': 'describe if the dimension of the input grid is different to its output grid', 
                'pixel_changes': 'describe the changes between the input and output pixels, focusing on movement or pattern changes', 
                'object_changes': 'describe the changes between the input and output objects, focusing on movement, object number, size, shape, position, value, cell count', 
                'overall_pattern': 'describe a broad input-output relationship for all input-output pairs',
                'instructions': 'describe the transformation actions step by step', 
                }
            prompt = cot_prompt.format(context=task_context, input="", output=output_format, special_instructions="") + y 
        elif current_step == 1:  # ARC-Plan - single_level - second step: get test outputs
            """
            ARC-Plan - single_level -> second step: get test outputs
                y: contains the llm aswer with the plan from the first step          
            """
            instruct = '''Moreover, you are given a plan of transformation actions that might help you.\n\n'''
            output_format = {
                'reflection': 'reflect on the answer', 
                'grid_changes': 'describe if the dimension of the input grid is different to its output grid', 
                'pixel_changes': 'describe the changes between the input and output pixels, focusing on movement or pattern changes', 
                'object_changes': 'describe the changes between the input and output objects, focusing on movement, object number, size, shape, position, value, cell count', 
                'overall_pattern': 'describe the simplest input-output relationship for all input-output pairs', 
                'instructions': 'describe the transformation actions step by step', 
                'test_output': 'Use the instructions to transform the test input grid and return only the resulting output grid in numpy array format.'
                }
            y = '''Transformation actions: ''' + get_content_from_llm_answer(y, "instructions")
            prompt = cot_prompt.format(context=task_context, input=task_input[0], output=output_format, special_instructions=instruct) + y 
        return prompt 

    @staticmethod
    def vote_prompt_wrap(x: str, ys: list, current_step: int=0, total_steps: int=1) -> str:
        task_context = get_context(x, DELIMITER)
        task_input, _ = get_tasks(x, DELIMITER) # TODO: currently just check first test case in case more exist

        if total_steps == 1:
            """
            baseline experiment - no ToT -> no vote
            """
            return ""
        elif current_step == 0: 
            """
            ARC-Plan - single_level -> first step: vote for plan
            """
            output_format = {
                'pattern_analysis': {
                    '1': 'analyze if the first given plan correctly describes the relation between all input and output pairs',
                    '2': '...'
                    },
                'vote': 'vote for the best choice by entering the number of the choice as integer'
                }
            instruct = '''Moreover, you are given a set of plans with transformation actions that might describe the relation between all input and output pairs. Evaluate the given plans and analyze which one describes the relation best.\n\n'''
            prompt = vote_prompt.format(context=task_context, input="", output=output_format, special_instructions=instruct)
            voting_object = "instructions"
        elif current_step == 1:  
            """
            ARC-Plan - single_level -> second step: vote for test output
            """
            output_format = {
                'pattern_analysis': {
                    '1': 'analyze if the first given test output is correct',
                    '2': '...'
                    },
                'vote': 'vote for the best choice by entering the number of the choice as integer'
                }
            instruct = '''Moreover, you are given a test input and a set of potential corresponding test outputs. Evaluate the given test outputs and analyze if they share the same relation compared to the provided training input and output pairs.\n\n'''
            prompt = vote_prompt.format(context=task_context, input=task_input[0], output=output_format, special_instructions=instruct)
            voting_object = "test_output"
            
        
        for i, y in enumerate(ys, 1):
            # get the needed information from LLM answer
            y = get_content_from_llm_answer(y, voting_object) 
            prompt += f'\nChoice {i}: {y}'
        return prompt
    
    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        vote_results = [0] * n_candidates
        for vote_output in vote_outputs:
            try:
                vote = int(get_content_from_llm_answer(vote_output, "vote"))
            except:
                vote = -1
                print("Vote from LLM not a single integer:", vote_output)
            if vote in range(n_candidates):
                vote_results[vote] += 1
        return vote_results

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