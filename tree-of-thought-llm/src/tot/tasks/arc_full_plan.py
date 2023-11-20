import os
import re
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.text import * # TODO: use ARC prompts
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
        tasks_json, tasks_names = load_arc_tasks(path)
        self.data = tasks_json
        self.names = tasks_names
        self.steps = 2
        self.stops = ['\Test output:\n', None] # TODO: adjust to prompt! 
        self.success = {} # saves success rates for each task
        self.full_success = 0 # counts completely solved tasks

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        task_json = self.data[idx]
        task_input = get_context(task_json, DELIMITER)
        
        return task_input
    
    def test_output(self, idx: int, output: str):
        """
        In this scenario, the output is a plan of how to solve the test task. 
        Now, apply the plan to the test task and see if it works.
        NOTE: Currently, apply plan once not 5 times like in the original code.
        """
        
        # get test case
        task_json = self.data[idx]
        task_name = self.names[idx]
        test_cases, solutions = get_tasks(task_json, DELIMITER)
        if task_name not in self.success: # TODO: works currently only if we have just one single plan, if we test 2 plans overwritten! 
            self.success[task_name] = 0
        for test_case, solution in zip(test_cases, solutions):
            prompt = test_output_prompt.format(input=test_case) # TODO: create Prompt to test plan
            test_output = gpt(prompt, n=1, stop=None)[0] 
            test_output_grid = test_output.split('Test output:\n')[-1] # TODO: extract output grid from json
            # Check answers and save success rates. 
            is_success = test_output_grid.strip() in solution
            self.success[task_name] += is_success / len(test_cases)
        
        if self.success[task_name] == 1:
            self.full_success += 1
            
        # print('------------')
        info = {'rs': self.success[task_name], 'r': self.full_success / len(self)}
        return info
    
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(input=x) + y

    @staticmethod
    def vote_prompt_wrap(x: str, ys: list) -> str:
        prompt = vote_prompt
        for i, y in enumerate(ys, 1):
            # y = y.replace('Plan:\n', '')
            # TODO: truncate the plan part?
            prompt += f'Choice {i}:\n{y}\n'
        return prompt
    
    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        vote_results = [0] * n_candidates
        for vote_output in vote_outputs:
            pattern = r".*best choice is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(n_candidates):
                    vote_results[vote] += 1
            else:
                print(f'vote no match: {[vote_output]}')
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