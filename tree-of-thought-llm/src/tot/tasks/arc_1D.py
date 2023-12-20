import os
import re
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.arc_1D import * # TODO: use ARC prompts
from tot.models import gpt
from tot.tasks.arc_full_plan import ARCTask
from tot.methods.arc_utils import *
from tot.methods.tree_nodes import Node

class ARC_1D(ARCTask):
    """
    Input (x)   : 1D list of pixels
    Output (y)  : 1D grid of pixels 
    Input Example:  [0, 1, 1, 1, 1, 0, 0, 0, 0]
    Output Example: [0, 0, 1, 1, 1, 1, 0, 0, 0]
    Pattern of Example: shift right by 1
    """
    # prompt_modules = prompt_modules
    def __init__(self):
        """
        several subfolders by task type
        """
        path = os.path.join(DATA_PATH, 'arc-1D')
        self.data, self.names = load_arc_tasks(path, "arc-1D")
        self.steps = int(list(prompt_modules.keys())[-1])+1 # +1 bc. steps start at 0
        self.stops = [None]*self.steps # TODO: adjust to prompt! 
        self.success = {} # saves success rates for each task
        self.full_success = 0 # counts completely solved tasks
        self.value_cache = {}

    def test_output(self, idx: int=0, output: str="", prompt_modules: dict=prompt_modules, dataset: str="arc", is_revision: bool=False, node: Node=None):      
        output_format = prompt_modules[str(self.steps-1)]["generation"]["output_format"]

        # if revision of abstraction based on examples, get task from revision node
        if is_revision:
            task_json = node.x.copy()
        # otherwise, get test case from task data & initialize success tracker
        else:
            # get test case
            task_json = self.data[idx]
            task_name = self.names[idx]
            if task_name not in self.success: # TODO: works currently only if we have just one try
                self.success[task_name] = 0
            
        _, solutions = get_tasks(task_json, DELIMITER[dataset])
        for solution in solutions[:1]: # TODO: currently just check first test case in case more exist
            output_key = list(output_format.keys())[-1]
            test_output_grid = extract_json_value(output, output_format, output_key) 
            test_output_grid = grid_to_2D_nparray(test_output_grid)
            solution = grid_to_2D_nparray(solution)
            is_success = np.array_equal(test_output_grid, solution)
            if not is_revision:
                self.success[task_name] += is_success / len(solutions)
                    
        # log the success if not revision
        if is_revision:
            node.thought = test_output_grid.tolist()
            return is_success
        else:
            if self.success[task_name] == 1:
                self.full_success += 1
            # print('------------')
            info = {'rs': self.success[task_name], 'r': self.full_success / len(self)}
        
        return info
   
    @staticmethod
    def update_node(node, prompt_modules: dict=prompt_modules):
        return ARCTask.update_node(node, prompt_modules)
    
    @staticmethod
    def standard_prompt_wrap(node, standard_prompt: str=standard_prompt, dataset: str="arc-1D") -> str:
        return ARCTask.standard_prompt_wrap(node, standard_prompt, dataset)

    @staticmethod
    def cot_prompt_wrap(node, total_steps: int=1, cot_prompt: str=cot_prompt, prompt_modules: dict=prompt_modules, dataset: str="arc-1D") -> str:
        return ARCTask.cot_prompt_wrap(node, total_steps, cot_prompt, prompt_modules, dataset)
    
    @staticmethod
    def value_prompt_wrap(node, total_steps: int=1, value_prompt: str=value_prompt, prompt_modules: dict=prompt_modules, dataset: str="arc-1D") -> str:
        return ARCTask.value_prompt_wrap(node, total_steps, value_prompt, prompt_modules, dataset)
           
    @staticmethod
    def value_outputs_unwrap(value_outputs: list, current_step: int=0, prompt_modules: dict=prompt_modules) -> float:
        return ARCTask.value_outputs_unwrap(value_outputs, current_step, prompt_modules)
        
    @staticmethod
    def failure_analysis_prompt_wrap(node, failure_analysis_prompt: str=failure_analysis_prompt, prompt_modules: dict=prompt_modules, dataset: str="arc-1D") -> str:
        return ARCTask.failure_analysis_prompt_wrap(node, failure_analysis_prompt, prompt_modules, dataset)    
    
    @staticmethod
    def failure_analysis_prompt_unwrap(output, node, prompt_modules: dict=prompt_modules) -> str:
        return ARCTask.failure_analysis_prompt_unwrap(output, node, prompt_modules)

    @staticmethod
    def revision_prompt_wrap(node, revision_prompt: str=revision_prompt, prompt_modules: dict=prompt_modules, dataset: str="arc-1D") -> str:
        return ARCTask.revision_prompt_wrap(node, revision_prompt, prompt_modules, dataset)
    
    @staticmethod
    def revision_prompt_unwrap(output, node, prompt_modules: dict=prompt_modules) -> str:
        return ARCTask.revision_prompt_unwrap(output, node, prompt_modules)
        
    @staticmethod
    def replace_revised_thoughts(node, prompt_modules: dict=prompt_modules):
        return ARCTask.replace_revised_thoughts(node, prompt_modules)
   
    @staticmethod
    def simulate_ex_as_test_case(original_x, currrent_test_idx):
        return ARCTask.simulate_ex_as_test_case(original_x, currrent_test_idx)