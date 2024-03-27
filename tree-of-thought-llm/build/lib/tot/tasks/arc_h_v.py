import os
import re
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.arc import *
from tot.tasks.arc_full_plan import ARCTask
from tot.methods.arc_utils import *
from tot.methods.tree_nodes import Node

class ARC_h_v(ARCTask):
    """
    Input (x)   : 2D grid of pixels
    Output (y)  : 2D grid of pixels 
    Input Example:  [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    Output Example: [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    """
    
    # class variable
    prompt_modules = prompt_modules
    few_shot_ex = few_shot_ex
    
    def __init__(self):
        """
        several subfolders by task type
        """
        self.path = os.path.join(DATA_PATH, 'arc_h_v')
        self.args = None
        self.data, self.names, self.categories = load_arc_tasks(self.path, "arc_h_v")
        self.steps = int(list(prompt_modules.keys())[-1])+1 # +1 bc. steps start at 0
        self.stops = [None]*self.steps # TODO: adjust to prompt! 
        self.success = {} # saves success rates for each task
        self.full_success = 0 # counts completely solved tasks
        self.cat_success, self.cat_failures = {}, {} # saves success cnt for each category
        self.object_representation_success_cnt = 0
        self.object_representation_success = {} # saves obj repres. success rates for each task
        self.object_representation_cat_success, self.object_representation_cat_failures = {}, {} # saves success cnt for each category
        self.too_long_prompts_no_output = {}
        self.too_long_prompts_all = {'sampling': [], 'value': [], 'vote': []}
        self.tasks_failed_solving = {}
        self.solved_tasks = []
        self.solved_tasks_str_comparison = []
        self.solved_tasks_object_representation = []
        self.value_cache = {}


    # NOTE No need to overwrite methods as logic, prompts, etc. are the same as in ARCTask (normal 2D ARC)
    
    # def test_output(self, idx: int=0, outputs: str=[""], prompt_modules: dict=prompt_modules, dataset: str="arc", is_revision: bool=False, node: Node=None):      
    #     return super().test_output(idx, outputs, dataset)

    # def test_output_naive(self, idx: int=0, output: str="", dataset: str="arc"):
    #     return super().test_output_naive(idx, output, dataset)
   
    # @staticmethod
    # def update_node(node, prompt_modules: dict=prompt_modules):
    #     return ARCTask.update_node(node, prompt_modules)
    
    # @staticmethod
    # def standard_prompt_wrap(node, standard_prompt: str=standard_prompt, dataset: str="arc-1D") -> str:
    #     return ARCTask.standard_prompt_wrap(node, standard_prompt, dataset)

    # @staticmethod
    # def cot_prompt_wrap(node, total_steps: int=1, cot_prompt: str=cot_prompt, prompt_modules: dict=prompt_modules, dataset: str="arc-1D") -> str:
    #     return ARCTask.cot_prompt_wrap(node, total_steps, cot_prompt, prompt_modules, dataset)
    
    # @staticmethod
    # def value_prompt_wrap(node, total_steps: int=1, value_prompt: str=value_prompt, prompt_modules: dict=prompt_modules, dataset: str="arc-1D") -> str:
    #     return ARCTask.value_prompt_wrap(node, total_steps, value_prompt, prompt_modules, dataset)
           
    # @staticmethod
    # def value_outputs_unwrap(value_outputs: list, current_step: int=0, prompt_modules: dict=prompt_modules) -> float:
    #     return ARCTask.value_outputs_unwrap(value_outputs, current_step, prompt_modules)
        
    # @staticmethod
    # def failure_analysis_prompt_wrap(node, failure_analysis_prompt: str=failure_analysis_prompt, prompt_modules: dict=prompt_modules, dataset: str="arc-1D") -> str:
    #     return ARCTask.failure_analysis_prompt_wrap(node, failure_analysis_prompt, prompt_modules, dataset)    
    
    # @staticmethod
    # def failure_analysis_prompt_unwrap(output, node, prompt_modules: dict=prompt_modules) -> str:
    #     return ARCTask.failure_analysis_prompt_unwrap(output, node, prompt_modules)

    # @staticmethod
    # def revision_prompt_wrap(node, revision_prompt: str=revision_prompt, prompt_modules: dict=prompt_modules, dataset: str="arc-1D") -> str:
    #     return ARCTask.revision_prompt_wrap(node, revision_prompt, prompt_modules, dataset)
    
    # @staticmethod
    # def revision_prompt_unwrap(output, node, prompt_modules: dict=prompt_modules) -> str:
    #     return ARCTask.revision_prompt_unwrap(output, node, prompt_modules)
        
    # @staticmethod
    # def replace_revised_thoughts(node, prompt_modules: dict=prompt_modules):
    #     return ARCTask.replace_revised_thoughts(node, prompt_modules)
   
    # @staticmethod
    # def simulate_ex_as_test_case(original_x, currrent_test_idx):
    #     return ARCTask.simulate_ex_as_test_case(original_x, currrent_test_idx)