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
    
    # class variable
    prompt_modules = prompt_modules
    few_shot_ex = few_shot_ex

    def __init__(self):
        """
        several subfolders by task type
        """    
        path = os.path.join(DATA_PATH, 'arc_1D')
        self.data, self.names, self.categories = load_arc_tasks(path, "arc_1D")
        self.steps = int(list(prompt_modules.keys())[-1])+1 # +1 bc. steps start at 0
        self.stops = [None]*self.steps # TODO: adjust to prompt! 
        self.success = {} # saves success rates for each task
        self.solved_tasks = []
        self.full_success = 0 # counts completely solved tasks
        self.cat_success, self.cat_failures = {}, {} # saves success rates for each category
        self.too_long_prompts_no_output = {}
        self.too_long_prompts_all = {'sampling': [], 'value': [], 'vote': []}
        self.tasks_failed_solving = {}
        self.solved_tasks = []
        self.solved_tasks_str_comparison = []
        self.value_cache = {}

    
    # Overwrite all methods to pass correct prompts and refer to correct dataset

    def test_output(self, idx: int=0, outputs: str=[""], prompt_modules: dict=None, dataset: str="arc", is_revision: bool=False, node: Node=None):      
        if prompt_modules is None:
            prompt_modules = ARC_1D.prompt_modules
        return super().test_output(idx, outputs, prompt_modules, dataset, is_revision, node)

    def test_output_naive(self, idx: int=0, outputs: list=[""], prompt_modules: dict=None, dataset: str="arc_1D"):
        if prompt_modules is None:
            prompt_modules = ARC_1D.prompt_modules
        return super().test_output_naive(idx, outputs, prompt_modules, dataset)
    
    def update_prompt_modules(self, type: str="naive", p: dict=prompt_modules_naive):
        if type == "naive":
            ARC_1D.prompt_modules = p
            self.steps = int(list(p.keys())[-1])+1 # +1 bc. steps start at 0
            
    
    @staticmethod
    def update_node(node, prompt_modules: dict=None):
        if prompt_modules is None:
            prompt_modules = ARC_1D.prompt_modules
        return ARCTask.update_node(node, prompt_modules)
    
    @staticmethod
    def standard_prompt_wrap(node, standard_prompt: str=standard_prompt, dataset: str="arc_1D") -> str:
        return ARCTask.standard_prompt_wrap(node, standard_prompt, dataset)

    @staticmethod
    def cot_prompt_wrap(node, total_steps: int=1, cot_prompt: str=cot_prompt, prompt_modules: dict=None, few_shot_ex: dict=None, dataset: str="arc_1D") -> str:
        if prompt_modules is None:
            prompt_modules = ARC_1D.prompt_modules
        if few_shot_ex is None:
            few_shot_ex = ARC_1D.few_shot_ex
        return ARCTask.cot_prompt_wrap(node, total_steps, cot_prompt, prompt_modules, few_shot_ex, dataset)
    
    @staticmethod
    def value_prompt_wrap(node, total_steps: int=1, value_prompt: str=value_prompt, prompt_modules: dict=None, dataset: str="arc_1D") -> str:
        if prompt_modules is None:
            prompt_modules = ARC_1D.prompt_modules
        return ARCTask.value_prompt_wrap(node, total_steps, value_prompt, prompt_modules, dataset)
           
    @staticmethod
    def value_outputs_unwrap(value_outputs: list, current_step: int=0, prompt_modules: dict=None) -> float:
        if prompt_modules is None:
            prompt_modules = ARC_1D.prompt_modules
        return ARCTask.value_outputs_unwrap(value_outputs, current_step, prompt_modules)
        
    @staticmethod
    def failure_analysis_prompt_wrap(node, failure_analysis_prompt: str=failure_analysis_prompt, prompt_modules: dict=None, dataset: str="arc_1D") -> str:
        if prompt_modules is None:
            prompt_modules = ARC_1D.prompt_modules
        return ARCTask.failure_analysis_prompt_wrap(node, failure_analysis_prompt, prompt_modules, dataset)    
    
    @staticmethod
    def failure_analysis_prompt_unwrap(output, node, prompt_modules: dict=None) -> str:
        if prompt_modules is None:
            prompt_modules = ARC_1D.prompt_modules
        return ARCTask.failure_analysis_prompt_unwrap(output, node, prompt_modules)

    @staticmethod
    def revision_prompt_wrap(node, revision_prompt: str=revision_prompt, prompt_modules: dict=None, dataset: str="arc_1D") -> str:
        if prompt_modules is None:
            prompt_modules = ARC_1D.prompt_modules
        return ARCTask.revision_prompt_wrap(node, revision_prompt, prompt_modules, dataset)
    
    @staticmethod
    def revision_prompt_unwrap(output, node, prompt_modules: dict=None) -> str:
        if prompt_modules is None:
            prompt_modules = ARC_1D.prompt_modules
        return ARCTask.revision_prompt_unwrap(output, node, prompt_modules)
        
    @staticmethod
    def replace_revised_thoughts(node, original_node, prompt_modules: dict=None):
        if prompt_modules is None:
            prompt_modules = ARC_1D.prompt_modules
        return ARCTask.replace_revised_thoughts(node, original_node, prompt_modules)
   
    @staticmethod
    def simulate_ex_as_test_case(original_x, currrent_test_idx):
        return ARCTask.simulate_ex_as_test_case(original_x, currrent_test_idx)