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
        self.stops = [None]*self.steps 
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
  