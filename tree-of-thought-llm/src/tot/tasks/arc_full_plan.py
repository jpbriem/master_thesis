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

    # class variable
    prompt_modules = prompt_modules
    
    def __init__(self):
        """
        2 subfolders: training, evaluation
        """
        super().__init__()
        path = os.path.join(DATA_PATH, 'arc')
        self.data, self.names, self.categories = load_arc_tasks(path)
        self.steps = int(list(prompt_modules.keys())[-1])+1 # +1 bc. steps start at 0
        self.stops = [None]*self.steps # TODO: adjust to prompt! 
        self.success = {} # saves success rates for each task
        self.full_success = 0 # counts completely solved tasks
        self.cat_success = {} # saves success rates for each category
        self.solved_tasks = []
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
    

    
    def test_output(self, idx: int=0, outputs: list=[""], prompt_modules: dict=None, dataset: str="arc", is_revision: bool=False, node: Node=None):      
        if prompt_modules is None:
            prompt_modules = ARCTask.prompt_modules
        output_format = prompt_modules[str(self.steps-1)]["generation"]["output_format"]
        
        # if revision of abstraction based on examples, get task from revision node
        if is_revision:
            task_json = node.x.copy()
        # otherwise, get test case from task data & initialize success tracker
        else:
            task_json = self.data[idx]
            task_name = self.names[idx]
            category = self.categories[idx]
            if task_name not in self.success: # TODO: works currently only if we have just one try
                self.success[task_name] = 0
        
        _, solutions = get_tasks(task_json, DELIMITER[dataset])
        solution = solutions[0] # TODO: currently just check first test case 
        
        try_cnt = 0
        for output in outputs:
            output = output.LLM_answer
            try_cnt += 1 
            output_key = list(output_format.keys())[-1]
            test_output_grid = extract_json_value(output, output_format, output_key) 
            test_output_grid = grid_to_2D_nparray(test_output_grid)
            solution = grid_to_2D_nparray(solution)
            is_success = np.array_equal(test_output_grid, solution)
            if is_success:
                break     
        
        # log the success if not revision
        if is_revision:
            node.thought = test_output_grid.tolist()
            return is_success
        else:
            self.success[task_name] += int(is_success) 
            if self.success[task_name] == 1:
                self.full_success += 1
                if category in self.cat_success:
                    self.cat_success[category] += 1
                else:
                    self.cat_success[category] = 1
            if self.success[task_name] > 0:
                self.solved_tasks.append(task_name)
            # print('------------')
            info = {'success': self.success[task_name], 'tries': try_cnt, 'total_result': self.full_success / len(self), 'category_result': self.cat_success[category] / self.categories.count(category)}
        
        return info
    
    def test_output_naive(self, idx: int=0, outputs: list=[""], dataset: str="arc"):
        task_json = self.data[idx]
        task_name = self.names[idx]
        category = self.categories[idx]
        if task_name not in self.success: # TODO: works currently only if we have just one try
            self.success[task_name] = 0
        if category not in self.cat_success:
            self.cat_success[category] = 0
            
        _, solutions = get_tasks(task_json, DELIMITER[dataset])
        solution = solutions[0] # TODO: currently just check first test case 

        try_cnt = 0
        for output in outputs:
            try_cnt += 1 
            is_success = re.sub(r'\s+', ' ', solution).strip() in re.sub(r'\s+', ' ', output.LLM_answer).strip()
            if is_success:
                break     
                        
        self.success[task_name] += is_success    
        if self.success[task_name] == 1:
            self.full_success += 1
            self.cat_success[category] += 1

        if self.success[task_name] > 0:
            self.solved_tasks.append(task_name)
        info = {'success': self.success[task_name], 'tries': try_cnt, 'total_result': self.full_success / len(self), 'category_result': self.cat_success[category] / self.categories.count(category)}
        return info
    
    def update_prompt_modules(self, type: str="naive", p: dict=prompt_modules_naive):
        if type == "naive":
            ARCTask.prompt_modules = p
            self.steps = int(list(p.keys())[-1])+1 # +1 bc. steps start at 0
    
    @staticmethod
    def update_node(node, prompt_modules: dict=None):
        if prompt_modules is None:
            prompt_modules = ARCTask.prompt_modules
        if node.level == len(prompt_modules):
            return
        
        # check if abstraction or application phase 
        node.phase = prompt_modules[str(node.level)]["phase"]
        
        # check if node level should spread
        isSpreader = prompt_modules[str(node.level)]["spread"]
        if not isSpreader:
            node.n_generate_children = 1
    
    @staticmethod 
    def standard_prompt_wrap(node, standard_prompt: str=standard_prompt, dataset: str="arc") -> str:
        task_context = get_context(node.x, DELIMITER[dataset], with_intro=False)
        task_input, _ = get_tasks(node.x, DELIMITER[dataset]) # TODO: currently just check first test case in case more exist        
        prompt = standard_prompt.copy()
        prompt["user"] = prompt["user"].format(context=task_context, test_input=task_input[0])
        return prompt

    @staticmethod # TODO: distingusih between abstraction & application
    def cot_prompt_wrap(node, total_steps: int=1, cot_prompt: str=cot_prompt, prompt_modules: dict=None, dataset: str="arc") -> str:
        if prompt_modules is None:
            prompt_modules = ARCTask.prompt_modules
        current_step = node.level
        
        # get arc examples
        task_context = get_context(node.x, DELIMITER[dataset])
        # get test case
        if current_step == total_steps-1:
            task_input, _ = get_tasks(node.x, DELIMITER[dataset]) # TODO: currently just check first test case in case more exist
            task_input[0] = "\n\n" + task_input[0]
        else:
            task_input = [""]        
        # get output format for current step
        output_format = prompt_modules[str(current_step)]["generation"]["output_format"]
        # get instructions for current step
        instruct = ""
        if total_steps == 1: # Naive run with COT prompt
            pass 
        # elif current_step == total_steps-1: #  For application don't use description of examples # NOTE: treat all the same!
        #     for i in range(current_step-2, current_step):
        #         instruct += prompt_modules[str(i)]["evaluation"]["instruct_previous_thoughts"]
        else:
            for i in range(current_step):
                instruct += prompt_modules[str(i)]["evaluation"]["instruct_previous_thoughts"]
        instruct += prompt_modules[str(current_step)]["generation"]["instruct_task"]
        
        # get previous thoughts
        if total_steps == 1: # Naive run with COT prompt
            previous_thoughts = ""
        elif node.current_test_idx: # we test an abstraction on an example
            # get previous thoughts
            previous_thoughts = get_previous_thoughts(node.parent.parent) # get only Example description of example under test
            previous_thoughts = previous_thoughts.split('\n')[node.current_test_idx+1] # +1 bc. first line is irrelevant TODO: does it work?= 
            previous_thoughts += get_previous_thoughts(node, 2) # use thoughts except description of examples   
        # elif current_step == total_steps-1: # NOTE: treat all the same! nod difference beteween abstraction & application phase
        #     previous_thoughts = f'{get_previous_thoughts(node, 2)}' # For application just take thoughts of nodes until 2 layers above 
        else:
            previous_thoughts = f'{get_previous_thoughts(node)}' # tak thoughts of all nodes above

        # get prompt template and fill
        prompt = cot_prompt.copy()
        prompt["system"] = cot_prompt["system"].format(output=output_format, special_instructions=instruct)
        prompt["user"] = cot_prompt["user"].format(context=task_context, test_input=task_input[0], previous_thoughts=previous_thoughts)

        return prompt 

    @staticmethod
    def value_prompt_wrap(node, total_steps: int=1, value_prompt: str=value_prompt, prompt_modules: dict=None, dataset: str="arc") -> str:
        if prompt_modules is None:
            prompt_modules = ARCTask.prompt_modules
        current_step = node.level-1 # -1 bc. node is the child to be evaluated
        
        # get arc examples
        task_context = get_context(node.x, DELIMITER[dataset])
        # get test case
        if current_step == total_steps-1:
            task_input, _ = get_tasks(node.x, DELIMITER[dataset]) # TODO: currently just check first test case in case more exist
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
        node.thought_before_revision = thought
        task_input[0] += f'{thought}'
        
        # get prompt template and fill 
        prompt = value_prompt.copy()
        prompt["system"] = value_prompt["system"].format(output=output_format, special_instructions=instruct)
        prompt["user"] = value_prompt["user"].format(context=task_context, test_input=task_input[0], previous_thoughts=previous_thoughts)

        return prompt

    @staticmethod
    def value_outputs_unwrap(value_outputs: list, current_step: int=0, prompt_modules: dict=None) -> float:
        if prompt_modules is None:
            prompt_modules = ARCTask.prompt_modules
        final_value = 0
        cnt_outputs = 0 # counter for number of outputs w valid value
        output_keys = extract_dict_keys(prompt_modules[str(current_step)]["evaluation"], "output_format")
        for value_output in value_outputs:
            value_output = get_json_from_text(value_output, output_keys)
            if isinstance(value_output, str): # error in json parsin
                continue
            cnt_examples = 0 # counter for number of examples w valid value
            value = 0 # sum up values over all examples
            example_id = 1
            if "Example_1" in value_output:
                while "Example_"+str(example_id) in value_output:
                    try:
                        value += get_int_from_dict_value(value_output["Example_"+str(example_id)], "value")
                        cnt_examples += 1
                    except:
                        log = "'value' from LLM not a single integer: " + str(value_output["Example_"+str(example_id)])
                        print(log)
                        path = "error_log/json_parsing_errors/"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+".txt"
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

    @staticmethod
    def failure_analysis_prompt_wrap(node, failure_analysis_prompt: str=failure_analysis_prompt, prompt_modules: dict=None, dataset: str="arc") -> str:
        if prompt_modules is None:
            prompt_modules = ARCTask.prompt_modules
        current_step = node.level - 1 # -1 bc. node is the child of the node under revision
        
        # get input and gt output
        delimmiter = DELIMITER.copy()
        delimmiter[dataset]["task_start"] = "" #  We dont want the prefix "Test Case:\n" (or similar) here
        delimmiter[dataset]["input_test"] = "" #  We dont want the prefix "input: " (or similar) here
        input, output_gt = get_tasks(node.x, DELIMITER[dataset])
        # get wrong output
        output_wrong = node.thought
        # get output format 
        output_format = prompt_modules[str(current_step)]["revision"]["analysis"]["output_format"]

        # get prompt template and fill
        prompt = failure_analysis_prompt.copy()
        prompt["system"] = failure_analysis_prompt["system"].format(output=output_format)
        prompt["user"] = failure_analysis_prompt["user"].format(test_input=input[0], output_gt=output_gt[0], output_wrong=output_wrong)

        return prompt
    
    @staticmethod
    def failure_analysis_prompt_unwrap(output, node, prompt_modules: dict=None) -> str:
        if prompt_modules is None:
            prompt_modules = ARCTask.prompt_modules
        current_step = node.level - 1 # -1 bc. node is the child of the node under revision
        output_keys = extract_dict_keys(prompt_modules[str(current_step)]["revision"]["analysis"], "output_format")   
        output_format = prompt_modules[str(current_step)]["revision"]["analysis"]["output_format"]
        thought_key = list(output_format.keys())[-1] # new thought is always last item in dict
        thought_data = extract_json_value(output[0], output_keys, thought_key)
        if isinstance(thought_data, dict):
            thought = ""
            for key, value in thought_data.items():
                thought += f'\n{" ".join(key.split("_"))}: {value}'
        else:
            thought = "\n" + " ".join(thought_key.split("_")) + ": "
            thought += f'{thought_data}'
        return thought
            
    @staticmethod
    def revision_prompt_wrap(node, revision_prompt: str=revision_prompt, prompt_modules: dict=None, dataset: str="arc") -> str:
        if prompt_modules is None:
            prompt_modules = ARCTask.prompt_modules
        current_step = node.level - 2 # -2 bc. node is the grand child of the node under revision
        
        # get the arc example as context that was tried to be solved
        node.x["train"] = node.x["test"]
        task_context = get_context(node.x, DELIMITER[dataset])
        
        # get output format for current step
        output_format = prompt_modules[str(current_step)]["revision"]["revision"]["output_format"]
        
        # get instructions for current step
        instruct = ""
        for i in range(current_step): # take instructions from all previous steps except test case generation step
            instruct += prompt_modules[str(i)]["evaluation"]["instruct_previous_thoughts"]
        instruct += prompt_modules[str(current_step)]["revision"]["revision"]["instruct_task"]
        
        # get previous thoughts
        previous_thoughts = get_previous_thoughts(node.parent.parent.parent.parent) # get only Example description of example under test
        previous_thoughts = previous_thoughts.split('\n')[node.current_test_idx+1] # +1 bc. first line is irrelevant # TODO: does it work? 
        previous_thoughts += get_previous_thoughts(node.parent.parent, 2) # use thoughts of node under revision and higher except description of examples

        # add hypotheses regarding potential mistakes 
        hypotheses = get_thought(node.LLM_answer, prompt_modules, current_step, isRevision=True)
        node.thought = hypotheses
        
        # get output format 
        output_format = prompt_modules[str(current_step)]["revision"]["revision"]["output_format"]

        # get prompt template and fill
        prompt = revision_prompt.copy()
        prompt["system"] = revision_prompt["system"].format(output=output_format, special_instructions=instruct)
        prompt["user"] = revision_prompt["user"].format(context=task_context, previous_thoughts=previous_thoughts, hypotheses=hypotheses)

        return prompt
    
    @staticmethod
    def revision_prompt_unwrap(output, node, prompt_modules: dict=None) -> str:
        if prompt_modules is None:
            prompt_modules = ARCTask.prompt_modules
        current_step = node.level - 2 # -2 bc. node is the grand child of the node under revision
        output_keys = extract_dict_keys(prompt_modules[str(current_step)]["revision"]["revision"], "output_format")   
        output_format = prompt_modules[str(current_step)]["revision"]["revision"]["output_format"]
        thought_key = list(output_format.keys())[-1] # new thought is always last item in dict
        thought_data = extract_json_value(output[0], output_keys, thought_key)
        if isinstance(thought_data, dict):
            thought = ""
            for key, value in thought_data.items():
                thought += f'\n{" ".join(key.split("_"))}: {value}'
        else:
            thought = "\n" + " ".join(thought_key.split("_")) + ": "
            thought += f'{thought_data}'
        return thought
        
    @staticmethod
    def replace_revised_thoughts(node, prompt_modules: dict=None):
        if node.level == 3:
            # this means the initial abstraction was the best: return to initial thoughts
            node.thought = node.thought_before_revision
            node.parent = node.parent.thought_before_revision
            return "Reset to initial thoughts."
        if prompt_modules is None:
            prompt_modules = ARCTask.prompt_modules
        replacement_log = ""
        current_step = node.level - 3 # -3 bc. node is the grand grand child of the node under revision
        revision_node = node.parent.parent.parent
        output_keys = extract_dict_keys(prompt_modules[str(current_step)]["revision"]["revision"], "output_format")   
        output_format = prompt_modules[str(current_step)]["revision"]["revision"]["output_format"]
        thought_key = list(output_format.keys())[-1] # new thought is always last item in dict
        thought_data = extract_json_value(node.LLM_answer, output_keys, thought_key)
        if isinstance(thought_data, dict):
            for i, (key, value) in enumerate(reversed(thought_data.items()), 1):
                thought = f'{" ".join(key.split("_"))}: {value}'                
                replacement_log += f'\n\n\nRevised node {i} layers above.\nOld thought: {revision_node.thought}\nNew thought: {thought}'
                revision_node.thought = thought
                revision_node = revision_node.parent
        else:
            thought = "\n" + " ".join(thought_key.split("_")) + ": "
            thought += f'{thought_data}'
            replacement_log += f'\n\n\nRevised node 1 layer above.\nOld thought: {revision_node.thought}\nNew thought: {thought}'
            revision_node.thought = thought
        return replacement_log
   
    @staticmethod
    def simulate_ex_as_test_case(original_x, currrent_test_idx):
        x = original_x.copy()
        # turn examples in context and test case
        context = original_x["train"][:currrent_test_idx]+original_x["train"][currrent_test_idx+1:]
        test_case = [original_x["train"][currrent_test_idx]]
        x["train"], x["test"] = context, test_case
        return x
    
       
            






    # TODO: NEEDED?!
    @staticmethod
    def vote_prompt_wrap(node, total_steps: int=1, dataset: str="arc") -> str:
        task_context = get_context(node.x, DELIMITER[dataset])
        task_input, _ = get_tasks(node.x, DELIMITER[dataset]) # TODO: currently just check first test case in case more exist
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
                path = "error_log/json_parsing_errors/"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+".txt"
                with open(path, "w") as text_file:
                    text_file.write(log)
            
            if (vote-1) in range(len(node.children)): # vote -1 bc. indexing starts at 0
                values[vote-1] += 1
        return values
    
    
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