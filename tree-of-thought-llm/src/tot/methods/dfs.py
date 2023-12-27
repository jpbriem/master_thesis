import itertools
import tkinter as tk
import numpy as np
from functools import partial
from tot.models import gpt
from tot.methods.tree_nodes import Node

# read multi line user inputs 
def read_multiline_input(query):
    lines = []
    eos = "eee"
    print(query)
    print(f'type {eos} after answer.')

    while True:
        line = input()
        if line == eos:
            break
        lines.append(line)

    text = "\n".join(lines)
    return text

# Evaluation: get value of a child
def get_value(args, task, child, cache_value=True):
    delimiter = "\n#############################\n"
    
    value_prompt = task.value_prompt_wrap(child, task.steps)
    
    # If just one child  
    if len(child.parent.children) == 1:
        value = 1
        prompt_log = delimiter.join(["Value Prompt:\n" + "\n\n".join(value_prompt.values()), "Value Outputs:\nNo Valuation - Only one candidate"])
        return value, prompt_log
    
    if cache_value and str(value_prompt) in task.value_cache:
        return task.value_cache[str(value_prompt)], value_prompt
    
    if args.use_api:
        value_outputs = gpt(value_prompt, n=args.n_evaluate_sample, stop=None)
    else: 
        # get values from chat interface
        value_outputs = []
        for i in range(args.n_evaluate_sample):
            print(value_prompt["system"] + "\n" + value_prompt["user"])
            value_output = read_multiline_input("Answer of LLM (type '<END>' after answer): ")
            value_outputs.append(value_output)
            
    value = task.value_outputs_unwrap(value_outputs, child.level-1)
    if cache_value:
        task.value_cache[str(value_prompt)] = value
    prompt_log = delimiter.join(["Value Prompt:\n" + "\n\n".join(value_prompt.values()), "Value Outputs:\n" + "\n------\n".join(value_outputs)])
    return value, prompt_log

# Evaluation: get values of children
def get_values(args, task, current_node, cache_value=True):
    prompt_log = []
    local_value_cache = {}
      
    # valuation
    for child in current_node.children:  # each partial output
        if child.LLM_answer in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value, value_prompt = get_value(args, task, child, cache_value=cache_value)
            child.value = value
            local_value_cache[child.LLM_answer] = value
        prompt_log.append(value_prompt)
   
    # log
    delimiter = "\n###########################################################\n"
    prompt_log = delimiter + delimiter.join([str(s) for s in prompt_log])
    return prompt_log

# Evaluation: get vote for a child
def get_votes(task, current_node, n_evaluate_sample):
    if len(current_node.children) == 1:
        current_node.children[0].value = 1
        return "\n###########################################################\nNo Valuation - Only one candidate\n"
 
    # voting
    vote_prompt = task.vote_prompt_wrap(current_node, task.steps) # TODO: add params to all calls
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(current_node, vote_outputs)
    for value, child in zip(values, current_node.children):
        child.value = value
        
    #log
    delimiter = "\n###########################################################\n"
    prompt_log = delimiter + delimiter.join(["Vote Prompt:\n" + "\n\n".join(vote_prompt.values()), "Vote Outputs:\n" + "\n------\n".join(vote_outputs), "Vote Values: "+ str([n.value for n in current_node.children])])
    
    return prompt_log

# Generation: get new samples
def get_samples(args, task, current_node, prompt_sample, stop):
    # sampling
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(current_node)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(current_node, task.steps) # TODO: add params to all calls
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    if args.use_api:
        samples = gpt(prompt, n=current_node.n_generate_children, stop=stop)
    else:
        # get samples from chat interface
        samples = []
        for i in range(current_node.n_generate_children):
            print(prompt["system"] + "\n" + prompt["user"])
            sample = read_multiline_input("Answer of LLM: ")
            samples.append(sample)
            
    # log
    delimiter = "\n###########################################################\n"
    prompt_log = delimiter.join(["Sample Prompt:\n" + "\n\n".join(prompt.values()), "Sample Outputs:\n" + "\n------\n".join(samples)])
    
    # turn samples in nodes
    if current_node.level+1 == task.steps:
        leaf = True
    else:
        leaf = False
    for sample in samples:
        new_node = Node(current_node.level+1, current_node.x, LLM_answer=sample, parent=current_node, n_generate_children=args.n_generate_sample, children=[], leaf=leaf)
        current_node.children.append(new_node)
    
    if task.__class__.__name__ in ["ARCTask", "ARC_1D"]:
        return prompt_log
    #return [y + _ for _ in samples], prompt_log # TODO: apply to old tasks


# Revision: anaylse failure of abstraction on example
def analyse_failure(args, task, node):
    prompt = task.failure_analysis_prompt_wrap(node)             
    if args.use_api:
        output = gpt(prompt, n=1)
    else:
        # get output from chat interface
        print(prompt["system"] + "\n" + prompt["user"])
        output = [read_multiline_input("Answer of LLM: ")]
    analysis_result = task.failure_analysis_prompt_unwrap(output, node)

    # create new node
    new_node = Node(node.level+1, node.x, LLM_answer=output[0], thought=analysis_result, parent=node, children=[])
    node.children.append(new_node)
    
    # log
    delimiter = "\n###########################################################\n"
    analysis_log = delimiter.join(["Analysis Prompt:\n" + "\n".join(prompt.values()),"Analysis Result:\n" + str(output)])    
    
    return analysis_log

# Revision: revise abstraction
def revise(args, task, node):
    # revise abstraction
    prompt = task.revision_prompt_wrap(node)
    if args.use_api:
        output = gpt(prompt, n=1)
    else:
        # get output from chat interface
        print(prompt["system"] + "\n" + prompt["user"])
        output = [read_multiline_input("Answer of LLM: ")]
    revision_result = task.revision_prompt_unwrap(output, node)
    
    # create new node
    new_node = Node(node.level+1, node.x, LLM_answer=output[0], thought=revision_result, parent=node, children=[])
    node.children.append(new_node)
    
    # replace old thoughts with revised thoughts - for each element in thought revise one more parent node one layer higher
    replacement_log = task.replace_revised_thoughts(new_node)    
    
    # log
    delimiter = "\n###########################################################\n"
    revision_log = delimiter.join(["Revision Prompt:\n" + "\n".join(prompt.values()),"Revision Result:\n" + str(output)])    
    revision_log += delimiter + replacement_log 
    
    return revision_log
    
# Revision: Loop
def revise_abstraction(args, task, original_node):
    # log
    delimiter = "\n###########################################################\n"
    revision_log = delimiter + "Abstraction Revision\n" + delimiter

    # work with copy of node
    node = original_node.copy()
    
    # tracker for example success
    n_examples = len(node.x["train"])
    example_success = [False]*n_examples
    
    # revision in a loop till termination
    current_test_idx = 0 
    revisions_in_a_row = 0 # for termination condition
    revision_last_iteration = False # for termination condition
    revisions_total = 0 # for termination condition
    max_revisions = 2*n_examples # for termination condition
    while True:
        # termination conditions
        revisions_in_a_row = revisions_in_a_row + 1 if revision_last_iteration else 0
        revision_last_iteration = False
        if example_success.count(True) == n_examples or revisions_in_a_row == n_examples or revisions_total >= max_revisions:
            break

        # change train and test samples in node.x to simulate current example as test case
        node.x = task.simulate_ex_as_test_case(original_node.x, current_test_idx)

        # apply abstraction to solve current example -> get child node
        revision_log += get_samples(args, task, node, prompt_sample=args.prompt_sample, stop=task.stops[node.level])
        example_test_node = node.children[0]
        
        # test the answer, which is in child 
        is_success = task.test_output(node=example_test_node, output=example_test_node.LLM_answer, is_revision=True)
        
        # if success: move to next example
        if is_success:
            example_success[current_test_idx] = True
            current_test_idx += 1
            revision_log += delimiter + "Example solved!\n" + delimiter
        # if failure: 
        else:
            current_test_idx += 1
            
            # compare wrong answer (which is in example_test_node) to gt
            revision_log += delimiter + analyse_failure(args, task, example_test_node)
            analysis_node = example_test_node.children[0]
            analysis_node.current_test_idx = current_test_idx
            # revise abstraction 
            revision_last_iteration = True
            example_success = [False]*n_examples
            revisions_total += 1
            revision_log += delimiter + revise(args, task, analysis_node)
        
        # reset children of node for next example iteration
        node.children = []
            
    return revision_log, example_success

def depth_first_search_prioritized(args, task, current_node, step, best_leaf_nodes=[], best_abstraction_node=[], example_success=[False], infos=[], to_print=True):
    if current_node.isLeaf:  # Leaf node
        return [current_node], best_abstraction_node, example_success, infos
    
    # generation  # TODO: Rename? Generate children?
    if args.method_generate == 'sample':
        # Sample: 1. Standard, 2. CoT, 3. Multiple CoT (w self-consistency)
        gen_prompts = get_samples(args, task, current_node, prompt_sample=args.prompt_sample, stop=task.stops[step])
    # elif args.method_generate == 'propose':
    #     # Propose potential next steps, define in promptamount of proposals
    #     new_ys, gen_prompts = get_proposals(task, x, y)
    new_ys = current_node.children
    
    # evaluation
    if args.method_evaluate == 'vote':
        # always vote for single best child, n_evalute_sample times
        eval_prompts = get_votes(task, current_node, args.n_evaluate_sample)
    elif args.method_evaluate == 'value':
        eval_prompts = get_values(args, task, current_node)
    values = [n.value for n in current_node.children]
    
    # selection / pruning
    if args.method_select == 'sample':
        ps = np.array([n.value for n in current_node.children]) / sum([n.value for n in current_node.children])
        current_node.children = np.random.choice(current_node.children, size=args.n_select_sample, p=ps, replace=False).tolist()
        current_node.children = sorted(current_node.children, key=lambda n: n.value, reverse=True)
    elif args.method_select == 'greedy':
        current_node.children = sorted(current_node.children, key=lambda n: n.value, reverse=True)[:args.n_select_sample]

    # if needed, update children nodes: phase & spreading
    for child in current_node.children:
        task.update_node(child)
    
    # log
    if to_print: 
        sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
        print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {current_node.children}\n')
    prompt_log = '\n'.join([gen_prompts, eval_prompts])
    infos.append({'step': step, 'x': current_node.x, 'ys': current_node.LLM_answer, 'new_ys': [str(y) for y in new_ys], 'values': values, 'select_new_ys': [str(child) for child in current_node.children], 'prompt_log': prompt_log})
    
    step += 1
    leaf_nodes = []
    for child in current_node.children:
        if all(example_success):
            # abstraction was already successful on examples -> no further search needed 
            continue
        
        if args.revision and child.phase == "application":
            revision_log, example_success = revise_abstraction(args, task, child) # TODO: Add params
            # TODO: what is in log? Add to infos?
            if all(example_success):
                # abstraction is successfull on examples -> apply to all test cases    
                leaf_nodes, best_abstraction_node, example_success, infos = depth_first_search_prioritized(args, task, child, step, best_leaf_nodes, best_abstraction_node, example_success, infos, to_print) 
            else:
                # TODO: WRONG loic #######################################################################################################################################
                # at least one example was wrong: value of instruction becomes # of solved examples
                child.value = example_success.count(True)
                if best_abstraction_node and best_abstraction_node.value < child.value:
                    best_abstraction_node = child
                # leaf_nodes.append(child) IST KEIN LEAF SONDER INSTRUCTIONS NODE 
        else:
            leaf_nodes, best_abstraction_node, example_success, infos = depth_first_search_prioritized(args, task, child, step, best_leaf_nodes, best_abstraction_node, example_success, infos, to_print) 
        for leaf_node in leaf_nodes:
            if leaf_node in best_leaf_nodes:
                continue
            if len(best_leaf_nodes) < args.n_select_sample:
                best_leaf_nodes.append(leaf_node)
            else:
                if leaf_node.value > min([n.value for n in best_leaf_nodes]):
                    best_leaf_nodes.remove(min(best_leaf_nodes, key=lambda n: n.value))
                    best_leaf_nodes.append(leaf_node)
    best_leaf_nodes = sorted(best_leaf_nodes, key=lambda n: n.value, reverse=True)

    # TODO: Muss anggepasst werden an Baum!!
    if len(best_leaf_nodes) == 0 and best_abstraction_node:
        # abstraction not successfull on examples -> take best abstraction so far
        leaf_nodes, best_abstraction_node, example_success, infos = depth_first_search_prioritized(args, task, best_abstraction_node, step, best_leaf_nodes, best_abstraction_node, example_success, infos, to_print) 
  
        
        
    return best_leaf_nodes, best_abstraction_node, example_success, infos

def solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    root = Node(0, x, n_generate_children=args.n_generate_sample, children=[])
    task.update_node(root)
    best_leaf_nodes, best_abstraction_node, revisionSuccess, infos = depth_first_search_prioritized(args, task, root, step=0, best_leaf_nodes=[], best_abstraction_node=[], infos=[], to_print=to_print)
    return best_leaf_nodes, {'steps': infos}
    

def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    root = Node(0, x, n_generate_children=args.n_generate_sample, children=[])
    prompt_log = get_samples(args, task, root, args.prompt_sample, stop=None)
    return root.children, {'x': root.x, 'LLM_answers': [str(y.LLM_answer) for y in root.children], 'new_ys': [str(y) for y in root.children], 'prompt_log': prompt_log}
 
