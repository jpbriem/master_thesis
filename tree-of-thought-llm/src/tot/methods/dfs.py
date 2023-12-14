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

def get_values(args, task, current_node, cache_value=True):
    prompt_log = []
    local_value_cache = {}
      
    # valuation
    for child in current_node.children:  # each partial output
        if child.LLM_answer in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value, value_prompt = get_value(args, task, child, args.n_evaluate_sample, cache_value=cache_value)
            child.value = value
            local_value_cache[child.LLM_answer] = value
        prompt_log.append(value_prompt)
   
    # log
    delimiter = "\n###########################################################\n"
    prompt_log = delimiter + delimiter.join([str(s) for s in prompt_log])
    return prompt_log

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


def depth_first_search_prioritized(args, task, current_node, step, best_leaf_nodes=[], abstractionSuccess=False, infos=[], to_print=True):
    if current_node.isLeaf:  # Leaf node
        return [current_node], infos
    
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
        child.update_node()
    
    # log
    if to_print: 
        sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
        print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {current_node.children}\n')
    prompt_log = '\n'.join([gen_prompts, eval_prompts])
    infos.append({'step': step, 'x': current_node.x, 'ys': current_node.LLM_answer, 'new_ys': [str(y) for y in new_ys], 'values': values, 'select_new_ys': [str(child) for child in current_node.children], 'prompt_log': prompt_log})
    
    step += 1
    for child in current_node.children:
        if abstractionSuccess:
            # abstraction was already successful on examples -> no further search needed 
            continue
        
        if args.revision and child.phase == "application":
            revision_log, abstractionSuccess = task.abstraction_revision_wrap(child)
            # TODO: what is in log? Add to infos?
            if abstractionSuccess:
                # abstraction is successfull on examples -> apply to test case    
                leaf_nodes, abstractionSuccess, infos = depth_first_search_prioritized(args, task, child, step, best_leaf_nodes, abstractionSuccess, infos, to_print) 
        
        else:
            leaf_nodes, abstractionSuccess, infos = depth_first_search_prioritized(args, task, child, step, best_leaf_nodes, abstractionSuccess, infos, to_print) 
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

    return best_leaf_nodes, abstractionSuccess, infos

def solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    x = task.get_input(idx)  # input
    root = Node(0, x, n_generate_children=args.n_generate_sample, children=[])
    best_leaf_nodes, revisionSuccess, infos = depth_first_search_prioritized(args, task, root, step=0, best_leaf_nodes=[], infos=[], to_print=to_print)
    return best_leaf_nodes, {'steps': infos}
    
    
 
