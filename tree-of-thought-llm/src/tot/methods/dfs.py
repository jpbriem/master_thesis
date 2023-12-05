import itertools
import numpy as np
from functools import partial
from tot.models import gpt
from tot.methods.tree_nodes import Node

def get_value(task, child, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(child, task.steps)
    if cache_value and str(value_prompt) in task.value_cache:
        return task.value_cache[str(value_prompt)], value_prompt
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(value_outputs, child.level-1)
    if cache_value:
        task.value_cache[str(value_prompt)] = value
    delimiter = "\n#############################\n"
    prompt_log = delimiter.join(["Value Prompt:\n" + "\n\n".join(value_prompt.values()), "Value Outputs:\n" + "\n------\n".join(value_outputs)])
    return value, prompt_log

def get_values(task, current_node, n_evaluate_sample, cache_value=True):
    prompt_log = []
    local_value_cache = {}

    # valuation
    for child in current_node.children:  # each partial output
        if child.LLM_answer in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value, value_prompt = get_value(task, child, n_evaluate_sample, cache_value=cache_value)
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

def get_samples(task, current_node, n_generate_sample, prompt_sample, stop):
    # sampling
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(current_node)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(current_node, task.steps) # TODO: add params to all calls
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    
    # log
    delimiter = "\n###########################################################\n"
    prompt_log = delimiter.join(["Sample Prompt:\n" + "\n\n".join(prompt.values()), "Sample Outputs:\n" + "\n------\n".join(samples)])
    
    # turn samples in nodes
    if current_node.level+1 == task.steps:
        leaf = True
    else:
        leaf = False
    for sample in samples:
        new_node = Node(current_node.level+1, current_node.x, LLM_answer=sample, parent=current_node, children=[], leaf=leaf)
        current_node.children.append(new_node)
    
    if task.__class__.__name__ == "ARCTask":
        return prompt_log
    #return [y + _ for _ in samples], prompt_log # TODO: apply to old tasks


def depth_first_search_prioritized(args, task, current_node, step, best_leaf_nodes=[], infos=[], to_print=True):
    if current_node.isLeaf:  # Leaf node
        return [current_node], infos
    
    # TODO: Check how to get information from intermediate_results
    # generation  # TODO: Rename? Generate children?
    if args.method_generate == 'sample':
        # Sample: 1. Standard, 2. CoT, 3. Multiple CoT (w self-consistency)
        gen_prompts = get_samples(task, current_node, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step])
    # elif args.method_generate == 'propose':
    #     # Propose potential next steps, define in promptamount of proposals
    #     new_ys, gen_prompts = get_proposals(task, x, y)
    new_ys = current_node.children
    
    # evaluation
    if args.method_evaluate == 'vote':
        # always vote for single best child, n_evalute_sample times
        eval_prompts = get_votes(task, current_node, args.n_evaluate_sample)
    elif args.method_evaluate == 'value':
        eval_prompts = get_values(task, current_node, args.n_evaluate_sample)
    values = [n.value for n in current_node.children]
    
    # selection / pruning
    if args.method_select == 'sample':
        ps = np.array([n.value for n in current_node.children]) / sum([n.value for n in current_node.children])
        current_node.children = np.random.choice(current_node.children, size=args.n_select_sample, p=ps, replace=False).tolist()
        current_node.children = sorted(current_node.children, key=lambda n: n.value, reverse=True)
    elif args.method_select == 'greedy':
        current_node.children = sorted(current_node.children, key=lambda n: n.value, reverse=True)[:args.n_select_sample]

    # log
    if to_print: 
        sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
        print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {current_node.children}\n')
    prompt_log = '\n'.join([gen_prompts, eval_prompts])
    infos.append({'step': step, 'x': current_node.x, 'ys': current_node.LLM_answer, 'new_ys': [str(y) for y in new_ys], 'values': values, 'select_new_ys': [str(child) for child in current_node.children], 'prompt_log': prompt_log})
    
    step += 1
    for child in current_node.children:
        leaf_nodes, infos = depth_first_search_prioritized(args, task, child, step, best_leaf_nodes, infos, to_print) 
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

    return best_leaf_nodes, infos

def solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    x = task.get_input(idx)  # input
    root = Node(0, x, children=[])
    best_leaf_nodes, infos = depth_first_search_prioritized(args, task, root, step=0, best_leaf_nodes=[], infos=[], to_print=to_print)
    return best_leaf_nodes, {'steps': infos}
    
    
    
##### old, w/o nodes #####


# def get_value(task, x, y, n_evaluate_sample, current_step, total_steps, intermediate_results, cache_value=True):
#     value_prompt = task.value_prompt_wrap(x, y, current_step, total_steps, intermediate_results)
#     if cache_value and value_prompt in task.value_cache:
#         return task.value_cache[value_prompt], value_prompt
#     value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
#     value = task.value_outputs_unwrap(x, y, value_outputs)
#     if cache_value:
#         task.value_cache[value_prompt] = value
#     delimiter = "\n#############################\n"
#     prompt_log = delimiter.join(["Value Prompt:\n" + value_prompt, "Value Outputs:\n" + "\n------\n".join(value_outputs)])
#     return value, prompt_log

# def get_values(task, x, ys, n_evaluate_sample, current_step, total_steps, intermediate_results, cache_value=True):
#     values = []
#     prompt_log = []
#     local_value_cache = {}
#     if len(ys) == 1:
#         return [1], "\n###########################################################\nNo Valuation - Only one candidate\n"
    
#     # valuation
#     for y in ys:  # each partial output
#         if y in local_value_cache:  # avoid duplicate candidates
#             value = 0
#         else:    
#             value, value_prompt = get_value(task, x, y, n_evaluate_sample, current_step, total_steps, intermediate_results, cache_value=cache_value)
#             local_value_cache[y] = value
#         values.append(value)
#         prompt_log.append(value_prompt)
   
#     # log
#     delimiter = "\n###########################################################\n"
#     prompt_log = delimiter + delimiter.join(prompt_log)
#     return values, prompt_log

# def get_votes(task, x, ys, n_evaluate_sample, current_step, total_steps):
#     # voting
#     vote_prompt = task.vote_prompt_wrap(x, ys, current_step, total_steps) # TODO: add params to all calls
#     vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
#     values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    
#     #log
#     delimiter = "\n###########################################################\n"
#     prompt_log = delimiter + delimiter.join(["Vote Prompt:\n" + vote_prompt, "Vote Outputs:\n" + "\n------\n".join(vote_outputs), "Vote Values: "+ str(values)])
    
#     return values, prompt_log

# def get_samples(task, x, y, n_generate_sample, current_step, total_steps, prompt_sample, stop):
#     # sampling
#     if prompt_sample == 'standard':
#         prompt = task.standard_prompt_wrap(x, y)
#     elif prompt_sample == 'cot':
#         prompt = task.cot_prompt_wrap(x, y, current_step, total_steps) # TODO: add params to all calls
#     else:
#         raise ValueError(f'prompt_sample {prompt_sample} not recognized')
#     samples = gpt(prompt, n=n_generate_sample, stop=stop)
    
#     # log
#     delimiter = "\n###########################################################\n"
#     prompt_log = delimiter.join(["Sample Prompt:\n" + prompt, "Sample Outputs:\n" + "\n------\n".join(samples)])
    
#     if task.__class__.__name__ == "ARCTask":
#         return samples, prompt_log
#     return [y + _ for _ in samples], prompt_log



# def depth_first_search_prioritized(args, task, x, intermediate_results, step, to_print=True):
#     if step == task.steps:  # Leaf node
#         return y
    
    
#     best_child = None
#     best_chance = -1
#     # TODO: Check how to get information from intermediate_results
#     # generation    
#     if args.method_generate == 'sample':
#         # Sample: 1. Naive, 2. CoT, 3. Multiple CoT (w self-consistency)
#         new_ys, gen_prompts = get_samples(task, x, y, args.n_generate_sample, step, task.steps, prompt_sample=args.prompt_sample, stop=task.stops[step])
#     # elif args.method_generate == 'propose':
#     #     # Propose potential next steps, define in promptamount of proposals
#     #     new_ys, gen_prompts = get_proposals(task, x, y)
#     ids = list(range(len(new_ys)))
    
#     # evaluation
#     if args.method_evaluate == 'vote':
#         values, eval_prompts = get_votes(task, x, new_ys, args.n_evaluate_sample, step, task.steps)
#     elif args.method_evaluate == 'value':
#         values, eval_prompts = get_values(task, x, new_ys, args.n_evaluate_sample, step, task.steps, intermediate_results)

#     # selection
#     if args.method_select == 'sample':
#         ps = np.array(values) / sum(values)
#         select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
#     elif args.method_select == 'greedy':
#         select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
#     select_new_ys = [new_ys[select_id] for select_id in select_ids]

#     # log
#     if to_print: 
#         sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
#         print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
#     prompt_log = '\n'.join([gen_prompts[0], eval_prompts])
#     infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys, 'prompt_log': prompt_log})
    
#     ys = select_new_ys

#     step += 1
#     for y in sorted_new_ys:
#         intermediate_results.append(y)
#         result = depth_first_search_prioritized(args, task, x, intermediate_results, step, to_print=to_print) 
#         intermediate_results = intermediate_results[:-1]
#         if result and result.chance > best_chance:
#             best_chance = result.chance
#             best_child = result

#     return best_child

