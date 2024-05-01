import numpy as np
from functools import partial
from tot.models import initialize_model
from tot.methods.tree_nodes import Node
from tot.methods import search_utils

def depth_first_search_prioritized(args, task, current_node, step, best_leaf_nodes=[], best_abstraction_nodes=[], example_success=[False], infos=[], to_print=True):
    if current_node.isLeaf:  # Leaf node
        return [current_node], best_abstraction_nodes, example_success, infos
    
    # generation  
    if args.method_generate == 'sample':
        # Sample: 1. Standard, 2. CoT, 3. Multiple CoT (w self-consistency)
        gen_prompts = search_utils.get_samples(args, task, current_node, prompt_sample=args.prompt_sample, stop=task.stops[step])
    # elif args.method_generate == 'propose':
    #     # Propose potential next steps, define in promptamount of proposals
    #     new_ys, gen_prompts = get_proposals(task, x, y)
    new_ys = current_node.children
    
    # evaluation
    if args.method_evaluate == 'vote':
        # always vote for single best child, n_evalute_sample times
        eval_prompts = search_utils.get_votes(task, current_node, args.n_evaluate_sample)
    elif args.method_evaluate == 'value':
        eval_prompts = search_utils.get_values(args, task, current_node)
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
            if child.isLeaf:
                best_leaf_nodes.append(child)
            continue
        
        # revise abstraction 
        if args.revision and child.phase == "application":
            revision_log, example_success = search_utils.revise_abstraction(args, task, child) 
            
            if all(example_success):
                # abstraction is successfull on examples -> apply to all test cases    
                leaf_nodes, best_abstraction_nodes, example_success, infos = depth_first_search_prioritized(args, task, child, step, best_leaf_nodes, best_abstraction_nodes, example_success, infos, to_print) 
            else:
                # at least one example was wrong: value of instruction becomes # of solved examples
                child.value = example_success.count(True)
                best_abstraction_nodes.append(child)
                best_abstraction_nodes = sorted(best_abstraction_nodes, key=lambda n: n.value, reverse=True)[:args.n_select_sample]

        else:
            leaf_nodes, best_abstraction_nodes, example_success, infos = depth_first_search_prioritized(args, task, child, step, best_leaf_nodes, best_abstraction_nodes, example_success, infos, to_print) 
        
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
       
    return best_leaf_nodes, best_abstraction_nodes, example_success, infos

def solve(args, task, idx, to_print=True):
    # search_utils.gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    search_utils.model = initialize_model(args)
    x = task.get_input(idx)  # input
    root = Node(0, x, n_generate_children=args.n_generate_sample, children=[], input_representation=args.input_representation)
    task.update_node(root)
    best_leaf_nodes, best_abstraction_nodes, example_success, infos = depth_first_search_prioritized(args, task, root, step=0, best_leaf_nodes=[], best_abstraction_nodes=[], infos=[], to_print=to_print)
    
    if len(best_leaf_nodes) == 0 and len(best_abstraction_nodes)>0:
        # abstraction not successfull on examples -> take best abstractions so far
        for best_abstraction_node in best_abstraction_nodes:
            best_leaf_nodes, _, _, infos = depth_first_search_prioritized(args, task, best_abstraction_node, task.steps-1, best_leaf_nodes, best_abstraction_nodes, example_success, infos, to_print) 

    return best_leaf_nodes, {'steps': infos}
    

def naive_solve(args, task, idx, to_print=True):
    # search_utils.gpt = partial(gpt, model=args.backend, temperature=args.temperature, response_format={ "type": "text" })
    search_utils.model = initialize_model(args)
    x = task.get_input(idx)  # input
    root = Node(0, x, n_generate_children=args.n_generate_sample, children=[])
    prompt_log = search_utils.get_samples(args, task, root, args.prompt_sample, stop=None)
    infos = [{'prompt_log': prompt_log}]
    return root.children, {'steps': infos}
 
