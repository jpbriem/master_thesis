import itertools
import numpy as np
from functools import partial
from tot.models import gpt
from tot.methods.tree_nodes import Node
from tot.methods import search_utils

def solve(args, task, idx, to_print=True):
    search_utils.gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    x = task.get_input(idx)  # input
    current_best_nodes = [Node(0, x, n_generate_children=args.n_generate_sample, children=[])]
    infos = []
    for step in range(task.steps):
        # generation  # TODO: Rename? Generate children?
        if args.method_generate == 'sample':
            gen_prompts = [search_utils.get_samples(args, task, current_node, prompt_sample=args.prompt_sample, stop=task.stops[step]) for current_node in current_best_nodes]
        # elif args.method_generate == 'propose':
        #     # Propose potential next steps, define in prompt the amount of proposals
        #     new_ys, gen_prompts = get_proposals(task, x, y)
        gen_prompts = "#############################\nFirst node, get samples:\n#############################" + '#############################\nNext node, get samples:\n#############################'.join(gen_prompts)
        new_ys = list(itertools.chain(*[current_node.children for current_node in current_best_nodes]))
        
        # evaluation
        if args.method_evaluate == 'vote':
            # always vote for single best child, n_evalute_sample times
            eval_prompts = [search_utils.get_votes(task, current_node, args.n_evaluate_sample) for current_node in current_best_nodes]
        elif args.method_evaluate == 'value':
            eval_prompts = [search_utils.get_values(args, task, current_node) for current_node in current_best_nodes]
        eval_prompts = "#############################\nFirst node, get values:\n#############################" + '#############################\nNext node, get values:\n#############################'.join(eval_prompts)
        values = [node.value for node in new_ys]
        
        # selection / pruning
        if args.method_select == 'sample':
            ps = np.array([n.value for n in new_ys]) / sum([n.value for n in new_ys])
            selected_best_nodes = np.random.choice(new_ys, size=args.n_select_sample, p=ps, replace=False).tolist()
        elif args.method_select == 'greedy':
            selected_best_nodes = sorted(new_ys, key=lambda n: n.value, reverse=True)[:args.n_select_sample]

        # if needed, update children nodes: phase & spreading
        for child in selected_best_nodes:
            task.update_node(child)

        # revise abstraction
        rev_log = ""
        revisions_total=0
        if args.revision and selected_best_nodes[0].phase == "application":
            for child in selected_best_nodes:
                revision_log, revisions_total, example_success = search_utils.revise_abstraction(args, task, child)
                rev_log += revision_log
                if all(example_success):
                    # abstraction is successfull on examples -> apply to all test cases! Assumption: Not multiple tries needed.
                    selected_best_nodes = [child]
                    break
                else:  
                    # at least one example was wrong: value of instruction becomes # of solved examples
                    child.value = example_success.count(True)
                    child.example_success = example_success
                    child.revisions_total = revisions_total
            if len(selected_best_nodes) != 1:
                # none of the abstractions was successfull on all examples -> take best abstraction on examples
                selected_best_nodes = sorted(selected_best_nodes, key=lambda n: n.value, reverse=True)[:1]
                revisions_total = selected_best_nodes[0].revisions_total
                example_success = selected_best_nodes[0].example_success
        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {selected_best_nodes}\n')
        log = {'step': step, 'x': x, 'ys': [str(y) for y in current_best_nodes], 'new_ys': [str(y) for y in new_ys], 'values': values, 'select_new_ys': [str(child) for child in selected_best_nodes]}
        if rev_log == "":
            prompt_log = '\n'.join([gen_prompts, eval_prompts])
            log['prompt_log'] = prompt_log
        else:
            prompt_log = '\n'.join([gen_prompts, eval_prompts, rev_log])
            log.update({'prompt_log': prompt_log, 'revision_success': all(example_success)})
        infos.append(log)
        
        current_best_nodes = selected_best_nodes
    if args.revision:
           return current_best_nodes, {'steps': infos, 'total_revisions': revisions_total}
    return current_best_nodes, {'steps': infos}
    
    
def naive_solve(args, task, idx, to_print=True):
    search_utils.gpt = partial(gpt, model=args.backend, temperature=args.temperature, response_format={ "type": "text" })
    x = task.get_input(idx)  # input
    root = Node(0, x, n_generate_children=args.n_generate_sample, children=[])
    prompt_log = search_utils.get_samples(args, task, root, args.prompt_sample, stop=None)
    infos = [{'prompt_log': prompt_log}]
    return root.children, {'steps': infos} 
