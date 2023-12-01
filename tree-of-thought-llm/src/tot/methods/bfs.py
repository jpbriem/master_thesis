import itertools
import numpy as np
from functools import partial
from tot.models import gpt

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt], value_prompt
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    delimiter = "\n#############################\n"
    prompt_log = delimiter.join(["Value Prompt:\n" + value_prompt, "Value Outputs:\n" + "\n------\n".join(value_outputs)])
    return value, prompt_log

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    prompt_log = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value, value_prompt = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
        prompt_log.append(value_prompt)
    delimiter = "\n###########################################################\n"
    prompt_log = delimiter + delimiter.join(prompt_log)
    return values, prompt_log

def get_votes(task, x, ys, n_evaluate_sample, current_step, total_steps):
    vote_prompt = task.vote_prompt_wrap(x, ys, current_step, total_steps) # TODO: add params to all calls
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    delimiter = "\n###########################################################\n"
    prompt_log = delimiter + delimiter.join(["Vote Prompt:\n" + vote_prompt, "Vote Outputs:\n" + "\n------\n".join(vote_outputs), "Vote Values: "+ str(values)])
    return values, prompt_log

def get_proposals(task, x, y): 
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')
    delimiter = "\n###########################################################\n"
    prompt_log = delimiter.join(["Proposal Prompt:\n" + propose_prompt, "Proposal Outputs:\n" + "\n------\n".join(proposals)])
    return [y + _ + '\n' for _ in proposals], prompt_log

def get_samples(task, x, y, n_generate_sample, current_step, total_steps, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y, current_step, total_steps) # TODO: add params to all calls
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    delimiter = "\n###########################################################\n"
    prompt_log = delimiter.join(["Sample Prompt:\n" + prompt, "Sample Outputs:\n" + "\n------\n".join(samples)])
    
    if task.__class__.__name__ == "ARCTask":
        return samples, prompt_log
    return [y + _ for _ in samples], prompt_log

def solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    x = task.get_input(idx)  # input
    ys = ['']  # current output candidates
    infos = []
    for step in range(task.steps):
        # generation
        if args.method_generate == 'sample':
            new_ys, gen_prompts = map(list, zip(*[get_samples(task, x, y, args.n_generate_sample, step, task.steps, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]))
        elif args.method_generate == 'propose':
            new_ys, gen_prompts = map(list, zip(*[get_proposals(task, x, y) for y in ys]))
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        # evaluation
        if args.method_evaluate == 'vote':
            values, eval_prompts = get_votes(task, x, new_ys, args.n_evaluate_sample, step, task.steps)
        elif args.method_evaluate == 'value':
            values, eval_prompts = get_values(task, x, new_ys, args.n_evaluate_sample)

        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        prompt_log = '\n'.join([gen_prompts[0], eval_prompts])
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys, 'prompt_log': prompt_log})
        
        ys = select_new_ys
    
    if to_print: 
        print(ys)
    return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}