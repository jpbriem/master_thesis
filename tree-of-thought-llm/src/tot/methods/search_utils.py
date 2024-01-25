from tot.methods.tree_nodes import Node
from tot.models import model, prompt_preprocessing_for_model

model = None

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
        value_prompt = prompt_preprocessing_for_model(value_prompt)
        value_outputs = model(value_prompt, n=args.n_evaluate_sample, stop=None)
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
    
    if isinstance(value_prompt, dict):
        prompt_log = delimiter.join(["Value Prompt:\n" + "\n\n".join(value_prompt.values()), "Value Outputs:\n" + "\n------\n".join(value_outputs)])
    elif isinstance(value_prompt, str):
        prompt_log = delimiter.join(["Value Prompt:\n" + value_prompt, "Value Outputs:\n" + "\n------\n".join(value_outputs)])
    return value, prompt_log


# Evaluation: get values of children
def get_values(args, task, current_node, cache_value=True):
    prompt_log = []
    local_value_cache = {}
      
    # valuation
    for child in current_node.children:  # each partial output
        if child.LLM_answer in local_value_cache:  # avoid duplicate candidates
            value = 0
            value_prompt = "No Valuation - Duplicate candidate"
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
    vote_prompt = prompt_preprocessing_for_model(vote_prompt)
    vote_outputs = model(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(current_node, vote_outputs)
    for value, child in zip(values, current_node.children):
        child.value = value
        
    #log
    delimiter = "\n###########################################################\n"
    if isinstance(vote_prompt, dict):
        prompt_log = delimiter + delimiter.join(["Vote Prompt:\n" + "\n\n".join(vote_prompt.values()), "Vote Outputs:\n" + "\n------\n".join(vote_outputs), "Vote Values: "+ str([n.value for n in current_node.children])])
    elif isinstance(vote_prompt, str):
        prompt_log = delimiter + delimiter.join(["Vote Prompt:\n" + vote_prompt, "Vote Outputs:\n" + "\n------\n".join(vote_outputs), "Vote Values: "+ str([n.value for n in current_node.children])])
    return prompt_log


# Generation: get new samples
def get_samples(args, task, current_node, prompt_sample, stop):
    # sampling
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(current_node)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(current_node, task.steps) # TODO: add params to all calls (old functions)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    if args.use_api:
        prompt = prompt_preprocessing_for_model(prompt)
        samples = model(prompt, n=current_node.n_generate_children, stop=stop)
    else:
        # get samples from chat interface
        samples = []
        for i in range(current_node.n_generate_children):
            if "system" in prompt:
                print(prompt["system"])
            if "user" in prompt:
                print("\n" + prompt["user"])
            sample = read_multiline_input("Answer of LLM: ")
            samples.append(sample)
            
    # log
    delimiter = "\n###########################################################\n"
    if isinstance(prompt, dict):
        prompt_log = delimiter.join(["Sample Prompt:\n" + "\n\n".join(prompt.values()), "Sample Outputs:\n" + "\n------\n".join(samples)])
    elif isinstance(prompt, str):
        prompt_log = delimiter.join(["Sample Prompt:\n" + prompt, "Sample Outputs:\n" + "\n------\n".join(samples)])
    elif isinstance(prompt, list):                             
        prompt_log = delimiter.join(["Sample Prompt:\n" + "\n\n".join(["#####\n"+v+":\n#####" if k == "role" else v for p in prompt for k, v in p.items()]), "Sample Outputs:\n" + "\n------\n".join(samples)])
   
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
        prompt = prompt_preprocessing_for_model(prompt)
        output = model(prompt, n=1)
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
    if isinstance(prompt, dict):
        analysis_log = delimiter.join(["Analysis Prompt:\n" + "\n".join(prompt.values()),"Analysis Result:\n" + str(output)])    
    elif isinstance(prompt, str):
        analysis_log = delimiter.join(["Analysis Prompt:\n" + prompt,"Analysis Result:\n" + str(output)])
        
    return analysis_log

# Revision: revise abstraction
def revise(args, task, node, original_node):
    # revise abstraction
    prompt = task.revision_prompt_wrap(node)
    if args.use_api:
        prompt = prompt_preprocessing_for_model(prompt)
        output = model(prompt, n=1)
    else:
        # get output from chat interface
        print(prompt["system"] + "\n" + prompt["user"])
        output = [read_multiline_input("Answer of LLM: ")]
    revision_result = task.revision_prompt_unwrap(output, node)
    
    # create new node
    new_node = Node(node.level+1, node.x, LLM_answer=output[0], thought=revision_result, parent=node, children=[])
    node.children.append(new_node)
    
    # replace old thoughts with revised thoughts - for each element in thought revise one more parent node one layer higher
    replacement_log = task.replace_revised_thoughts(new_node, original_node)    
    
    # log
    delimiter = "\n###########################################################\n"
    if isinstance(prompt, dict):
        revision_log = delimiter.join(["Revision Prompt:\n" + "\n".join(prompt.values()),"Revision Result:\n" + str(output)])    
    elif isinstance(prompt, str):
        revision_log = delimiter.join(["Revision Prompt:\n" + prompt,"Revision Result:\n" + str(output)])
    revision_log += delimiter + replacement_log 
    
    return revision_log
    
# Revision: Loop
def revise_abstraction(args, task, original_node):
    # reset thoughts of higher levels in case already modified in due to revision of sibling with same parent
    _ = task.replace_revised_thoughts(original_node, original_node)
    
    # log
    delimiter = "\n###########################################################\n"
    revision_log = delimiter + "Abstraction Revision\n" + delimiter

    # work with copy of node
    node = original_node.copy()
    
    # tracker for example success
    n_examples = len(node.x["train"])
    example_success = [False]*n_examples
    
    # track partly correct abstractions
    best_abstraction_node = [None, [False]]
    # examples_solved_so_far = 0
    
    # revision in a loop till termination
    current_test_idx = 0 
    revisions_in_a_row = 0 # for termination condition
    revision_last_iteration = False # for termination condition
    revisions_total = 0 # for termination condition
    max_revisions = 2*n_examples # for termination condition
    while True:
        # termination conditions
        if example_success.count(True) == n_examples:
            break

        # change train and test samples in node.x to simulate current example as test case
        node.x = task.simulate_ex_as_test_case(original_node.x, current_test_idx)
        node.current_test_idx = current_test_idx
        node.n_generate_children = 1 # in revision just 1 child
        
        # apply abstraction to solve current example -> get child node
        revision_log += get_samples(args, task, node, prompt_sample=args.prompt_sample, stop=task.stops[node.level])
        example_test_node = node.children[0] # TODO: use multiple children?
        
        # test the answer, which is in child 
        is_success = task.test_output(node=example_test_node, outputs=[example_test_node], is_revision=True)
        
        # if success: move to next example
        if is_success:
            example_success[current_test_idx] = True
            current_test_idx = (current_test_idx+1) % len(example_success)
            revision_log += delimiter + "Example solved!\n" + delimiter

        # if failure: 
        else:          
            # track best abstraction so far
            if best_abstraction_node[0] is None:
                best_abstraction_node = [original_node, example_success.copy()]
            elif example_success.count(True) > best_abstraction_node[1].count(True):
                best_abstraction_node = [analysis_node.children[0], example_success] # analysis_node.child cotains best revised abstraction so far

            # termination conditions
            revisions_in_a_row = revisions_in_a_row + 1 if revision_last_iteration else 0
            revision_last_iteration = False
            if revisions_in_a_row == n_examples or revisions_total >= max_revisions:
                break
            
            # compare wrong answer (which is in example_test_node) to gt
            revision_log += delimiter + analyse_failure(args, task, example_test_node)
            analysis_node = example_test_node.children[0]
            analysis_node.current_test_idx = current_test_idx
            # revise abstraction 
            revision_last_iteration = True
            example_success = [False]*n_examples
            revisions_total += 1
            revision_log += delimiter + revise(args, task, analysis_node, original_node)
            # update index of example to be tested
            current_test_idx = (current_test_idx+1) % len(example_success)
            
        # reset children of node for next example iteration
        node.children = []
            
            
    # if not successful, return best abstraction so far
    if example_success.count(True) != n_examples:
        if best_abstraction_node[0] == original_node:
            revision_log += delimiter + "Initial abstraction was best.\n"
        else:
            revision_log += delimiter + "One of the revisions was best, reset to best thoughts.\n"
        revision_log += task.replace_revised_thoughts(best_abstraction_node[0], original_node)
        example_success = best_abstraction_node[1]
        original_node.best_abstraction_node = best_abstraction_node[0]
        
    return revision_log, revisions_total, example_success

