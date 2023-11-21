import os
OPENAI_KEY = "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"
os.environ['OPENAI_API_KEY'] = OPENAI_KEY

import argparse
from tot.methods.bfs import solve
from tot.tasks.game24 import Game24Task
from tot.tasks.text import TextTask

########## Game 24 ##########
# args = argparse.Namespace(
#     backend='gpt-3.5-turbo', 
#     temperature=0.7, 
#     task='game24', 
#     naive_run=False, 
#     prompt_sample=None, 
#     method_generate='propose', 
#     method_evaluate='value', 
#     method_select='greedy', 
#     n_generate_sample=1, 
#     n_evaluate_sample=3, 
#     n_select_sample=5)

# task = Game24Task()
# ys, infos = solve(args, task, 900)
# print(ys[0])

########## Text ##########
args = argparse.Namespace(
    backend='gpt-3.5-turbo', 
    temperature=0.7, 
    task='text', 
    naive_run=False, 
    prompt_sample='cot', 
    method_generate='sample', 
    method_evaluate='vote', 
    method_select='greedy', 
    n_generate_sample=2, 
    n_evaluate_sample=2, 
    n_select_sample=1)

task = TextTask()

ys, infos = solve(args, task, 0)
print(ys[0])