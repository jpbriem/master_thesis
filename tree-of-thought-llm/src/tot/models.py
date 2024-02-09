import os
from functools import partial
import openai
import backoff 
import datetime
import json
from tot.methods.arc_config import MODEL_CONFIGS, GPU
from tot.methods.credentials import OPENAI_KEY, HF_TOKEN
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] =  GPU
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline
from auto_gptq import exllama_set_max_input_length, AutoGPTQForCausalLM
import tiktoken
import torch

completion_tokens = prompt_tokens = 0
model = tokenizer = llm = backend = prompt_sample = None
naive_run = False
responses = [] # TODO: Delete
idx = 0 # TODO: Delete
date = datetime.datetime.now() # TODO: Delete


def initialize_model(args):
    global model, tokenizer, llm, backend, naive_run, prompt_sample
    backend = args.backend
    naive_run = args.naive_run
    prompt_sample = args.prompt_sample
    if "gpt" in backend:
        os.environ['OPENAI_API_KEY'] = OPENAI_KEY
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key != "":
            openai.api_key = api_key
        else:
            print("Warning: OPENAI_API_KEY is not set")
            
        api_base = os.getenv("OPENAI_API_BASE", "")
        if api_base != "":
            print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
            openai.api_base = api_base
            
        if args.prompt_sample == "standard":
            response_format = { "type": "text" }
        else:
            response_format = { "type": "json_object" }
        llm = partial(gpt, model=backend, temperature=args.temperature, response_format=response_format)
        return call_model
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available")
        return None
    try:
        if backend in ["TheBloke/Falcon-7B-Instruct-GPTQ", "TheBloke/Falcon-40B-Instruct-GPTQ"]:
            model, tokenizer = load_falcon(backend, args.model_revision)
            llm = run_falcon
        elif backend in ["Qwen/Qwen-14B-Chat", "Qwen/Qwen-7B-Chat", "Qwen/Qwen-72B-Chat"]:
            llm, tokenizer = load_qwen(backend, MODEL_CONFIGS[backend]["model_config"])
        else:
            tokenizer, model, llm = load_llama(backend, args.model_revision, MODEL_CONFIGS[backend]["max_token"], MODEL_CONFIGS[backend]["model_config"])
        return call_model
    except Exception as e:
        print(e)
        return None

def prompt_preprocessing_for_model(prompt):
    global backend, naive_run, prompt_sample
    if "gpt" in backend:
        messages = []
        if "system" in prompt:
            messages.append({"role": "system", "content": prompt["system"]})
        if "few_shot_ex" in prompt:
            for k, v in prompt["few_shot_ex"].items():
                messages.append({"role": "user", "content": v[0]}) # ex task
                messages.append({"role": "assistant", "content": v[1]}) # ex answer
        messages.append({"role": "user", "content": prompt["user"]})
        return messages
    elif not naive_run or not prompt_sample == "standard":
        if "chat" in backend.lower() and "llama" in backend.lower():
            # use prompting template for llama chat models
            if "system" in prompt:
                return "[INST] <<SYS>>\n" + prompt["system"] + "\n<</SYS>>\n" + prompt["user"] + "[/INST]"
            else:
                return "[INST]\n" + prompt["user"] + "[/INST]\n"
        elif "platypus2" in backend.lower():
            # use prompting template for Platypus2 models
            if "system" in prompt:
                return prompt["system"] + "\n\n### Instruction:\n" + prompt["user"] + "\n\n### Response:"
            else:
                return "### Instruction:\n" + prompt["user"] + "\n\n### Response:"
        elif "falcon" in backend.lower():
            # use prompting template for Falcon models
            if "system" in prompt:
                return prompt["system"] + "\n\nUser: " + prompt["user"] + "\n\nAssistant:"
            else:
                return "User: " + prompt["user"] + "\n\nAssistant:"
        elif "mixtral" in backend.lower() and "DPO" in backend.lower():
            # use prompting template for Mixtral DPO model
            if "system" in prompt:
                return "<|im_start|>system\n" + prompt["system"] + "<|im_end|>\n<|im_start|>user\n" + prompt["user"] + "<|im_end|>\n<|im_start|>assistant"
            else:
                return "<|im_start|>user\n" + prompt["user"] + "<|im_end|>\n<|im_start|>assistant"
        elif ("mistral" in backend.lower() or "mixtral" in backend.lower()) and "instruct" in backend.lower():
            # use prompting template for Mistral models
            if "system" in prompt:
                return "[INST] " + prompt["system"] + "\n" + prompt["user"] + "\n[/INST]"
            else:
                return "[INST] " + prompt["user"] + "\n[/INST]"
        elif "Qwen" in backend:
            if "system" not in prompt:
                prompt["system"] = ""
            return prompt 
            
    # use naive prompting template, instead 
    if "system" in prompt:
        return prompt["system"] + "\n" + prompt["user"]
    else:
        return prompt["user"]

def check_prompt_size(prompt):
    global backend, tokenizer
    if isinstance(prompt, dict):
        prompt = "\n".join(prompt.values())
    num_tokens, token_limit = count_tokens(prompt, backend, tokenizer)
    if num_tokens > token_limit:
        return False, num_tokens, token_limit
    return True, num_tokens, token_limit
   

def call_model(prompt, max_tokens=2000, n=1, stop=None):
    global model, tokenizer, llm, completion_tokens, prompt_tokens, backend, naive_run
    if llm is None:
        raise Exception("Model is not initialized")
    if "gpt" in backend:
        return llm(prompt, max_tokens=max_tokens, n=n, stop=stop)
    elif backend in ["TheBloke/Falcon-7B-Instruct-GPTQ", "TheBloke/Falcon-40B-Instruct-GPTQ"]:
        outputs = []
        for i in range(n):
            output = llm(tokenizer, model, prompt, **MODEL_CONFIGS[backend]["model_config"])
            prompt_tokens += count_tokens(prompt, backend, tokenizer)[0]
            completion_tokens += count_tokens(output, backend, tokenizer)[0]
            outputs.append(output)
    elif backend in ["Qwen/Qwen-14B-Chat", "Qwen/Qwen-7B-Chat", "Qwen/Qwen-72B-Chat"]:
        outputs = []
        for i in range(n):
            output, _ = llm(tokenizer, prompt["user"], history=None, system=prompt["system"])
            prompt_tokens += count_tokens("\n".join(prompt), backend, tokenizer)[0]
            completion_tokens += count_tokens(output, backend, tokenizer)[0]
            outputs.append(output)
    else:
        outputs = []
        for i in range(n):
            output = llm(prompt)
            prompt_tokens += count_tokens(prompt, backend, tokenizer)[0]
            completion_tokens += count_tokens(output, backend, tokenizer)[0]
            outputs.append(output)
    return outputs 

#################### OpenAI #####################

class WrongFinishReasonError(Exception):
    pass

# log openai errors
def log_exception(details):
    current_datetime = datetime.datetime.now()
    log = "Exception occurred: {}\nRetry number: {}\nNext retry in: {}\nKwargs: {}\n".format(details['exception'], details.get('tries', 0), details.get('wait', 0), details["kwargs"], )
    path = "error_log/gpt_output_errors/"+current_datetime.strftime("%Y-%m-%d_%H-%M-%S")+".txt"
    with open(path, "w") as text_file:
        text_file.write(log)
    print("Backoff: {}".format(details['exception']))

@backoff.on_exception(backoff.expo, openai.error.OpenAIError, on_backoff=log_exception)
@backoff.on_exception(backoff.expo, WrongFinishReasonError, max_tries=5, on_backoff=log_exception)
def completions_with_backoff(**kwargs):
    print("call openai API")
    res = openai.ChatCompletion.create(**kwargs)
    for choice in res["choices"]:
        if choice["finish_reason"] != "stop":
            raise WrongFinishReasonError("finish_reason is {}".format(res["finish_reason"]))
    return res

def gpt(messages, model="gpt-3.5-turbo-1106", temperature=0.7, response_format={ "type": "json_object" }, max_tokens=2000, n=1, stop=None) -> list:
    return chatgpt(messages, model=model, temperature=temperature, response_format=response_format, max_tokens=max_tokens, n=n, stop=stop)
    
def chatgpt(messages, model="gpt-3.5-turbo-1106", temperature=0.7, response_format={ "type": "json_object" }, max_tokens=2000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens, responses, idx, date  # TODO: Delete
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        try:
            res = completions_with_backoff(model=model, messages=messages, temperature=temperature, response_format=response_format, max_tokens=max_tokens, n=cnt, stop=stop)
        except:
            res = {"choices": [{"message": {"content": "ERROR"}}], "usage": {"prompt_tokens": 0,"completion_tokens": 0}}
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        # log completion tokens
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]
    return outputs
    
def gpt_usage(backend="gpt-4", input_tokens=None, output_tokens=None):
    global completion_tokens, prompt_tokens

    if input_tokens is None:
        input_tokens = prompt_tokens
    if output_tokens is None:
        output_tokens = completion_tokens

    if input_tokens != prompt_tokens and output_tokens != completion_tokens:
        prompt_tokens = input_tokens
        completion_tokens = output_tokens
        
    if backend == "gpt-4-1106-preview":
        cost = completion_tokens / 1000 * 0.03 + prompt_tokens / 1000 * 0.01
    elif backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo-1106":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.001
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    else: 
        cost = None
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}


#################### Open-Source #############
# LLama Models and Llama-like models
def load_llama(model_name, revision, max_token, model_config):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.model_max_length is None or tokenizer.model_max_length > 9999999999:
        tokenizer.model_max_length = max_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16, revision=revision
    )

    # fix bug for certain models - fixed in new Optimum version
    if model_name in ["TheBloke/Camel-Platypus2-70B-GPTQ", "TheBloke/Platypus2-70B-GPTQ", "TheBloke/Llama-2-70b-Chat-GPTQ", "TheBloke/Mistral-7B-v0.1-GPTQ", "TheBloke/Llama-2-70B-GPTQ"]:
        model = exllama_set_max_input_length(model, 4096)

    # make pipeline
    # Docs for config: https://huggingface.co/docs/transformers/v4.33.3/en/main_classes/configuration#transformers.PretrainedConfig
    # https://www.promptingguide.ai/introduction/settings
    generation_config = GenerationConfig.from_pretrained(model_name)
    generation_config.max_new_tokens = model_config["max_new_tokens"]
    generation_config.temperature = model_config["temperature"]
    #generation_config.top_p = 0.9 #  If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    generation_config.do_sample = True # Whether or not to use sampling ; use greedy decoding otherwise.
    generation_config.repetition_penalty = model_config["repetition_penalty"] # 1.0 means no penalty.

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        generation_config=generation_config,
        # num_workers = 2, # Default=8, When the pipeline will use DataLoader [..] the number of workers to be used.
        # batch_size=2, # Default=1, When the pipeline will use DataLoader [..] the size of the batch to use.
    )

    # make pipeline compatbile with langchain and return
    hf_pipeline = HuggingFacePipeline(pipeline=text_pipeline) #, model_kwargs={"temperature": 0})
    return tokenizer, model, hf_pipeline

# Falcon Models
def load_falcon(model_name, revision):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(model_name,
            model_basename=revision,
            use_safetensors=True,
            trust_remote_code=True,
            #device="cuda:0",
            use_triton=False,
            quantize_config=None)
    # fix bug for certain models - fixed in new Optimum version
    if model_name in ["TheBloke/Falcon-40B-Instruct-GPTQ"]:
        model = exllama_set_max_input_length(model, 4096)
    return model, tokenizer

def run_falcon(tokenizer, model, prompt, max_new_tokens, temperature):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=temperature, max_new_tokens=max_new_tokens)
    return [tokenizer.decode(output[0])]


# Qwen Models
def load_qwen(model_name, model_config):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True, bf16=True).eval()
    # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True).eval()
    # Docs for config: https://huggingface.co/docs/transformers/v4.33.3/en/main_classes/configuration#transformers.PretrainedConfig
    # https://www.promptingguide.ai/introduction/settings
    generation_config = GenerationConfig.from_pretrained(model_name)
    generation_config.max_new_tokens = model_config["max_new_tokens"]
    generation_config.temperature = model_config["temperature"]
    #generation_config.top_p = 0.9 #  If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    generation_config.do_sample = True # Whether or not to use sampling ; use greedy decoding otherwise.
    generation_config.repetition_penalty = model_config["repetition_penalty"] # 1.0 means no penalty.
    model.generation_config = generation_config
    return model.chat, tokenizer

#################### Utils #####################

def count_tokens(prompt, model_name, tokenizer):
    if "gpt" in model_name:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = 0
        tokens_per_message = 3 # for model gpt-3.5-turbo-0613 & gpt-4-0613
        tokens_per_name = 1
        for message in prompt:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    else: 
        num_tokens = len(tokenizer.encode(prompt, add_special_tokens=True))
    token_limit = MODEL_CONFIGS[backend]["max_token"]
    return num_tokens, token_limit