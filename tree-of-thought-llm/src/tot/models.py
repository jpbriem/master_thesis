import os
import openai
import backoff 
import datetime

completion_tokens = prompt_tokens = 0

api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")
    
api_base = os.getenv("OPENAI_API_BASE", "")
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base

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

def gpt(prompt, model="gpt-3.5-turbo-1106", temperature=0.7, response_format={ "type": "json_object" }, max_tokens=2000, n=1, stop=None) -> list:
    if "system" in prompt:
        messages =[
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]}
        ]
    else:
        messages =[
            {"role": "user", "content": prompt["user"]}
        ]
    return chatgpt(messages, model=model, temperature=temperature, response_format=response_format, max_tokens=max_tokens, n=n, stop=stop)
    
def chatgpt(messages, model="gpt-3.5-turbo-1106", temperature=0.7, response_format={ "type": "json_object" }, max_tokens=2000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
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
    
def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4-1106-preview":
        cost = completion_tokens / 1000 * 0.03 + prompt_tokens / 1000 * 0.01
    elif backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo-1106":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.001
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
