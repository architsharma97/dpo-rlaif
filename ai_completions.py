import os
import openai
openai.api_key = os.environ['OPENAI_API_KEY']
import anthropic
import hashlib
import json

import numpy as np
import random
import copy
import logging
import time

from preference_datasets import get_batch_iterator
import transformers

import argparse

def _cached_function(fn_to_cache, cache_dir='/ebs/.cache/ubuntu/gpt4_completions/'):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    def wrapped(*args, **kwargs):
        no_cache = False
        if 'no_cache' in kwargs:
            no_cache = kwargs['no_cache']
            del kwargs['no_cache']

        json_dump_args_kwargs = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True)
        hash = hashlib.sha256(json_dump_args_kwargs.encode('utf-8')).hexdigest()
        cache_path = os.path.join(cache_dir, hash)
        if os.path.exists(cache_path) and not no_cache:
            with open(cache_path, 'r') as f:
                return json.load(f)
        else:
            result = fn_to_cache(*args, **kwargs)
            with open(cache_path, 'w') as f:
                json.dump(result, f)
            return result

    return wrapped


def get_openai_completion(prompt,
                          cache=True,
                          model='gpt-4-0314',
                          system_prompt='You are a helpful assistant.'):
    c = _openai_chat_completion(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ],
        no_cache=not cache,
    )
    if len(c['choices']) == 1:
        return c['choices'][0]['message']['content'], c['choices'][0]['finish_reason']
    else:
        return [c_['message']['content'] for c_ in c['choices']]


def get_anthropic_completion(
    prompt: str,
    sleep_time: int = 2,
    max_tokens_to_sample = 2048,
    temperature = 1.0,
    **kwargs,
) -> str:
    kwargs.update(dict(model='claude-v1', max_tokens_to_sample=max_tokens_to_sample, temperature=temperature))
    curr_kwargs = copy.deepcopy(kwargs)
    prompt = f'\n\nHuman: {prompt}\n\nAssistant:'

    while True:
        try:
            completion = _claude_chat_completion(prompt=prompt, **curr_kwargs)

            if completion == "":
                completion = " " # annoying doesn't allow empty string
            break

        except anthropic.RateLimitError as e:
            logging.warning(f"API RateLimitError: {e}.")
            logging.warning(f"Rate limit hit. Sleeping for {sleep_time} seconds.")
            time.sleep(sleep_time)

        except anthropic.APITimeoutError as e:
            logging.warning(f"API TimeoutError: {e}. Retrying request.")

        except anthropic.BadRequestError as e:
            logging.warning(f"API BadRequestError: {e}. Retrying request.")
            completion = " " # input context length exceeded, just return an empty string
            break

    return completion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ff', type=int, default=0)
    parser.add_argument('--cache_dir', type=str, default='/ebs/.cache/ubuntu/')
    parser.add_argument('--data_fraction', type=float, default=1.0)
    parser.add_argument('--ai_model', type=str, default='gpt4')
    parser.add_argument('--base_output_dir', type=str, default='/ebs/.cache/ubuntu/sharegpt_data')
    args = parser.parse_args()

    _openai_chat_completion = _cached_function(openai.ChatCompletion.create)

    client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'], max_retries=3)
    def claude_chat_completion_helper(prompt=None, **kwargs):
        return client.completions.create(prompt=prompt, **kwargs).completion

    _claude_chat_completion = _cached_function(claude_chat_completion_helper, cache_dir='/ebs/.cache/ubuntu/claude_completions/')

    tokenizer = transformers.AutoTokenizer.from_pretrained('huggyllama/llama-7b', cache_dir=args.cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    prompt_iterator = get_batch_iterator(['sharegpt'], tokenizer=tokenizer, split='combined', batch_size=1, sft_mode=True,
                                         seed=0, n_epochs=1, cache_dir=args.cache_dir, shuffle=False,
                                         max_prompt_length=256, max_length=512, # don't matter, as we use complete prompt for GPT-4/Claude
                                         num_turns=1, data_fraction=args.data_fraction, prefs_path=None, sampled_data_dir=None)


    def _get_prompt_from_sharegpt(instruction):
        return instruction[len('Human: '):-len('\n\nAssistant: ')]

    def _dump_files(responses):
        with open(os.path.join(args.base_output_dir, f'sharegpt1turn_df{args.data_fraction}_ff{args.ff}_{args.ai_model}_completions.json'), 'w') as f:
            json.dump(responses, f, indent=2)
        print('Saved to file')

    responses = {}
    prompt_idx = 0
    if args.ff > 0:
        print(f'fastforwarding {args.ff} prompts')

    for batch in prompt_iterator:
        prompt_idx += 1
        if prompt_idx < args.ff:
            continue
        print(f'prompt_idx: {prompt_idx}')
        prompt = _get_prompt_from_sharegpt(batch['prompt'][0])
        if len(prompt.split()) >= 2000 or len(prompt) >= 8000:
            print('Skipping due to length')
            continue
        if args.ai_model == 'gpt4':
            completion, reason = get_openai_completion(prompt)
            if reason != 'stop':
                continue
        elif args.ai_model == 'claude':
            completion = get_anthropic_completion(prompt)
            if completion == ' ':
                continue

        responses[prompt] = [completion.strip()]
        if prompt_idx % 10 == 0:
            print(f'finished generating {prompt_idx} prompts')
            _dump_files(responses)

    _dump_files(responses)
