import transformers
from preference_datasets import get_batch_iterator

import torch

import os
import json
import random
import numpy as np
import argparse

def dump_files(responses, all_models, temp):
    if args.prompt_set == 'alpaca_eval':
        def _process_instruction(instruction):
            return instruction[len('Human: '):-len('\n\nAssistant: ')]

        def _process_output(o):
            return o.strip(' ')

        json_out = [{'instruction': _process_instruction(k), 'output': _process_output(v[0])} for k, v in responses.items()]
    else:
        json_out = responses

    with open(all_models[temp], 'w') as f:
        json.dump(json_out, f, indent=2)
    print('dumped to file')

def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if args.model_name == 'llama7b':
        model_path = 'huggyllama/llama-7b'
    elif args.model_name == 'mistral7b':
        model_path = 'mistralai/Mistral-7B-v0.1'
    elif args.model_name == 'deepseek7b':
        model_path = 'deepseek-ai/deepseek-llm-7b-base'
    elif args.model_name == "llama2_13b":
        model_path = "meta-llama/Llama-2-13b-hf"
    elif args.model_name == "yi_6b":
        model_path = "01-ai/Yi-6B"
    elif args.model_name == "mixtral":
        model_path = "mistralai/Mixtral-8x7B-v0.1"

    policy = transformers.AutoModelForCausalLM.from_pretrained(model_path, cache_dir=args.cache_dir, device_map='balanced')
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, cache_dir=args.cache_dir)
    print(tokenizer)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.archive is not None:
        print('loading pre-trained weights')
        state_dict = torch.load(os.path.join(args.archive, 'policy.pt'), map_location='cpu')
        policy.load_state_dict(state_dict['state'])
        print('loaded pre-trained weights')

    all_models = {}
    temps = [float(t) for t in args.temperatures.split(',')]
    if args.prompt_set == 'alpaca_eval':
        max_length = args.max_length
        max_prompt_length = args.max_prompt_length
        chunk_size = args.chunk_size
        assert args.num_samples_per_prefix == 1
        if args.archive is not None:
            output_dir = args.archive
        else:
            output_dir = os.path.join(args.cache_dir, args.model_name + '_samples', f'alpaca_eval_nsamples{args.num_samples_per_prefix}_maxlen{max_length}')
            os.makedirs(output_dir, exist_ok=True)
        for temp in temps:
            all_models[temp] = os.path.join(output_dir, f'alpaca_eval_temp{temp}.json')

    elif args.prompt_set == 'sharegpt':
        max_length = args.max_length
        max_prompt_length = args.max_prompt_length
        chunk_size = args.chunk_size
        assert len(temps) == 1
        sample_folder_name = f'sharegpt2turn_noeos_maxlen{max_length}_temp{temps[0]}'
        if args.archive is not None:
            output_dir = os.path.join(args.archive, sample_folder_name)
        else:
            output_dir = os.path.join(args.cache_dir, args.model_name + '_samples', sample_folder_name)
        os.makedirs(output_dir, exist_ok=True)

        for temp in temps:
            all_models[temp] = os.path.join(output_dir, f'fastforward{args.ff}.json')

    for temp in temps:
        print(f'generating samples at temperature {temp}')
        policy.eval()
        policy.half()
        policy.to('cuda')
        if args.prompt_set == 'alpaca_eval':
            prompt_iterator = get_batch_iterator(['alpaca_eval'], tokenizer=tokenizer, split='eval', batch_size=chunk_size, sft_mode=True,
                                                 seed=0, n_epochs=1, cache_dir=args.cache_dir, shuffle=False,
                                                 max_prompt_length=max_prompt_length, max_length=max_length)
        elif args.prompt_set == 'sharegpt':
            prompt_iterator = get_batch_iterator(['sharegpt'], tokenizer=tokenizer, split='combined', batch_size=chunk_size, sft_mode=True,
                                                 seed=0, n_epochs=1, cache_dir=args.cache_dir, shuffle=False,
                                                 max_prompt_length=max_prompt_length, max_length=max_length,
                                                 num_turns=1, data_fraction=args.data_fraction, prefs_path=None, sampled_data_dir=None)

        responses = {}
        batch_idx = 0
        if args.ff > 0:
            print(f'fastforwarding {args.ff} prompts')
        for batch in prompt_iterator:
            batch_idx += 1
            if batch_idx * chunk_size < args.ff:
                continue

            generator_input = {'input_ids': batch['prompt_input_ids'].to('cuda'),
                               'attention_mask': batch['prompt_attention_mask'].to('cuda'),}
            for _ in range(args.num_samples_per_prefix):
                with torch.no_grad():
                    if temp > 0.0:
                        outputs = policy.generate(**generator_input, max_length=max_length, do_sample=True, top_p=0.9, temperature=temp, pad_token_id=tokenizer.pad_token_id)
                    else:
                        outputs = policy.generate(**generator_input, max_length=max_length, do_sample=False, pad_token_id=tokenizer.pad_token_id)

                for idx, output in enumerate(outputs):
                    # cur_prompt -> complete prompt
                    # cur_truncated_prompt -> truncated prompt
                    # cur_truncated response is used for generation, cur_prompt is used for indexing in the json file
                    cur_prompt = batch['prompt'][idx]
                    cur_truncated_prompt = tokenizer.decode(generator_input['input_ids'][idx], skip_special_tokens=True)
                    cur_response = tokenizer.decode(output, skip_special_tokens=True)
                    cur_response_only = cur_response[len(cur_truncated_prompt):].strip() # remove the space at the beginning of the response
                    if cur_prompt not in responses:
                        responses[cur_prompt] = [cur_response_only]
                    else:
                        responses[cur_prompt].append(cur_response_only)

            if batch_idx % 10 == 0:
                print(f'finished generating {batch_idx * chunk_size} prompts')
                dump_files(responses, all_models, temp)

        dump_files(responses, all_models, temp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama7b')
    parser.add_argument('--archive', type=str, default=None)
    parser.add_argument('--num_samples_per_prefix', type=int, default=1)
    parser.add_argument('--temperatures', type=str, default='0.7')
    parser.add_argument('--prompt_set', type=str, default='sharegpt')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ff', type=int, default=0)
    parser.add_argument('--cache_dir', type=str, default='/ebs/.cache/ubuntu/')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_prompt_length', type=int, default=256)
    parser.add_argument('--chunk_size', type=int, default=32)
    parser.add_argument('--data_fraction', type=float, default=1.0)
    args = parser.parse_args()

    main()
