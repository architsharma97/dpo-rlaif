import json
import os
from preference_datasets import get_batch_iterator
import transformers
import alpaca_eval
import argparse

def get_chatgpt_outputs(max_prompt_length=256, max_length=1024, data_fraction=1.0, num_turns=1):
    tokenizer = transformers.AutoTokenizer.from_pretrained('huggyllama/llama-7b', cache_dir=args.cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    prompt_iterator = get_batch_iterator(['sharegpt'], tokenizer=tokenizer, split='train', batch_size=1, sft_mode=True,
                                        seed=0, n_epochs=1, cache_dir=args.cache_dir, shuffle=False,
                                        max_prompt_length=max_prompt_length, max_length=max_length, data_fraction=data_fraction, num_turns=num_turns)

    chatgpt_instruction_truncoutput_pair = []
    instructions = []
    for batch in prompt_iterator:
        instruction = batch['prompt'][0]
        instruction = instruction[len('Human: '):-len('\n\nAssistant: ')]
        prompt_token_count = batch['prompt_input_ids'][0].shape[0]
        output = tokenizer.decode(batch['chosen_input_ids'][0][prompt_token_count:], skip_special_tokens=True)
        chatgpt_instruction_truncoutput_pair.append({'instruction': instruction,
                                                    'output': output})
        instructions.append(instruction)

    return chatgpt_instruction_truncoutput_pair, instructions

def process_llama_samples_from_dir(sample_folder):
    sft_instruction_truncoutput_pair = []
    sft_instructions = []
    for sample_file in os.listdir(sample_folder):
        sft_outputs = json.load(open(os.path.join(sample_folder, sample_file), 'r'))
        for instruction, sft_output in sft_outputs.items():
            instruction_trimmed = instruction[len('Human: '):-len('\n\nAssistant: ')]
            sft_instruction_truncoutput_pair.append({'instruction': instruction_trimmed,
                                                    'output': sft_output[0]})
            sft_instructions.append(instruction_trimmed)
    return sft_instruction_truncoutput_pair, sft_instructions

def match_instruction_outputs(instruct_out_1, instruction_set_1, instruct_out_2, instruction_set_2):
    matched_instruction_outputs = []
    for idx, instruction in enumerate(instruction_set_1):
        if instruction in instruction_set_2:
            matched_instruction_outputs.append({'instruction': instruction,
                                                'output_1': instruct_out_1[idx]['output'],
                                                'output_2': instruct_out_2[instruction_set_2.index(instruction)]['output']})
    return matched_instruction_outputs

def main(base_dir, model1_name, model2_name, max_length, data_fraction, max_num_comparisons):
    def _get_model_outputs_and_instructions(name):
        if name == 'chatgpt':
            return get_chatgpt_outputs(max_prompt_length=256, max_length=max_length, data_fraction=data_fraction, num_turns=1)
        else:
            return process_llama_samples_from_dir(os.path.join(base_dir, f'sharegpt2turn_noeos_maxlen{max_length}_{name}'))

    out1, inst1 = _get_model_outputs_and_instructions(model1_name)
    out2, inst2 = _get_model_outputs_and_instructions(model2_name)
    matched_instruction_outputs = match_instruction_outputs(out1, inst1, out2, inst2)
    comparison_name = f'{model1_name}_vs_{model2_name}'
    comparison_folder = os.path.join(base_dir, 'comparisons', comparison_name)
    os.makedirs(comparison_folder, exist_ok=True)

    model1_outputs = []
    model2_outputs = []
    for idx, matched_instruction_output in enumerate(matched_instruction_outputs):
        model1_outputs.append({'instruction': matched_instruction_output['instruction'],
                            'output': matched_instruction_output['output_1'],
                            "generator": model1_name,})
        model2_outputs.append({'instruction': matched_instruction_output['instruction'],
                            'output': matched_instruction_output['output_2'],
                            "generator": model2_name,})
        if idx > max_num_comparisons:
            break

    model1_output_path = os.path.join(comparison_folder, model1_name + '.json')
    with open(model1_output_path, 'w') as f:
        json.dump(model1_outputs, f, indent=4)

    model2_output_path = os.path.join(comparison_folder, model2_name + '.json')
    with open(model2_output_path, 'w') as f:
        json.dump(model2_outputs, f, indent=4)

    alpaca_eval.evaluate(model_outputs=model1_output_path, annotators_config='claude', name=comparison_name, output_path=comparison_folder, reference_outputs=model2_output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1_name', type=str, default='temp1.0')
    parser.add_argument('--model2_name', type=str, default='chatgpt')
    parser.add_argument('--base_dir', type=str) # example: /ebs/.cache/ubuntu/sharegpt2turn_llama7b_sft_maxlen1024_2023-09-11_21-52-36_584206/step-10000/
    parser.add_argument('--cache_dir', type=str, default='/ebs/.cache/ubuntu/')
    parser.add_argument('--max_num_comparisons', type=int, default=200)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--max_prompt_length', type=int, default=256)
    parser.add_argument('--num_turns', type=int, default=1)
    parser.add_argument('--data_fraction', type=float, default=1.0)
    args = parser.parse_args()

    main(base_dir=args.base_dir, model1_name=args.model1_name, model2_name=args.model2_name, max_length=args.max_length, data_fraction=args.data_fraction, max_num_comparisons=args.max_num_comparisons)
    
