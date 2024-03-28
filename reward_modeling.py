# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
python examples/scripts/reward_modeling.py \
    --model_name_or_path=facebook/opt-350m \
    --output_dir="reward_modeling_anthropic_hh" \
    --per_device_train_batch_size=64 \
    --num_train_epochs=1 \
    --gradient_accumulation_steps=16 \
    --gradient_checkpointing=True \
    --learning_rate=1.41e-5 \
    --report_to="wandb" \
    --remove_unused_columns=False \
    --optim="adamw_torch" \
    --logging_steps=10 \
    --evaluation_strategy="steps" \
    --max_length=512 \
"""
import warnings

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from datasets import Dataset

from preference_datasets import get_sharegpt_aiprefs
from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config
from peft import AutoPeftModelForSequenceClassification
import os

tqdm.pandas()


if __name__ == "__main__":
    parser = HfArgumentParser((RewardConfig, ModelConfig))
    parser.add_argument("--prefs_path", default="/data/dpo_rlaif/mistral7bsft0.1/comparisons_gpt4/temp1.0_vs_chatgpt/annotations.json", type=str)
    parser.add_argument("--data_fraction", type=float, default=1.0)
    # parser.add_argument("--tokenizer", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument('--archive', type=str, default="/data/dpo_rlaif/mistral7bsft0.1/comparisons_gpt4/temp1.0_vs_chatgpt/annotations.json")
    parser.add_argument('--token', type=str, default=None)

    # parser.add_argument("--max_length", type=int, default=512)
    reward_config, model_config, script_args = parser.parse_args_into_dataclasses()
    reward_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        token=script_args.token,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, 
                    use_fast=True, 
                    truncation_side="left", 
                    model_max_length=reward_config.max_length,
                    torch_dtype=torch_dtype)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, num_labels=1, **model_kwargs
    )
    model.config.pad_token_id = model.config.eos_token_id

    if script_args.archive is not None:
        state_dict = torch.load(script_args.archive, map_location='cpu')
        model.load_state_dict(state_dict['state'], strict=False)
        print('loaded pre-trained weights')

    if model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script."
        )

    ################
    # Dataset
    ################

    def simplify_aipref_dataset(dataset):
        prompts = []
        chosen = []
        rejected = []
        sft_targets = []

        for prompt, data in dataset.items():
            prompts.append(prompt)
            sft_targets.append(data['sft_target'])
            responses = data['responses']
            pairs = data['pairs'][0]

            chosen.append(responses[pairs[0]])
            rejected.append(responses[pairs[1]])

        return Dataset.from_dict({
            'prompts': prompts,
            'chosen': chosen,
            'rejected': rejected,
            'sft_targets': sft_targets,
        })

    raw_train_dataset = simplify_aipref_dataset(get_sharegpt_aiprefs(split='train', silent=False, prefs_path=script_args.prefs_path, data_fraction=script_args.data_fraction))

    raw_eval_dataset = simplify_aipref_dataset(get_sharegpt_aiprefs(split='test', silent=False, prefs_path=script_args.prefs_path, data_fraction=script_args.data_fraction))

    # Tokenize chosen/rejected pairs of inputs
    # Adapt this section to your needs for custom datasets

    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for prompt, chosen, rejected in \
            zip(examples['prompts'], examples['chosen'], examples['rejected']):
            chosen = prompt + chosen
            rejected = prompt + rejected

            # max_length = reward_config.max_length or 512     # min(script_args.max_length, reward_config.max_length)
            tokenized_chosen = tokenizer(chosen, truncation=True)
            tokenized_rejected = tokenizer(rejected, truncation=True)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

    # Preprocess the dataset and filter out examples that are longer than args.max_length
    train_dataset = raw_train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=64,
    )


    eval_dataset = raw_eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=64,
    )

    # lora_task_type needs to be SEQ_CLS if using peft with reward model training.
    model_config.lora_task_type = "SEQ_CLS"

    ################
    # Training
    ################
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_config),
    )
    trainer.train()
    trainer.save_model(reward_config.output_dir)

    # Free memory for merging weights
    del model

    torch.cuda.empty_cache()

    model = AutoPeftModelForSequenceClassification.from_pretrained(reward_config.output_dir, device_map="auto", torch_dtype=torch.bfloat16, num_labels=1)
    model = model.merge_and_unload()

    output_merged_dir = os.path.join(reward_config.output_dir, "final_merged_checkpoint")
    model.save_pretrained(output_merged_dir, safe_serialization=True)
