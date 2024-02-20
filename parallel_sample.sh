#!/bin/bash
cd ~/dpo-rlaif
# examples
# /home/ubuntu/.cache/rlaif/sharegpt2turn_llama7b_sft_maxlen512_2023-07-24_16-38-00_740003/step-550000
model_ckpt_dir=$1
model_name="${2:-llama7b}"
temperature="${3:-1.0}"
prompt_set="${4:-sharegpt}"

for i in {0..7}
do
    ff_idx=$(((i-0)*5000 + 6000))
    screen -dmS samp$i bash -c "conda run -n rlhf --no-capture-output CUDA_VISIBLE_DEVICES=$i python3 generate_samples.py --archive $model_ckpt_dir --temperatures $temperature --ff $ff_idx --data_fraction 1.0 --model_name $model_name --prompt_set $prompt_set"
done