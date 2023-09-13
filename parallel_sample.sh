#!/bin/bash
cd ~/dpo-rlaif
# examples
# /home/archit/.cache/archit/llama7b_sharegpt_sft/step-250016
# /ebs/.cache/ubuntu/sharegpt2turn_llama7b_sft_maxlen512_2023-07-24_16-38-00_740003/step-550000
model_ckpt_dir=$1
temperature="${2:-1.0}"

for i in {0..6}
do
    ff_idx=$(((i-0)*16000 + 5000))
    screen -dmS samp$i bash -c "conda run -n rlhf --no-capture-output CUDA_VISIBLE_DEVICES=$i python3 generate_samples.py --archive $model_ckpt_dir --temperatures $temperature --ff $ff_idx --data_fraction 1.0"
done