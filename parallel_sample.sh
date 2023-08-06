#!/bin/bash
cd ~/dpo-rlaif
# examples
# /home/archit/.cache/archit/llama7b_sharegpt_sft/step-250016
# /ebs/.cache/ubuntu/sharegpt2turn_llama7b_sft_maxlen512_2023-07-24_16-38-00_740003/step-550000
model_ckpt_dir=$1

for i in {6..7}
do
    ff_idx=$(((i-6)*100000))
    screen -dmS samp$i bash -c "conda run -n rlhf --no-capture-output CUDA_VISIBLE_DEVICES=$i python3 generate_samples.py --archive $model_ckpt_dir --temperatures 0.7 --ff $ff_idx"
done