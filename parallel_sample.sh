#!/bin/bash
cd ~/dpo-rlaif
model_ckpt_dir="/ebs/.cache/ubuntu/sharegpt2turn_llama7b_sft_maxlen512_2023-07-24_16-38-00_740003/step-550000"

for i in {0..7}
do
    ff_idx=$((i*16000))
    screen -dmS samp$i bash -c "conda run -n rlhf --no-capture-output CUDA_VISIBLE_DEVICES=$i python3 generate_samples.py --archive $model_ckpt_dir --temperatures 0.7 --ff $ff_idx"
done