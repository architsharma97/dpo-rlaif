#!/bin/bash
cd ~/dpo-rlaif
conda run -n rlhf --no-capture-output bash -c "CUDA_VISIBLE_DEVICES=$1 python3 generate_samples.py --archive $2 --prompt_set alpaca_eval --temperatures 0.7"
echo "starting alpaca evaluation" > $2/eval_log.txt
conda run -n rlhf --no-capture-output bash -c "alpaca_eval --model_outputs $2/alpaca_eval_temp0.7.json --annotators_config claude --name $3 --output_path $2"
echo "finished alpaca evaluation" > $2/eval_log.txt