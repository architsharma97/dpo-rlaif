#!/bin/bash
cd ~/dpo-rlaif
annotator="${4:-claude_2}"
temperature="${5:-0.7}"

if ![-f $2/"alpaca_eval_temp$temperature.json"]; then
    echo "generating samples" > $2/eval_log.txt
    conda run -n rlhf --no-capture-output bash -c "CUDA_VISIBLE_DEVICES=$1 python3 generate_samples.py --archive $2 --prompt_set alpaca_eval --temperatures $temperature"
fi
echo "starting alpaca evaluation" > $2/eval_log.txt
mkdir -p $2/$annotator
conda run -n rlhf --no-capture-output bash -c "alpaca_eval --model_outputs $2/alpaca_eval_temp$temperature.json --annotators_config $annotator --name $3 --output_path $2/$annotator"
echo "finished alpaca evaluation" > $2/eval_log.txt