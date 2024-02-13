which_gpus=(0 1 2 3 4 5 6 7)
models=(llama7b ultralm wizardlm falcon)
ff_idxs=(0 15000)
temperature=1.0
prompt_set=ultrachat

exp_num=0
which_exp=${1:--1}
dryrun=${2:-false}

for model in "${models[@]}"; do
for ff_idx in "${ff_idxs[@]}"; do
    if [[ $exp_num != $which_exp && $which_exp -ge 0 ]]; then
        exp_num=$((exp_num+1))
        continue
    fi
    which_gpu=${which_gpus[$exp_num % ${#which_gpus[@]}]}
    export CUDA_VISIBLE_DEVICES=$which_gpu

    echo "Using GPU $which_gpu"
    echo "Generating samples for $model"
    command="python3 generate_samples.py \
    --temperatures $temperature \
    --ff $ff_idx \
    --data_fraction 1.0 \
    --model_name $model \
    --prompt_set $prompt_set
    "

    if [[ $which_exp -lt 0 ]]; then
        command+=" &"
    fi
    echo -e "$command\n"
    if [ $dryrun = false ]; then
        eval $command
        sleep 20
    fi
    exp_num=$((exp_num+1))
done
done
