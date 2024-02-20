# A Critical Evaluation of AI Feedback for Aligning Language Models

## What is this repo?

This repo is the official implementation for aligning language models with AI feedback and instruction-tuning, described in the paper [A Critical Evaluation of AI Feedback for Aligning Language Models](https://arxiv.org/abs/2402.12366). This codebase builds on the official [DPO](https://github.com/eric-mitchell/direct-preference-optimization) implementation.

This repository supports the following functions:
1. Generating completions using the OpenAI and Anthropic API to create instruction-tuning datasets.
2. Fine-tuning LLMs using SFT.
3. Sampling pairs of completions for an instruction, and labeling the preferred answer using OpenAI or Anthropic API. AI Feedback generation builds on AlpacaEval.
4. Fine-tuning LLMs using DPO on AI Feedback datasets.
5. Evaluating instruction-tuned models using AlpacaEval.


Bonus files:
1. Training reward models
2. Bandit experiments comparing RLAIF and SFT (Figure 6 in the [paper](https://arxiv.org/abs/2402.12366)).

We first give a general description for every function, and then we work through an exact example to fine-tune a LLM by SFT and DPO on AI feedback. The code here supports full fine-tuning for causal LLMs accessible through the HuggingFace transformers API. Check out [Customizing training](#customizing-training) to fine-tune your own LLMs.

## Generating Instruction-tuning Data
Ensure that `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is setup before using the script. This script will generate completions for a filtered set of instructions from ShareGPT using the API, and dump the instruction-completion pairs in a json file in `<PROJECT_CACHE>`. An example command is given below:

    python ai_completions.py --ai_model gpt4 --data_fraction 1.0

You can pass `claude` to `--ai_model` to generate completions using `Claude-v1`, and pass a fraction < 1 to `--data_fraction` to generate completions for fewer instructions. Note, that we already provide the instruction-tuning datasets used for our experiments along with this repository.

## Running SFT

To instruction-tune a base model using SFT, we can run the following command:

    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py model=mistral7b exp_name=<exp_name> batch_size=8 eval_batch_size=16 sample_during_eval=false loss=sft lr=1e-6 trainer=FSDPTrainer activation_checkpointing=True data_fraction=1.0 save_every=epoch_2 n_epochs=6 datasets=[sharegpt4]

The above command will instruction-tune a base mistral 7B model on the GPT-4 generated completions, using a `batch_size=8` and a learning rate `lr=1e-6`. IF debugging, it might be useful to pass `debug=True` and `trainer=BasicTrainer`. The dataset size can be adjusted using `data_fraction=1.0`, and the number of epochs and saving frequency can be adjusted using `save_every` and `n_epochs`.

## Generating AI Feedback

To generate a preference dataset, we first need to sample completions from the SFT model (or whatever LLM you may want to use). Sampling LLMs is done through `generate_samples.py`, for example:

    CUDA_VISIBLE_DEVICES=0 python3 generate_samples.py --archive <path_to_model_ckpt> --temperatures 0.7 --data_fraction 1.0 --model_name mistral7b --prompt_set sharegpt

Next, we will label the pairs of completions using GPT-4 or Claude:

    python label_ai_preferences.py --model1_name mistral7b_1.0 --model2_name chatgpt --base_dir ${PROJECT_CACHE}/sharegpt_mistral7b_2024-02-19_16-55-49_904051/epoch-9/ --max_num_comparisons 50000 --llm gpt4

By default, `--model2_name` uses GPT-3.5 completions in the ShareGPT dataset. It can alternately accept `gpt4`, `claude` or `<model_name>_<temp>`, where the last option can be used to label preferences on pairs of completions generated from the model itself. To change the critic labeling the preferences, pass the option to `--llm`. Only `gpt4` and `claude` are supported right now.

## Running DPO

To fine-tune the model using DPO, we can use the following command:

    CUDA_VISIBLE_DEVICES=4,5,6,7 python3 train.py loss=dpo loss.beta=0.05 model.archive=<path_to_model>/policy.pt prefs_path=<path_to_ai_feedback>/annotations.json exp_name=<exp_name> model=mistral7b save_every=epoch_1 n_epochs=3 fsdp_port=1234

To train on newly generated annotations, pass the output directory from the previous step to `--prefs_path`. Alternately, you can use a pre-generated preference dataset, as shown in our worked example.


# A Complete Example

Let's work through a complete example training Mistral-7B on the ShareGPT dataset.

### Step 1: Set up environment and paths

First, create a virtualenv and install the dependencies. Python 3.8+ is recommended, though Python 3.10+ is required to do evaluation with alpaca-eval.

    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt

Set up the PROJECT_CACHE path. All trained models and data will be saved to the PROJECT_CACHE.
    
    export PROJECT_CACHE=~/.cache/rlaif

### Step 2: Download/generate SFT data

    wget -P ${PROJECT_CACHE}/sharegpt_data https://huggingface.co/datasets/TRI-ML/dpo-rlaif-data/resolve/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json 

### Step 3: Run SFT

We'll take advantage of FSDP's mixed precision in bfloat16 to speed up training; we usually see about a 50% speedup. By default, SFT will run for a single epoch over a mixture of the selected datasets. Datasets will be downloaded on the fly and cached locally.

    python -u train.py loss=sft model=mistral7b datasets=[sharegpt] exp_name=sharegpt_mistral7b eval_batch_size=16 sample_during_eval=false debug=false lr=1e-6 trainer=FSDPTrainer activation_checkpointing=True data_fraction=0.1 save_every=epoch_3 n_epochs=9

This runs SFT on 10% of the prompts in ShareGPT, training for 9 epochs and saving every 3 epochs. Assume that the auto-generated output directory is `${PROJECT_CACHE}/sharegpt_mistral7b_2024-02-19_16-55-49_904051/`

> Note: this command is run on a machine with 8 80GB A100s; on this hardware, SFT takes about 2 hours. If you have less compute available, you might need to increase the number of gradient accumulation steps, and SFT will take longer.

### Step 4: Download/generate preference data for DPO

You can directly download our previously generated samples below:

    wget -P ${PROJECT_CACHE}/sharegpt_data/comparisons_gpt4/mistral7bsft_vs_chatgpt https://huggingface.co/datasets/TRI-ML/dpo-rlaif-data/resolve/main/comparisons_gpt4/mistral7bsft_vs_chatgpt/annotations.json

These preferences are generated using previously fine-tuned models such as Llama and Mistral. 

Alternatively, if you wish to generate your own preference data using your newly SFT-ed model, you can do so by following the steps below.

    bash parallel_sample.sh ${PROJECT_CACHE}/sharegpt_mistral7b_2024-02-19_16-55-49_904051/epoch-9/ mistral7b

Note that the script above assumes 8 GPUs. If you are using less GPUs, then you may need to modify the script accordingly. Specifically, lines 10 (`for i in {0..7}`) and 12 (`ff_idx`) will need to be changed. Afterwards, you can run the script below to perform the preference labeling using GPT4.

    python label_ai_preferences.py --model1_name mistral7b_1.0 --base_dir ${PROJECT_CACHE}/sharegpt_mistral7b_2024-02-19_16-55-49_904051/epoch-9/ --max_num_comparisons 50000 --llm gpt4

If you generate your own AI feedback dataset, it will be stored by default at `${PROJECT_CACHE}/sharegpt_mistral7b_2024-02-19_16-55-49_904051/epoch-9/comparisons_gpt4/temp1.0_vs_chatgpt/annotations.json`.

### Step 5: Run DPO

Check either wandb (if enabled, it is by default) or your output log to find the local run directory. To run DPO, you'll need the path to the final weights, which will look something like `${PROJECT_CACHE}/sharegpt_mistral7b_2024-02-19_16-55-49_904051/epoch-9/policy.pt`.

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train.py loss=dpo loss.beta=0.05 model.archive=~/.cache/rlaif/sharegpt_pythia28_2024-02-19_16-55-49_904051/epoch-3/policy.pt prefs_path=${PROJECT_CACHE}/sharegpt_data/comparisons_gpt4/mistral7bsft_vs_chatgpt/annotations.json exp_name=pythia28 data_fraction=1.0 model=pythia28 save_every=epoch_1 n_epochs=3

> On 8 80GB A100s, DPO training took about 2hrs 45min.

### Step 6: Run AlpacaEval evaluations:
Evaluation is done using an oracle annotator with [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval). To run the evaluation script, the general pattern is

    bash ./eval_ckpt.sh (gpu-num) (path-to-checkpoint) (eval-run-name) (oracle-annotator-name) (temperature) (model-name)

Below is a specific example:

    bash ./eval_ckpt.sh 0 ${PROJECT_CACHE}/sharegpt_mistral7b_2024-02-19_16-55-49_904051/epoch-9 mistral7b_epoch_9_sft_0.1 gpt4 0.7 mistral7b

## Bandit experiment

Section 5 of the paper discusses mechanistic explanations for the failure of RLAIF to provide substantial improvements over SFT on a strong teacher. To provide intuition, we conduct a synthetic experiment in a simple bandit setting (i.e., no prompt, only a small number of possible synthetic responses). This experiment illustrates how the improvement of RLAIF may not be sufficient to overcome a sufficiently poor initial model, relative to a strong teacher model.

Run all cells in `synthetic-rlaif.ipynb` to run this experiment. This experiment requires only `torch`, `numpy`, and `matplotlib` to run.


### Customizing training
The options for training are in `config/config.yaml`, `config/model/blank_model.yaml`, and `config/loss/dpo.yaml`. See the comments in these files for more information on what they do.

You can use one of the pre-configured models by passing `model=some_model`, where `config/model/some_model.yaml` exists. We have a few examples already given.

If you want to use another model, just create a new config for that model (following our examples; it must be a `.yaml` file!), or use `model=blank_model` with `model.name_or_path=NAME_OR_PATH`, optionally `model.tokenizer_name_or_path=TOKENIZER_NAME_OR_PATH` if it is different than the model's name/path, and `model.block_name=NAME_OF_TRANSFORMER_BLOCK` (if you are using FSDP). The only other options you might want to change are the dpo loss options, which are `loss.beta` and `loss.reference_free` (see `config/loss/dpo.yaml`).

# Citation
If our paper or this repository is useful for your research, you can use the following BibTeX entry:

    @article{
        sharma2024critical,
        title={A Critical Evaluation of AI Feedback for Aligning Large Language Models},
        author={Archit Sharma and Sedrick Keh and Eric Mitchell and Chelsea Finn and Kushal Arora and Thomas Kollar},
        journal={arXiv preprint arXiv:2402.12366},
        year={2024},
        url={http://arxiv.org/abs/2402.12366}
    }
