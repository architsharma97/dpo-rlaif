# A Critical Evaluation of AI Feedback for Aligning Language Models
This codebase mostly builds on the existing code from the [DPO](https://github.com/eric-mitchell/direct-preference-optimization) repo.

## What is this repo?

This repo includes reference code used to investigate training with AI feedback, as described in the paper [A Critical Evaluation of AI Feedback for Aligning Language Models](https://arxiv.org/abs/2305.18290).

The code here supports any causal HuggingFace model - look at our examples in `config/model` to add your own. Adding your own datasets is also easy. See [the README section](#adding-new-datasets) on adding datasets.

Our AI feedback DPO pipeline is outlined below:

1. Run supervised fine-tuning (SFT) on the dataset(s) of interest.
2. Run preference learning on the model from step 1, using preference data (ideally from the same distribution as the SFT examples).


Repo files:

- Slightly modified from DPO script:
    - `train.py`: the main entry point for training (either SFT or DPO preference-based training)
    - `trainers.py`: the trainer classes (e.g., implementing the loop of learning as well as multi-GPU logic)
    - `utils.py`: some convenience functions used by multiple other files
    - `preference_datasets.py`: dataset processing logic for both SFT and DPO preference-based training; **this is where you'll need to make some additions to train on your own data**

- New scripts:
    - `ai_completions.py`: used to generate SFT completions
    - `generate_samples.py`: 
    - `label_ai_preferences.py`: 
    - `reward_trainer.py`: 

## Running SFT
> This README section is mostly copied from the [DPO README](https://github.com/eric-mitchell/direct-preference-optimization), with a few changes.

For DPO, the SFT stage essentially ensures that the preference data we train on is in-distribution for our policy before we actually do the learning from preferences part.

Run SFT for Pythia 6.9B on Anthropic-HH data with batch size 64:
    python train.py model=llama2_7b exp_name=sharegpt1turn_llama2_7b_sft0.1 batch_size=8 eval_batch_size=16 sample_during_eval=false loss=sft debug=false lr=1e-6 trainer=FSDPTrainer activation_checkpointing=True data_fraction=0.1 save_every=epoch_3 n_epochs=9
    python -u train.py model=pythia69 datasets=[hh] loss=sft exp_name=anthropic_dpo_pythia69 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false

Run SFT for a custom model (for example, Llama at a local path) on Anthropic-HH + Stanford Human Preference data with batch size 64:

    python -u train.py model=blank_model model.name_or_path=/PATH/TO/LLAMA/WEIGHTS model.block_name=LlamaDecoderLayer datasets=[hh,shp] loss=sft exp_name=anthropic_shp_sft_llama_7b gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false

> Note: Since we're not using one of our predefined model configs, we also need to pass `model.block_name` to tell FSDP what modules to wrap.

By default, evaluation will run every 20k **examples**. You can change this arg with `eval_every` arg. If you don't pass `sample_during_eval=false`, sampling will happen during each eval as well.

To run a different model, either add a new model config to `config/model`, or use the `blank_model` option for `model` and pass `model.name_or_path` (and `model.block_name` if training with FSDP trainer) explicitly. For example, for GPT-2, this would look like:
    
    python -u train.py ... model=blank_model model.name_or_path=gpt2-xl model.block=GPT2Block

## Running DPO
> This README section is mostly copied from the [DPO README](https://github.com/eric-mitchell/direct-preference-optimization), with a few changes.

To run DPO, use the same command as SFT, but pass `loss=dpo`, `loss.beta=DESIRED_BETA` (0.1-0.5 is a good starting point), and `model.archive=/path/to/checkpoint/from/sft/step-XXXX/policy.pt`. If SFT completed successfully, you should also have a `/.../LATEST/policy.pt` from the end of training.

Run DPO on Pythia 6.9B with effective batch size 64:

    python -u train.py model=pythia69 datasets=[hh] loss=dpo loss.beta=0.1 model.archive=/path/to/checkpoint/from/sft/step-XXXX/policy.pt exp_name=anthropic_dpo_pythia69 gradient_accumulation_steps=2 batch_size=32 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false

> Note: `eval_every` is measured in **examples**.

## A complete example

Let's work through a complete example training Llama2-7B on the ShareGPT dataset.

<!-- See sample wandb outputs for this example [here](https://wandb.ai/eric_anthony_mitchell/dpo-demos) (tagged `readme-example`). -->

### Step 1: Set up environment and paths

First, create a virtualenv and install the dependencies. Python 3.8+ is recommended, though Python 3.10+ is required to do evaluation with alpaca-eval.

    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt

Set up the PROJECT_CACHE path. All trained models and data will be saved to the PROJECT_CACHE.
    
    export PROJECT_CACHE=~/.cache/rlaif

### Step 2: Download/generate SFT data

    wget -P ${PROJECT_CACHE}/sharegpt_data https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json 

### Step 3: Run SFT

We'll take advantage of FSDP's mixed precision in bfloat16 to speed up training; we usually see about a 50% speedup. By default, SFT will run for a single epoch over a mixture of the selected datasets. Datasets will be downloaded on the fly and cached locally.

    python -u train.py loss=sft model=llama2_7b datasets=[sharegpt] exp_name=sharegpt_llama2_7b eval_batch_size=16 sample_during_eval=false debug=false lr=1e-6 trainer=FSDPTrainer activation_checkpointing=True data_fraction=0.1 save_every=epoch_3 n_epochs=9

This runs SFT on 10% of the prompts in ShareGPT, training for 9 epochs and saving every 3 epochs.

> Note: this command is run on a machine with 8 80GB A100s; on this hardware, SFT takes about 2 hours. If you have less compute available, you might need to increase the number of gradient accumulation steps, and SFT will take longer.


### Step 4: Download/generate preference data for DPO

You can directly download our previously generated samples below:

    wget -P ${PROJECT_CACHE}/sharegpt_data/comparisons_gpt4/mistral7bsft_vs_chatgpt/annotations.json https://huggingface.co/datasets/TRI-ML/dpo-rlaif-data/resolve/main/comparisons_gpt4/mistral7bsft_vs_chatgpt/annotations.json

These preferences are generated using previously fine-tuned models such as Llama and Mistral. 

Alternatively, if you wish to generate your own preference data using your newly SFT-ed model, you can do so by following the steps below.

    bash parallel_sample.sh ${PROJECT_CACHE}/sharegpt_llama2_7b_2024-02-19_16-55-49_904051/epoch-9/ llama2_7b

Then,

    python label_ai_preferences.py --model1_name llama2_7b_1.0 --base_dir ${PROJECT_CACHE}/sharegpt_llama2_7b_2024-02-19_16-55-49_904051/epoch-9/ --max_num_comparisons 50000 --llm gpt4


### Step 5: Run DPO

Check either wandb (if enabled, it is by default) or your output log to find the local run directory. To run DPO, you'll need the path to the final weights, which will look something like `/some/cache/dir/pythia28_hh_sft_bf16_2023-06-21_16-58-17_973996/LATEST/policy.pt`. The `LATEST` directory contains the final set of weights from the end of training.

    python -u train.py loss=dpo loss.beta=0.05 model.archive=${PROJECT_CACHE}/sharegpt_llama2_7b_2024-02-19_16-55-49_904051/epoch-9/policy.pt prefs_path=${PROJECT_CACHE}/sharegpt_data/comparisons_gpt4/mistral7bsft_vs_chatgpt/annotations.json exp_name=llama2_7b_dpo data_fraction=1.0 model=llama2_7b save_every=epoch_1 n_epochs=3

On 8 80GB A100s, DPO training took about 2hrs 45min.


### Step 6: Run AlpacaEval evaluations:
Evaluation is done using an oracle annotator with [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval). To run the evaluation script, the general pattern is

    bash ./eval_ckpt.sh (gpt-num) (path-to-checkpoint) (eval-run-name) (oracle-annotator-name) (temperature) (model-name)

Below is a specific example:

    bash ./eval_ckpt.sh 0 ${PROJECT_CACHE}/sharegpt_llama2_7b_2024-02-19_16-55-49_904051/epoch-9 llama2_7b_epoch_9_sft_0.1 gpt4 0.7 llama2_7b


### Customizing training
The options for training are in `config/config.yaml`, `config/model/blank_model.yaml`, and `config/loss/dpo.yaml`. See the comments in these files for more information on what they do.

You can use one of the pre-configured models by passing `model=some_model`, where `config/model/some_model.yaml` exists. We have a few examples already given.

If you want to use another model, just create a new config for that model (following our examples; it must be a `.yaml` file!), or use `model=blank_model` with `model.name_or_path=NAME_OR_PATH`, optionally `model.tokenizer_name_or_path=TOKENIZER_NAME_OR_PATH` if it is different than the model's name/path, and `model.block_name=NAME_OF_TRANSFORMER_BLOCK` (if you are using FSDP). The only other options you might want to change are the dpo loss options, which are `loss.beta` and `loss.reference_free` (see `config/loss/dpo.yaml`).

## Trainer classes
> This README section is mostly copied from the [DPO README](https://github.com/eric-mitchell/direct-preference-optimization), with a few changes.

We implement three different trainer classes in `trainers.py`:
- `BasicTrainer`: For multiple GPUs, naively partition the model among them. e.g., for two GPUs, the first half of the model layers will be on GPU 0, the second half will be on GPU 1. This trainer effectively increases your available GPU memory without using multiple GPUs are once for compute (so you get no speedup).
- `FSDPTrainer`: Use PyTorch's [Fully Sharded Data Parallel](https://pytorch.org/docs/stable/fsdp.html) (FSDP) implementation to shard each transformer block amongst available GPUs. Should give a significant speedup over `BasicTrainer` with batch size per GPU >1. The batch size per gpu is equal to `batch_size / (gradient_accumulation_steps * num_gpus)`. **You may need to run `ulimit -n 64000` in your launch script before calling `train.py` with this trainer; e.g., `ulimit -n 64000; python train.py ...`.**
- `TensorParallelTrainer`: Use PyTorch tensor parallelism (with [this wrapper](https://github.com/BlackSamorez/tensor_parallel)) to shard each linear layer amongst available GPUs. This trainer is experimental, but should work.

**Warning:** Sampling may be very slow for `FSDPTrainer` and especially `TensorParallelTrainer` (see [this issue](https://github.com/pytorch/pytorch/issues/100069) and [this issue](https://github.com/BlackSamorez/tensor_parallel/issues/66), respectively for `FSDPTrainer` and `TensorParallelTrainer`). Passing `sample_during_eval=false` is recommended for these trainers.

### Which trainer do I use?
 For single GPU training, use `BasicTrainer`. For many-GPU setups, `FSDPTrainer` will most likely be the best choice, though these haven't been benchmarked yet.

# Adding new datasets
> This README section is mostly copied from the [DPO README](https://github.com/eric-mitchell/direct-preference-optimization), with a few changes.

Adding new/custom datasets is easy, and shouldn't take more than 10 minutes or so. Add your dataset to `preference_datasets.py` (we've implemented Anthropic-HH, Stanford Human Preferences, and StackExchange as references). Follow our reference datasets (in the functions `get_se()`, `get_shp()`, `get_hh()`); you essentially need to return a dict mapping each prompt to another dict containing three values:

- `responses: List[str]`: the list of responses on which preferences are given
- `pairs: List[Tuple[int]]`: the preference pairs, where the first value in each tuple is the preferred response and the second value is the dispreferred response
- `sft_target: str`: the response to use for this prompt during SFT (this response may or may not be one of the values in `responses`)

Once you've added your dataset, for example `xyz`, you can train on it by passing it to `datasets=[xyz]` to an SFT or DPO train command.

**Make sure you've updated `preference_datasets:get_dataset()` to return your new dataset when its name is passed in!**

# Tips for faster training on multiple GPUs
> This README section is mostly copied from the [DPO README](https://github.com/eric-mitchell/direct-preference-optimization), with a few changes.

FSDP is recommended for faster training when multiple GPUs are available. In general, you should try to use a batch size of at least 2 on each GPU (i.e., `batch_size // (grad_accumulation_steps * N_GPUS)` is at least 2) to see a speedup from FSDP compared to the `BasicTrainer`. One way to do this is to use mixed precision. This repo implements mixed precision through [FSDP](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision). Enable mixed precision (only supported for `FSDPTrainer`, currently) by passing `model.fsdp_policy_mp=bfloat16` or `model.fsdp_policy_mp=float16` (only `bfloat16` has been tested). Another way to reduce memory usage is activation checkpointing (or *gradient checkpointing*), which can be enabled with `activation_checkpointing=true` (also implemented only for `FSDPTrainer`). Activation checkpointing doesn't always increase throughput, but if you're stuck at batch size per GPU of 1, it's worth a try.

See [this article](https://pytorch.org/blog/efficient-large-scale-training-with-pytorch/) for more information about optimizing FSDP.

# Citation
If DPO or this repository is useful in your own research, you can use the following BibTeX entry:

    @misc{rafailov2023direct,
        title={Direct Preference Optimization: Your Language Model is Secretly a Reward Model}, 
        author={Rafael Rafailov and Archit Sharma and Eric Mitchell and Stefano Ermon and Christopher D. Manning and Chelsea Finn},
        year={2023},
        eprint={2305.18290},
        archivePrefix={arXiv},
    }
