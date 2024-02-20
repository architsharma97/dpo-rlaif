import transformers
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
from preference_datasets import get_batch_iterator, get_dataset
import argparse
import wandb
import os
from collections import defaultdict
import time

from utils import pad_to_length, get_local_run_dir

def get_logits(model, batch):
    chosen_batch = {'attention_mask': batch['chosen_attention_mask'], 'input_ids': batch['chosen_input_ids']}
    rejected_batch = {'attention_mask': batch['rejected_attention_mask'], 'input_ids': batch['rejected_input_ids']}
    # combine the chosen and rejected batches into one batch
    # combined_batch = {}
    # max_length = max(chosen_batch['attention_mask'].shape[1], rejected_batch['attention_mask'].shape[1])
    # pad_vals = {'attention_mask': 0, 'input_ids': tokenizer.pad_token_id}
    # for key in chosen_batch.keys():
    #     chosen_vals = pad_to_length(chosen_batch[key], max_length, pad_value=pad_vals[key])
    #     rejected_vals = pad_to_length(rejected_batch[key], max_length, pad_value=pad_vals[key])
    #     combined_batch[key] = torch.cat([chosen_vals, rejected_vals], dim=0)

    # logits = model(**combined_batch)['logits']
    # split_size = chosen_batch['attention_mask'].shape[0]
    # return logits[:split_size], logits[split_size:]

    chosen_logits = model(**chosen_batch)['logits']
    rejected_logits = model(**rejected_batch)['logits']
    return chosen_logits, rejected_logits

def train_step(model, batch, optimizer):
    metrics = {}
    chosen_logits, rejected_logits = get_logits(model, batch)

    loss = -F.logsigmoid(chosen_logits - rejected_logits).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    metrics['train/loss'] = loss.item()
    metrics['train/accuracy'] = (chosen_logits > rejected_logits).float().mean().item()
    return metrics


def eval_loop(model, eval_batches):
    def _eval_step(model, batch):
        metrics = {}
        with torch.no_grad():
            chosen_logits, rejected_logits = get_logits(model, batch)

        metrics['eval/loss'] = (-F.logsigmoid(chosen_logits - rejected_logits)).squeeze(1).cpu().numpy().tolist()
        metrics['eval/accuracy'] = (chosen_logits > rejected_logits).float().squeeze(1).cpu().numpy().tolist()
        return metrics

    eval_start = time.time() 
    model.eval()
    metrics = defaultdict(list)
    for batch in eval_batches:
        batch_metrics = _eval_step(model, batch)
        for k, v in batch_metrics.items():
            metrics[k].extend(v)
    eval_time = time.time() - eval_start
    wandb.log({k: sum(v) / len(v) for k, v in metrics.items()},
                step=(idx+1)*args.batch_size)
    wandb.log({'time/eval_time': eval_time}, step=idx * args.batch_size)
    model.train()
    return metrics

def save(model, optimizer, test_metrics, epoch, step, exp_dir):
    save_dir = os.path.join(exp_dir, f'epoch-{epoch}')
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'state': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
    }, os.path.join(save_dir, 'checkpoint.pt'))
    with open(os.path.join(save_dir, 'test_predictions.txt'), 'w') as f:
        for acc, loss in zip(test_metrics['eval/accuracy'], test_metrics['eval/loss']):
            f.write(f'{acc},{loss}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=str, default=None)
    parser.add_argument('--prompt_set', type=str, default='sharegpt')
    parser.add_argument('--prefs_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cache_dir', type=str, default=os.getenv("PROJECT_CACHE", "~/.cache"))
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_prompt_length', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--data_fraction', type=float, default=1.0)
    parser.add_argument('--exp_name', type=str, default='reward_training_test')
    parser.add_argument('--eval_frequency', type=int, default=10000)
    parser.add_argument('--n_eval_examples', type=int, default=None)
    parser.add_argument('--save_frequency', type=int, default=1, help='number of epochs between saves')
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=1e-6)
    args = parser.parse_args()

    if args.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    exp_dir = get_local_run_dir(args.exp_name, [os.getenv("PROJECT_CACHE", "~/.cache")])
    os.makedirs(exp_dir, exist_ok=True)
    print(f'experiment directory: {exp_dir}')
    os.environ['WANDB_CACHE_DIR'] = args.cache_dir
    wandb.init(
        entity=None,
        project='dpo-rlaif',
        dir=args.cache_dir,
        name=args.exp_name,
    )

    model = transformers.AutoModelForSequenceClassification.from_pretrained('mistralai/Mistral-7B-v0.1', num_labels=1, 
                                                                            torch_dtype=torch.float32, low_cpu_mem_usage=True, device_map='balanced')
    tokenizer = transformers.AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    if args.archive is not None:
        state_dict = torch.load(args.archive, map_location='cpu')
        model.load_state_dict(state_dict['state'], strict=False)
        print('loaded pre-trained weights')

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    if 'sharegpt' in args.prompt_set:
        assert args.prefs_path is not None
    train_iterator = get_batch_iterator([args.prompt_set], tokenizer=tokenizer, split='train', batch_size=args.batch_size, sft_mode=False,
                                         seed=0, n_epochs=args.num_epochs, cache_dir=args.cache_dir, shuffle=False,
                                         max_prompt_length=args.max_prompt_length, max_length=args.max_length,
                                         num_turns=1, data_fraction=args.data_fraction, prefs_path=args.prefs_path, sampled_data_dir=None)
    if args.n_eval_examples is None:
        n_eval_epochs = 1
    else:
        n_eval_epochs = None
    eval_iterator = get_batch_iterator([args.prompt_set], tokenizer=tokenizer, split='test', batch_size=args.batch_size*2, sft_mode=False,
                                        seed=0, n_examples=args.n_eval_examples, n_epochs=n_eval_epochs, cache_dir=args.cache_dir, shuffle=False,
                                        max_prompt_length=args.max_prompt_length, max_length=args.max_length,
                                        num_turns=1, data_fraction=args.data_fraction, prefs_path=args.prefs_path, sampled_data_dir=None)
    eval_batches = list(eval_iterator)
    model.train()

    # compute the number of examples per epoch: TODO(works for single dataset only)
    all_data = get_dataset(args.prompt_set, cache_dir=args.cache_dir, split='train', prefs_path=args.prefs_path, num_turns=1, data_fraction=args.data_fraction)
    n_examples_per_epoch = (len(all_data) // args.batch_size) * args.batch_size
    next_save = n_examples_per_epoch

    for idx, batch in enumerate(train_iterator):
        if (idx * args.batch_size) % args.eval_frequency == 0:
            eval_loop(model, eval_batches)

        train_start = time.time()
        train_metrics = train_step(model, batch, optimizer)
        train_time = time.time() - train_start
        wandb.log(train_metrics, step=(idx+1)*args.batch_size)
        wandb.log({'time/train_time': train_time}, step=(idx+1)*args.batch_size)
        wandb.log({'time/time_per_example': train_time / args.batch_size}, step=(idx+1)*args.batch_size)
        wandb.log({'time/examples_per_second': args.batch_size / train_time}, step=(idx+1)*args.batch_size)
        print(f'step: {idx+1}, train loss: {train_metrics["train/loss"]:.4f}, train acc: {train_metrics["train/accuracy"]:.4f}')

        if (idx + 1) * args.batch_size >= next_save:
            print(f'saving checkpoint at step {(idx + 1) * args.batch_size}')
            test_metrics = eval_loop(model, eval_batches)
            save(model, optimizer, test_metrics, ((idx + 1) * args.batch_size) // n_examples_per_epoch, idx, exp_dir)
            next_save += args.save_frequency * n_examples_per_epoch