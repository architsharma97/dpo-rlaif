import os
import json
from datasets import Dataset
import tqdm

path = '/ebs/.cache/ubuntu'
subpaths = dict(
    llama7b='llama7b_samples/ultrachat_maxlen512_temp1.0/fastforward0.json',
    ultralm='ultralm_samples/ultrachat_maxlen512_temp1.0/fastforward0.json',
    falcon='falcon_samples/ultrachat_maxlen512_temp1.0/fastforward0.json'
)
hug_token = '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'
# Load the dataset
all_rows = []
for name, subpath in tqdm.tqdm(subpaths.items()):
    with open(os.path.join(path, subpath), 'r') as f:
        data = json.load(f)
    for prompt, response_list in tqdm.tqdm(data.items()):
        all_rows.append((name, prompt, response_list))

def gen():
    for name, prompt, response_list in all_rows:
        assert len(response_list) == 1
        response = response_list[0]
        yield {
            'name': name,
            'prompt': prompt,
            'response': response
        }

ds = Dataset.from_generator(gen)
ds.save_to_disk('/ebs/.cache/ubuntu/ultrachat_samples')
ds.push_to_hub('Asap7772/ultrachat_samples', token=hug_token)