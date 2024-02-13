import os
import csv

location='/ebs/.cache/ubuntu'
gpt4 = [
    'llama7b_samples',
]

claude = [
    'llama7b_samples'
]
all_files = os.listdir(location)

filtered_gpt = [x for x in all_files if any([y in x for y in gpt4])]
filtered_claude = [x for x in all_files if any([y in x for y in claude])]


print('GPT4')
for f in filtered_gpt:
    full_path = os.path.join(location, f)
    print(full_path)
    epochs = [x for x in os.listdir(full_path) if 'epoch' in x]
    for e in epochs:
        epoch_path = os.path.join(full_path, e)
        print(e)
        files = os.listdir(epoch_path)
        
        if 'name.txt' not in files:
            continue
        name = os.popen(f'cat {epoch_path}/name.txt').read().strip()
        if 'gpt4' not in files or not os.path.exists(f'{epoch_path}/gpt4/leaderboard.csv'):
            continue
        print(name)
        grep = os.popen(f'cat {epoch_path}/gpt4/leaderboard.csv | grep {name}').read().strip()
        win = grep.split(',')[1]
        std = grep.split(',')[2]
        print(win, std)
    print('---')


print('Claude')
for f in filtered_claude:
    full_path = os.path.join(location, f)
    print(full_path)
    epochs = [x for x in os.listdir(full_path) if 'epoch' in x]
    for e in epochs:
        epoch_path = os.path.join(full_path, e)
        print(e)
        files = os.listdir(epoch_path)
        
        if 'name.txt' not in files:
            continue
        name = os.popen(f'cat {epoch_path}/name.txt').read().strip()
        print(name)

        claude_files = [x for x in files if 'claude' in x]
        for claude_file in claude_files:
            if not os.path.exists(f'{epoch_path}/{claude_file}/leaderboard.csv'):
                continue
            if claude_file == 'claude_2':
                continue
            grep = os.popen(f'cat {epoch_path}/{claude_file}/leaderboard.csv | grep {name}').read().strip()
            win = grep.split(',')[1]
            std = grep.split(',')[2]
            print(win, std)
    print('---')