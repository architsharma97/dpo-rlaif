import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

models = []
winrate_method = {}

def add_model(model_name, display_name, n1, n2, n3):
    models.append(model_name)
    winrate_method[f"{model_name} display name"] = display_name
    winrate_method[f"{model_name} gpt4 sft 10p"] = n1
    winrate_method[f"{model_name} gpt4 sft 10p dpo 90p"] = n2
    winrate_method[f"{model_name} gpt4 sft 100p"] = n3

add_model("llama 7b", "LLaMA 7B", [51.8125, 1.764], [52.875, 1.766], [57.4375, 1.7435])
add_model("yi 6b", "Yi 6B", [59.25, 1.73], [61.25, 1.72], [62.25, 1.71])
add_model("mistral 7b", "Mistral 7B", [72.375, 1.577], [77.875, 1.458], [75.3125, 1.53])



color_palette = ["#1AC0C6", "#DEE0E6", "#134E6F", "#FFA822", "#134E6F", "#134E6F"]

# First Subplot
fig, ax = plt.subplots(figsize=(11.2, 5.5))
ax.set_ylim([40, 80])
# ax.set_xlim([-1, 6])
ax.set_ylabel('Alpaca Eval Winrate %', fontsize=16)
ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
ax.set_title('SFT and DPO Using Completions From Stronger Oracle Models (GPT-4)', fontsize=16)

# ax.set_xlabel('Method')


xticks = []
xticklabels = []
BAR_WIDTH = 0.5
MODEL_SPACING = 0.2
for idx, i in enumerate(models):
    if len(xticks)==0:
        xticks.append(BAR_WIDTH)
    else:
        xticks.append(xticks[-1] + 2*BAR_WIDTH + MODEL_SPACING)
    ax.set_xticks(xticks)
    xticklabels.append(winrate_method[f"{i} display name"])
    ax.set_xticklabels(xticklabels, fontsize=16)
    
    pos = xticks[-1] - BAR_WIDTH / 2
    n1, n2, n3 = winrate_method[f"{i} gpt4 sft 10p"], winrate_method[f"{i} gpt4 sft 10p dpo 90p"], winrate_method[f"{i} gpt4 sft 100p"]
    ax.bar(pos, n1[0], width=BAR_WIDTH, color=color_palette[idx], edgecolor='black', capsize=5)
    ax.bar(pos, n2[0]-n1[0], yerr=n2[1], width=BAR_WIDTH, color=color_palette[idx], edgecolor='black', hatch='o', capsize=5, bottom=n1[0])
    ax.bar(pos+BAR_WIDTH, n3[0], yerr=n3[1], width=BAR_WIDTH, color=color_palette[idx], edgecolor='black', hatch='//\\\\', capsize=5)


handle1 = mpatches.Patch(facecolor='#FFFFFF', edgecolor='#000000', label='SFT w/ GPT-4 (10% of total examples)')
handle2 = mpatches.Patch(facecolor='#FFFFFF', edgecolor='#000000', hatch='o', label='+ RLAIF w/ GPT-4 preferences')
handle3 = mpatches.Patch(facecolor='#FFFFFF', edgecolor='#000000', hatch='//\\\\', label='SFT w/ GPT-4 (100% of total examples)')

ax.legend(handles=[handle1, handle2, handle3], handlelength=3, handleheight=3, fontsize=13)


plt.savefig('fig5.png', dpi=300, bbox_inches='tight')
plt.show()