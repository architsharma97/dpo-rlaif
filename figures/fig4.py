import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

models = []
winrate_method = {}

def add_model(model_name, display_name, n1, n2, n3, n4):
    models.append(model_name)
    winrate_method[f"{model_name} display name"] = display_name
    winrate_method[f"{model_name} chatgpt sft 100p"] = n1
    winrate_method[f"{model_name} chatgpt sft 10p"] = n2
    winrate_method[f"{model_name} chatgpt sft 10p dpo 90p"] = n3
    winrate_method[f"{model_name} gpt4 sft 100p"] = n4

add_model("llama-7b", "LLaMA 7B", [57.5625, 1.74], [55.00, 1.76], [66.5, 1.67], [64.6875, 1.69])
add_model("mistral-7b", "Mistral 7B", [59.31, 1.73], [59.32, 1.74], [71.38, 1.60], [69.4375, 1.626])


color_palette = ["#FF6150", "#134E6F", "#DEE0E6", "#1AC0C6", "#FFA822", "#134E6F"]
fig, ax = plt.subplots(figsize=(11.4, 6))
ax.set_ylim([40, 80])
# ax.set_xlim([-1, 6])
ax.set_ylabel('Alpaca Eval Winrate % (Measured by Claude)', fontsize=14)
ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
# ax.set_xlabel('Method')
ax.set_title('Fine-tuning Base LLMs on ShareGPT Prompts (Claude version)', fontsize=18)


xticks = []
xticklabels = []
BAR_WIDTH = 0.5
MODEL_SPACING = 0.3
for idx, i in enumerate(models):
    if len(xticks)==0:
        xticks.append(2*BAR_WIDTH)
    else:
        xticks.append(xticks[-1] + 3*BAR_WIDTH + MODEL_SPACING)
    ax.set_xticks(xticks)
    xticklabels.append(winrate_method[f"{i} display name"])
    ax.set_xticklabels(xticklabels, fontsize=18)
    ax.yaxis.set_tick_params(labelsize=12)
    
    pos = xticks[-1] - BAR_WIDTH
    n1, n2, n3, n4 = winrate_method[f"{i} chatgpt sft 100p"], winrate_method[f"{i} chatgpt sft 10p"], winrate_method[f"{i} chatgpt sft 10p dpo 90p"], winrate_method[f"{i} gpt4 sft 100p"]
    ax.bar(pos, n1[0], yerr=n1[1], width=BAR_WIDTH, color=color_palette[idx], edgecolor='black', capsize=5)
    ax.bar(pos+BAR_WIDTH, n2[0], width=BAR_WIDTH, color=color_palette[idx], edgecolor='black', capsize=5)
    ax.bar(pos+BAR_WIDTH, n3[0]-n2[0], yerr=n3[1], width=BAR_WIDTH, color=color_palette[idx], edgecolor='black', hatch='o', capsize=5, bottom=n2[0])
    ax.bar(pos+2*BAR_WIDTH, n4[0], yerr=n4[1], width=BAR_WIDTH, color=color_palette[idx], edgecolor='black', hatch='//\\\\', capsize=5)


handle1 = mpatches.Patch(facecolor='#FFFFFF', edgecolor='#000000', label='SFT w/ GPT-3.5 (default)')
handle2 = mpatches.Patch(facecolor='#FFFFFF', edgecolor='#000000', hatch='o', label='+ RLAIF w/ Claude preferences')
handle3 = mpatches.Patch(facecolor='#FFFFFF', edgecolor='#000000', hatch='//\\\\', label='SFT w/ Claude')

ax.legend(handles=[handle1, handle2, handle3], handlelength=3, handleheight=3, fontsize=11, loc='upper center')
plt.savefig('fig4.png', dpi=300, bbox_inches='tight')
plt.show()
