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

add_model("deepseek 7b", "DeepSeek 7B", [44.75, 1.76], [41.25, 1.74], [57.50, 1.75], [55.50, 1.76])
add_model("yi 6b", "Yi 6B", [50.75, 1.76], [49.00, 1.76], [59.62, 1.74], [62.25, 1.71])
add_model("llama1-7b", "LLaMA1 7B", [47.0625, 1.76], [42.5625, 1.75], [47.625, 1.77], [57.4375, 1.74])
add_model("llama2-13b", "LLaMA2 13B", [61.62, 1.72], [58.50, 1.74], [65.06, 1.68], [70.06, 1.61])
add_model("mistral-7b", "Mistral 7B", [56.8125, 1.75], [55.125, 1.75], [71.125, 1.60], [75.3125, 1.53])
add_model("mixtral-7b", "Mixtral 8x7B", [70.19, 1.61], [70.12, 1.62], [72.75, 1.57], [77.75, 1.46])


color_palette = ["#FF6150", "#DEE0E6", "#1AC0C6", "#FFA822", "#134E6F", "#60A87B"]
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.set_ylim([35, 80])
# ax.set_xlim([-1, 6])
ax.set_ylabel('Alpaca Eval Winrate % (Measured by GPT-4)', fontsize=14)
ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)

# ax.set_xlabel('Method')
ax.set_title('Fine-tuning Base LLMs on ShareGPT Prompts (GPT-4 version)', fontsize=16)


xticks = []
xticklabels = []
BAR_WIDTH = 0.5
MODEL_SPACING = 0.7
for idx, i in enumerate(models):
    if len(xticks)==0:
        xticks.append(2*BAR_WIDTH)
    else:
        xticks.append(xticks[-1] + 3*BAR_WIDTH + MODEL_SPACING)
    ax.set_xticks(xticks)
    xticklabels.append(winrate_method[f"{i} display name"])
    ax.set_xticklabels(xticklabels, fontsize=14)
    
    pos = xticks[-1] - BAR_WIDTH
    n1, n2, n3, n4 = winrate_method[f"{i} chatgpt sft 100p"], winrate_method[f"{i} chatgpt sft 10p"], winrate_method[f"{i} chatgpt sft 10p dpo 90p"], winrate_method[f"{i} gpt4 sft 100p"]
    ax.bar(pos, n1[0], yerr=n1[1], width=BAR_WIDTH, color=color_palette[idx], edgecolor='black', capsize=5)
    ax.bar(pos+BAR_WIDTH, n2[0], width=BAR_WIDTH, color=color_palette[idx], edgecolor='black', capsize=5)
    ax.bar(pos+BAR_WIDTH, n3[0]-n2[0], yerr=n3[1], width=BAR_WIDTH, color=color_palette[idx], edgecolor='black', hatch='o', capsize=5, bottom=n2[0])
    ax.bar(pos+2*BAR_WIDTH, n4[0], yerr=n4[1], width=BAR_WIDTH, color=color_palette[idx], edgecolor='black', hatch='//\\\\', capsize=5)


handle1 = mpatches.Patch(facecolor='#FFFFFF', edgecolor='#000000', label='SFT w/ GPT-3.5 (default)')
handle2 = mpatches.Patch(facecolor='#FFFFFF', edgecolor='#000000', hatch='o', label='+ RLAIF w/ GPT-4 preferences')
handle3 = mpatches.Patch(facecolor='#FFFFFF', edgecolor='#000000', hatch='//\\\\', label='SFT w/ GPT-4')

ax.legend(handles=[handle1, handle2, handle3], handlelength=3, handleheight=3)
plt.savefig('fig2.png', dpi=300, bbox_inches='tight')
plt.show()
