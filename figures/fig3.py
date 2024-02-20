import matplotlib.pyplot as plt
import numpy as np
color_palette = ["#FF6150", "#134E6F", "#1AC0C6", "#FFA822", "#DEE0E6", "#091A29"]

# Data preparation
x_values = [0, 5, 10, 25, 50, 100]  # Numerical values representing the labels including 'Base' as 0
gpt_evals_adjusted = [0.375, 40.1875, 42.5625, 42.9375, 47.0625, 47.0625]
gpt_evals_vars = [0.2162352095110562, 1.7333473168521505, 1.7480754995501935, 1.7500209524045185, 1.7624920112726044, 1.7624920112726044]
claude_evals_adjusted = [6.758448060075094, 53.125, 55.00000000000001, 55.0625, 55.50688360450563, 57.56250000000001]
claude_evals_vars = [0.8886419095405922, 1.76541463395635, 1.757782622990428, 1.7586710243540082, 1.7558664647524238, 1.7451644595642437]

# Create the plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_ylim([-3, 65])

# Plotting both lines with specified colors and increased line width
ax.plot(x_values, gpt_evals_adjusted, label='Winrates evaluated using GPT-4', marker='o', color=color_palette[0], linewidth=4, markersize=10)
ax.plot(x_values, claude_evals_adjusted, label='Winrates evaluated using Claude', marker='s', color=color_palette[1], linewidth=4, markersize=10)

# Add labels, title, and custom x-axis tick labels
ax.set_xlabel('Number of Training Examples for SFT (% of Total Examples)', fontsize=16)
ax.set_ylabel('Alpaca Eval Winrate \n(Measured by GPT-4)', fontsize=16)
ax.set_title('Data Scaling Curve for SFT on ShareGPT (GPT3.5 completions)', fontsize=16)
ax.set_xticks(x_values)
ax.set_xticklabels(['0', '5', '10', '25', '50', '100'], fontsize=12)
ax.yaxis.set_tick_params(labelsize=14)
ax.legend(fontsize=16)

# Add dotted grid lines
ax.grid(True, linestyle='--', which='both', color='grey', alpha=0.7)
ax.axvline(x=10, color='black', linestyle='--', linewidth=4, label='10% Mark')

# Adjust layout
fig.tight_layout()

# Display the plot
plt.savefig('fig3.png', dpi=300, bbox_inches='tight')
plt.show()
