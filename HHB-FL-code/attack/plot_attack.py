import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV file
file_path = 'E:\\桌面\\攻击\\attack_success_rates.csv'
results_df = pd.read_csv(file_path)

# Define attack types and methods
attack_types = ['Inference Attack', 'Data Leakage', 'Replay Attack']
methods = ['No Protection', 'Differential Privacy', 'Homomorphic Encryption', 'HHB-FL']

# Extract success rates and convert to percentage
success_rates = [
    [results_df[results_df['Unnamed: 0'] == attack].iloc[0][method] * 100 for method in methods]
    for attack in attack_types
]

# Set up the plot
x = np.arange(len(methods))
width = 0.2
fig, ax = plt.subplots(figsize=(12, 6))

colors = ['#4E79A7', '#F28E2B', '#E15759']  # Professional color palette

# Draw bars
for i, (attack, color) in enumerate(zip(attack_types, colors)):
    bars = ax.bar(x + i * width - width, success_rates[i], width, label=attack, color=color)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontname='Times New Roman')

# Set labels
ax.set_xlabel('Protection Methods', fontname='Times New Roman', fontsize=18)
ax.set_ylabel('Success Rate (%)', fontname='Times New Roman', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontname='Times New Roman', fontsize=16)

# Configure grid: emulate dashed lines
ax.yaxis.grid(True, linestyle='--', linewidth=1.5, color='lightgray', dashes=(5, 5))
ax.xaxis.grid(True, linestyle='--', linewidth=1.5, color='lightgray', dashes=(5, 5))
ax.set_axisbelow(True)

# Update legend: larger font, move to bottom center outside plot
legend = ax.legend(loc='upper right', fontsize=20, frameon=False, prop={'family': 'Times New Roman'})


# Tight layout with space for legend
fig.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at bottom for legend

# Save figure
plt.savefig('Attack_Success_Rates_DashedGrid.eps', dpi=300, bbox_inches='tight')
plt.show()
