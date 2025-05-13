import json
import matplotlib.pyplot as plt
import os

# Load the JSON file
folder = "outputs_context_model_grpo_with_prompt_and_sigmoid"
with open(f"{folder}/checkpoint-200/trainer_state.json", "r") as f:
    data = json.load(f)

log_history = data["log_history"][1:]

# Use 'step' as the x-axis for all plots
steps = [entry["step"] for entry in log_history]

# Get all metric keys from the first log entry (excluding 'step')
keys = [key for key in log_history[0].keys() if key != "step"]

# Create a plot for each metric
if not os.path.exists(f'plots/{folder}'):
    os.mkdir(f'plots/{folder}')
for key in keys:
    values = [entry[key] for entry in log_history]
    plt.figure()
    plt.plot(steps, values, marker="o")
    plt.xlabel("Step")
    plt.ylabel(key)
    plt.title(f"{key} vs Step")
    plt.grid(True)
    plt.savefig(f'plots/{folder}/{key.replace('/', '_')}.png')
