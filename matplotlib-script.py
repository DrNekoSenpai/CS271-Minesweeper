import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import shutil

# config
ROOT_DIRS = ["metrics/s4-m5", "metrics/s5-m5", "metrics/s5-m8", "metrics/s5-m10"]
MODEL_ORDER = ["1-random", "2-heuristic", "3-bayesian", "4-dqn"] # text output order
MODEL_X_ORDER = ["Random", "Heuristic", "Bayesian", "DQN"] # plot output order
MODEL_NAMES = {
    "1-random": "Random",
    "2-heuristic": "Heuristic",
    "3-bayesian": "Bayesian",
    "4-dqn": "DQN",
}

# mapping for better legend
ENV_NAMES = {
    "metrics/s4-m5": "Size 4, Mines 5",
    "metrics/s5-m5": "Size 5, Mines 5",
    "metrics/s5-m8": "Size 5, Mines 8",
    "metrics/s5-m10": "Size 5, Mines 10"
}

TARGET_METRICS = {
    ("Total Reward", "mean"),
    ("Total Reward", "stdev"),
    ("Number of Moves", "mean"),
    ("Number of Moves", "stdev"),
    ("Win / Loss", "Win rate"),
    ("Timing", "Avg per ep"),
}

# extraction regex and parsing
VALUE_RE = re.compile(r"([\w\s/]+)=\s*([-+]?[0-9]*\.?[0-9]+)")

def parse_metrics(path):
    results = {}
    with open(path, "r") as f:
        text = f.read()

    current_section = None
    for line in text.split("\n"):
        line = line.strip()
        if line.endswith(":") and "=" not in line:
            current_section = line.replace(":", "").strip()
            continue
        m = VALUE_RE.search(line)
        if m and current_section:
            key = m.group(1).strip()
            val = float(m.group(2))
            results[(current_section, key)] = val
    return results

# output dir cleanup
PLOT_DIR = "plots"
if os.path.exists(PLOT_DIR):
    shutil.rmtree(PLOT_DIR)
os.makedirs(PLOT_DIR, exist_ok=True)

# df aggregation
rows = []
for env in ROOT_DIRS:
    for model_file in MODEL_ORDER:
        path = os.path.join(env, model_file + ".txt")
        metrics = parse_metrics(path)

        row = {
            "environment": env,
            "model": MODEL_NAMES[model_file],
        }

        for (section, key) in TARGET_METRICS:
            row[f"{section} - {key}"] = metrics.get((section, key), None)

        rows.append(row)

df = pd.DataFrame(rows)
df["model"] = pd.Categorical(df["model"], categories=MODEL_X_ORDER, ordered=True)
df = df.sort_values(by=["environment", "model"])
print(df)

# plot
def plot_metric(df, metric_name, ylabel, output_name):
    plt.figure(figsize=(10, 6))

    for env in ROOT_DIRS:
        sub = df[df["environment"] == env]
        plt.plot(
            sub["model"],
            sub[metric_name],
            marker="o",
            label=ENV_NAMES[env]  # human-readable legend
        )

    plt.title(metric_name)
    plt.xlabel("Model")
    plt.ylabel(ylabel)
    plt.legend(title="Environment")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, output_name))
    plt.close()

# plot gen
plot_metric(df, "Total Reward - mean", "Mean Reward", "reward_mean.png")
plot_metric(df, "Total Reward - stdev", "Reward Stdev", "reward_stdev.png")
plot_metric(df, "Number of Moves - mean", "Mean Moves", "moves_mean.png")
plot_metric(df, "Number of Moves - stdev", "Moves Stdev", "moves_stdev.png")
plot_metric(df, "Win / Loss - Win rate", "Win Rate", "winrate.png")
plot_metric(df, "Timing - Avg per ep", "Avg Time per Episode (sec)", "avg_time.png")

print("done!")
