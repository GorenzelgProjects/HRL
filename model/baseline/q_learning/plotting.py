import json
import numpy as np
import matplotlib.pyplot as plt

# ---- Load JSON ----
path = ...
with open(path, "r") as f:
    data = json.load(f)

# ---- Extract data ----
episodes = [ep["episode"] for ep in data["episodes"]]
steps = [ep["steps"] for ep in data["episodes"]]

episodes = np.array(episodes)
steps = np.array(steps)

# ---- Plot ----
plt.figure(figsize=(8, 4))

plt.plot(episodes, steps, alpha=0.6, linewidth=1, color="green")

moving_avg = np.convolve(steps, np.ones(10) / 10, mode="same")
plt.plot(
    episodes,
    moving_avg,
    color="darkgreen",
    linewidth=2,
    label="Moving Average (10)"
)

plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Steps per Episode")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()