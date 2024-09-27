import matplotlib.pyplot as plt
import numpy as np


def plot_pred_confidence(scores_per_frame, title="Confidence Scores of SAM2 on Cube Dataset"):
    plt.figure(figsize=(10, 5))
    plt.plot(np.array(scores_per_frame)[:, :2])

    plt.title(title)
    plt.legend(["top-1", "top-2"])
    plt.xlabel("Frame")
    plt.ylabel("Confidence")
