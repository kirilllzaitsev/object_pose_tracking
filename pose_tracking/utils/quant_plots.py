import matplotlib.pyplot as plt
import numpy as np


def plot_pred_confidence(scores_per_frame, title="Confidence Scores of SAM2 on Cube Dataset"):
    plt.figure(figsize=(10, 5))
    plt.plot(np.array(scores_per_frame)[:, :2])

    plt.title(title)
    plt.legend(["top-1", "top-2"])
    plt.xlabel("Frame")
    plt.ylabel("Confidence")


def plot_auc_res(auc: dict):
    # auc contains thresholds, recall, auc keys
    thresholds = auc["thresholds"]
    recall = auc["recall"]
    auc_val = auc["auc"]
    plt.plot(thresholds, recall)
    plt.title(f"Recall curve, AUC: {auc_val:.2f}")
    plt.xlabel("Threshold, cm")
    plt.ylabel("Recall")


def plot_history(history, title="History", xlabel="Epoch"):
    keys = list(history.keys())
    num_rows = len(history)
    num_cols = len(history[keys[0]])
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    for ridx, (key, values) in enumerate(history.items()):
        ax = axs[ridx] if num_rows > 1 else axs
        for cidx, (k, v) in enumerate(values.items()):
            ax[cidx].plot(v, label=k)
            ax[cidx].set_title(k)
            ax[cidx].set_xlabel(xlabel)
            ax[cidx].set_ylabel(k)

    plt.legend()
    plt.suptitle(title)
    return fig
