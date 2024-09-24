import copy
from collections import defaultdict
import os

import cv2
import numpy as np
import pandas as pd
from pose_tracking.metrics import calc_auc


def get_metrics_per_obj(metrics_all):
    metrics_all_per_obj = defaultdict(lambda: defaultdict(list))
    for scene_id, scene_metrics in metrics_all.items():
        for obj_id, obj in scene_metrics.items():
            for metric, values in obj.items():
                metrics_all_per_obj[obj_id][metric].extend(values)

    return metrics_all_per_obj


def agg_metrics_per_obj(metrics_all_per_obj):
    metrics_all_per_obj = copy.deepcopy(metrics_all_per_obj)
    for obj_id, obj_metrics in metrics_all_per_obj.items():
        for metric, values in obj_metrics.items():
            metrics_all_per_obj[obj_id][metric] = np.mean(values)
    return metrics_all_per_obj


def calc_aucs_from_metrics_per_obj(metrics_all_per_obj, *args, metrics_for_auc=["add", "adds", "t_err"], **kwargs):
    aucs_per_obj = {}
    for metric_name_for_auc in metrics_for_auc:
        auc_per_obj = calc_auc_from_metrics_per_obj(
            metrics_all_per_obj, *args, **kwargs, metric_name_for_auc=metric_name_for_auc
        )
        aucs_per_obj[metric_name_for_auc] = auc_per_obj
    return aucs_per_obj


def calc_auc_from_metrics_per_obj(metrics_all_per_obj, metric_name_for_auc, max_val_cm=10, step_cm=1):
    metrics_all_per_obj = copy.deepcopy(metrics_all_per_obj)
    auc_per_obj = defaultdict(dict)
    for obj_id, obj_metrics in metrics_all_per_obj.items():
        values = np.array(obj_metrics[metric_name_for_auc])
        values_cm = values / 10
        auc_res = calc_auc(values_cm, max_val=max_val_cm, step=step_cm)
        auc = auc_res["auc"]
        thresholds = auc_res["thresholds"]
        recall = auc_res["recall"]
        auc_per_obj[obj_id] = {"thresholds": thresholds, "recall": recall, "auc": auc}

    return auc_per_obj


def metrics_all_per_obj_to_df(metrics_all_per_obj, id_to_name):
    df = pd.DataFrame(metrics_all_per_obj)
    df.columns = [id_to_name[int(i) if str.isnumeric(i) else i] for i in df.columns]
    df["avg"] = df.mean(axis=1)
    df = df.round(3)
    return df


def save_df(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=True)


def load_df(path):
    return pd.read_csv(path, index_col=0)


def save_video(images, save_path, frame_height=480, frame_width=640, fps=1):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))

    for image in images:
        cv2.imshow("Video", image)
        video_writer.write(image)
        if cv2.waitKey(1000) & 0xFF == ord("q"):
            break

    video_writer.release()
    cv2.destroyAllWindows()

    print(f"Video saved as {save_path}")
