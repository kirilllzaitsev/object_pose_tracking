import copy
import os
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from pose_tracking.config import PROJ_DIR, WORKSPACE_DIR
from pose_tracking.metrics import calc_auc
from pose_tracking.utils.common import create_dir


def get_metrics_per_obj(metrics_all):
    # get metrics per object from all scenes it appears in
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


def calc_auc_bt(errors):
    # auc that works for bundle* and foundation_pose
    errors = np.sort(np.array(errors))
    n = len(errors)
    prec = np.arange(1, n + 1) / float(n)
    errors = errors.reshape(-1)
    prec = prec.reshape(-1)
    index = np.where(errors < 0.1)[0]
    errors = errors[index]
    prec = prec[index]

    mrec = [0, *list(errors), 0.1]
    mpre = [0, *list(prec), prec[-1]]

    for i in range(1, len(mpre)):
        mpre[i] = max(mpre[i], mpre[i - 1])
    mpre = np.array(mpre)
    mrec = np.array(mrec)
    i = np.where(mrec[1:] != mrec[0 : len(mrec) - 1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i]) * 10
    return {
        "thresholds": mrec,
        "recall": mpre,
        "auc": ap,
    }


def metrics_all_per_obj_to_df(metrics_all_per_obj, id_to_name):
    df = pd.DataFrame(metrics_all_per_obj)
    df.columns = [id_to_name[int(i) if str.isnumeric(i) else i] for i in df.columns]
    df["avg"] = df.mean(axis=1)
    df = df.round(3)
    return df


def save_df(df, path):
    create_dir(path)
    df.to_csv(path, index=True)


def load_df(path):
    return pd.read_csv(path, index_col=0)


def get_preds_path_benchmark(model_name, obj_name, ds_name=None):
    if model_name == "bundletrack":
        preds_path = f"{WORKSPACE_DIR}/related_work/BundleTrack/results/ycbineoat/{obj_name}/poses"
    elif model_name == "bundlesdf":
        preds_path = f"{WORKSPACE_DIR}/related_work/BundleSDF/data/{obj_name}_out/ob_in_cam"
    elif model_name == "se3tracknet":
        assert ds_name is not None, "ds_name must be provided for se3tracknet"
        preds_path = f"{WORKSPACE_DIR}/related_work/iros20-6d-pose-tracking/results/{ds_name}/model_free_tracking_model_{obj_name}/poses"
    elif model_name == "foundation_pose":
        preds_path = f"{WORKSPACE_DIR}/related_work/FoundationPoseRSL/demo_data/{obj_name}/ob_in_cam"
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return preds_path
