import argparse
import functools
import glob
import json
import os
import re
import typing as t
from pathlib import Path

import comet_ml
import yaml
from comet_ml.api import API
from pose_tracking.config import ARTIFACTS_DIR, COMET_WORKSPACE, PROJ_DIR, PROJ_NAME
from pose_tracking.utils.args_parsing import load_args_from_file
from pose_tracking.utils.misc import wrap_with_futures
from tqdm import tqdm


def log_tags(args: argparse.Namespace, exp: comet_ml.Experiment, args_to_group_map=None) -> None:
    """Logs tags to the experiment."""

    extra_tags = []
    if os.path.exists("/home/kirillz"):
        extra_tags.append("e_remote")
    elif os.path.exists("/cluster"):
        extra_tags.append("e_eu")
    else:
        extra_tags.append("e_local")
    for k, v in vars(args).items():
        if k in ["use_cuda", "use_test_set", "use_rnn"] or any(
            [x in k for x in ["save_", "use_es", "do_log", "print", "ignore_file_args_with_provided", "focal_loss"]]
        ):
            continue
        tag_prefix = get_tag_pref(k, args_to_group_map)
        p = r"^(use_|do_)|.*(_use_|_do_)"
        if re.match(p, k) and v:
            extra_tags.append(f'{tag_prefix}{re.sub(p, "", k)}')
        p = r"^(disable_|no_)"
        if re.match(p, k) and v:
            extra_tags.append(f"{tag_prefix}no_{re.sub(p, '', k)}")
    for k in ["ds_name", "model_name", "ds_alias", "encoder_name", "ds_alias"]:
        tag_prefix = get_tag_pref(k, args_to_group_map)
        v = getattr(args, k)
        if not v:
            continue
        if isinstance(v, list):
            v = "_".join(v)
        extra_tags.append(f"{tag_prefix}{v}")

    tags_to_log = extra_tags
    if len(args.exp_tags) > 0:
        # et=exp tags
        tag_prefix = "et_"
        tags_to_log += [f"{tag_prefix}{x}" for x in args.exp_tags if x]
    exp.add_tags(tags_to_log)


def get_tag_pref(k, args_to_group_map=None):
    group_alias = args_to_group_map.get(k) if args_to_group_map is not None else None
    tag_prefix = f"{group_alias}_" if group_alias is not None else ""
    return tag_prefix


def get_latest_ckpt_epoch(
    exp_name: str,
    model_name_regex: str = r"model_(\d+)\.pt*",
    project_name: str = PROJ_NAME,
) -> int:
    """Infers the latest ckpt epoch from the experiment's assets."""

    api = API(api_key=os.environ["COMET_API_KEY"])
    exp_api = api.get(f"{COMET_WORKSPACE}/{project_name}/{exp_name}")
    ckpt_epochs = [
        int(re.match(model_name_regex, x["fileName"]).group(1))
        for x in exp_api.get_asset_list(asset_type="all")
        if re.match(model_name_regex, x["fileName"])
    ]
    return max(ckpt_epochs)


def load_artifacts_from_comet(
    exp_name,
    args_file_path: str = None,
    model_ckpt_path: str = None,
    local_artifacts_dir: str = ARTIFACTS_DIR,
    artifact_suffix: str = "best",
    args_filename: str = "args",
    session_artifact_name: t.Optional[str] = "session",
    api: t.Optional[API] = None,
    epoch: t.Optional[int] = None,
    use_epoch: bool = False,
    do_load_model: bool = True,
    do_load_session: bool = False,
    do_force_download=False,
    do_raise_if_missing=False,
):
    """Downloads artifacts from comet.ml if they don't exist locally and returns the paths to them.
    Args:
        exp_name: The name of the Comet experiment.
        args_file_path: The local path to the args file.
        model_ckpt_path: The local path to the model ckpt.
        local_artifacts_dir: The directory to save the artifacts to.
        model_artifact_name: The path to the model artifact in the experiment Assets.
        args_filename: The regex to match the args file name.
        session_artifact_name: The name of the session artifact.
        session_ckpt_path: The path to the session ckpt.
        project_name: The name of the Comet project.
        api: The comet.ml API object (takes time to initialize, so it's better to pass it as an argument if done multiple times).
        epoch: The epoch of the model ckpt to download. It will be inferred if not provided.
    Returns:
        A dictionary containing the paths to the artifacts.
    """

    if isinstance(exp_name, list):
        load_artifacts_from_comet_fn = functools.partial(
            load_artifacts_from_comet,
            args_file_path=args_file_path,
            model_ckpt_path=model_ckpt_path,
            local_artifacts_dir=local_artifacts_dir,
            artifact_suffix=artifact_suffix,
            args_filename=args_filename,
            session_artifact_name=session_artifact_name,
            api=api,
            epoch=epoch,
            use_epoch=use_epoch,
            do_load_model=do_load_model,
            do_load_session=do_load_session,
        )
        load_ress = wrap_with_futures(exp_name, load_artifacts_from_comet_fn)
        return load_ress

    artifact_suffix = artifact_suffix.replace("model_", "")
    exp_dir = f"{local_artifacts_dir}/{exp_name}"
    args_file_path = args_file_path or f"{exp_dir}/{args_filename}.yaml"
    model_artifact_name = f"model_{artifact_suffix}"
    model_ckpt_path = model_ckpt_path or f"{exp_dir}/{model_artifact_name}.pth"
    alt_artifact_suffix = "last" if artifact_suffix == "best" else "best"
    alt_model_artifact_name = f"model_{alt_artifact_suffix}"
    alt_model_ckpt_path = f"{exp_dir}/{alt_model_artifact_name}.pth"
    # session
    session_ckpt_path = f"{exp_dir}/{session_artifact_name}.pth"
    session_not_exist = not os.path.exists(session_ckpt_path)
    if session_not_exist:
        session_ckpt_path = f"{exp_dir}/{session_artifact_name}_{artifact_suffix}.pth"  # v2 includes suffix
        session_not_exist = not os.path.exists(session_ckpt_path)

    weights_not_exist = do_load_model and do_force_download
    if do_load_model and not os.path.exists(model_ckpt_path):
        weights_not_exist = True
    args_not_exist = not os.path.exists(args_file_path)
    model_step = None
    session_step = None

    need_load_model = weights_not_exist and do_load_model
    need_load_session = session_not_exist and do_load_session

    os.makedirs(exp_dir, exist_ok=True)

    if any([args_not_exist, need_load_model, need_load_session]):
        os.makedirs(exp_dir, exist_ok=True)
        if api is None:
            api = API(api_key=os.environ["COMET_API_KEY"])
        exp_api = api.get(
            workspace=COMET_WORKSPACE,
            project_name=PROJ_NAME,
            experiment=exp_name,
        )
        print(f"loading artifacts for {exp_name=}")
        assert exp_api is not None, f"Experiment {exp_name} not found"

        # args
        if args_not_exist:
            try:
                asset_id = [
                    x for x in exp_api.get_asset_list(asset_type="all") if f"{args_filename}.yaml" in x["fileName"]
                ][0]["assetId"]
                api.download_experiment_asset(
                    exp_api.id,
                    asset_id,
                    args_file_path,
                )
            except IndexError:
                print(f"No args found with name {args_filename}")
                args_file_path = None

        if need_load_model or need_load_session:
            if use_epoch and epoch is None:
                epoch = get_latest_ckpt_epoch(exp_name)
                model_artifact_name = f"{model_artifact_name}_{epoch}"
                alt_model_artifact_name = f"{alt_model_artifact_name}_{epoch}"
            logged_models = exp_api.get_model_asset_list("ckpt")
            sorted_assets = sorted(logged_models, key=lambda x: x["step"], reverse=True)
            model_assets = [x for x in sorted_assets if model_artifact_name in x["fileName"]]
            if len(model_assets) == 0:
                print(
                    f"WARN: No model found with name {model_artifact_name}. Trying alternative: {alt_model_artifact_name}"
                )
                artifact_suffix = alt_artifact_suffix
                model_ckpt_path = alt_model_ckpt_path
                model_artifact_name = alt_model_artifact_name
                model_assets = [x for x in sorted_assets if model_artifact_name in x["fileName"]]
                assert (
                    len(model_assets) > 0
                ), f"No model found with name {alt_model_artifact_name}"  # no artifacts logged
            if need_load_model:
                model_asset = model_assets[0]
                model_step = model_asset["step"]
                print(f"Loading the model from step {model_step}")
                load_asset(exp_api, model_asset["assetId"], model_ckpt_path)
                weights_not_exist = False
            if need_load_session:
                try:
                    filenames = [x["fileName"] for x in sorted_assets]
                    if "session" in filenames:
                        session_artifact_name = "session"
                    else:
                        session_artifact_name = f"session_{artifact_suffix}"  # v2
                    session_ckpt_path = f"{exp_dir}/{session_artifact_name}.pth"
                    session_asset = [x for x in sorted_assets if session_artifact_name in x["fileName"]][0]
                    session_step = session_asset["step"]
                    load_asset(exp_api, session_asset["assetId"], session_ckpt_path)
                    session_not_exist = False
                except IndexError as e:
                    print(f"No session found with name {session_artifact_name}")
                    session_ckpt_path = None
                    if do_raise_if_missing:
                        raise e
    args = load_args_from_file(args_file_path)
    results = {
        "args_path": args_file_path,
        "args": args,
    }

    if do_load_model:
        results["checkpoint_path"] = model_ckpt_path
        if model_step is not None:
            results["model_step"] = model_step
        assert os.path.exists(model_ckpt_path), f"Model ckpt not found at {model_ckpt_path}"
        args.ckpt_path = model_ckpt_path
    assert os.path.exists(args_file_path), f"Args file not found at {args_file_path}"

    if do_load_session and not session_not_exist:
        results["session_checkpoint_path"] = session_ckpt_path
        if session_step is not None:
            results["session_step"] = session_step
    else:
        results["session_checkpoint_path"] = None
    return results


def load_asset(exp_api, assetId, save_path):
    asset_response = exp_api.get_asset(assetId, "response")
    with open(save_path, "wb") as f:
        for chunk in asset_response.iter_content(chunk_size=1024**2):
            f.write(chunk)
    return save_path


def load_metrics_from_comet(
    exp_name: str,
    save_name: str = "metrics.json",
    api: t.Optional[API] = None,
):
    if api is None:
        api = API(api_key=os.environ["COMET_API_KEY"])
    exp_api = api.get(
        workspace=COMET_WORKSPACE,
        project_name=PROJ_NAME,
        experiment=exp_name,
    )
    metrics = exp_api.get_metrics()
    save_path = f"{ARTIFACTS_DIR}/{exp_name}/{save_name}"
    with open(save_path, "w") as f:
        json.dump(metrics, f)
    return save_path


def log_args(exp: comet_ml.Experiment, args: argparse.Namespace, save_path: str) -> None:
    """Logs the args to the experiment and saves them to a file."""
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    args = {k: v for k, v in sorted(args.items())}
    with open(save_path, "w") as f:
        yaml.dump(
            args,
            f,
            default_flow_style=False,
        )
    exp.log_asset(save_path)


def log_params_to_exp(experiment: comet_ml.Experiment, params: dict, prefix: str = "") -> None:
    """Logs the given parameter map to the experiment, putting the given prefix before each key."""
    experiment.log_parameters({f"{prefix}/{str(k)}": v for k, v in params.items()})
    if "SLURM_JOBID" in os.environ:
        experiment.log_parameter("slurm_job_id", os.environ["SLURM_JOBID"])


def log_ckpt_to_exp(experiment: comet_ml.Experiment, ckpt_path: str, model_name: str) -> None:
    experiment.log_model(model_name, ckpt_path, overwrite=True)


def create_tracking_exp(
    args: argparse.Namespace,
    exp_disabled: bool = True,
    force_disabled: bool = False,
    project_name: str = PROJ_NAME,
) -> comet_ml.Experiment:
    """Creates a Comet.ml experiment if args.resume_exp is False, otherwise resumes the experiment with the given name. Logs the package code."""

    if "COMET_GIT_DIRECTORY" not in os.environ:
        os.environ["COMET_GIT_DIRECTORY"] = str(PROJ_DIR)
    disabled = getattr(args, "exp_disabled", exp_disabled) or force_disabled
    api_key = get_comet_api_key()
    exp_init_args = dict(
        api_key=api_key,
        auto_output_logging="native",
        auto_metric_logging=True,
        auto_param_logging=False,
        log_env_details=True,
        log_env_host=False,
        log_env_gpu=True,
        log_env_cpu=True,
        log_code=False,
        parse_args=False,
        display_summary_level=0,
        disabled=disabled,
    )
    if getattr(args, "resume_exp", False):
        from comet_ml.api import API

        api = API(api_key=api_key)
        exp_api = api.get(f"{COMET_WORKSPACE}/{project_name}/{args.exp_name}")
        experiment = comet_ml.ExistingExperiment(**exp_init_args, experiment_key=exp_api.id)
    else:
        experiment = comet_ml.Experiment(**exp_init_args, project_name=project_name)

    if not experiment.name and not args.exp_disabled:
        print("WARN: Experiment failed to init. Doing an offline experiment instead")
        experiment = comet_ml.OfflineExperiment(**exp_init_args, project_name=project_name)

    return experiment


def get_comet_api_key():
    return os.environ.get("COMET_API_KEY", os.environ["COMET_API_TOKEN"])


def log_pkg_code(exp: comet_ml.Experiment, overwrite: bool = False) -> None:
    """Recursively logs the package code to the experiment."""
    import pose_tracking

    import memotr.models
    import trackformer.models.detr

    pkg_dir_pt = Path(pose_tracking.__file__).parent
    pkg_dir_memotr = Path(memotr.models.__file__).parent
    current_dir = os.getcwd()
    for pkg_dir in [pkg_dir_pt, pkg_dir_memotr]:
        os.chdir(pkg_dir)
        for code_file in glob.glob(str(pkg_dir / "**/*.py"), recursive=True):
            code_file = Path(code_file)
            exp.log_code(file_name=code_file.relative_to(pkg_dir), overwrite=overwrite)
    tf_path = Path(trackformer.models.detr.__file__).parent
    os.chdir(tf_path)
    for code_file in ["detr.py", "__init__.py"]:
        exp.log_code(file_name=code_file, overwrite=overwrite)
    os.chdir(current_dir)


def load_artifacts_from_comet_v2(exp_name, save_path_model=None, save_path_args=None, comet_api=None):
    if comet_api is None:
        comet_api = API(api_key=get_comet_api_key())

    if save_path_model is None:
        save_path_model = f"{ARTIFACTS_DIR}/{exp_name}/model.pt"
    if save_path_args is None:
        save_path_args = f"{ARTIFACTS_DIR}/{exp_name}/args.yaml"
    save_dir = os.path.dirname(save_path_model)
    model_ckpt_regex = r".*model_(\d+).pt"
    model_paths = [x for x in glob.glob(f"{save_dir}/*.pt") if re.match(model_ckpt_regex, x)]
    model_paths = sorted(model_paths, key=lambda x: int(re.match(model_ckpt_regex, x).groups()[0]))
    if len(model_paths) > 0:
        model_path = model_paths[-1]
        # rename to save_path_model
        os.rename(model_path, save_path_model)

    do_load_model = not os.path.exists(save_path_model)
    do_load_args = not os.path.exists(save_path_args)
    save_path_dir = os.path.dirname(save_path_model)
    os.makedirs(save_path_dir, exist_ok=True)
    if do_load_model or do_load_args:
        experiment = comet_api.get(
            COMET_WORKSPACE,
            project_name="pose_tracking",
            experiment=exp_name,
        )
    if do_load_model:
        model_artifacts = [
            {"id": x["assetId"], "name": f'{x["dir"]}/{x["fileName"]}'}
            for x in experiment.get_asset_list()
            if re.match(model_ckpt_regex, x["fileName"])
        ]
        model_artifacts = sorted(
            model_artifacts,
            key=lambda x: int(re.match(model_ckpt_regex, x["name"]).group(1)),
        )
        artifact_id_model = model_artifacts[-1]["id"]
        print(f"Using {model_artifacts[-1]=} for the model")
        model_binary = experiment.get_asset(artifact_id_model)
        # Save the asset to a file
        with open(save_path_model, "wb") as f:
            f.write(model_binary)
    if do_load_args:
        args_p = r".*args.yaml"
        args_artifacts = [
            {"id": x["assetId"], "name": f'{x["dir"]}/{x["fileName"]}'}
            for x in experiment.get_asset_list()
            if re.match(args_p, x["fileName"])
        ]
        artifact_id_args = args_artifacts[-1]["id"]
        print(f"Using {artifact_id_args=} for the args")
        args_binary = experiment.get_asset(artifact_id_args)
        with open(save_path_args, "wb") as f:
            f.write(args_binary)
    args_yaml = yaml.safe_load(open(save_path_args))
    if "args" in args_yaml:
        args_yaml = args_yaml["args"]
    artifacts = {
        "args": argparse.Namespace(**args_yaml),
        "model_path": save_path_model,
    }
    return artifacts
