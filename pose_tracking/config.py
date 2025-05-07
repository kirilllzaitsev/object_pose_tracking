import functools
import os
import sys
import traceback
from pathlib import Path
from types import TracebackType

from dotenv import load_dotenv

load_dotenv()

IS_REMOTE = os.path.exists("/home/kirillz")
IS_CLUSTER = os.path.exists("/cluster")
IS_LOCAL = not (IS_REMOTE or IS_CLUSTER)
PROJ_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJ_DIR if (IS_REMOTE or IS_CLUSTER) else PROJ_DIR.parent

DATA_DIR = Path(os.environ["DATA_DIR"]) if "DATA_DIR" in os.environ else PROJ_DIR / "data"
RELATED_DIR = WORKSPACE_DIR / "related_work"
TF_DIR = RELATED_DIR / "obj_det/trackformer"
MEMOTR_DIR = RELATED_DIR / "obj_det/MeMOTR/memotr"
if os.path.exists("/home/arjun"):
    MEMOTR_DIR = Path("/home/arjun/Desktop/repo/nvidia-dexterous-repos/pose_tracking_inference")

ARTIFACTS_DIR = PROJ_DIR / "artifacts"
BENCHMARK_DIR = PROJ_DIR.parent / "benchmark"
YCBINEOAT_SCENE_DIR = DATA_DIR / "ycbineoat"
YCBV_SCENE_DIR = DATA_DIR / "ycbv"
YCBV_SYNT_SCENE_DIR = DATA_DIR / "ycbv_synt"
NOCS_SCENE_DIR = DATA_DIR / "nocs"

YCB_MESHES_DIR = DATA_DIR / "ycb/models"
HO3D_ROOT = DATA_DIR / "ho3d_v3"

PROJ_NAME = "pose-tracking"
COMET_WORKSPACE = "kirilllzaitsev"


def prepare_logger(logpath=None, level="INFO"):
    from loguru import logger

    # Define handlers
    common_cfg = {
        "level": level,
        "colorize": True,
    }
    stdout_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {level} | {file}:{line} | <level>{message}</level>"
    file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}"
    handlers = [
        {
            "sink": sys.stdout,
            **common_cfg,
            "format": stdout_format,
        }
    ]

    if logpath:
        handlers.append(
            {
                "sink": logpath,
                **common_cfg,
                "format": file_format,
            }
        )

    try:
        from tqdm import tqdm

        def tqdm_sink(message):
            tqdm.write(message, end="")

        handler_tqdm = {
            "sink": tqdm_sink,
            **common_cfg,
            "format": stdout_format,
        }
        handlers[0] = handler_tqdm
    except ModuleNotFoundError:
        logger.debug("tqdm not found. Skipping tqdm integration.")

    # Configure logger with the defined handlers
    logger.remove()
    logger.configure(handlers=handlers)
    sys.excepthook = functools.partial(log_exception, logger)
    return logger


def log_exception(*args):
    if len(args) == 1:
        e = args[0]
        etype, value, tb = type(e), e, e.__traceback__
    elif len(args) == 3:
        etype, value, tb = args
    elif len(args) == 4:
        _, etype, value, tb = args
    else:
        logger.error(
            "Not able to log exception. Wrong number of arguments given. Should either receive 1 argument "
            "- an exception, or 3 arguments: exc type, exc value and traceback"
        )
        logger.error(f"{len(args)=}, {args=}")
        msg = ""
        for arg in args:
            if isinstance(arg, TracebackType):
                msg += f"{traceback.format_tb(arg)}\n"
        logger.error(msg)
        return

    tb_msg = "".join(traceback.format_exception(etype, value, tb))
    logger.exception(f"{tb_msg}")


logger = prepare_logger()
default_logger = logger

if __name__ == "__main__":
    logger.info(f"PROJ_DIR: {PROJ_DIR}")
    logger.warning(f"PROJ_DIR: {PROJ_DIR}")
    logger.error(f"PROJ_DIR: {PROJ_DIR}")
    logger.critical(f"PROJ_DIR: {PROJ_DIR}")
    logger.debug(f"PROJ_DIR: {PROJ_DIR}")
    raise ValueError("This is a test exception")
