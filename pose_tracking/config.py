import functools
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJ_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJ_DIR.parent

DATA_DIR = PROJ_DIR / "data"
RELATED_DIR = WORKSPACE_DIR / "related_work"

ARTIFACTS_DIR = PROJ_DIR / "artifacts"
BENCHMARK_DIR = PROJ_DIR.parent / "benchmark"
YCBINEOAT_SCENE_DIR = DATA_DIR / "ycbineoat"

YCB_MESHES_DIR = DATA_DIR / "ycb/models"
HO3D_ROOT = DATA_DIR / "ho3d"


def prepare_logger(logpath=None, level="INFO"):
    from loguru import logger

    # Define handlers
    common_cfg = {
        "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {level} | {file}:{line} | <level>{message}</level>",
        "level": level,
        "colorize": True,
    }
    handlers = [
        {
            "sink": sys.stdout,
            **common_cfg,
        }
    ]

    if logpath:
        handlers.append(
            {
                "sink": logpath,
                **common_cfg,
            }
        )

    try:
        from tqdm import tqdm

        def tqdm_sink(message):
            tqdm.write(message, end="")

        handler_tqdm = {
            "sink": tqdm_sink,
            **common_cfg,
        }
        handlers[0] = handler_tqdm
    except ModuleNotFoundError:
        logger.debug("tqdm not found. Skipping tqdm integration.")

    # Configure logger with the defined handlers
    logger.remove()
    logger.configure(handlers=handlers)
    sys.excepthook = functools.partial(log_exception, logger)
    return logger


def log_exception(logger, *args):
    if len(args) == 1:
        e = args[0]
        etype, value, tb = type(e), e, e.__traceback__
    elif len(args) == 3:
        etype, value, tb = args
    else:
        logger.error(
            "Not able to log exception. Wrong number of arguments given. Should either receive 1 argument "
            "- an exception, or 3 arguments: exc type, exc value and traceback"
        )
        return

    tb_msg = "".join(traceback.format_exception(etype, value, tb))
    logger.exception(f"{tb_msg}")


logger = prepare_logger()

if __name__ == "__main__":
    logger.info(f"PROJ_DIR: {PROJ_DIR}")
    logger.warning(f"PROJ_DIR: {PROJ_DIR}")
    logger.error(f"PROJ_DIR: {PROJ_DIR}")
    logger.critical(f"PROJ_DIR: {PROJ_DIR}")
    logger.debug(f"PROJ_DIR: {PROJ_DIR}")
    raise ValueError("This is a test exception")
