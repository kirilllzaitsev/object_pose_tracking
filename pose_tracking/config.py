from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PROJ_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJ_DIR.parent

DATA_DIR = PROJ_DIR / "data"

ARTIFACTS_DIR = PROJ_DIR / "artifacts"
BENCHMARK_DIR = PROJ_DIR.parent / "benchmark"
YCBINEOAT_SCENE_DIR = DATA_DIR / "ycbineoat"

YCB_MESHES_DIR = f"{DATA_DIR}/ycb/models_bt"

try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
