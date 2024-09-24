from pathlib import Path

from loguru import logger
from pose_tracking.config import MODELS_DIR, PROCESSED_DATA_DIR, PROJ_DIR
from tqdm import tqdm


def main():
    logger.info(f"PROJ_ROOT path is: {PROJ_DIR}")
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")


if __name__ == "__main__":
    ...
