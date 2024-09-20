import os

from pose_tracking.dataset.bop_loaders import load_cad, load_list_scene, load_metadata
from torch.utils.data import Dataset
from tqdm import tqdm


class BaseBOP(Dataset):
    def __init__(
        self,
        root_dir,
        split,
        rot_repr="rotation6d",
        cad_dir=None,
        do_load_cad=False,
    ):
        """
        Read a dataset in the BOP format.
        See https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md
        """
        self.root_dir = root_dir
        self.split = split
        self.rot_repr = rot_repr
        self.cad_dir = cad_dir
        self.list_scenes = load_list_scene(root_dir, split)
        self.metadata = load_metadata(root_dir, split)
        if do_load_cad:
            if cad_dir is None:
                possible_cad_subdir_names = ["models_cad", "models"]
                for subdir_name in possible_cad_subdir_names:
                    cad_dir = root_dir / subdir_name
                    if cad_dir.exists():
                        break
                assert cad_dir is not None, "CAD dir must be provided"
            self.cads = load_cad(cad_dir)
            self.cad_dir = cad_dir
        else:
            self.cads = None


if __name__ == "__main__":
    from pose_tracking.config import DATA_DIR

    for dataset_name, split_ in zip(["tless/test", "tless/train"], ["test_primesense", "train_primesense"]):
        ds_dir = DATA_DIR / dataset_name
        dataset = BaseBOP(ds_dir, split_, cad_dir=DATA_DIR / "tless/models_cad", do_load_cad=True)

    dataset_names = ["lmo"]
    for dataset_name in tqdm(dataset_names):
        ds_dir = DATA_DIR / dataset_name
        splits = [s for s in os.listdir(ds_dir) if os.path.isdir(ds_dir / s)]
        splits = [s for s in splits if s.startswith("train") or s.startswith("val") or s.startswith("test")]
        for split_ in splits:
            dataset = BaseBOP(ds_dir, split_)
