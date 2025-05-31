# Multi-object 6D Tracking with Pose Confidence

6-DoF tracking of multiple objects with pose confidence estimation.

## Code structure

```
├── data               <- data directory
├── artifacts               <- directory for experiment artifacts
├── pyproject.toml     <- project configuration file with package metadata for 
│                         pose_tracking and configuration for tools like black
│
├── requirements.txt   <- minimal python dependencies
├── requirements_full.txt   <- versioned python dependencies
│
├── setup.cfg          <- configuration file for flake8 and others
│
└── pose_tracking   <- source code
    ├── models
        ├── baselines.py       <- CNN and Keypoint-CNN
        ├── cnnlstm.py       <- belief-based CNNLSTM
        ├── cvae.py       <- Conditional VAE
        ├── detr.py       <- DETR and Keypoint-DETR
        ├── encoders.py      <- encoders for images and depth maps
        ├── matcher.py     <- Hungarian matcher
        ├── set_criterion.py  <- criterion for DETR-based models
        ├── pizza.py       <- PIZZA (based on [link](https://github.com/nv-nguyen/pizza/tree/main))
        ├── pos_encoding.py  <- positional encoding for transformer-based models
        ├── unet.py       <- UNet for depth estimation
    ├── dataset               <- transforms dataset classes for loading data
    ├── utils          <- utils for various functionalities            
    ├── chamferdist               <- package containing a CUDA implementation of Chamfer distance
    ├── trainer*.py               <- trainers for different models
    └── train.py                <- main script for training models
```

There are multiple `trainer` modules for training CNNLSTM, Keypoint-DETR, Trackformer, Memotr, and other models:

```
trainer.py <- (main) trainer for CNNLSTM
trainer_detr.py  <- trainers for DETR, Deformable-DETR, and Trackformer
trainer_memotr.py <- trainer for Memotr
trainer_others.py <- trainers for VideoPose and PIZZA models
trainer_cvae.py  <- trainer for Conditional VAE model
trainer_kpt.py <- trainer for keypoint-based CNN (Kabsch algorithm for pose prediction)
trainer_rnd.py <- trainer for random network distillation for sim2real gap quantification
```

## Installation

```
git submodule update --init --recursive
conda create -n pose_tracking python=3.10
pip install -r requirements.txt
cd libs || exit 1
cd trackformer && pip install -e . && cd ..
cd bop_toolkit && pip install -e . && cd ..
cd MeMOTR && pip install -e . && cd ..
export COMET_API_KEY=W5npcWDiWeNPoB2OYkQvwQD0C
```

`COMET_API_KEY` should be set to the SDK key obtained based on [these instructions](https://www.comet.com/docs/v2/guides/experiment-management/configure-sdk/).

To finish the Comet setup, modify the `PROJ_NAME` and `COMET_WORKSPACE` variables in the `pose_tracking/config.py` to match your desired Comet project and workspace ([link](https://www.comet.com/docs/v2/guides/quickstart/)).

For Memotr, Trackformer, and Deformable DETR, compile the CUDA kernel for deformable attention ([link](https://github.com/fundamentalvision/Deformable-DETR)):

```shell
cd ./libs/MeMOTR/memotr/models/ops/
sh make.sh
cd ./libs/trackformer/src/trackformer/models/ops/
sh make.sh
```

If installing from `requirements.txt` resulted in package-related issues, you can install the full requirements file:

```
pip install -r requirements_full.txt
```

## Training a model

### Generating synthetic data

Simulation code can be found in the `IsaacLab-Internal` [repository](https://github.com/leggedrobotics/IsaacLab-Internal) on the branch `dev/kzait/pose_tracking`. `IsaacLab-Internal` is assumed to be the root directory for the remaining instructions on data generation.

In `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/inhand/inhand_env_cfg.py`, set `usd_path` to the Dextreme cube USD.

In `source/isaaclab_assets/isaaclab_assets/robots/allegro.py`, set `usd_path` of the `ALLEGRO_HAND_CFG` to the Allegro hand USD (its RSL version and not the default one from Omniverse).

**Note.** To use Euler for data generation, the aforementioned USD paths should be valid inside the Docker container.

**Note.** USD assets should be accompanied by the `asset_info.json` metadata file in the same folder as the USD. For example, the JSON file for `dex_cube_instanceable.usd` USD with a Dextreme cube contains the following:

```json
{
    "mesh_assets": [
        {
            "bbox": {
                "min": [
                    -0.03249999880790713,
                    -0.03249999880790713,
                    -0.03249999880790713
                ],
                "max": [
                    0.03249999880790713,
                    0.03249999880790713,
                    0.03249999880790713
                ],
                "center": [
                    0.0,
                    0.0,
                    0.0
                ],
                "extends": [
                    0.06499999761581426,
                    0.06499999761581426,
                    0.06499999761581426
                ]
            }
        }
    ],
    "metadata": {
        "name": "/data/custom/allegro_real/mesh",
        "root_dir": "/data/custom/allegro_real",
        "path": "/data/custom/allegro_real/mesh/cube.obj"
    }
}
```

While the mesh bounding box parameters are mandatory, `metadata` field can have an arbitrary structure.

This metadata file is either generated automatically (e.g., for IKEA by the IKEA data scraper) or should be created by hand.

#### In-hand manipulation

The script to generate the data for this use case is found at `scripts/reinforcement_learning/rsl_rl/play_dextreme_ds.py`.

To generate data for the in-hand manipulation with an Allegro hand and a Dextreme cube, run the following from the `IsaacLab-Internal` root directory:

```shell
./docker/cluster/cluster_interface.sh job --job_type dextreme --headless --enable_cameras --matching_assets_set dextreme --num_envs 40 --assets_each_env 1 --warmup_cam_steps 10 --max_size 0.5 --min_size 0.05 --max_assets 1000 --randomizations 'light' --save_start_idx_offset 0 --num_iters 50 --intrinsics_setup rsl --task Isaac-Repose-Cube-Allegro-v0 --run_name test --checkpoint /workspace/isaaclab/logs/rsl_rl/allegro_cube/allegro_rsl_pretrained_big/model_14999.pt --save_output --do_randomize --do_sample_cam_intrinsics --ds_name custom_sim_dextreme_2k_v2 --num_steps 40
```

#### IKEA

Clone the repository for `isaaclab_pose_tracking_sim_ext` package from [this link](https://github.com/kirilllzaitsev/isaaclab_pose_tracking_sim_ext). Install the `pose_tracking_sim` package from the root of the cloned repository via `pip install -e .`.

The script to generate the data for single-/multi-object IKEA and RSL cube asset sets can be found at `isaaclab_pose_tracking_sim_ext/pose_tracking_sim/ikea_asset_ds.py`.

The command to generate a training dataset with three identical RSL cubes and randomization of the environment (floor and table textures, lighting) and object motion is:

```shell
export DS_NAME=custom_sim_cube_scaled_1k_multiobj
export MATCHING_ASSETS_SET=cube_scaled
export NUM_ASSETS_PER_ENV=3
./docker/cluster/cluster_interface.sh job --job_type standard --headless --enable_cameras --matching_assets_set ${MATCHING_ASSETS_SET} --num_envs 1000 --assets_each_env ${NUM_ASSETS_PER_ENV} --warmup_cam_steps 10 --randomizations 'ground_texture table_texture' --save_start_idx_offset 0 --num_iters 20 --intrinsics_setup rsl --save_output --do_randomize --do_sample_cam_intrinsics --use_motion_with_noise --use_hemisphere_cam_xyz --ds_name ${DS_NAME} --num_steps 100
```

To get the corresponding validation dataset, append `_val` to `DS_NAME` and set an appropriate value for `num_envs` argument.

If `use_motion_with_noise` is set, the delta pose for objects between two subsequent timesteps is sampled from the Gaussian distribution. Otherwise, object's velocity will be resampled every timestep with a given probability.

`matching_assets_set` defines a set of assets which can appear in an environment. Available values are specified in the CLI arguments of `ikea_asset_ds.py`. To use the assets from IKEA, you should set up the dataset according to the instructions on the `ikea_isaac_lab` branch of the `rsl_assets` [repository](https://github.com/leggedrobotics/rsl_assets). Set `MATCHING_ASSETS_SET=mixed` to generate data using the RSL cube and some IKEA assets in one dataset.

**Note.** Ensure the `intrinsics_setup` corresponds to the camera intrinsics matrix that matches that of the eventual real-world camera.

`.obj` meshes required for training the model are generated by the script by converting the respective USD assets.

### General

The datasets are published at [this link](https://drive.google.com/drive/folders/1Owm-B_i82UaVaSJ008p1miareTXGJhFp) and should be extracted to the `data` directory.

To train with multiple GPUs, add the `--use_ddp` flag to the `train.py` arguments from below. A SLURM environment is required for this to work.

To disable logging to Comet, use `--exp_disabled`.

Available CLI arguments can be viewed in `args_parsing.py`. `model_name` argument defines a set of models that can be selected for training.

### Single-object pose tracking

Extract the dataset `custom_sim_dextreme_2k_cam1`.

To train a Memotr-based model from scratch, first download the checkpoint from [this link](https://drive.google.com/file/d/17FxIGgIZJih8LWkGdlIOe9ZpVZ9IRxSj/view?usp=sharing), placing it into the `libs/MeMOTR/memotr` folder. From the `pose_tracking` directory, run the following command:

```shell
python train.py --args_path configs/memotr_dextreme.yaml
```

To override some arguments from the file, add `--do_ignore_file_args_with_provided` followed by the respective arguments. For example, `--do_ignore_file_args_with_provided --exp_disabled --batch_size 2`.

To resume training from a Comet experiment assigned the name `striking_plank_5540`, optionally modifying some of its arguments, run:

```shell
export EXP_NAME=striking_plank_5540
export DS_NAME=dextreme_2k_cam1
python train.py --do_ignore_file_args_with_provided --args_from_exp_name ${EXP_NAME} --device cuda --exp_name args_${DS_NAME}_memotr_continue_exp_${EXP_NAME} --ckpt_exp_name ${EXP_NAME} --num_samples_val 30 --ds_name ikea --ds_alias ${DS_NAME} --ds_folder_name_train custom_sim_${DS_NAME} --ds_folder_name_val custom_sim_${DS_NAME}_val --max_train_videos 10000 --max_val_videos 24 --num_workers 4 --mask_pixels_prob 0.0 --transform_names brightness motion_blur gamma iso bg --transform_prob 0.8
```

where `args_from_exp_name` sets the source experiment and `do_ignore_file_args_with_provided` allows overriding the arguments from the experiment with the ones provided in the command.

To train a Keypoint-DETR model via CLI arguments:

```shell
export DS_NAME=dextreme_2k_cam1
python train.py --use_es --do_save_artifacts --use_lrs --exp_tags ablation --num_epochs 300 --val_epoch_freq 2 --save_epoch_freq 2 --device cuda --exp_name args_${DS_NAME}_detr_kpt --es_patience_epochs 25 --es_delta 0 --lrs_gamma 0.4 --lrs_min_lr 1e-6 --lrs_patience 10 --lrs_delta 0 --lr_encoders 1e-5 --do_predict_6d_rot --do_vis --vis_epoch_freq 20 --rt_hidden_dim 384 --lr 1e-4 --model_name detr_kpt --encoder_img_weights imagenet --encoder_depth_weights imagenet --t_loss_name mse --rot_loss_name mse --encoder_name resnet50 --rt_mlps_num_layers 3 --encoder_out_dim 512 --mt_encoding_type spatial --mt_num_queries 20 --mt_d_model 256 --mt_n_layers 6 --tf_use_focal_loss --tf_bbox_loss_coef 1 --tf_giou_loss_coef 1 --tf_ce_loss_coef 1 --tf_rot_loss_coef 1 --tf_t_loss_coef 1 --tf_depth_loss_coef 1 --seq_len 1 --batch_size 8 --num_samples_val 30 --ds_name ikea --ds_alias ${DS_NAME} --ds_folder_name_train custom_sim_${DS_NAME} --ds_folder_name_val custom_sim_${DS_NAME}_val --max_train_videos 10000 --max_val_videos 24 --num_workers 4 --mask_pixels_prob 0.0 --transform_names brightness motion_blur gamma iso bg --transform_prob 0.8
```

**Note.** Data augmentations are set by the `transform_names` argument. `bg` augmentation randomizes the background using the [MIT indoor dataset](https://web.mit.edu/torralba/www/indoor.html), treating both the hand and the cube as foreground.

### Multi-object pose tracking

Extract the dataset `custom_sim_cube_scaled_1k_random_multiobj`.

```shell
python train.py --args_path configs/memotr_multiobj.yaml
```

## Acknowledgements

These repositories were used as references for some of the implemented functionality:

- https://github.com/S-JingTao/Categorical_Pose_Tracking
- https://github.com/ylabbe/cosypose
- https://github.com/NVlabs/FoundationPose
- https://github.com/timmeinhardt/trackformer
- https://github.com/MCG-NJU/MeMOTR