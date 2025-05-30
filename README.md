# Multi-object 6D Tracking with Pose Confidence

6-DoF tracking of multiple objects with pose confidence estimation.

## Code structure

```
├── data               <- data directory
├── pyproject.toml     <- project configuration file with package metadata for 
│                         pose_tracking and configuration for tools like black
│
├── requirements.txt   <- python dependencies
│
├── setup.cfg          <- configuration file for flake8
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

For Memotr and Deformable DETR, compile the CUDA kernel for deformable attention ([link](https://github.com/fundamentalvision/Deformable-DETR)) from `libs/MeMOTR`:

```shell
cd ./libs/MeMOTR/memotr/models/ops/
sh make.sh
```

For Trackformer, run the same command to install the deformable attention from `libs/trackformer/trackformer`. 

If installing from `requirements.txt` resulted in package-related issues, you can install the full requirements file:

```
pip install -r requirements_full.txt
```

## Training a model

### Single-object pose tracking

Extract the dataset `custom_sim_dextreme_2k_cam1` published at [this link](https://drive.google.com/drive/folders/1Owm-B_i82UaVaSJ008p1miareTXGJhFp) to the `data` directory.

To train a Memotr-based model from scratch, first download the checkpoint from [this link](https://drive.google.com/file/d/17FxIGgIZJih8LWkGdlIOe9ZpVZ9IRxSj/view?usp=sharing), placing it into the `libs/MeMOTR/memotr` folder. From the `pose_tracking` directory, run the following command:

```
python train.py --use_ddp --do_ignore_file_args_with_provided --args_from_exp_name striking_plank_5540 --device cuda --exp_name args_dextreme_2k_cam1_memotr_continue_exp_striking_plank_5540 --ckpt_exp_name striking_plank_5540 --num_samples_val 30 --ds_name ikea --ds_alias dextreme_2k_cam1 --ds_folder_name_train custom_sim_dextreme_2k_cam1 --ds_folder_name_val custom_sim_dextreme_2k_cam1_val --max_train_videos 10000 --max_val_videos 24 --num_workers 4 --mask_pixels_prob 0.0 --transform_names brightness motion_blur gamma iso bg --transform_prob 0.8
```

To train a Keypoint-DETR model:

```
python train.py --use_ddp --use_es --do_save_artifacts --use_lrs --exp_tags ablation --num_epochs 300 --val_epoch_freq 2 --save_epoch_freq 2 --device cuda --exp_name args_dextreme_2k_cam1_detr_kpt --es_patience_epochs 25 --es_delta 0 --lrs_gamma 0.4 --lrs_min_lr 1e-6 --lrs_patience 10 --lrs_delta 0 --lr_encoders 1e-5 --do_predict_6d_rot --do_vis --vis_epoch_freq 20 --rt_hidden_dim 384 --lr 1e-4 --model_name detr_kpt --encoder_img_weights imagenet --encoder_depth_weights imagenet --t_loss_name mse --rot_loss_name mse --encoder_name resnet50 --rt_mlps_num_layers 3 --encoder_out_dim 512 --mt_encoding_type spatial --mt_num_queries 20 --mt_d_model 256 --mt_n_layers 6 --tf_use_focal_loss --tf_bbox_loss_coef 1 --tf_giou_loss_coef 1 --tf_ce_loss_coef 1 --tf_rot_loss_coef 1 --tf_t_loss_coef 1 --tf_depth_loss_coef 1 --seq_len 1 --batch_size 8 --num_samples_val 30 --ds_name ikea --ds_alias dextreme_2k_cam1 --ds_folder_name_train custom_sim_dextreme_2k_cam1 --ds_folder_name_val custom_sim_dextreme_2k_cam1_val --max_train_videos 10000 --max_val_videos 24 --num_workers 4 --mask_pixels_prob 0.0 --transform_names brightness motion_blur gamma iso bg --transform_prob 0.8
```

### Multi-object pose tracking

Extract the dataset `custom_sim_cube_scaled_1k_random_multiobj` published at [this link](https://drive.google.com/drive/folders/1Owm-B_i82UaVaSJ008p1miareTXGJhFp) to the `data` directory.

```
python train.py --use_ddp --use_es --do_save_artifacts --num_epochs 300 --val_epoch_freq 2 --save_epoch_freq 4 --device cuda --exp_name args_cube_scaled_1k_random_multiobj_memotr_best --es_patience_epochs 60 --es_delta 0 --lrs_min_lr 1e-6 --lrs_patience 30 --weight_decay 1e-4 --lr_encoders 2e-5 --do_predict_6d_rot --use_rnn --tf_use_pretrained_model --do_vis --vis_epoch_freq 50 --rt_hidden_dim 384 --lr 2e-4 --model_name memotr --encoder_img_weights imagenet --encoder_depth_weights imagenet --t_loss_name mse --rot_loss_name mse --encoder_name resnet50 --rt_mlps_num_layers 3 --do_load_session --mt_num_queries 5 --tf_use_focal_loss --tf_bbox_loss_coef 1 --tf_giou_loss_coef 1 --tf_ce_loss_coef 2 --tf_rot_loss_coef 1 --tf_t_loss_coef 1 --seq_len 5 --num_samples 10 --end_frame_idx 100 --batch_size 4 --ds_name ikea_multiobj --ds_alias cube_scaled_1k_random_multiobj --ds_folder_name_train custom_sim_cube_scaled_1k_random_multiobj --ds_folder_name_val custom_sim_cube_scaled_1k_random_multiobj_val --max_train_videos 10000 --max_val_videos 24 --num_workers 16 --mask_pixels_prob 0.1 --transform_names brightness motion_blur gamma iso --transform_prob 0.8
```

## Acknowledgements

These repositories served as references for some implemented functionality:

- https://github.com/S-JingTao/Categorical_Pose_Tracking
- https://github.com/ylabbe/cosypose
- https://github.com/NVlabs/FoundationPose
- https://github.com/timmeinhardt/trackformer
- https://github.com/MCG-NJU/MeMOTR