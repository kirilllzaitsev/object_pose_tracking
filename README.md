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
    ├── utils          <- utils for various functionalities            
    ├── dataset               <- transforms dataset classes for loading data
    ├── chamferdist               <- package containing a CUDA implementation of Chamfer distance
    ├── trainer*.py               <- trainers for different models
    └── train.py                <- main script for training models
```

## Installation

```
conda create -n pose_tracking python=3.10
pip install -r requirements.txt
cd trackformer && pip install -e . && cd -
cd memotr && pip install -e . && cd -
export COMET_API_KEY=W5npcWDiWeNPoB2OYkQvwQD0C
```

Extract the dataset published at [this link](https://drive.google.com/drive/folders/1Owm-B_i82UaVaSJ008p1miareTXGJhFp) to the `data` directory.

## Training a model

From the `pose_tracking` directory, run the following command:

```
python train.py --use_ddp --use_es --do_save_artifacts --use_lrs --exp_tags ablation --num_epochs 300 --val_epoch_freq 2 --save_epoch_freq 2 --device cuda --exp_name args_dextreme_2k_cam1_detr_kpt --es_patience_epochs 25 --es_delta 0 --lrs_gamma 0.4 --lrs_min_lr 1e-6 --lrs_patience 10 --lrs_delta 0 --lr_encoders 1e-5 --do_predict_6d_rot --do_vis --vis_epoch_freq 20 --rt_hidden_dim 384 --lr 1e-4 --model_name detr_kpt --encoder_img_weights imagenet --encoder_depth_weights imagenet --t_loss_name mse --rot_loss_name mse --encoder_name resnet50 --rt_mlps_num_layers 3 --encoder_out_dim 512 --mt_encoding_type spatial --mt_num_queries 20 --mt_d_model 256 --mt_n_layers 6 --tf_use_focal_loss --tf_bbox_loss_coef 1 --tf_giou_loss_coef 1 --tf_ce_loss_coef 1 --tf_rot_loss_coef 1 --tf_t_loss_coef 1 --tf_depth_loss_coef 1 --seq_len 1 --batch_size 8 --num_samples_val 30 --ds_name ikea --ds_alias dextreme_2k_cam1 --ds_folder_name_train custom_sim_dextreme_2k_cam1 --ds_folder_name_val custom_sim_dextreme_2k_cam1_val --max_train_videos 10000 --max_val_videos 24 --num_workers 4 --mask_pixels_prob 0.0 --transform_names brightness motion_blur gamma iso bg --transform_prob 0.8
```

## Acknowledgements

These repositories served as references for some implemented functionality:

- https://github.com/S-JingTao/Categorical_Pose_Tracking
- https://github.com/ylabbe/cosypose
- https://github.com/NVlabs/FoundationPose
- https://github.com/timmeinhardt/trackformer
- https://github.com/MCG-NJU/MeMOTR