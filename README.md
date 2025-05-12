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
    ├── trainer*.py               <- trainers for different models
    └── train.py                <- main script for training models
```

## Installation

```
conda create -n pose_tracking
pip install -r requirements.txt
```

## Training a model

TBD

## Acknowledgements

These repositories served as references for some implemented functionality:

- https://github.com/S-JingTao/Categorical_Pose_Tracking
- https://github.com/ylabbe/cosypose
- https://github.com/NVlabs/FoundationPose
- https://github.com/timmeinhardt/trackformer
- https://github.com/MCG-NJU/MeMOTR