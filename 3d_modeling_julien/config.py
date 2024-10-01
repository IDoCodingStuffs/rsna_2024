import torch
from dataclasses import dataclass


@dataclass
class Config:
    # Data
    data_basepath = "/Data/RSNA/"
    model_dir = "./models/"
    df_basepath = "./dataframes/"

    # Constants
    CONDITIONS = {
        "Sagittal T2/STIR": ["Spinal Canal Stenosis"],
        "Axial T2": ["Left Subarticular Stenosis", "Right Subarticular Stenosis"],
        "Sagittal T1": [
            "Left Neural Foraminal Narrowing",
            "Right Neural Foraminal Narrowing",
        ],
    }
    LABEL_MAP = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
    LEVELS = ["L1", "L2", "L3", "L4", "L5", "S1"]
    LEVELS_DISCS = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
    LEVEL_MAP = {
        level: idx for idx, level in enumerate(LEVELS_DISCS)
    }

    # Preprocess
    num_workers = 1
    crop_shape = (64, 64, 64)
    data_path = "/Data/RSNA/processed_studies/"

    # Model
    backbone = "tf_efficientnet_b0.in1k"
    drop_rate = 0.3
    drop_path_rate = 0.3
    drop_rate_last = 0.
    out_dim = 3

    # Transforms
    image_interpolation = "bspline"
    aug_prob = 0.9
    vol_size = (128, 128, 128)

    # Training 
    n_epochs = 45
    debug = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_folds = 5
    batch_size = 4
    lr = 3e-4