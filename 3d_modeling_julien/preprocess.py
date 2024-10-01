import math
import glob
import argparse

import pandas as pd
from tqdm import tqdm
from multiprocessing import Process

from utils import *
from config import Config

print(
    f"Preprocessing data with the following arguments:\n"
    f"  Crop shape: {Config.crop_shape}"
    f"  Number of workers: {Config.num_workers}"
    f" "
)

# Retrieve training data : all the levels for all conditions are annotated with x, y position
# Columns : 'study_id', 'condition', 'level', 'severity', 'series_id','instance_number', 'x', 'y', 'series_description', 'row_id','image_path'
train_df = retrieve_training_data(Config.data_basepath)
print("Number of studies : ", len(train_df.groupby("study_id")))
series_descs = {
    e[0]: e[1]
    for e in train_df[["series_id", "series_description"]].drop_duplicates().values
}

train_descs_df = pd.read_csv(Config.data_basepath + "train_series_descriptions.csv")
train_descs_df = train_descs_df[
    (train_descs_df["study_id"].isin(train_df.study_id.unique()))
    & (train_descs_df["series_description"] == "Sagittal T1")
]

print("Using spinenet to find the centers of vertebral bones...")
patient_coords_df = calc_center_per_study(train_descs_df)

print("Calculate bounding boxes...")
bb_3d_df = calc_bounding_boxes_3d(patient_coords_df)


def worker_loop(dirslice):
    for dir in tqdm(dirslice):
        study_id = dir.split("/")[-1]
        study_bounds = bb_3d_df[bb_3d_df["study_id"] == int(study_id)].sort_values(
            by="level", ascending=True
        )

        levels = []
        min_bounds = []
        max_bounds = []

        for i in range(len(study_bounds)):
            row = study_bounds.iloc[i]
            levels.append(row["level"].replace("/", "").lower())
            min_bounds.append(np.array([row["x_min"], row["y_min"], row["z_min"]]))
            max_bounds.append(np.array([row["x_max"], row["y_max"], row["z_max"]]))

        read_vertebral_levels_as_voxel_grids(
            dir,
            study_id,
            vertebral_levels=levels,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            series_type_dict=series_descs,
            voxel_size=Config.crop_shape,
        )


if __name__ == "__main__":

    # print("Caching 3d volumes and train dataframe...")
    # dirs = [
    #     Config.data_basepath + "train_images/" + str(study_id)
    #     for study_id in train_df.study_id.unique()
    # ]
    # dirs = sorted(dirs)

    # slice_size = math.ceil(len(dirs) / Config.num_workers)

    # workers = []
    # for worker_index in range(Config.num_workers):
    #     dirslice = dirs[slice_size * worker_index : slice_size * (worker_index + 1)]
    #     p = Process(target=worker_loop, args=(dirslice,))
    #     p.start()
    #     workers.append(p)

    # for p in workers:
    #     p.join()

    # Create the train dataframe with columns : study_id level path label
    train_labels_df = pd.read_csv(Config.data_basepath + "train.csv").replace(Config.LABEL_MAP)

    preprocessed_train = {
        "study_id": [],
        "level": [],
        "path": [],
        "label": [],
    }
    for index, row in bb_3d_df.iterrows():
        preprocessed_train["study_id"].append(row["study_id"])
        preprocessed_train["level"].append(row["level"])
        preprocessed_train["path"].append(
            Config.data_basepath
            + f"processed_studies/{row['study_id']}_{row['level'].lower()}_{Config.crop_shape[0]}_{Config.crop_shape[1]}_{Config.crop_shape[2]}.npy.gz"
        )
        label = train_labels_df[train_labels_df["study_id"] == row["study_id"]].values[0][
            1 + Config.LEVEL_MAP[row["level"].lower()] :: 5
        ]
        preprocessed_train["label"].append(label)

    preprocessed_train_df = pd.DataFrame.from_dict(preprocessed_train)
    preprocessed_train_df.to_csv(
        Config.data_basepath + "processed_studies/preprocessed_train.csv"
    )
