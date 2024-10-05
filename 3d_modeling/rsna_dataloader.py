import math
import random
from pathlib import Path
from typing import List, Tuple

import open3d as o3d
import nibabel as nib
import os
import numpy as np
import pandas as pd
import glob
import pydicom
import torch
from pydicom import dcmread
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import cv2
import torchio as tio
import itk
from transformers import SamModel, SamProcessor
import torch.nn.functional as F
import pgzip
import copy

LABEL_MAP = {'normal_mild': 0, 'moderate': 1, 'severe': 2}
CONDITIONS = {
    "Sagittal T2/STIR": ["Spinal Canal Stenosis"],
    "Axial T2": ["Left Subarticular Stenosis", "Right Subarticular Stenosis"],
    "Sagittal T1": ["Left Neural Foraminal Narrowing", "Right Neural Foraminal Narrowing"],
}
MAX_IMAGES_IN_SERIES = {
    "Sagittal T2/STIR": 29,
    # !TODO: Might need a 3D model for this one
    "Axial T2": 192,
    "Sagittal T1": 38,
}

CENTERS = ["L1", "L2", "L3", "L4", "L5", "S1"]
LEVELS = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]

device = "cuda" if torch.cuda.is_available() else "cpu"


def rotation_matrix_from_vectors(vec1, vec2=np.array([0, 0, -1])):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


class StudyLevelDataset(Dataset):
    def __init__(self,
                 base_path: str,
                 dataframe: pd.DataFrame,
                 transform_3d=None,
                 is_train=False,
                 vol_size=(192, 192, 192),
                 use_mirror_trick=False):
        self.base_path = base_path
        self.is_train = is_train
        self.use_mirror_trick = use_mirror_trick

        self.dataframe = (dataframe[['study_id', "series_id", "series_description", "condition", "severity", "level"]]
                          .drop_duplicates())

        self.subjects = self.dataframe[['study_id']].drop_duplicates().reset_index(drop=True)
        self.series = self.dataframe[["study_id", "series_id"]].drop_duplicates().groupby("study_id")[
            "series_id"].apply(list).to_dict()
        self.series_descs = {e[0]: e[1] for e in
                             self.dataframe[["series_id", "series_description"]].drop_duplicates().values}

        self.transform_3d = transform_3d

        self.levels = sorted(self.dataframe["level"].unique())
        self.labels = self._get_labels()
        self.vol_size = vol_size

    def __len__(self):
        return len(self.subjects) * (2 if self.use_mirror_trick else 1)

    def __getitem__(self, index):
        is_mirror = index >= len(self.subjects)
        curr = self.subjects.iloc[index % len(self.subjects)]

        label = np.array(self.labels[(curr["study_id"])])
        study_path = os.path.join(self.base_path, str(curr["study_id"]))

        study_images = read_study_as_voxel_grid_v2(study_path,
                                                   series_type_dict=self.series_descs,
                                                   img_size=(self.vol_size[0], self.vol_size[1]))

        if is_mirror:
            temp = label[:10].copy()
            label[:10] = label[10:20].copy()
            label[10:20] = temp

        if self.transform_3d is not None:
            study_images = torch.FloatTensor(study_images)

            if is_mirror:
                study_images = torch.flip(study_images, [1])

            study_images = self.transform_3d(study_images)  # .data
            return study_images.to(torch.half), torch.tensor(label, dtype=torch.long)

        print("loaded")
        return torch.HalfTensor(study_images.copy()), torch.tensor(label, dtype=torch.long)

    def _get_labels(self):
        labels = dict()
        for name, group in self.dataframe.groupby(["study_id"]):
            group = group[["condition", "level", "severity"]].drop_duplicates().sort_values(["condition", "level"])
            label_indices = []
            for index, row in group.iterrows():
                if row["severity"] in LABEL_MAP:
                    label_indices.append(LABEL_MAP[row["severity"]])
                else:
                    raise ValueError()

            study_id = name[0]

            labels[study_id] = label_indices

        return labels


class StudyPerVertebraLevelDataset(Dataset):
    def __init__(self,
                 base_path: str,
                 dataframe: pd.DataFrame,
                 bounds_dataframe: pd.DataFrame,
                 coords_dataframe: pd.DataFrame,
                 transform_3d=None,
                 is_train=False,
                 vol_size=(192, 192, 192),
                 use_mirror_trick=False):
        self.base_path = base_path
        self.is_train = is_train
        self.use_mirror_trick = use_mirror_trick

        self.dataframe = (dataframe[['study_id', "series_id", "series_description", "condition", "severity", "level"]]
                          .drop_duplicates())

        self.bounds_dataframe = bounds_dataframe
        self.coords_dataframe = coords_dataframe

        self.subjects = self.dataframe[['study_id', 'level']].drop_duplicates().reset_index(drop=True)
        self.series = self.dataframe[["study_id", "series_id"]].drop_duplicates().groupby("study_id")[
            "series_id"].apply(list).to_dict()
        self.series_descs = {e[0]: e[1] for e in
                             self.dataframe[["series_id", "series_description"]].drop_duplicates().values}

        self.transform_3d = transform_3d

        self.levels = sorted(self.dataframe["level"].unique())
        self.labels = self._get_labels()
        self.vol_size = vol_size

    def __len__(self):
        return len(self.subjects) * (2 if self.use_mirror_trick else 1)

    def __getitem__(self, index):
        is_mirror = index >= len(self.subjects)
        curr = self.subjects.iloc[index % len(self.subjects)]

        label = np.array(self.labels[(curr["study_id"], curr["level"])])
        study_path = os.path.join(self.base_path, str(curr["study_id"]))

        level = curr["level"].replace("/", "_").lower()
        level_idx = LEVELS.index(level)

        curr_bounds = self.bounds_dataframe[
            (self.bounds_dataframe["study_id"] == curr["study_id"]) & (self.bounds_dataframe["level"] == curr["level"])
            ]
        curr_centers = self.coords_dataframe[
            (self.coords_dataframe["study_id"] == curr["study_id"]) &
            (self.coords_dataframe["level"].isin((CENTERS[level_idx], CENTERS[level_idx + 1])))
            ].sort_values(by=["level"])

        if len(curr_centers) != 2:
            raise ValueError("Malformed centers")

        center_1 = np.array(curr_centers.iloc[0][["x", "y", "z"]].values)
        center_2 = np.array(curr_centers.iloc[1][["x", "y", "z"]].values)

        study_images = read_vertebral_level_as_voxel_grid_nonaligned(study_path,
                                                                vertebral_level=level,
                                                                center_point_pair=(center_1, center_2),
                                                                min_bound=np.array(
                                                                    [curr_bounds['x_min'], curr_bounds['y_min'],
                                                                     curr_bounds['z_min']]),
                                                                max_bound=np.array(
                                                                    [curr_bounds['x_max'], curr_bounds['y_max'],
                                                                     curr_bounds['z_max']]),
                                                                series_type_dict=self.series_descs,
                                                                voxel_size=self.vol_size
                                                                )

        if is_mirror:
            temp = label[:2].copy()
            label[:2] = label[2:4].copy()
            label[2:4] = temp

        if self.transform_3d is not None:
            study_images = torch.FloatTensor(study_images)

            if is_mirror:
                study_images = torch.flip(study_images, [1])

            study_images = self.transform_3d(study_images)  # .data
            return study_images.to(torch.half), torch.tensor(label, dtype=torch.long)

        return torch.HalfTensor(study_images.copy()), torch.tensor(label, dtype=torch.long)

    def _get_labels(self):
        labels = dict()
        for name, group in self.dataframe.groupby(["study_id", "level"]):
            group = group[["condition", "level", "severity"]].drop_duplicates().sort_values(["condition"])
            label_indices = []
            for index, row in group.iterrows():
                if row["severity"] in LABEL_MAP:
                    label_indices.append(LABEL_MAP[row["severity"]])
                else:
                    raise ValueError()

            study_id = name[0]
            level = name[1]

            labels[(study_id, level)] = label_indices

        return labels


# region creation_funcs

def create_study_level_datasets_and_loaders(df: pd.DataFrame,
                                            base_path: str,
                                            transform_3d_train=None,
                                            transform_3d_val=None,
                                            vol_size=None,
                                            split_factor=0.2,
                                            random_seed=42,
                                            batch_size=1,
                                            num_workers=0,
                                            pin_memory=True,
                                            use_mirroring_trick=True):
    df = df.dropna()
    # This drops any subjects with nans

    filtered_df = pd.DataFrame(columns=df.columns)
    for series_desc in CONDITIONS.keys():
        subset = df[df['series_description'] == series_desc]
        if series_desc == "Sagittal T2/STIR":
            subset = subset[subset.groupby(["study_id"]).transform('size') == 5]
        else:
            subset = subset[subset.groupby(["study_id"]).transform('size') == 10]
        filtered_df = pd.concat([filtered_df, subset])

    filtered_df = filtered_df[filtered_df.groupby(["study_id"]).transform('size') == 25]

    train_studies, val_studies = train_test_split(filtered_df["study_id"].unique(), test_size=split_factor,
                                                  random_state=random_seed)
    val_studies, test_studies = train_test_split(val_studies, test_size=0.25, random_state=random_seed)

    train_df = filtered_df[filtered_df["study_id"].isin(train_studies)]
    val_df = filtered_df[filtered_df["study_id"].isin(val_studies)]
    test_df = filtered_df[filtered_df["study_id"].isin(test_studies)]

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    random.seed(random_seed)
    train_dataset = StudyLevelDataset(base_path, train_df,
                                      transform_3d=transform_3d_train,
                                      is_train=True,
                                      use_mirror_trick=use_mirroring_trick,
                                      vol_size=vol_size
                                      )
    val_dataset = StudyLevelDataset(base_path, val_df,
                                    transform_3d=transform_3d_val,
                                    vol_size=vol_size
                                    )
    test_dataset = StudyLevelDataset(base_path, test_df,
                                     transform_3d=transform_3d_val,
                                     vol_size=vol_size
                                     )

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             persistent_workers=num_workers > 0)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def create_study_level_datasets_and_loaders_k_fold(df: pd.DataFrame,
                                                   base_path: str,
                                                   transform_3d_train=None,
                                                   transform_3d_val=None,
                                                   vol_size=None,
                                                   split_k=4,
                                                   random_seed=42,
                                                   batch_size=1,
                                                   num_workers=0,
                                                   pin_memory=True,
                                                   use_mirroring_trick=True):
    df = df.dropna()
    # This drops any subjects with nans

    filtered_df = pd.DataFrame(columns=df.columns)
    for series_desc in CONDITIONS.keys():
        subset = df[df['series_description'] == series_desc]
        if series_desc == "Sagittal T2/STIR":
            subset = subset[subset.groupby(["study_id"]).transform('size') == 5]
        else:
            subset = subset[subset.groupby(["study_id"]).transform('size') == 10]
        filtered_df = pd.concat([filtered_df, subset])

    filtered_df = filtered_df[filtered_df.groupby(["study_id"]).transform('size') == 25]

    np.random.seed(random_seed)
    ids = filtered_df["study_id"].unique()
    np.random.shuffle(ids)

    ret = []
    folds = np.array_split(ids, split_k)

    for index, fold in enumerate(folds):
        val_studies = fold

        train_df = filtered_df[~filtered_df["study_id"].isin(val_studies)]
        val_df = filtered_df[filtered_df["study_id"].isin(val_studies)]

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        train_dataset = StudyLevelDataset(base_path, train_df,
                                          transform_3d=transform_3d_train,
                                          is_train=True,
                                          use_mirror_trick=use_mirroring_trick,
                                          vol_size=vol_size
                                          )
        val_dataset = StudyLevelDataset(base_path, val_df,
                                        transform_3d=transform_3d_val,
                                        vol_size=vol_size
                                        )

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  persistent_workers=num_workers > 0)
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                persistent_workers=num_workers > 0)

        ret.append((train_loader, val_loader, train_dataset, val_dataset))

    return ret


def create_vertebra_level_datasets_and_loaders_k_fold(df: pd.DataFrame,
                                                      boundaries_df: pd.DataFrame,
                                                      coords_df: pd.DataFrame,
                                                      base_path: str,
                                                      transform_3d_train=None,
                                                      transform_3d_val=None,
                                                      vol_size=None,
                                                      split_k=4,
                                                      random_seed=42,
                                                      batch_size=1,
                                                      num_workers=0,
                                                      pin_memory=True,
                                                      use_mirroring_trick=True):
    df = df.dropna()
    # This drops any subjects with nans

    filtered_df = pd.DataFrame(columns=df.columns)
    for series_desc in CONDITIONS.keys():
        subset = df[df['series_description'] == series_desc]
        if series_desc == "Sagittal T2/STIR":
            subset = subset[subset.groupby(["study_id"]).transform('size') == 5]
        else:
            subset = subset[subset.groupby(["study_id"]).transform('size') == 10]
        filtered_df = pd.concat([filtered_df, subset])

    filtered_df = filtered_df[filtered_df.groupby(["study_id"]).transform('size') == 25]

    np.random.seed(random_seed)
    ids = filtered_df["study_id"].unique()
    np.random.shuffle(ids)

    ret = []
    folds = np.array_split(ids, split_k)

    for index, fold in enumerate(folds):
        val_studies = fold

        train_df = filtered_df[~filtered_df["study_id"].isin(val_studies)]
        val_df = filtered_df[filtered_df["study_id"].isin(val_studies)]

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        train_dataset = StudyPerVertebraLevelDataset(base_path, train_df, boundaries_df, coords_df,
                                                     transform_3d=transform_3d_train,
                                                     is_train=True,
                                                     use_mirror_trick=use_mirroring_trick,
                                                     vol_size=vol_size
                                                     )
        val_dataset = StudyPerVertebraLevelDataset(base_path, val_df, boundaries_df, coords_df,
                                                   transform_3d=transform_3d_val,
                                                   vol_size=vol_size
                                                   )

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  persistent_workers=num_workers > 0)
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                persistent_workers=num_workers > 0)

        ret.append((train_loader, val_loader, train_dataset, val_dataset))

    return ret


# endregion


def read_study_as_pcd(dir_path,
                      series_types_dict=None,
                      downsampling_factor=1,
                      resize_slices=True,
                      resize_method="nearest",
                      stack_slices_thickness=True,
                      img_size=(256, 256)):
    pcd_sagittal = o3d.geometry.PointCloud()
    pcd_axial = o3d.geometry.PointCloud()

    for path in glob.glob(os.path.join(dir_path, "**/*.dcm"), recursive=True):
        dicom_slice = dcmread(path)

        series_id = os.path.basename(os.path.dirname(path))
        study_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if series_types_dict is None or int(series_id) not in series_types_dict:
            series_desc = dicom_slice.SeriesDescription
        else:
            series_desc = series_types_dict[int(series_id)]
            series_desc = series_desc.split(" ")[-1]

        x_orig, y_orig = dicom_slice.pixel_array.shape
        if resize_slices:
            if resize_method == "nearest":
                img = np.expand_dims(cv2.resize(dicom_slice.pixel_array, img_size, interpolation=cv2.INTER_AREA), -1)
            elif resize_method == "maxpool":
                img_tensor = torch.tensor(dicom_slice.pixel_array).float()
                img = F.adaptive_max_pool2d(img_tensor.unsqueeze(0), img_size).numpy()
            else:
                raise ValueError(f"Invalid resize_method {resize_method}")
        else:
            img = np.expand_dims(np.array(dicom_slice.pixel_array), -1)
        x, y, z = np.where(img)

        downsampling_factor_iter = max(downsampling_factor, int(math.ceil(len(x) / 6e6)))

        index_voxel = np.vstack((x, y, z))[:, ::downsampling_factor_iter]
        grid_index_array = index_voxel.T
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid_index_array.astype(np.float64)))

        vals = np.expand_dims(img[x, y, z][::downsampling_factor_iter], -1)
        if series_desc == "T1":
            vals = np.pad(vals, ((0, 0), (0, 2)))
        elif series_desc == "T2":
            vals = np.pad(vals, ((0, 0), (1, 1)))
        elif series_desc == "T2/STIR":
            vals = np.pad(vals, ((0, 0), (2, 0)))
        else:
            raise ValueError(f"Unknown series desc: {series_desc}")

        pcd.colors = o3d.utility.Vector3dVector(vals.astype(np.float64))

        if resize_slices:
            transform_matrix_factor = np.matrix(
                [[0, y_orig / img_size[1], 0, 0],
                 [x_orig / img_size[0], 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]
            )
        else:
            transform_matrix_factor = np.matrix(
                [[0, 1, 0, 0],
                 [1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]
            )

        dX, dY = dicom_slice.PixelSpacing
        dZ = dicom_slice.SliceThickness

        X = np.array(list(dicom_slice.ImageOrientationPatient[:3]) + [0]) * dX
        Y = np.array(list(dicom_slice.ImageOrientationPatient[3:]) + [0]) * dY

        S = np.array(list(dicom_slice.ImagePositionPatient) + [1])

        transform_matrix = np.array([X, Y, np.zeros(len(X)), S]).T
        transform_matrix = transform_matrix @ transform_matrix_factor

        if stack_slices_thickness:
            for z in range(int(dZ) + 1):
                pos = list(dicom_slice.ImagePositionPatient)
                if series_desc == "T2":
                    pos[-1] += z
                else:
                    pos[0] += z
                S = np.array(pos + [1])

                transform_matrix = np.array([X, Y, np.zeros(len(X)), S]).T
                transform_matrix = transform_matrix @ transform_matrix_factor

                if series_desc == "T2":
                    pcd_axial += copy.deepcopy(pcd).transform(transform_matrix)
                else:
                    pcd_sagittal += copy.deepcopy(pcd).transform(transform_matrix)

        else:
            if series_desc == "T2":
                pcd_axial += pcd.transform(transform_matrix)
            else:
                pcd_sagittal += pcd.transform(transform_matrix)

    bbox = pcd_sagittal.get_oriented_bounding_box()
    pcd_axial = pcd_axial.crop(bbox)

    return pcd_axial + pcd_sagittal


def read_study_as_voxel_grid(dir_path, channel_count=3, downsampling_factor=1, series_type_dict=None,
                             img_size=(256, 256)):
    cache_path = os.path.join(dir_path, f"cached_grid_{img_size[0]}.npy.gz")
    # cache_path = os.path.join(dir_path, f"cached_grid_2_down3.npy.gz")
    f = None
    if os.path.exists(cache_path):
        try:
            f = pgzip.PgzipFile(cache_path, "r")
            ret = np.load(f, allow_pickle=True)
            f.close()
            return ret
        except Exception as e:
            print(dir_path, "\n", e)
            if f:
                f.close()
            os.remove(cache_path)

    pcd_overall = read_study_as_pcd(dir_path, series_types_dict=series_type_dict,
                                    downsampling_factor=downsampling_factor, img_size=img_size)

    path = next(glob.iglob(os.path.join(dir_path, "**/*.dcm"), recursive=True))
    dicom_slice = dcmread(path)

    dX, dY = dicom_slice.PixelSpacing
    # size = dX * dicom_slice.pixel_array.shape[0] / img_size[0]
    # size = max(size, dY * dicom_slice.pixel_array.shape[1] / img_size[1])
    size = 1
    voxel_grid = o3d.geometry.VoxelGrid().create_from_point_cloud(pcd_overall, size)

    coords = []
    vals = []

    for voxel in voxel_grid.get_voxels():
        coords.append(voxel.grid_index)
        vals.append(voxel.color)

    coords = np.array(coords)
    vals = np.array(vals, dtype=np.float16)

    size = np.max(coords, axis=0) + 1
    # 1 channel per pulse sequence type, CHWD (I think?)
    grid = np.zeros((channel_count, size[0], size[1], size[2]), dtype=np.float16)

    for i in range(channel_count):
        grid[i, coords[:, 0], coords[:, 1], coords[:, 2]] = vals[:, i]

    f = pgzip.PgzipFile(cache_path, "w")
    np.save(f, grid)
    f.close()

    del pcd_overall
    del voxel_grid

    return grid


def read_study_as_voxel_grid_v2(dir_path, series_type_dict=None, downsampling_factor=1, img_size=(256, 256)):
    cache_path = os.path.join(dir_path, f"cached_grid_v2_{img_size[0]}.npy.gz")
    f = None
    if os.path.exists(cache_path):
        try:
            f = pgzip.PgzipFile(cache_path, "r")
            ret = np.load(f, allow_pickle=True)
            f.close()
            return ret
        except Exception as e:
            print(dir_path, "\n", e)
            if f:
                f.close()
            os.remove(cache_path)

    pcd_overall = read_study_as_pcd(dir_path,
                                    series_types_dict=series_type_dict,
                                    downsampling_factor=downsampling_factor,
                                    img_size=img_size,
                                    stack_slices_thickness=True)
    box = pcd_overall.get_axis_aligned_bounding_box()

    max_b = np.array(box.get_max_bound())
    min_b = np.array(box.get_min_bound())

    pts = (np.array(pcd_overall.points) - (min_b)) * (
            (img_size[0] - 1, img_size[0] - 1, img_size[0] - 1) / (max_b - min_b))
    coords = np.round(pts).astype(np.int32)
    vals = np.array(pcd_overall.colors, dtype=np.float16)

    grid = np.zeros((3, img_size[0], img_size[0], img_size[0]), dtype=np.float16)
    indices = coords[:, 0], coords[:, 1], coords[:, 2]

    np.maximum.at(grid[0], indices, vals[:, 0])
    np.maximum.at(grid[1], indices, vals[:, 1])
    np.maximum.at(grid[2], indices, vals[:, 2])

    f = pgzip.PgzipFile(cache_path, "w")
    np.save(f, grid)
    f.close()

    return grid


def read_vertebral_level_as_voxel_grid(dir_path,
                                       vertebral_level: str,
                                       max_bound: np.array,
                                       min_bound: np.array,
                                       pcd_overall: o3d.geometry.PointCloud = None,
                                       series_type_dict=None,
                                       downsampling_factor=1,
                                       voxel_size=(128, 128, 32),
                                       caching=True,
                                       ):
    cache_path = os.path.join(dir_path,
                              f"cached_grid_{vertebral_level}_{voxel_size[0]}_{voxel_size[1]}_{voxel_size[2]}.npy.gz")
    f = None
    if caching and os.path.exists(cache_path):
        try:
            f = pgzip.PgzipFile(cache_path, "r")
            ret = np.load(f, allow_pickle=True)
            f.close()
            return ret
        except Exception as e:
            print(dir_path, "\n", e)
            if f:
                f.close()
            os.remove(cache_path)

    if pcd_overall is None:
        pcd_overall = read_study_as_pcd(dir_path,
                                        series_types_dict=series_type_dict,
                                        downsampling_factor=downsampling_factor,
                                        img_size=(voxel_size[0], voxel_size[1]),
                                        stack_slices_thickness=True,
                                        resize_slices=False)

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    pcd_level = pcd_overall.crop(bbox)

    bbox = pcd_level.get_axis_aligned_bounding_box()

    max_b = np.array(bbox.get_max_bound())
    min_b = np.array(bbox.get_min_bound())

    pts = (np.array(pcd_level.points) - min_b) * (
            (voxel_size[0] - 1, voxel_size[1] - 1, voxel_size[2] - 1) / (max_b - min_b))
    coords = np.round(pts).astype(np.int32)
    vals = np.array(pcd_level.colors, dtype=np.float16)

    grid = np.zeros((3, voxel_size[0], voxel_size[1], voxel_size[2]), dtype=np.float16)
    indices = coords[:, 0], coords[:, 1], coords[:, 2]

    np.maximum.at(grid[0], indices, vals[:, 0])
    np.maximum.at(grid[1], indices, vals[:, 1])
    np.maximum.at(grid[2], indices, vals[:, 2])

    if caching:
        f = pgzip.PgzipFile(cache_path, "w")
        np.save(f, grid)
        f.close()

    return grid


def read_vertebral_level_as_voxel_grid_alt(dir_path,
                                           vertebral_level: str,
                                           max_bound: np.array,
                                           min_bound: np.array,
                                           pcd_overall: o3d.geometry.PointCloud = None,
                                           series_type_dict=None,
                                           downsampling_factor=1,
                                           voxel_size=(128, 128, 32),
                                           caching=True,
                                           ):
    cache_path = os.path.join(dir_path,
                              f"cached_grid_alt_{vertebral_level}_{voxel_size[0]}_{voxel_size[1]}_{voxel_size[2]}.npy.gz")
    f = None
    if caching and os.path.exists(cache_path):
        try:
            f = pgzip.PgzipFile(cache_path, "r")
            ret = np.load(f, allow_pickle=True)
            f.close()
            return ret
        except Exception as e:
            print(dir_path, "\n", e)
            if f:
                f.close()
            os.remove(cache_path)

    resize = tio.Resize(voxel_size)

    if pcd_overall is None:
        pcd_overall = read_study_as_pcd(dir_path,
                                        series_types_dict=series_type_dict,
                                        downsampling_factor=downsampling_factor,
                                        img_size=(voxel_size[0], voxel_size[1]),
                                        stack_slices_thickness=True,
                                        resize_slices=False)

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    pcd_level = pcd_overall.crop(bbox)

    size = 1
    voxel_level = o3d.geometry.VoxelGrid().create_from_point_cloud(pcd_level, size,
                                                                   color_mode=o3d.geometry.VoxelGrid.VoxelColorMode.MAX)

    coords = np.array([voxel.grid_index for voxel in voxel_level.get_voxels()])
    vals = np.array([voxel.color for voxel in voxel_level.get_voxels()], dtype=np.float16)

    size = np.max(coords, axis=0) + 1
    grid = np.zeros((3, size[0], size[1], size[2]), dtype=np.float32)
    indices = coords[:, 0], coords[:, 1], coords[:, 2]

    for i in range(3):
        grid[i][indices] = vals[:, i]

    grid = resize(grid)

    if caching:
        f = pgzip.PgzipFile(cache_path, "w")
        np.save(f, grid)
        f.close()

    return grid


def read_vertebral_level_as_voxel_grid_nonaligned(dir_path,
                                             vertebral_level: str,
                                             center_point_pair: tuple,
                                             max_bound: np.array,
                                             min_bound: np.array,
                                             pcd_overall: o3d.geometry.PointCloud = None,
                                             series_type_dict=None,
                                             downsampling_factor=1,
                                             voxel_size=(128, 128, 32),
                                             caching=True,
                                             ):
    cache_path = os.path.join(dir_path,
                              f"cached_grid_nonaligned_{vertebral_level}_{voxel_size[0]}_{voxel_size[1]}_{voxel_size[2]}.npy.gz")
    f = None
    if caching and os.path.exists(cache_path):
        try:
            f = pgzip.PgzipFile(cache_path, "r")
            ret = np.load(f, allow_pickle=True)
            f.close()
            return ret
        except Exception as e:
            print(cache_path, "\n", e)
            if f:
                f.close()
            os.remove(cache_path)

    resize = tio.Resize(voxel_size)

    if pcd_overall is None:
        pcd_overall = read_study_as_pcd(dir_path,
                          series_types_dict=series_type_dict,
                          downsampling_factor=downsampling_factor,
                          img_size=(voxel_size[0], voxel_size[2]),
                          stack_slices_thickness=True,
                          resize_slices=False)

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    pcd_overall = pcd_overall.crop(bbox)

    pts = np.array(pcd_overall.points)
    vals = np.array(pcd_overall.colors)

    dist_vec = center_point_pair[1] - center_point_pair[0]

    plane_1 = np.sum((pts - center_point_pair[0]) * dist_vec, axis=-1)
    plane_2 = np.sum((pts - center_point_pair[1]) * dist_vec, axis=-1)

    in_bounds = (plane_1 > 0) & (plane_2 < 0)

    pts_level = pts[in_bounds]
    vals_level = vals[in_bounds]

    pcd_level = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_level))
    pcd_level.colors = o3d.utility.Vector3dVector(vals_level)

    size = 1
    voxel_level = o3d.geometry.VoxelGrid().create_from_point_cloud(pcd_level, size,
                                                                   color_mode=o3d.geometry.VoxelGrid.VoxelColorMode.MAX)

    coords = np.array([voxel.grid_index for voxel in voxel_level.get_voxels()])
    vals = np.array([voxel.color for voxel in voxel_level.get_voxels()], dtype=np.float16)

    size = np.max(coords, axis=0) + 1
    grid = np.zeros((3, size[0], size[1], size[2]), dtype=np.float32)
    indices = coords[:, 0], coords[:, 1], coords[:, 2]

    for i in range(3):
        grid[i][indices] = vals[:, i]

    grid = resize(grid)

    if caching:
        f = pgzip.PgzipFile(cache_path, "w")
        np.save(f, grid)
        f.close()

    return grid


def read_vertebral_level_as_voxel_grid_aligned(dir_path,
                                             vertebral_level: str,
                                             center_point_pair: tuple,
                                             max_bound: np.array,
                                             min_bound: np.array,
                                             pcd_overall: o3d.geometry.PointCloud = None,
                                             series_type_dict=None,
                                             downsampling_factor=1,
                                             voxel_size=(128, 128, 32),
                                             caching=True,
                                             ):
    cache_path = os.path.join(dir_path,
                              f"cached_grid_aligned_{vertebral_level}_{voxel_size[0]}_{voxel_size[1]}_{voxel_size[2]}.npy.gz")
    f = None
    if caching and os.path.exists(cache_path):
        try:
            f = pgzip.PgzipFile(cache_path, "r")
            ret = np.load(f, allow_pickle=True)
            f.close()
            return ret
        except Exception as e:
            print(cache_path, "\n", e)
            if f:
                f.close()
            os.remove(cache_path)

    resize = tio.Resize(voxel_size)

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    if pcd_overall is None:
        pcd_overall = read_study_as_pcd(dir_path,
                          series_types_dict=series_type_dict,
                          downsampling_factor=downsampling_factor,
                          img_size=(voxel_size[0], voxel_size[2]),
                          stack_slices_thickness=True,
                          resize_slices=False)

    pcd_overall = pcd_overall.crop(bbox)

    pts = np.array(pcd_overall.points)
    vals = np.array(pcd_overall.colors)

    dist_vec = center_point_pair[1] - center_point_pair[0]

    plane_1 = np.sum((pts - center_point_pair[0]) * dist_vec, axis=-1)
    plane_2 = np.sum((pts - center_point_pair[1]) * dist_vec, axis=-1)

    in_bounds = (plane_1 > 0) & (plane_2 < 0)

    pts_level = pts[in_bounds]
    vals_level = vals[in_bounds]

    pcd_level = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_level))
    pcd_level = pcd_level.rotate(rotation_matrix_from_vectors(dist_vec))

    pcd_level.colors = o3d.utility.Vector3dVector(vals_level)

    size = 1
    voxel_level = o3d.geometry.VoxelGrid().create_from_point_cloud(pcd_level, size,
                                                                   color_mode=o3d.geometry.VoxelGrid.VoxelColorMode.MAX)

    coords = np.array([voxel.grid_index for voxel in voxel_level.get_voxels()])
    vals = np.array([voxel.color for voxel in voxel_level.get_voxels()], dtype=np.float16)

    size = np.max(coords, axis=0) + 1
    grid = np.zeros((3, size[0], size[1], size[2]), dtype=np.float32)
    indices = coords[:, 0], coords[:, 1], coords[:, 2]

    for i in range(3):
        grid[i][indices] = vals[:, i]

    grid = resize(grid)

    if caching:
        f = pgzip.PgzipFile(cache_path, "w")
        np.save(f, grid)
        f.close()

    return grid


def read_vertebral_levels_as_voxel_grids(dir_path,
                                         vertebral_levels: list[str],
                                         max_bounds: list[np.array],
                                         min_bounds: list[np.array],
                                         pcd_overall: o3d.geometry.PointCloud = None,
                                         series_type_dict=None,
                                         downsampling_factor=1,
                                         voxel_size=(128, 128, 42)):
    ret = {}

    for index, vertebral_level in enumerate(vertebral_levels):
        cache_path = os.path.join(dir_path,
                                  f"cached_grid_{vertebral_level}_{voxel_size[0]}_{voxel_size[1]}_{voxel_size[2]}.npy.gz")
        f = None
        if os.path.exists(cache_path):
            try:
                f = pgzip.PgzipFile(cache_path, "r")
                ret[vertebral_level] = np.load(f, allow_pickle=True)
                f.close()
            except Exception as e:
                print(dir_path, "\n", e)
                if f:
                    f.close()
                os.remove(cache_path)

        else:
            if pcd_overall is None:
                pcd_overall = read_study_as_pcd(dir_path,
                                                series_types_dict=series_type_dict,
                                                downsampling_factor=downsampling_factor,
                                                img_size=(voxel_size[0], voxel_size[2]),
                                                stack_slices_thickness=True,
                                                resize_slices=False)

            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bounds[index], max_bound=max_bounds[index])
            pcd_level = pcd_overall.crop(bbox)

            box = pcd_level.get_axis_aligned_bounding_box()

            max_b = np.array(box.get_max_bound())
            min_b = np.array(box.get_min_bound())

            pts = (np.array(pcd_level.points) - (min_b)) * (
                    (voxel_size[0] - 1, voxel_size[1] - 1, voxel_size[2] - 1) / (max_b - min_b))
            coords = np.round(pts).astype(np.int32)
            vals = np.array(pcd_level.colors, dtype=np.float16)

            grid = np.zeros((3, voxel_size[0], voxel_size[1], voxel_size[2]), dtype=np.float16)
            indices = coords[:, 0], coords[:, 1], coords[:, 2]

            for i in range(3):
                grid[i][indices] = vals[:, i]

            f = pgzip.PgzipFile(cache_path, "w")
            np.save(f, grid)
            f.close()

            ret[vertebral_level] = grid

    return ret


def read_vertebral_levels_as_voxel_grids_alt(dir_path,
                                             vertebral_levels: list[str],
                                             max_bounds: list[np.array],
                                             min_bounds: list[np.array],
                                             pcd_overall: o3d.geometry.PointCloud = None,
                                             series_type_dict=None,
                                             downsampling_factor=1,
                                             voxel_size=(128, 128, 42)):
    ret = {}

    resize = tio.Resize(voxel_size)

    for index, vertebral_level in enumerate(vertebral_levels):
        cache_path = os.path.join(dir_path,
                                  f"cached_grid_alt_{vertebral_level}_{voxel_size[0]}_{voxel_size[1]}_{voxel_size[2]}.npy.gz")
        f = None
        if os.path.exists(cache_path):
            try:
                f = pgzip.PgzipFile(cache_path, "r")
                ret[vertebral_level] = np.load(f, allow_pickle=True)
                f.close()
            except Exception as e:
                print(dir_path, "\n", e)
                if f:
                    f.close()
                os.remove(cache_path)

        else:
            if pcd_overall is None:
                pcd_overall = read_study_as_pcd(dir_path,
                                                series_types_dict=series_type_dict,
                                                downsampling_factor=downsampling_factor,
                                                img_size=(voxel_size[0], voxel_size[2]),
                                                stack_slices_thickness=True,
                                                resize_slices=False)

            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bounds[index], max_bound=max_bounds[index])
            pcd_level = pcd_overall.crop(bbox)

            size = 1
            voxel_level = o3d.geometry.VoxelGrid().create_from_point_cloud(pcd_level, size,
                                                                           color_mode=o3d.geometry.VoxelGrid.VoxelColorMode.MAX)

            coords = np.array([voxel.grid_index for voxel in voxel_level.get_voxels()])
            vals = np.array([voxel.color for voxel in voxel_level.get_voxels()], dtype=np.float16)

            size = np.max(coords, axis=0) + 1
            grid = np.zeros((3, size[0], size[1], size[2]), dtype=np.float32)
            indices = coords[:, 0], coords[:, 1], coords[:, 2]

            for i in range(3):
                grid[i][indices] = vals[:, i]

            grid = resize(grid)

            f = pgzip.PgzipFile(cache_path, "w")
            np.save(f, grid)
            f.close()

            ret[vertebral_level] = grid

    return ret


def read_vertebral_levels_as_voxel_grids_nonaligned(dir_path,
                                               vertebral_levels: list[str],
                                               max_bounds: list[np.array],
                                               min_bounds: list[np.array],
                                               center_point_pairs: list[tuple[np.array, np.array]],
                                               pcd_overall: o3d.geometry.PointCloud = None,
                                               series_type_dict=None,
                                               downsampling_factor=1,
                                               voxel_size=(128, 128, 42)):
    ret = {}

    resize = tio.Resize(voxel_size)

    for index, vertebral_level in enumerate(vertebral_levels):
        cache_path = os.path.join(dir_path,
                                  f"cached_grid_nonaligned_{vertebral_level}_{voxel_size[0]}_{voxel_size[1]}_{voxel_size[2]}.npy.gz")
        f = None
        if os.path.exists(cache_path):
            try:
                f = pgzip.PgzipFile(cache_path, "r")
                ret[vertebral_level] = np.load(f, allow_pickle=True)
                f.close()
            except Exception as e:
                print(dir_path, "\n", e)
                if f:
                    f.close()
                os.remove(cache_path)

        else:
            if pcd_overall is None:
                pcd_overall = read_study_as_pcd(dir_path,
                                                series_types_dict=series_type_dict,
                                                downsampling_factor=downsampling_factor,
                                                img_size=(voxel_size[0], voxel_size[2]),
                                                stack_slices_thickness=True,
                                                resize_slices=False)

            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bounds[index], max_bound=max_bounds[index])
            pcd_level = pcd_overall.crop(bbox)

            pts = np.array(pcd_level.points)
            vals = np.array(pcd_level.colors)

            dist_vec = center_point_pairs[index][1] - center_point_pairs[index][0]

            plane_1 = np.sum((pts - center_point_pairs[index][0]) * dist_vec, axis=-1)
            plane_2 = np.sum((pts - center_point_pairs[index][1]) * dist_vec, axis=-1)

            in_bounds = (plane_1 > 0) & (plane_2 < 0)

            pts_level = pts[in_bounds]
            vals_level = vals[in_bounds]

            pcd_level = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_level))
            pcd_level.colors = o3d.utility.Vector3dVector(vals_level)

            size = 1
            voxel_level = o3d.geometry.VoxelGrid().create_from_point_cloud(pcd_level, size,
                                                                           color_mode=o3d.geometry.VoxelGrid.VoxelColorMode.MAX)

            coords = np.array([voxel.grid_index for voxel in voxel_level.get_voxels()])
            vals = np.array([voxel.color for voxel in voxel_level.get_voxels()], dtype=np.float16)

            size = np.max(coords, axis=0) + 1
            grid = np.zeros((3, size[0], size[1], size[2]), dtype=np.float32)
            indices = coords[:, 0], coords[:, 1], coords[:, 2]

            for i in range(3):
                grid[i][indices] = vals[:, i]

            grid = resize(grid)

            f = pgzip.PgzipFile(cache_path, "w")
            np.save(f, grid)
            f.close()

            ret[vertebral_level] = grid

    return ret


def read_vertebral_levels_as_voxel_grids_aligned(dir_path,
                                               vertebral_levels: list[str],
                                               max_bounds: list[np.array],
                                               min_bounds: list[np.array],
                                               center_point_pairs: list[tuple[np.array, np.array]],
                                               pcd_overall: o3d.geometry.PointCloud = None,
                                               series_type_dict=None,
                                               downsampling_factor=1,
                                               voxel_size=(128, 128, 42)):
    ret = {}

    resize = tio.Resize(voxel_size, image_interpolation="bspline")

    for index, vertebral_level in enumerate(vertebral_levels):
        cache_path = os.path.join(dir_path,
                                  f"cached_grid_aligned_{vertebral_level}_{voxel_size[0]}_{voxel_size[1]}_{voxel_size[2]}.npy.gz")
        f = None
        if os.path.exists(cache_path):
            try:
                f = pgzip.PgzipFile(cache_path, "r")
                ret[vertebral_level] = np.load(f, allow_pickle=True)
                f.close()
            except Exception as e:
                print(dir_path, "\n", e)
                if f:
                    f.close()
                os.remove(cache_path)

        else:
            if pcd_overall is None:
                pcd_overall = read_study_as_pcd(dir_path,
                                                series_types_dict=series_type_dict,
                                                downsampling_factor=downsampling_factor,
                                                img_size=(voxel_size[0], voxel_size[2]),
                                                stack_slices_thickness=True,
                                                resize_slices=False)

            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bounds[index], max_bound=max_bounds[index])
            pcd_level = pcd_overall.crop(bbox)

            pts = np.array(pcd_level.points)
            vals = np.array(pcd_level.colors)

            dist_vec = center_point_pairs[index][1] - center_point_pairs[index][0]

            plane_1 = np.sum((pts - center_point_pairs[index][0]) * dist_vec, axis=-1)
            plane_2 = np.sum((pts - center_point_pairs[index][1]) * dist_vec, axis=-1)

            in_bounds = (plane_1 > 0) & (plane_2 < 0)

            pts_level = pts[in_bounds]
            vals_level = vals[in_bounds]

            pcd_level = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_level))
            pcd_level = pcd_level.rotate(rotation_matrix_from_vectors(dist_vec))

            pcd_level.colors = o3d.utility.Vector3dVector(vals_level)

            size = 1
            voxel_level = o3d.geometry.VoxelGrid().create_from_point_cloud(pcd_level, size,
                                                                           color_mode=o3d.geometry.VoxelGrid.VoxelColorMode.MAX)

            coords = np.array([voxel.grid_index for voxel in voxel_level.get_voxels()])
            vals = np.array([voxel.color for voxel in voxel_level.get_voxels()], dtype=np.float16)

            size = np.max(coords, axis=0) + 1
            grid = np.zeros((3, size[0], size[1], size[2]), dtype=np.float32)
            indices = coords[:, 0], coords[:, 1], coords[:, 2]

            for i in range(3):
                grid[i][indices] = vals[:, i]

            grid = resize(grid)

            f = pgzip.PgzipFile(cache_path, "w")
            np.save(f, grid)
            f.close()

            ret[vertebral_level] = grid

    return ret


def retrieve_training_data(train_path):
    # !TODO: refactor
    def reshape_row(row):
        data = {col: [] for col in row.axes[0] if col in
                ['study_id', 'series_id', 'instance_number', 'x', 'y', 'series_description', 'image_paths']}
        data["level"] = []
        data["condition"] = []
        data["severity"] = []

        for column, value in row.items():
            if column not in ['study_id', 'series_id', 'instance_number', 'x', 'y', 'series_description',
                              'image_paths']:
                parts = column.split('_')
                condition = ' '.join([word.capitalize() for word in parts[:-2]])
                level = parts[-2].capitalize() + '/' + parts[-1].capitalize()
                data['condition'].append(condition)
                data['level'].append(level)
                data['severity'].append(value)
            else:
                # !TODO: Seriously, refactor
                for i in range(25):
                    data[column].append(value)

        return pd.DataFrame(data)

    train = pd.read_csv(train_path + 'train.csv')
    train_desc = pd.read_csv(train_path + 'train_series_descriptions.csv')

    train_df = pd.merge(train, train_desc, on="study_id")

    train_df = pd.concat([reshape_row(row) for _, row in train_df.iterrows()], ignore_index=True)
    train_df['severity'] = train_df['severity'].map(
        {'Normal/Mild': 'normal_mild', 'Moderate': 'moderate', 'Severe': 'severe'})

    return train_df


def retrieve_coordinate_training_data(train_path):
    def reshape_row(row):
        data = {'study_id': [], 'condition': [], 'level': [], 'severity': []}

        for column, value in row.items():
            if column not in ['study_id', 'series_id', 'instance_number', 'x', 'y', 'series_description']:
                parts = column.split('_')
                condition = ' '.join([word.capitalize() for word in parts[:-2]])
                level = parts[-2].capitalize() + '/' + parts[-1].capitalize()
                data['study_id'].append(row['study_id'])
                data['condition'].append(condition)
                data['level'].append(level)
                data['severity'].append(value)

        return pd.DataFrame(data)

    train = pd.read_csv(train_path + 'train.csv')
    label = pd.read_csv(train_path + 'train_label_coordinates.csv')
    train_desc = pd.read_csv(train_path + 'train_series_descriptions.csv')
    test_desc = pd.read_csv(train_path + 'test_series_descriptions.csv')
    sub = pd.read_csv(train_path + 'sample_submission.csv')

    new_train_df = pd.concat([reshape_row(row) for _, row in train.iterrows()], ignore_index=True)
    merged_df = pd.merge(new_train_df, label, on=['study_id', 'condition', 'level'], how='inner')
    final_merged_df = pd.merge(merged_df, train_desc, on=['series_id', 'study_id'], how='inner')
    final_merged_df['severity'] = final_merged_df['severity'].map(
        {'Normal/Mild': 'normal_mild', 'Moderate': 'moderate', 'Severe': 'severe'})

    final_merged_df['row_id'] = (
            final_merged_df['study_id'].astype(str) + '_' +
            final_merged_df['condition'].str.lower().str.replace(' ', '_') + '_' +
            final_merged_df['level'].str.lower().str.replace('/', '_')
    )

    # Create the image_path column
    final_merged_df['image_path'] = (
            f'{train_path}/train_images/' +
            final_merged_df['study_id'].astype(str) + '/' +
            final_merged_df['series_id'].astype(str) + '/' +
            final_merged_df['instance_number'].astype(str) + '.dcm'
    )

    return final_merged_df
