import os
import math
import time
import cv2
import copy
import glob
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pydicom import dcmread
import pgzip
import open3d as o3d
import torchio as tio
import torch.nn.functional as F

import sys

sys.path.insert(0, "../../SpineNet")
import spinenet
from spinenet import SpineNet, download_example_scan
from spinenet.io import load_dicoms_from_folder

from config import Config


def retrieve_training_data(train_path) -> pd.DataFrame:
    def reshape_row(row):
        data = {"study_id": [], "condition": [], "level": [], "severity": []}

        for column, value in row.items():
            if column not in [
                "study_id",
                "series_id",
                "instance_number",
                "x",
                "y",
                "series_description",
            ]:
                parts = column.split("_")
                condition = " ".join([word.capitalize() for word in parts[:-2]])
                level = parts[-2].capitalize() + "/" + parts[-1].capitalize()
                data["study_id"].append(row["study_id"])
                data["condition"].append(condition)
                data["level"].append(level)
                data["severity"].append(value)

        return pd.DataFrame(data)

    train = pd.read_csv(train_path + "train.csv")
    label = pd.read_csv(train_path + "train_label_coordinates.csv")
    train_desc = pd.read_csv(train_path + "train_series_descriptions.csv")

    new_train_df = pd.concat(
        [reshape_row(row) for _, row in train.iterrows()], ignore_index=True
    )
    merged_df = pd.merge(
        new_train_df, label, on=["study_id", "condition", "level"], how="inner"
    )
    final_merged_df = pd.merge(
        merged_df, train_desc, on=["series_id", "study_id"], how="inner"
    )
    final_merged_df["severity"] = final_merged_df["severity"].map(
        {"Normal/Mild": "normal_mild", "Moderate": "moderate", "Severe": "severe"}
    )

    final_merged_df["row_id"] = (
        final_merged_df["study_id"].astype(str)
        + "_"
        + final_merged_df["condition"].str.lower().str.replace(" ", "_")
        + "_"
        + final_merged_df["level"].str.lower().str.replace("/", "_")
    )

    # Create the image_path column
    final_merged_df["image_path"] = (
        f"{train_path}train_images/"
        + final_merged_df["study_id"].astype(str)
        + "/"
        + final_merged_df["series_id"].astype(str)
        + "/"
        + final_merged_df["instance_number"].astype(str)
        + ".dcm"
    )

    final_merged_df = final_merged_df.dropna()

    filtered_df = pd.DataFrame(columns=final_merged_df.columns)
    for series_desc in Config.CONDITIONS.keys():
        subset = final_merged_df[final_merged_df["series_description"] == series_desc]
        if series_desc == "Sagittal T2/STIR":
            subset = subset[subset.groupby(["study_id"]).transform("size") == 5]
        else:
            subset = subset[subset.groupby(["study_id"]).transform("size") == 10]
        filtered_df = pd.concat([filtered_df, subset])

    filtered_df = filtered_df[filtered_df.groupby(["study_id"]).transform("size") == 25]

    return filtered_df


def convert_coords_to_patient(x, y, dicom_slice):

    dX, dY = dicom_slice.PixelSpacing

    X = np.array(list(dicom_slice.ImageOrientationPatient[:3]) + [0]) * dY
    Y = np.array(list(dicom_slice.ImageOrientationPatient[3:]) + [0]) * dX

    S = np.array(list(dicom_slice.ImagePositionPatient) + [1])

    transform_matrix = np.array([Y, X, np.zeros(len(X)), S]).T

    return transform_matrix @ np.array([y, x, 0, 1]).T


def calc_center_per_study(train_descs_df) -> pd.DataFrame:

    patient_coords_path = Config.df_basepath + "patient_coords.csv"

    if os.path.exists(patient_coords_path):
        return pd.read_csv(patient_coords_path)

    spnt = SpineNet(device="cuda:0", verbose=True)

    def calculate_centers(data):
        centers = {}
        for item in data:
            level = item["predicted_label"]
            if level in Config.LEVELS:
                average_polygon = item["average_polygon"]
                centroid_x = np.mean(average_polygon[:, 0])
                centroid_y = np.mean(average_polygon[:, 1])
                centroid_z = item["slice_nos"][len(item["slice_nos"]) // 2]
                centers[level] = (centroid_x, centroid_y, centroid_z)
        return centers

    centers_per_study = {
        "study_id": [],
        "series_id": [],
        "x": [],
        "y": [],
        "instance_number": [],
        "level": [],
    }

    for index, row in tqdm(train_descs_df.iterrows(), total=train_descs_df.shape[0]):
        scan = load_dicoms_from_folder(
            f"{Config.data_basepath}train_images/{row['study_id']}/{row['series_id']}",
            require_extensions=False,
        )
        num_slices = scan.volume.shape[-1]

        vert_dicts = spnt.detect_vb(scan.volume, scan.pixel_spacing)
        centers = calculate_centers(vert_dicts)

        for level in centers:
            centers_per_study["study_id"].append(row["study_id"])
            centers_per_study["series_id"].append(row["series_id"])
            centers_per_study["level"].append(level)

            centers_per_study["x"].append(centers[level][0])
            centers_per_study["y"].append(centers[level][1])
            centers_per_study["instance_number"].append(centers[level][2])

    centers_per_study = pd.DataFrame.from_dict(centers_per_study)

    patient_coords_dict = {
        "study_id": [],
        "level": [],
        "x": [],
        "y": [],
        "instance_number": [],
        "middle_slice_path": [],
        "x_patient": [],
        "y_patient": [],
        "z_patient": [],
    }

    for index, group in centers_per_study.groupby("study_id"):
        for row_index, row in group.iterrows():
            dicom_slice_path = f"{Config.data_basepath}train_images/{row['study_id']}/{row['series_id']}/{row['instance_number']}.dcm"
            dicom_slice = dcmread(dicom_slice_path)
            coords = convert_coords_to_patient(row["x"], row["y"], dicom_slice)

            patient_coords_dict["study_id"].append(row["study_id"])
            patient_coords_dict["level"].append(row["level"])
            patient_coords_dict["x"].append(row["x"])
            patient_coords_dict["y"].append(row["y"])
            patient_coords_dict["instance_number"].append(row["instance_number"])
            patient_coords_dict["x_patient"].append(coords[0])
            patient_coords_dict["y_patient"].append(coords[1])
            patient_coords_dict["z_patient"].append(coords[2])
            patient_coords_dict["middle_slice_path"].append(dicom_slice_path)

    patient_coords_df = pd.DataFrame.from_dict(patient_coords_dict)
    patient_coords_df.to_csv(patient_coords_path)

    return patient_coords_df


def get_normal_vector_from_dicom(dicom_file_path):
    dicom_data = dcmread(dicom_file_path)
    orientation = dicom_data.ImageOrientationPatient
    if len(orientation) != 6:
        raise ValueError("Invalid Image Orientation data.")

    row_vector = np.array(orientation[:3])
    col_vector = np.array(orientation[3:])

    normal_vector = np.cross(row_vector, col_vector)

    norm = np.linalg.norm(normal_vector)
    if norm == 0:
        raise ValueError("Normal vector has zero length; cannot normalize.")

    normalized_normal_vector = normal_vector / norm

    return normalized_normal_vector


def calc_bounding_boxes_3d(patient_coords_df):
    bb_path = Config.df_basepath + "bb_3d.csv"
    if os.path.exists(bb_path):
        return pd.read_csv(bb_path)

    patient_bounding_boxes_dict = {
        "study_id": [],
        "level": [],
        "x_min": [],
        "y_min": [],
        "z_min": [],
        "x_max": [],
        "y_max": [],
        "z_max": [],
    }
    for index, group in patient_coords_df.groupby("study_id"):
        ordered_group = group.sort_values(by="level", ascending=True)
        print("new study")
        if len(ordered_group) != 6:
            continue
        for level_index in range(5):
            patient_bounding_boxes_dict["study_id"].append(
                ordered_group["study_id"].iloc[0]
            )
            level_label = (
                ordered_group["level"].iloc[level_index].lower()
                + "_"
                + ordered_group["level"].iloc[level_index + 1]
            )
            patient_bounding_boxes_dict["level"].append(level_label)

            # Middle vertebra points
            pt_0 = np.array(
                ordered_group.iloc[level_index][
                    ["x_patient", "y_patient", "z_patient"]
                ],
                dtype=float,
            )
            pt_1 = np.array(
                ordered_group.iloc[level_index + 1][
                    ["x_patient", "y_patient", "z_patient"]
                ],
                dtype=float,
            )
            path_0 = ordered_group.iloc[level_index + 1]["middle_slice_path"]
            path_1 = ordered_group.iloc[level_index + 1]["middle_slice_path"]

            d = np.linalg.norm(pt_0 - pt_1)
            z = np.array(pt_0 - pt_1) / d
            print("z : ", z)
            x = (
                get_normal_vector_from_dicom(path_0)
                + get_normal_vector_from_dicom(path_1)
            ) / 2
            y = np.cross(z, x)
            c = (pt_0 + pt_1) / 2
            min_bound = c - d * (1.7*x + 2*y + z/2) 
            max_bound = c + d * (1.7*x + y + z/2) 

            patient_bounding_boxes_dict["x_min"].append(min_bound[0])
            patient_bounding_boxes_dict["y_min"].append(min_bound[1])
            patient_bounding_boxes_dict["z_min"].append(min_bound[2])
            patient_bounding_boxes_dict["x_max"].append(max_bound[0])
            patient_bounding_boxes_dict["y_max"].append(max_bound[1])
            patient_bounding_boxes_dict["z_max"].append(max_bound[2])

            # # Distance vector to the next vertebra
            # d_vec = np.array(pt_0 - pt_1)
            # d_size = np.linalg.norm(d_vec)
            # d_unit = d_vec / d_size

            # # Get a pair of orthogonal vectors to find x and y boundary candidates
            # orth_1 = np.random.randn(3).astype(np.float64)
            # orth_1 = orth_1 - orth_1.dot(d_unit) * d_unit
            # orth_1 = orth_1 / np.linalg.norm(orth_1)

            # orth_1 = orth_1.astype(np.float64)
            # d_unit = d_unit.astype(np.float64)

            # orth_2 = np.cross(orth_1, d_unit)
            # orth_2 = orth_2.astype(np.float64)

            # orth_1 *= d_size
            # orth_2 *= d_size

            # # Get candidate points (10 of them, 2 per orthogonal per each vertebra center, and the centers themselves)
            # c_pts = np.array(
            #     [pt - vec for pt in (pt_0, pt_1) for vec in (orth_1, orth_2)]
            #     + [pt + vec for pt in (pt_0, pt_1) for vec in (orth_1, orth_2)]
            #     + [pt_0, pt_1]
            # )

            # # x_min and x_max are just the min and max from all this
            # x_min = np.min(c_pts[:, 0])
            # x_max = np.max(c_pts[:, 0])

            # # y_max is going to be over the center ys
            # # And we're going to get y_min by getting y_min over c_pts and then extending the y_min over center ys
            # c_pts_y_min = np.min(c_pts[:, 1])
            # c_pts_y_max = np.max(c_pts[:, 1])

            # y_max = max(pt_0[1], pt_1[1])
            # y_min = min(pt_0[1], pt_1[1])

            # y_max += abs(c_pts_y_max - y_max) * 2
            # y_min -= abs(c_pts_y_min - y_min) / 2

            # # z_max and z_min will be the same as x_min and x_max
            # z_min = np.min(c_pts[:, 2])
            # z_max = np.max(c_pts[:, 2])

            # patient_bounding_boxes_dict["x_min"].append(x_min)
            # patient_bounding_boxes_dict["y_min"].append(y_min)
            # patient_bounding_boxes_dict["z_min"].append(z_min)
            # patient_bounding_boxes_dict["x_max"].append(x_max)
            # patient_bounding_boxes_dict["y_max"].append(y_max)
            # patient_bounding_boxes_dict["z_max"].append(z_max)

    bb_3d_df = pd.DataFrame.from_dict(patient_bounding_boxes_dict)
    bb_3d_df.to_csv(bb_path)
    return bb_3d_df


def read_study_as_pcd(
    dir_path,
    series_types_dict=None,
    downsampling_factor=1,
    resize_slices=True,
    resize_method="nearest",
    stack_slices_thickness=True,
    img_size=(256, 256),
):
    pcd_overall = o3d.geometry.PointCloud()

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
                img = np.expand_dims(
                    cv2.resize(
                        dicom_slice.pixel_array, img_size, interpolation=cv2.INTER_AREA
                    ),
                    -1,
                )
            elif resize_method == "maxpool":
                img_tensor = torch.tensor(dicom_slice.pixel_array).float()
                img = F.adaptive_max_pool2d(img_tensor.unsqueeze(0), img_size).numpy()
            else:
                raise ValueError(f"Invalid resize_method {resize_method}")
        else:
            img = np.expand_dims(np.array(dicom_slice.pixel_array), -1)
        x, y, z = np.where(img)

        downsampling_factor_iter = max(
            downsampling_factor, int(math.ceil(len(x) / 6e6))
        )

        index_voxel = np.vstack((x, y, z))[:, ::downsampling_factor_iter]
        grid_index_array = index_voxel.T
        pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(grid_index_array.astype(np.float64))
        )

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
                [
                    [0, y_orig / img_size[1], 0, 0],
                    [x_orig / img_size[0], 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
        else:
            transform_matrix_factor = np.matrix(
                [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            )

        dX, dY = dicom_slice.PixelSpacing
        dZ = dicom_slice.SliceThickness

        X = np.array(list(dicom_slice.ImageOrientationPatient[:3]) + [0]) * dX
        Y = np.array(list(dicom_slice.ImageOrientationPatient[3:]) + [0]) * dY

        S = np.array(list(dicom_slice.ImagePositionPatient) + [1])

        transform_matrix = np.array([X, Y, np.zeros(len(X)), S]).T
        transform_matrix = transform_matrix @ transform_matrix_factor

        if stack_slices_thickness:
            for z in range(int(dZ)):
                pos = list(dicom_slice.ImagePositionPatient)
                if series_desc == "T2":
                    pos[-1] += z
                else:
                    pos[0] += z
                S = np.array(pos + [1])

                transform_matrix = np.array([X, Y, np.zeros(len(X)), S]).T
                transform_matrix = transform_matrix @ transform_matrix_factor

                pcd_overall += copy.deepcopy(pcd).transform(transform_matrix)

        else:
            pcd_overall += copy.deepcopy(pcd).transform(transform_matrix)

    return pcd_overall


def read_vertebral_levels_as_voxel_grids(
    dir_path,
    study_id,
    vertebral_levels: list[str],
    max_bounds: list[np.array],
    min_bounds: list[np.array],
    pcd_overall: o3d.geometry.PointCloud = None,
    series_type_dict=None,
    downsampling_factor=1,
    voxel_size=(128, 128, 42),
):
    ret = {}

    for index, vertebral_level in enumerate(vertebral_levels):
        cache_path = os.path.join(
            Config.data_path,
            f"{study_id}_{vertebral_level}_{voxel_size[0]}_{voxel_size[1]}_{voxel_size[2]}.npy.gz",
        )
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
                pcd_overall = read_study_as_pcd(
                    dir_path,
                    series_types_dict=series_type_dict,
                    downsampling_factor=downsampling_factor,
                    img_size=(voxel_size[0], voxel_size[2]),
                    stack_slices_thickness=True,
                    resize_slices=False,
                )

            bbox = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=min_bounds[index], max_bound=max_bounds[index]
            )
            pcd_level = pcd_overall.crop(bbox)

            box = pcd_level.get_axis_aligned_bounding_box()

            max_b = np.array(box.get_max_bound())
            min_b = np.array(box.get_min_bound())

            pts = (np.array(pcd_level.points) - (min_b)) * (
                (voxel_size[0] - 1, voxel_size[1] - 1, voxel_size[2] - 1)
                / (max_b - min_b)
            )
            coords = np.round(pts).astype(np.int32)
            vals = np.array(pcd_level.colors, dtype=np.float16)

            grid = np.zeros(
                (3, voxel_size[0], voxel_size[1], voxel_size[2]), dtype=np.float16
            )
            indices = coords[:, 0], coords[:, 1], coords[:, 2]

            np.maximum.at(grid[0], indices, vals[:, 0])
            np.maximum.at(grid[1], indices, vals[:, 1])
            np.maximum.at(grid[2], indices, vals[:, 2])

            f = pgzip.PgzipFile(cache_path, "w")
            np.save(f, grid)
            f.close()

            ret[vertebral_level] = grid

    return ret


# ------------------------------------------------- Without Open3d


def read_study_as_np(
    dir_path,
    series_types_dict=None,
    downsampling_factor=1,
    resize_slices=True,
    resize_method="nearest",
    img_size=(256, 256),
    stack_slices_thickness=True,
):
    all_points = []  # To store all transformed points
    all_vals = []  # To store all values

    for path in glob.glob(os.path.join(dir_path, "**/*.dcm"), recursive=True):
        dicom_slice = dcmread(path)

        series_id = os.path.basename(os.path.dirname(path))
        if series_types_dict is None or int(series_id) not in series_types_dict:
            series_desc = dicom_slice.SeriesDescription
        else:
            series_desc = series_types_dict[int(series_id)].split(" ")[-1]

        x_orig, y_orig = dicom_slice.pixel_array.shape
        if resize_slices:
            if resize_method == "nearest":
                img = np.expand_dims(
                    cv2.resize(
                        dicom_slice.pixel_array, img_size, interpolation=cv2.INTER_AREA
                    ),
                    -1,
                )
            elif resize_method == "maxpool":
                img_tensor = torch.tensor(dicom_slice.pixel_array).float()
                img = F.adaptive_max_pool2d(img_tensor.unsqueeze(0), img_size).numpy()
            else:
                raise ValueError(f"Invalid resize_method {resize_method}")
        else:
            img = np.expand_dims(np.array(dicom_slice.pixel_array), -1)

        x, y, z = np.where(img)

        downsampling_factor_iter = max(
            downsampling_factor, int(math.ceil(len(x) / 6e6))
        )

        index_voxel = np.vstack((x, y, z))[:, ::downsampling_factor_iter]
        grid_index_array = index_voxel.T  # Shape: [n, 3]
        ones = np.ones((grid_index_array.shape[0], 1))
        points = np.hstack([grid_index_array, ones])  # Shape: [n, 4]

        vals = np.expand_dims(
            img[x, y, z][::downsampling_factor_iter], -1
        )  # Shape: [n, 1]
        if series_desc == "T1":
            vals = np.pad(vals, ((0, 0), (0, 2)))
        elif series_desc == "T2":
            vals = np.pad(vals, ((0, 0), (1, 1)))
        elif series_desc == "T2/STIR":
            vals = np.pad(vals, ((0, 0), (2, 0)))
        else:
            raise ValueError(f"Unknown series desc: {series_desc}")

        # Create transformation matrix
        if resize_slices:
            transform_matrix_factor = np.matrix(
                [
                    [0, y_orig / img_size[1], 0, 0],
                    [x_orig / img_size[0], 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
        else:
            transform_matrix_factor = np.matrix(
                [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            )

        dX, dY = dicom_slice.PixelSpacing
        dZ = dicom_slice.SliceThickness

        X = np.array(list(dicom_slice.ImageOrientationPatient[:3]) + [0]) * dX
        Y = np.array(list(dicom_slice.ImageOrientationPatient[3:]) + [0]) * dY
        S = np.array(list(dicom_slice.ImagePositionPatient) + [1])

        transform_matrix = np.array([X, Y, np.zeros(len(X)), S]).T
        transform_matrix = transform_matrix @ transform_matrix_factor

        # Instead of appending, initialize a list for transformed points in this iteration
        transformed_points_list = []

        if stack_slices_thickness:
            for z in range(int(dZ)):
                pos = list(dicom_slice.ImagePositionPatient)
                pos[0] += z if series_desc != "T2" else 0
                pos[-1] += z if series_desc == "T2" else 0
                S = np.array(pos + [1])

                transform_matrix = np.array([X, Y, np.zeros(len(X)), S]).T
                transform_matrix = transform_matrix @ transform_matrix_factor
                transformed_points = points.copy() @ transform_matrix.T

                # Directly add transformed points and vals to the current list
                all_points.append(transformed_points[:, :3])  # Keep only x, y, z
                all_vals.append(vals)

        else:
            transformed_points = points.copy() @ transform_matrix.T
            transformed_points_list.append(
                transformed_points[:, :3]
            )  # Keep only x, y, z
            all_vals.append(vals)

    # Convert lists to arrays at once
    return np.vstack(all_points), np.vstack(all_vals)


def voxelize_point_cloud(points, colors, grid_size, voxel_size):
    """
    Discretizes a point cloud into a voxel grid of shape (grid_size, grid_size, grid_size)
    and performs max pooling on the color values.

    Args:
        points (np.ndarray): (n, 3) array of 3D points.
        colors (np.ndarray): (n, 3) array of colors corresponding to each point.
        grid_size (int): The size of the voxel grid (e.g., 64 for a 64x64x64 grid).
        voxel_size (float): The size of each voxel.

    Returns:
        voxel_grid (np.ndarray): A (grid_size, grid_size, grid_size, 3) array containing
                                 the max-pooled colors for each voxel.
    """
    # Step 1: Translate points so that the minimum coordinate is aligned with the grid's origin
    min_coords = np.min(points, axis=0)
    translated_points = points - min_coords

    # Step 2: Initialize the voxel grid with -infinity to perform max pooling
    voxel_grid = np.full((*grid_size, 3), -np.inf)

    # Step 3: Compute voxel indices for each point
    voxel_indices = np.floor(translated_points / voxel_size).astype(int)

    # Clamp the voxel indices to the grid size
    voxel_indices = np.clip(voxel_indices, 0, grid_size[0] - 1)

    # Step 4: Iterate over points and max-pool the colors into the voxel grid
    for idx, voxel in enumerate(voxel_indices.tolist()):
        x, y, z = voxel
        voxel_grid[x, y, z] = np.maximum(voxel_grid[x, y, z], colors[idx])

    # Step 5: Replace -inf values with zeros (or any other background color) where no point exists
    voxel_grid[voxel_grid == -np.inf] = 0

    return voxel_grid


def crop_point_cloud(points, colors, min_bound, max_bound):
    mask = np.all(
        (points.tolist() >= min_bound) & (points.tolist() <= max_bound), axis=1
    )

    cropped_points = points[mask]
    cropped_colors = colors[mask]

    return cropped_points, cropped_colors


def read_vertebral_levels_as_voxel_grids_np(
    dir_path,
    study_id,
    vertebral_levels: list[str],
    max_bounds: list[np.array],
    min_bounds: list[np.array],
    pcd_overall=None,
    series_type_dict=None,
    downsampling_factor=1,
    voxel_size=(128, 128, 42),
):
    ret = {}
    resize = tio.Resize(voxel_size)

    for index, vertebral_level in enumerate(vertebral_levels):
        cache_path = os.path.join(
            Config.data_path,
            f"{study_id}_{vertebral_level}_{voxel_size[0]}_{voxel_size[1]}_{voxel_size[2]}_numpy.npy.gz",
        )
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
                points, colors = read_study_as_np(
                    dir_path,
                    series_types_dict=series_type_dict,
                    downsampling_factor=downsampling_factor,
                    img_size=(voxel_size[0], voxel_size[2]),
                )
            print(points.shape, colors.shape)
            cropped_points, cropped_colors = crop_point_cloud(
                points, colors, np.array(min_bounds[index]), np.array(max_bounds[index])
            )
            voxel_grid = voxelize_point_cloud(
                cropped_points, cropped_colors, voxel_size, 1
            )

            # Save the voxel grid
            f = pgzip.PgzipFile(cache_path, "w")
            np.save(f, voxel_grid)
            f.close()

    return ret


transform_train = tio.Compose(
    [
        tio.CropOrPad(target_shape=Config.vol_size),
        tio.ZNormalization(),
        tio.RandomAffine(
            translation=10,
            image_interpolation=Config.image_interpolation,
            p=Config.aug_prob,
        ),
        tio.RandomNoise(p=Config.aug_prob),
        tio.RandomSpike(1, intensity=(-0.5, 0.5), p=Config.aug_prob),
        tio.RescaleIntensity((0, 1)),
    ]
)

transform_val = tio.Compose(
    [
        tio.CropOrPad(target_shape=Config.vol_size),
        tio.RescaleIntensity((0, 1)),
    ]
)
