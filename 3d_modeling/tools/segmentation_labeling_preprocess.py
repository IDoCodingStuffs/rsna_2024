import os.path

import numpy as np
import matplotlib.pyplot as plt
import pydicom
import cv2
from glob import glob

from data_retrieval import *

data_basepath = "../../data/rsna-2024-lumbar-spine-degenerative-classification/"

def retrieve_sagittal_midslices(type="T1"):
    data = retrieve_training_data(data_basepath)
    data = data[data["series_description"] == f"Sagittal {type}"]
    data = data[['study_id', 'series_id']].drop_duplicates()

    midslice_ids = []
    midslice_paths = []

    for index, row in data.iterrows():
        series_path = os.path.join(data_basepath, "train_images", str(row["study_id"]), str(row["series_id"]))
        image_paths = glob(series_path + "/*.dcm")

        image_ids = [int(img_path.split("/")[-1].split("\\")[-1].replace(".dcm", "")) for img_path in image_paths]
        max_id = max(image_ids)
        midslice_id = (max_id + 1) // 2

        midslice_ids.append(midslice_id)
        midslice_paths.append(os.path.join(series_path, f"{midslice_id}.dcm"))

    data["midslice_ids"] = midslice_ids
    data["midslice_paths"] = midslice_paths
    data = data.reset_index(drop=True)

    return data


def visualize_mid_slices():
    data = retrieve_sagittal_midslices()
    data = data.sort_values(by="midslice_paths", ascending=True)

    for i in range(1, 51):
        img_path = data["midslice_paths"].iloc[-i]
        study_id = data["study_id"].iloc[-i]

        img = pydicom.dcmread(img_path).pixel_array
        canny_edges = get_canny_edges_for_img(img, th=20)

        f, ax = plt.subplots(1, 2)

        ax[0].imshow(img, cmap="gist_yarg")
        ax[1].imshow(canny_edges, cmap="gist_yarg")

        plt.title(study_id)
        plt.show()

def dump_canny_edges(out_path=data_basepath + "/canny_edges/"):
    os.makedirs(out_path, exist_ok=True)

    data = retrieve_sagittal_midslices()
    data = data.sort_values(by="midslice_paths", ascending=True)

    for i in range(51, 101):
        img_path = data["midslice_paths"].iloc[-i]
        study_id = data["study_id"].iloc[-i]

        if not os.path.exists(os.path.join(out_path, f"{study_id}_canny.png")):
            img = pydicom.dcmread(img_path).pixel_array
            canny_edges = get_canny_edges_for_img(img)

            dilated_canny_edges = cv2.dilate(canny_edges, None, iterations=1)

            plt.imsave(out_path + f"{study_id}_canny.png", dilated_canny_edges, cmap="gist_yarg")
            plt.imsave(out_path + f"{study_id}_orig.png", img, cmap="gist_yarg")


def get_canny_edges_for_img(img, th=40, upper_bound_factor=3, l2_grad=True):
    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    cvt_img = cv2.convertScaleAbs(img)

    blurred_img = cv2.medianBlur(cvt_img, ksize=3)
    sharpened_img = cv2.filter2D(blurred_img, -1, sharpening_kernel)
    ret, binary_image = cv2.threshold(cvt_img, 0, 150, cv2.THRESH_OTSU)

    canny_out = cv2.Canny(binary_image, th, th * upper_bound_factor, L2gradient=l2_grad)
    return canny_out

# visualize_mid_slices()
dump_canny_edges()
