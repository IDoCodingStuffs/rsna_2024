import math
from tqdm import tqdm
import argparse
from multiprocessing import Process

from rsna_dataloader import *

DATA_BASEPATH = "./data/rsna-2024-lumbar-spine-degenerative-classification/"
TRAINING_DATA = retrieve_coordinate_training_data(DATA_BASEPATH)

df = TRAINING_DATA.dropna()
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

series_descs = {e[0]: e[1] for e in df[["series_id", "series_description"]].drop_duplicates().values}

bounding_boxes = pd.read_csv("./data/lumbar-coordinate-pretraining-dataset/bounding_boxes_3d.csv")


def worker_loop(dirslice):
    for dir in tqdm(dirslice):
        study_id = dir.split("/")[-1].split("\\")[-2]
        study_bounds = bounding_boxes[bounding_boxes['study_id'] == int(study_id)].sort_values(by="level", ascending=True)

        read_study_as_voxel_grid_v2(dir, series_descs)

        # levels = []
        # min_bounds = []
        # max_bounds = []
        #
        # for i in range(len(study_bounds)):
        #     row = study_bounds.iloc[i]
        #
        #     if row['level'] == "L5/S1":
        #         read_vertebral_level_as_voxel_grid(
        #             dir,
        #             vertebral_level="l5s1",
        #             min_bound=np.array([row['x_min'], row['y_min'], row['z_min']]),
        #             max_bound=np.array([row['x_max'], row['y_max'], row['z_max']]),
        #             series_type_dict=series_descs,
        #             voxel_size=(128, 128, int(128 / 3 * 2)))
        #
        #     else:
        #         levels.append(row['level'].replace("/", "").lower())
        #         min_bounds.append(np.array([row['x_min'], row['y_min'], row['z_min']]))
        #         max_bounds.append(np.array([row['x_max'], row['y_max'], row['z_max']]))
        #
        # read_vertebral_levels_as_voxel_grids(
        #     dir,
        #     vertebral_levels=levels,
        #     min_bounds=min_bounds,
        #     max_bounds=max_bounds,
        #     series_type_dict=series_descs,
        #     voxel_size=(128, 128, int(128 / 3)))


if __name__ == "__main__":
    dirs = glob.glob("./data/rsna-2024-lumbar-spine-degenerative-classification/train_images/*/")
    dirs = sorted(dirs)

    parser = argparse.ArgumentParser(
        prog='GridCachePrepop',
    )
    parser.add_argument('count', type=int, default=5)
    args = parser.parse_args()

    slice_size = math.ceil(len(dirs)/args.count)

    workers = []
    for worker_index in range(args.count):
        dirslice = dirs[slice_size * worker_index:slice_size * (worker_index + 1)]
        p = Process(target=worker_loop, args=(dirslice,))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()