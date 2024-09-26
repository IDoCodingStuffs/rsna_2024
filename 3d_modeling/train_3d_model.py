import timm_3d
import torch.optim.lr_scheduler
from spacecutter.losses import CumulativeLinkLoss
from spacecutter.models import LogisticCumulativeLink
from spacecutter.callbacks import AscensionCallback

from training_utils import *
from rsna_dataloader import *
from constants import *

_logger = logging.getLogger(__name__)

CONFIG = dict(
    n_levels=5,
    num_classes=25,
    num_conditions=5,
    image_interpolation="bspline",
    backbone="coatnet_rmlp_3_rw_224",
    # backbone="maxxvit_rmlp_small_rw_256",
    head_type="softmax", # {"sigmoid", "cumulog"}
    vol_size=(128, 128, 128),
    # vol_size=(256, 256, 256),
    # loss_weights=CLASS_RELATIVE_WEIGHTS_MIRROR_CLIPPED,
    loss_weights=CONDITION_LOGN_RELATIVE_WEIGHTS_MIRROR,
    num_workers=12,
    gradient_acc_steps=4,
    drop_rate=0.3,
    drop_rate_last=0.,
    drop_path_rate=0.3,
    aug_prob=0.9,
    out_dim=3,
    epochs=45,
    tune_epochs=15,
    batch_size=5,
    split_rate=0.25,
    split_k=5,
    device=torch.device("cuda") if torch.cuda.is_available() else "cpu",
    seed=2024
)
DATA_BASEPATH = "./data/rsna-2024-lumbar-spine-degenerative-classification/"
TRAINING_DATA = retrieve_coordinate_training_data(DATA_BASEPATH)

def noop():
    pass

class Classifier3dMultihead(nn.Module):
    def __init__(self,
                 backbone="efficientnet_lite0",
                 head_type="cumulog",
                 in_chans=1,
                 out_classes=5,
                 cutpoint_margin=0,
                 pretrained=False):
        super(Classifier3dMultihead, self).__init__()
        self.out_classes = out_classes

        self.backbone = timm_3d.create_model(
            backbone,
            features_only=False,
            drop_rate=CONFIG["drop_rate"],
            drop_path_rate=CONFIG["drop_path_rate"],
            pretrained=pretrained,
            in_chans=in_chans,
            global_pool="max",
        )
        if "efficientnet" in backbone:
            head_in_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.LayerNorm(head_in_dim),
                nn.Dropout(CONFIG["drop_rate_last"]),
            )

        elif "vit" in backbone or "coat" in backbone:
            self.backbone.head.drop = nn.Dropout(p=CONFIG["drop_rate_last"])
            head_in_dim = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()

        if head_type == "softmax":
            self.heads = nn.ModuleList(
                [nn.Sequential(
                    nn.Linear(head_in_dim, CONFIG["out_dim"]),
                    nn.Sigmoid()
                ) for i in range(out_classes)]
            )

            self.ascension_callback = noop

        elif head_type == "cumulog":
            self.heads = nn.ModuleList(
                [nn.Sequential(
                    nn.Linear(head_in_dim, 1),
                    LogisticCumulativeLink(CONFIG["out_dim"])
                ) for i in range(out_classes)]
            )

            self.ascension_callback = AscensionCallback(margin=cutpoint_margin)

    def forward(self, x):
        feat = self.backbone(x)
        return torch.swapaxes(torch.stack([head(feat) for head in self.heads]), 0, 1)

    def _ascension_callback(self):
        for head in self.heads:
            self.ascension_callback.clip(head[-1])


def train_model_3d(backbone, model_label: str):
    transform_3d_train = tio.Compose([
        tio.ZNormalization(),
        tio.RandomAffine(translation=10, image_interpolation=CONFIG["image_interpolation"], p=CONFIG["aug_prob"]),
        tio.RandomNoise(p=CONFIG["aug_prob"]),
        tio.RandomSpike(1, intensity=(-0.5, 0.5), p=CONFIG["aug_prob"]),
        tio.RescaleIntensity((0, 1)),
    ])

    transform_3d_val = tio.Compose([
        tio.RescaleIntensity((0, 1)),
    ])

    dataset_folds = create_study_level_datasets_and_loaders_k_fold(TRAINING_DATA,
                                                                   transform_3d_train=transform_3d_train,
                                                                   transform_3d_val=transform_3d_val,
                                                                   base_path=os.path.join(
                                                                    DATA_BASEPATH,
                                                                    "train_images"),
                                                                   vol_size=CONFIG["vol_size"],
                                                                   num_workers=CONFIG["num_workers"],
                                                                   split_k=CONFIG["split_k"],
                                                                   batch_size=CONFIG["batch_size"],
                                                                   pin_memory=True,
                                                                   use_mirroring_trick=True
                                                                   )

    schedulers = [
    ]
    criteria = {
        "train": [
            CumulativeLinkLoss(class_weights=CONFIG["loss_weights"][i]).to(device) for i in range(CONFIG["num_classes"])
        ],
        "unweighted_val": [
            CumulativeLinkLoss().to(device) for i in range(CONFIG["num_classes"])
        ],
        "alt_val": [
            CumulativeLinkLoss(class_weights=COMP_WEIGHTS[i]).to(device) for i in range(CONFIG["num_classes"])
        ],
        "weighted_alt_val": [
            nn.CrossEntropyLoss(weight=COMP_WEIGHTS[i]).to(device) for i in range(CONFIG["num_classes"])
        ],
        "unweighted_alt_val": [
            nn.CrossEntropyLoss().to(device) for i in range(CONFIG["num_classes"])
        ]
    }

    for index, fold in enumerate(dataset_folds):
        if index != 3:
            continue
        model = Classifier3dMultihead(backbone=backbone, in_chans=3, out_classes=CONFIG["num_classes"]).to(device)
        model.load_state_dict(torch.load("models/maxxvit_rmlp_small_rw_256_256_3d_fold_3_pt1/maxxvit_rmlp_small_rw_256_256_3d_fold_3_20.pt"))
        optimizers = [
            torch.optim.AdamW(model.parameters(), lr=3e-4),
        ]

        trainloader, valloader, trainset, testset = fold

        train_model_with_validation(model,
                                    optimizers,
                                    schedulers,
                                    criteria,
                                    trainloader,
                                    valloader,
                                    model_desc=model_label + f"_fold_{index}",
                                    train_loader_desc=f"Training {model_label} fold {index}",
                                    epochs=CONFIG["epochs"],
                                    freeze_backbone_initial_epochs=-1,
                                    freeze_backbone_after_epochs=-1,
                                    loss_weights=CONFIG["loss_weights"],
                                    callbacks=[model._ascension_callback],
                                    gradient_accumulation_per=CONFIG["gradient_acc_steps"]
                                    )

    return model


def train_stage_2_model_3d(backbone, model_label: str):
    bounds_dataframe = pd.read_csv(os.path.join("data/SpineNet/bounding_boxes_3d.csv"))

    transform_3d_train = tio.Compose([
        tio.ZNormalization(),
        # tio.RandomAffine(translation=10, image_interpolation=CONFIG["image_interpolation"], p=CONFIG["aug_prob"]),
        tio.RandomAffine(translation=10, scales=0, p=CONFIG["aug_prob"]),
        tio.RandomNoise(p=CONFIG["aug_prob"]),
        tio.RandomSpike(1, intensity=(-0.5, 0.5), p=CONFIG["aug_prob"]),
        tio.RescaleIntensity((0, 1)),
    ])

    transform_3d_val = tio.Compose([
        tio.RescaleIntensity((0, 1)),
    ])

    train_data = TRAINING_DATA[TRAINING_DATA["study_id"].isin(bounds_dataframe["study_id"])]
    dataset_folds = create_vertebra_level_datasets_and_loaders_k_fold(train_data,
                                                                      boundaries_df=bounds_dataframe,
                                                                   transform_3d_train=transform_3d_train,
                                                                   transform_3d_val=transform_3d_val,
                                                                   base_path=os.path.join(
                                                                    DATA_BASEPATH,
                                                                    "train_images"),
                                                                   vol_size=CONFIG["vol_size"],
                                                                   num_workers=CONFIG["num_workers"],
                                                                   split_k=CONFIG["split_k"],
                                                                   batch_size=CONFIG["batch_size"],
                                                                   pin_memory=True,
                                                                   use_mirroring_trick=True
                                                                   )

    schedulers = [

    ]
    criteria = {
        "train": [
            CumulativeLinkLoss(class_weights=CONFIG["loss_weights"][i]) for i in range(CONFIG["num_conditions"])
        ],
        "unweighted_val": [
            CumulativeLinkLoss() for i in range(CONFIG["num_conditions"])
        ],
        "alt_val": [
            CumulativeLinkLoss(class_weights=COMP_WEIGHTS[i]) for i in range(CONFIG["num_conditions"])
        ],
        "weighted_alt_val": [
            nn.CrossEntropyLoss(weight=COMP_WEIGHTS[i]).to(device) for i in range(CONFIG["num_conditions"])
        ],
        "unweighted_alt_val": [
            nn.CrossEntropyLoss().to(device) for i in range(CONFIG["num_conditions"])
        ]
    }

    for index, fold in enumerate(dataset_folds):
        model = Classifier3dMultihead(backbone=backbone, in_chans=3, out_classes=CONFIG["num_conditions"]).to(device)
        if index == 0:
            model.load_state_dict(torch.load("models/coatnet_rmlp_3_rw_224_128_vertebrae_fold_0_pt3/coatnet_rmlp_3_rw_224_128_vertebrae_fold_0_35.pt"))
        optimizers = [
            torch.optim.AdamW(model.parameters(), lr=3e-4),
        ]

        trainloader, valloader, trainset, testset = fold

        train_model_with_validation(model,
                                    optimizers,
                                    schedulers,
                                    criteria,
                                    trainloader,
                                    valloader,
                                    model_desc=model_label + f"_fold_{index}",
                                    train_loader_desc=f"Training {model_label} fold {index}",
                                    epochs=CONFIG["epochs"],
                                    freeze_backbone_initial_epochs=-1,
                                    freeze_backbone_after_epochs=-1,
                                    loss_weights=CONFIG["loss_weights"],
                                    callbacks=[model._ascension_callback],
                                    gradient_accumulation_per=CONFIG["gradient_acc_steps"]
                                    )

    return model


def tune_stage_2_model_3d(backbone, model_label: str, model_path: str, fold_index: int):
    bounds_dataframe = pd.read_csv(os.path.join("data/SpineNet/bounding_boxes_3d.csv"))

    transform_3d_train = tio.Compose([
        tio.CropOrPad(target_shape=CONFIG["vol_size"]),
        tio.ZNormalization(),
        tio.RandomAffine(translation=10, image_interpolation=CONFIG["image_interpolation"], p=CONFIG["aug_prob"]),
        tio.RandomNoise(p=CONFIG["aug_prob"]),
        tio.RandomSpike(1, intensity=(-0.5, 0.5), p=CONFIG["aug_prob"]),
        tio.RescaleIntensity((0, 1)),
    ])

    transform_3d_val = tio.Compose([
        tio.CropOrPad(target_shape=CONFIG["vol_size"]),
        tio.RescaleIntensity((0, 1)),
    ])

    train_data = TRAINING_DATA[TRAINING_DATA["study_id"].isin(bounds_dataframe["study_id"])]
    dataset_folds = create_vertebra_level_datasets_and_loaders_k_fold(train_data,
                                                                      boundaries_df=bounds_dataframe,
                                                                   transform_3d_train=transform_3d_train,
                                                                   transform_3d_val=transform_3d_val,
                                                                   base_path=os.path.join(
                                                                    DATA_BASEPATH,
                                                                    "train_images"),
                                                                   vol_size=CONFIG["vol_size"],
                                                                   num_workers=CONFIG["num_workers"],
                                                                   split_k=CONFIG["split_k"],
                                                                   batch_size=CONFIG["batch_size"],
                                                                   pin_memory=True,
                                                                   use_mirroring_trick=True
                                                                   )

    schedulers = [
    ]
    criteria = {
        "train": [
            # CumulativeLinkLoss(class_weights=CONFIG["loss_weights"][i]) for i in range(CONFIG["num_conditions"])
            nn.CrossEntropyLoss(weight=COMP_WEIGHTS[i]).to(device) for i in range(CONFIG["num_conditions"])
        ],
        "unweighted_val": [
            CumulativeLinkLoss() for i in range(CONFIG["num_conditions"])
        ],
        "alt_val": [
            CumulativeLinkLoss(class_weights=COMP_WEIGHTS[i]) for i in range(CONFIG["num_conditions"])
        ],
        "weighted_alt_val": [
            # nn.CrossEntropyLoss(weight=COMP_WEIGHTS[i]).to(device) for i in range(CONFIG["num_conditions"])
        ],
        "unweighted_alt_val": [
            # nn.CrossEntropyLoss().to(device) for i in range(CONFIG["num_conditions"])
        ]
    }

    fold = dataset_folds[fold_index]
    initial_model = Classifier3dMultihead(backbone=backbone, in_chans=3, out_classes=CONFIG["num_conditions"], head_type="cumulog").to(device)
    initial_model.load_state_dict(torch.load(model_path))

    model = Classifier3dMultihead(backbone=backbone, in_chans=3, out_classes=CONFIG["num_conditions"], head_type=CONFIG["head_type"]).to(device)
    model.backbone = initial_model.backbone

    optimizers = [
        # torch.optim.AdamW(model.backbone.parameters(), lr=5e-6),
        torch.optim.AdamW(model.parameters(), lr=3e-4),
    ]

    trainloader, valloader, trainset, testset = fold

    train_model_with_validation(model,
                                optimizers,
                                schedulers,
                                criteria,
                                trainloader,
                                valloader,
                                model_desc=model_label + f"_fold_{fold_index}",
                                train_loader_desc=f"Tuning {model_label} fold {fold_index}",
                                epochs=CONFIG["tune_epochs"],
                                freeze_backbone_initial_epochs=-1,
                                freeze_backbone_after_epochs=0,
                                loss_weights=CONFIG["loss_weights"],
                                # callbacks=[model._ascension_callback],
                                gradient_accumulation_per=CONFIG["gradient_acc_steps"]
                                )

    return model


def train():
    # model = train_stage_2_model_3d(CONFIG['backbone'], f"{CONFIG['backbone']}_{CONFIG['vol_size'][0]}_vertebrae_{CONFIG['head_type']}")
    # model = train_model_3d(CONFIG['backbone'], f"{CONFIG['backbone']}_{CONFIG['vol_size'][0]}_3d")
    model = tune_stage_2_model_3d(CONFIG['backbone'],
                                  f"{CONFIG['backbone']}_{CONFIG['vol_size'][0]}_vertebrae_{CONFIG['head_type']}_retuned",
                                  "models/coatnet_rmlp_3_rw_224_128_vertebrae_fold_0_pt5/coatnet_rmlp_3_rw_224_128_vertebrae_fold_0_15.pt",
                                  fold_index=0)

if __name__ == '__main__':
    train()
