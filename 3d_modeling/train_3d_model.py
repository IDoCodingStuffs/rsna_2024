import timm_3d
import torch.optim.lr_scheduler
from spacecutter.losses import CumulativeLinkLoss
from spacecutter.models import LogisticCumulativeLink
from spacecutter.callbacks import AscensionCallback
from timm_3d.models import MaxxVitCfg
from timm_3d.models.maxxvit import _rw_coat_cfg

from training_utils import *
from rsna_dataloader import *
from constants import *

_logger = logging.getLogger(__name__)

CONFIG = dict(
    n_levels=5,
    num_classes=25,
    num_conditions=5,
    image_interpolation="linear",
    backbone="coatnet_rmlp2_reg_rw",
    # backbone="maxxvit_rmlp_small_rw_256",
    # backbone="coatnet_nano_cc",
    vol_size=(96, 96, 96),
    # vol_size=(256, 256, 256),
    # loss_weights=CLASS_RELATIVE_WEIGHTS_MIRROR_CLIPPED,
    loss_weights=CONDITION_RELATIVE_WEIGHTS_MIRROR,
    num_workers=15,
    gradient_acc_steps=2,
    drop_rate=0.35,
    drop_rate_last=0.,
    drop_path_rate=0.,
    aug_prob=0.85,
    out_dim=3,
    stage_1_epochs=6,
    stage_2_epochs=12,
    stage_3_epochs=24,
    epochs=35,
    tune_epochs=5,
    batch_size=10,
    split_rate=0.25,
    split_k=5,
    device=torch.device("cuda") if torch.cuda.is_available() else "cpu",
    seed=2024
)
DATA_BASEPATH = "./data/rsna-2024-lumbar-spine-degenerative-classification/"
TRAINING_DATA = retrieve_coordinate_training_data(DATA_BASEPATH)


class CustomMaxxVit3dClassifier(nn.Module):
    def __init__(self,
                 backbone,
                 in_chans=3,
                 out_classes=5,
                 cutpoint_margin=0):
        super(CustomMaxxVit3dClassifier, self).__init__()
        self.out_classes = out_classes

        # self.config = timm_3d.models.maxxvit.model_cfgs[backbone]

        self.backbone = timm_3d.models.MaxxVit(
            img_size=CONFIG["vol_size"],
            in_chans=in_chans,
            num_classes=out_classes,
            drop_rate=CONFIG["drop_rate"],
            drop_path_rate=CONFIG["drop_path_rate"],
            cfg=MaxxVitCfg(
                embed_dim=(96, 384, 768, 1536),
                # embed_dim=(256, 512, 1280, 2048),
                depths=(2, 16, 32, 2),
                # stem_width=(128, 256),
                stem_width=(48, 96),
                **_rw_coat_cfg(
                    stride_mode='dw',
                    conv_attn_act_layer='silu',
                    init_values=1e-6,
                    rel_pos_type='mlp',
                    rel_pos_dim=2048,
                ),
            )
        )
        self.backbone.head.drop = nn.Dropout(p=CONFIG["drop_rate_last"])
        head_in_dim = self.backbone.head.fc.in_features
        self.backbone.head.fc = nn.Identity()

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


class Classifier3dMultihead(nn.Module):
    def __init__(self,
                 backbone="efficientnet_lite0",
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
        tio.RandomAffine(translation=10, degrees=25, image_interpolation=CONFIG["image_interpolation"],
                         p=CONFIG["aug_prob"]),
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
        model = CustomMaxxVit3dClassifier()
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
    coords_dataframe = pd.read_csv(os.path.join("data/SpineNet/coords_3d.csv"))

    transform_3d_train = tio.Compose([
        tio.ZNormalization(),
        tio.RandomAffine(translation=(10, 10, 10),
                         degrees=(25, 25, 25),
                         isotropic=True,
                         image_interpolation=CONFIG["image_interpolation"],
                         p=CONFIG["aug_prob"]),
        tio.RandomNoise(p=CONFIG["aug_prob"]),
        tio.RescaleIntensity((0, 1)),
    ])

    transform_3d_val = tio.Compose([
        tio.RescaleIntensity((0, 1)),
    ])

    train_data = TRAINING_DATA[TRAINING_DATA["study_id"].isin(bounds_dataframe["study_id"])]
    dataset_folds = create_vertebra_level_datasets_and_loaders_k_fold(train_data,
                                                                      boundaries_df=bounds_dataframe,
                                                                      coords_df=coords_dataframe,
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
        "train_2": [
            CumulativeLinkLoss(class_weights=CONDITION_ELOGN_RELATIVE_WEIGHTS_MIRROR[i]) for i in range(CONFIG["num_conditions"])
        ],
        "train_3": [
            CumulativeLinkLoss(class_weights=CONDITION_LOGN_RELATIVE_WEIGHTS_MIRROR[i]) for i in range(CONFIG["num_conditions"])
        ],
        "unweighted_val": [
            CumulativeLinkLoss() for i in range(CONFIG["num_conditions"])
        ],
        "alt_val": [
            CumulativeLinkLoss(class_weights=COMP_WEIGHTS[i]) for i in range(CONFIG["num_conditions"])
        ],
        "weighted_alt_val": [
            nn.BCELoss(weight=COMP_WEIGHTS[i]).to(device) for i in range(CONFIG["num_conditions"])
        ],
        "unweighted_alt_val": [
            nn.BCELoss().to(device) for i in range(CONFIG["num_conditions"])
        ]
    }

    for index, fold in enumerate(dataset_folds):
        model = CustomMaxxVit3dClassifier(backbone=backbone).to(device)
        optimizers = [
            torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2),
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
                                    stage_1_epochs=CONFIG["stage_1_epochs"],
                                    stage_2_epochs=CONFIG["stage_2_epochs"],
                                    stage_3_epochs=CONFIG["stage_3_epochs"],
                                    epochs=CONFIG["epochs"] + 1,
                                    freeze_backbone_initial_epochs=-1,
                                    freeze_backbone_after_epochs=-1,
                                    loss_weights=CONFIG["loss_weights"],
                                    callbacks=[model._ascension_callback],
                                    gradient_accumulation_per=CONFIG["gradient_acc_steps"]
                                    )

    return model


def tune_stage_2_model_3d(backbone, model_label: str, model_path: str, fold_index: int):
    bounds_dataframe = pd.read_csv(os.path.join("data/SpineNet/bounding_boxes_3d.csv"))
    coords_dataframe = pd.read_csv(os.path.join("data/SpineNet/coords_3d.csv"))

    transform_3d_train = tio.Compose([
        tio.ZNormalization(),
        tio.RandomAffine(translation=(10, 10, 10),
                         degrees=(25, 25, 25),
                         isotropic=True,
                         image_interpolation=CONFIG["image_interpolation"],
                         p=CONFIG["aug_prob"]),
        tio.RandomNoise(p=CONFIG["aug_prob"]),
        tio.RescaleIntensity((0, 1)),
    ])

    transform_3d_val = tio.Compose([
        tio.RescaleIntensity((0, 1)),
    ])

    train_data = TRAINING_DATA[TRAINING_DATA["study_id"].isin(bounds_dataframe["study_id"])]
    dataset_folds = create_vertebra_level_datasets_and_loaders_k_fold(train_data,
                                                                      boundaries_df=bounds_dataframe,
                                                                      coords_df=coords_dataframe,
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
            nn.BCELoss(weight=COMP_WEIGHTS[i]).to(device) for i in range(CONFIG["num_conditions"])
        ],
        "unweighted_alt_val": [
            nn.BCELoss().to(device) for i in range(CONFIG["num_conditions"])
        ]
    }

    fold = dataset_folds[fold_index]
    model = CustomMaxxVit3dClassifier(backbone=backbone).to(device)
    model.load_state_dict(torch.load(model_path))
    optimizers = [
        # torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True),
        torch.optim.Adam(model.parameters(), lr=1e-3),
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
                                callbacks=[model._ascension_callback],
                                gradient_accumulation_per=CONFIG["gradient_acc_steps"]
                                )

    return model


def train():
    model = train_stage_2_model_3d(CONFIG['backbone'], f"{CONFIG['backbone']}_{CONFIG['vol_size'][0]}_v2")
    # model = train_model_3d(CONFIG['backbone'], f"{CONFIG['backbone']}_{CONFIG['vol_size'][0]}_3d")
    # model = tune_stage_2_model_3d(CONFIG['backbone'],
    #                               f"{CONFIG['backbone']}_{CONFIG['vol_size'][0]}_16_nonaligned",
    #                               "models/coatnet_rmlp2_reg_rw_96_fold_0/coatnet_rmlp2_reg_rw_96_fold_0_16.pt",
    #                               fold_index=0)


if __name__ == '__main__':
    train()
