import os.path
from typing import Iterable

import numpy as np
import timm_3d
import torch.optim.lr_scheduler
from spacecutter.losses import CumulativeLinkLoss
from spacecutter.models import LogisticCumulativeLink
from spacecutter.callbacks import AscensionCallback
from timm_3d.layers import NormMlpClassifierHead
from timm_3d.models import MaxxVitCfg
from timm_3d.models.maxxvit import _rw_coat_cfg, _rw_max_cfg

from training_utils import *
from rsna_dataloader import *
from constants import *

_logger = logging.getLogger(__name__)

CONFIG = dict(
    n_levels=5,
    num_classes=25,
    num_conditions=5,
    image_interpolation="linear",
    backbone="maxvit_rmlp_bc_rw",
    vol_size=(96, 96, 96),
    loss_weights=CONDITION_RELATIVE_WEIGHTS_MIRROR,
    num_workers=18,
    gradient_acc_steps=1,
    drop_rate=0.,
    drop_rate_last=0.,
    drop_path_rate=0.,
    aug_prob=0.4,
    out_dim=3,
    stage_1_epochs=20,
    stage_2_epochs=20,
    stage_3_epochs=20,
    epochs=20,
    tune_epochs=4,
    batch_size=12,
    split_rate=0.25,
    split_k=5,
    device=torch.device("cuda") if torch.cuda.is_available() else "cpu",
    seed=2024
)
DATA_BASEPATH = f"{os.path.dirname(os.path.dirname(__file__))}/data/rsna-2024-lumbar-spine-degenerative-classification/"
TRAINING_DATA = retrieve_coordinate_training_data(DATA_BASEPATH)


class CustomMaxxVit3dClassifier(nn.Module):
    def __init__(self,
                 backbone,
                 in_chans=3,
                 out_classes=5,
                 cutpoint_margin=0):
        super(CustomMaxxVit3dClassifier, self).__init__()
        self.out_classes = out_classes

        self.backbone = timm_3d.models.MaxxVit(
            img_size=CONFIG["vol_size"],
            in_chans=in_chans,
            num_classes=out_classes,
            drop_rate=CONFIG["drop_rate"],
            drop_path_rate=CONFIG["drop_path_rate"],
            cfg=MaxxVitCfg(
                embed_dim=(192, 384, 768, 1536),
                depths=(2, 16, 32, 2),
                stem_width=96,
                stem_bias=True,
                head_hidden_size=1536,
                **_rw_max_cfg(
                    rel_pos_type='mlp',
                )
            )
        )

        head_in_dim = self.backbone.head.fc.in_features + 5

        self.backbone.head = LevelInjectorHead(self.backbone.head.fc.in_features, head_in_dim)
        self.backbone.head.drop = nn.Dropout(p=CONFIG["drop_rate_last"])

        self.heads = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(head_in_dim, 3),
            ) for i in range(out_classes)]
        )


    def forward(self, x, level):
        feat = self.backbone.stem(x)
        feat = self.backbone.stages(feat)
        feat = self.backbone.head(feat, level, pre_logits=True)
        # feat = torch.concat([feat, level], dim=1)
        return torch.swapaxes(torch.stack([head(feat) for head in self.heads]), 0, 1)


class LevelInjectorHead(NormMlpClassifierHead):
    def __init__(self, in_feats, out_feats):
        super(LevelInjectorHead, self).__init__(in_feats, out_feats)

    def forward(self, x, level=None, pre_logits=False):
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        if level is not None:
            x = torch.concat([x, level], dim=1)
        x = self.pre_logits(x)
        x = self.drop(x)
        if pre_logits:
            return x
        x = self.fc(x)
        return x


CustomMaxxVit3dClassifier("foo")


class CustomMaxxVit3dClassifierEnsemble(nn.Module):
    def __init__(self,
                 models=Iterable[CustomMaxxVit3dClassifier],
                 out_classes=5
                 ):
        super(CustomMaxxVit3dClassifierEnsemble, self).__init__()
        self.backbones = [model.backbone for model in models]

        for backbone in self.backbones:
            for param in backbone.parameters():
                param.requires_grad = False

        head_in_dim = self.backbones[0].head.fc.out_features

        self.heads = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(head_in_dim * len(self.backbones), 1),
                LogisticCumulativeLink(CONFIG["out_dim"])
            ) for i in range(out_classes)]
        )

    def forward(self, x, level):
        feats = []
        for backbone in self.backbones:
            feat = backbone.stem(x)
            feat = backbone.stages(feat)
            feat = backbone.head(feat, level, pre_logits=True)
            feats.append(feat)

        feat = torch.cat(feats, dim=1)
        return torch.swapaxes(torch.stack([head(feat) for head in self.heads]), 0, 1)


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


def train_stage_2_model_3d(backbone, model_label: str):
    bounds_dataframe = pd.read_csv(os.path.join("data/SpineNet/bounding_boxes_3d.csv"))
    coords_dataframe = pd.read_csv(os.path.join("data/SpineNet/coords_3d.csv"))

    transform_3d_train = tio.Compose([
        tio.RescaleIntensity((0, 1)),
        tio.RandomAffine(translation=(10, 10, 10),
                         degrees=(25, 25, 25),
                         isotropic=True,
                         image_interpolation=CONFIG["image_interpolation"],
                         p=CONFIG["aug_prob"]),
        tio.RandomNoise(p=CONFIG["aug_prob"]),
        tio.RandomSpike(1, intensity=(-0.5, 0.5), p=CONFIG["aug_prob"]),
        tio.RandomBlur(p=CONFIG["aug_prob"]),
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
            nn.CrossEntropyLoss(weight=torch.tensor([1, 2, 4])) for i in range(CONFIG["num_conditions"])
        ],
        "unweighted_val": [
            nn.CrossEntropyLoss() for i in range(CONFIG["num_conditions"])
        ],
    }

    for index, fold in enumerate(dataset_folds):
        model = CustomMaxxVit3dClassifier(backbone=backbone).to(device)
        # model = CustomEfficientformer3dClassifier(backbone=backbone).to(device)
        if index < 2:
            continue
        optimizers = [
            torch.optim.Adam(model.parameters(), lr=3e-4),
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
                                    epochs=CONFIG["epochs"],
                                    freeze_backbone_initial_epochs=-1,
                                    freeze_backbone_after_epochs=-1,
                                    loss_weights=CONFIG["loss_weights"],
                                    gradient_accumulation_per=CONFIG["gradient_acc_steps"]
                                    )

    return model


def tune_stage_2_model_3d(backbone, model_label: str, model_path: str, fold_index: int):
    bounds_dataframe = pd.read_csv(os.path.join("data/SpineNet/bounding_boxes_3d.csv"))
    coords_dataframe = pd.read_csv(os.path.join("data/SpineNet/coords_3d.csv"))

    transform_3d_train = tio.Compose([
        tio.RescaleIntensity((0, 1)),
        tio.RandomAffine(translation=(5, 5, 5),
                         degrees=(5, 5, 5),
                         isotropic=True,
                         image_interpolation=CONFIG["image_interpolation"],
                         p=CONFIG["aug_prob"]),
        tio.RandomNoise(p=CONFIG["aug_prob"]),
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

    fold = dataset_folds[fold_index]
    model = CustomMaxxVit3dClassifier(backbone=backbone).to(device)
    model.load_state_dict(torch.load(model_path))
    optimizers = [
        # torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True),
        torch.optim.Adam(model.parameters(), lr=1e-4),
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


def train_ensemble(backbone, model_label: str, model_path: str, fold_index: int):
    bounds_dataframe = pd.read_csv(os.path.join("data/SpineNet/bounding_boxes_3d.csv"))
    coords_dataframe = pd.read_csv(os.path.join("data/SpineNet/coords_3d.csv"))

    transform_3d_train = tio.Compose([
        tio.ZNormalization(),
        tio.RescaleIntensity((0, 1)),
    ])

    transform_3d_val = tio.Compose([
        tio.ZNormalization(),
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

    fold = dataset_folds[fold_index]
    model = CustomMaxxVit3dClassifier(backbone=backbone).to(device)
    model.load_state_dict(torch.load(model_path))
    optimizers = [
        # torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True),
        torch.optim.Adam(model.parameters(), lr=5e-5),
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
                                freeze_backbone_after_epochs=-1,
                                stage_3_epochs=2,
                                loss_weights=CONFIG["loss_weights"],
                                callbacks=[model._ascension_callback],
                                gradient_accumulation_per=CONFIG["gradient_acc_steps"]
                                )

    return model


def train():
    model = train_stage_2_model_3d(CONFIG['backbone'], f"{CONFIG['backbone']}_{CONFIG['vol_size'][0]}_v2")
    # model = train_model_3d(CONFIG['backbone'], f"{CONFIG['backbone']}_{CONFIG['vol_size'][0]}_3d")

def tune():
    model = tune_stage_2_model_3d(CONFIG['backbone'],
                                  f"{CONFIG['backbone']}_{CONFIG['vol_size'][0]}_9_v2",
                                  "models/maxvit_rmlp_bc_rw_96_v2_fold_1/maxvit_rmlp_bc_rw_96_v2_fold_1_9.pt",
                                  fold_index=1)


if __name__ == '__main__':
    train()
    # tune()
