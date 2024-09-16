import torch.nn.functional as F
from torch import Tensor
import torchvision.transforms as transforms
import segmentation_models_pytorch_3d as smp3d
import segmentation_models_pytorch as smp
from torchvision.transforms import v2

from training_utils import *
from rsna_dataloader import *

_logger = logging.getLogger(__name__)

CONFIG = dict(
    n_levels=5,
    interpolation="bspline",
    backbone="efficientnet-b5",
    segmentation_type="per_vertebrae", # {per_vertebrae, binary}
    vol_size=(96, 96, 96),
    img_size=(512, 512),
    num_workers=20,
    drop_rate=0.5,
    drop_rate_last=0.1,
    drop_path_rate=0.5,
    aug_prob=0.9,
    out_dim=3,
    epochs=200,
    batch_size=8,
    device=torch.device("cuda") if torch.cuda.is_available() else "cpu",
    seed=2024
)
DATA_BASEPATH = "./data/spine_segmentation_nnunet_v2/"


# TRAINING_DATA = retrieve_segmentation_training_data(DATA_BASEPATH)

# region segment_loss
class DiceCELoss(nn.Module):
    def __init__(self, multiclass=False):
        super().__init__()
        self.multiclass = multiclass
        # self.dice = smp.losses.DiceLoss(mode="multiclass")

    def forward(self, input, target):
        if self.multiclass:
            ce_loss = F.cross_entropy(input, target)
        else:
            ce_loss = F.binary_cross_entropy_with_logits(input, target)
        dice_loss = self.dice_loss(F.softmax(input, dim=1) if self.multiclass else F.sigmoid(input), target)
        return (ce_loss + dice_loss) / 2

    def dice_coeff(self, input: Tensor, target: Tensor, epsilon: float = 1e-6):
        # Average of Dice coefficient for all batches, or for a single mask
        assert input.size() == target.size()
        sum_dim = (-1, -2) if input.dim() == 2 else (-1, -2, -3)

        inter = 2 * (input * target).sum(dim=sum_dim)
        sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

        dice = (inter + epsilon) / (sets_sum + epsilon)
        return dice.mean()

    def multiclass_dice_coeff(self, input: Tensor, target: Tensor, reduce_batch_first: bool = False,
                              epsilon: float = 1e-6):
        # Average of Dice coefficient for all classes
        num_classes = input.shape[1]
        return self.dice_coeff(input.flatten(0, 1),
                               F.one_hot(target, num_classes=num_classes).permute(0, -1, 1, 2, 3).flatten(0, 1),
                               epsilon)

    def dice_loss(self, input: Tensor, target: Tensor):
        # Dice loss (objective to minimize) between 0 and 1
        fn = self.multiclass_dice_coeff if self.multiclass else self.dice_coeff
        return 1 - fn(input, target, reduce_batch_first=True)

class AdaptiveWingLoss(nn.Module):
    """
    Inputs are assumed to be raw logits, and will be normalized+ReLUed
    """
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        '''
        :param pred: BxNxHxHxH
        :param target: BxNxHxHxH
        :return:
        '''

        num_classes = pred.shape[1]
        y = F.one_hot(target, num_classes=num_classes).permute(0, -1, 1, 2, 3)
        y_hat = F.relu(F.normalize(pred))

        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))
# endregion


class RandomFlipIntensity:
    def __call__(self, input: tio.Subject, p=0.5):
        if np.random.random() <= p:
            input = tio.Subject(
                one_image=tio.ScalarImage(tensor=1 - input["one_image"].tensor),
                a_segmentation=tio.LabelMap(tensor=input["a_segmentation"].tensor),
            )
        return input


def train_segmentation_model_3d(model_label: str):
    transform_3d_train = tio.Compose([
        tio.Resize(CONFIG["vol_size"], image_interpolation=CONFIG["interpolation"]),
        tio.RandomAffine(p=CONFIG["aug_prob"]),
        tio.RandomNoise(p=CONFIG["aug_prob"]),
        tio.RandomBlur(p=CONFIG["aug_prob"]),
        tio.RandomAnisotropy(p=CONFIG["aug_prob"]),
        tio.RandomSpike(p=CONFIG["aug_prob"]),
        tio.RandomGamma(p=CONFIG["aug_prob"]),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        RandomFlipIntensity(),
    ])

    transform_3d_val = tio.Compose([
        tio.Resize(CONFIG["vol_size"], image_interpolation=CONFIG["interpolation"]),
        tio.RescaleIntensity(out_min_max=(0, 1)),
    ])

    (trainloader, valloader, test_loader,
     trainset, valset, testset) = create_segmentation_datasets_and_loaders(
        transform_3d_train=transform_3d_train,
        transform_3d_val=transform_3d_val,
        base_path=DATA_BASEPATH,
        num_workers=CONFIG[
            "num_workers"],
        split_factor=0.3,
        batch_size=CONFIG[
            "batch_size"],
        pin_memory=False
    )

    NUM_EPOCHS = CONFIG["epochs"]

    model = smp3d.Unet(
        encoder_name=CONFIG["backbone"],  # choose encoder, e.g. resnet34
        classes=1 if CONFIG["segmentation_type"] == "binary" else 26,  # model input channels (1 for gray-scale volumes, 3 for RGB, etc.)
        in_channels=1,  # model output channels (number of classes in your dataset)
    ).to(device)

    optimizers = [
        torch.optim.Adam(model.parameters(), lr=1e-3),
    ]

    schedulers = [
    ]

    criteria = {
        "train": [
            DiceCELoss(multiclass=True)
        ],
        "val": [
        ]
    }

    train_model_with_validation(model,
                                optimizers,
                                schedulers,
                                criteria,
                                trainloader,
                                valloader,
                                model_desc=model_label,
                                train_loader_desc=f"Training {model_label}",
                                epochs=NUM_EPOCHS,
                                freeze_backbone_initial_epochs=0,
                                )


def train_segmentation_model_2d(data_type: str, model_label: str):
    transform_2d_train = transforms.Compose([
        # transforms.Resize(CONFIG["img_size"]),
        # tio.RandomAffine(p=CONFIG["aug_prob"]),
        # transforms.RandomNoise(p=CONFIG["aug_prob"]),
        # transforms.RandomBlur(p=CONFIG["aug_prob"]),
        # transforms.RandomAnisotropy(p=CONFIG["aug_prob"]),
        # transforms.RandomSpike(p=CONFIG["aug_prob"]),
        # transforms.RandomGamma(p=CONFIG["aug_prob"]),
        # v2.AutoAugment(),
        v2.RandomChoice((
            v2.GaussianBlur(kernel_size=(3, 3)),
            v2.GaussianBlur(kernel_size=(5, 5)),
            v2.GaussianBlur(kernel_size=(7, 7)),
            v2.Identity()
        ),
            p=[0.2, 0.2, 0.2, 0.4]),
        v2.RandomChoice((
            v2.ColorJitter(),
            v2.Identity()
        ), p=[0.7, 0.3]),
        transforms.Normalize(mean=0.5, std=0.5),
    ])

    transform_2d_val = transforms.Compose([
        # transforms.Resize(CONFIG["img_size"]),
        transforms.Normalize(mean=0.5, std=0.5),
    ])

    (trainloader, valloader, test_loader,
     trainset, valset, testset) = create_series_level_segmentation_datasets_and_loaders(TRAINING_DATA,
                                                                                        data_type=data_type,
                                                                                        transform_2d_train=transform_2d_train,
                                                                                        transform_2d_val=transform_2d_val,
                                                                                        base_path=os.path.join(
                                                                                            DATA_BASEPATH,
                                                                                            "train_images"),
                                                                                        num_workers=CONFIG[
                                                                                            "num_workers"],
                                                                                        split_factor=0.3,
                                                                                        batch_size=CONFIG[
                                                                                            "batch_size"],
                                                                                        pin_memory=False
                                                                                        )

    NUM_EPOCHS = CONFIG["epochs"]
    # model = UNet3D(n_channels=num_channels, n_classes=CONFIG["n_levels"]).to(device)

    model = smp.Unet(
        classes=CONFIG["n_levels"]
    ).to(device)
    optimizers = [
        torch.optim.Adam(model.parameters(), lr=5e-2),
    ]

    schedulers = [
    ]

    criteria = {
        "train": [
            DiceCELoss() for i in range(CONFIG["n_levels"])
        ],
        "val": [
            nn.BCEWithLogitsLoss() for i in range(CONFIG["n_levels"])
        ]
    }

    train_model_with_validation(model,
                                optimizers,
                                schedulers,
                                criteria,
                                trainloader,
                                valloader,
                                model_desc=model_label,
                                train_loader_desc=f"Training {model_label}",
                                epochs=NUM_EPOCHS,
                                freeze_backbone_initial_epochs=0,
                                )


def train():
    # model = train_segmentation_model_3d(f"{CONFIG['backbone']}_unet_segmentation_{CONFIG['vol_size'][0]}_3d_{CONFIG['segmentation_type']}")
    model = train_segmentation_model_2d(
        f"{CONFIG['backbone']}_unet_segmentation_{CONFIG['img_size'][0]}_2d_{CONFIG['segmentation_type']}")
    # torch.multiprocessing.set_start_method('spawn')


if __name__ == '__main__':
    train()
