import math

import timm
import timm_3d
import torchio as tio
from spacecutter.losses import CumulativeLinkLoss
from spacecutter.models import LogisticCumulativeLink
from spacecutter.callbacks import AscensionCallback
from torch.autograd import Variable

from training_utils import *
from rsna_dataloader import *

_logger = logging.getLogger(__name__)

# See: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/

# region utils
class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.shape
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x #, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1) #, trans, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
# endregion



CONFIG = dict(
    n_levels=5,
    num_classes=25,
    num_workers=6,
    drop_rate=0.5,
    drop_rate_last=0.1,
    drop_path_rate=0.5,
    aug_prob=0.7,
    out_dim=3,
    epochs=25,
    batch_size=2,
    gradient_accumulation_per=12,
    split_rate=0.2,
    downsampling_rate=3,
    device=torch.device("cuda") if torch.cuda.is_available() else "cpu",
    seed=2024
)
DATA_BASEPATH = "./data/rsna-2024-lumbar-spine-degenerative-classification/"
TRAINING_DATA = retrieve_coordinate_training_data(DATA_BASEPATH)

CLASS_RELATIVE_WEIGHTS = torch.Tensor([[1., 29.34146341, 601.5, ],
                                       [1., 10.46296296, 141.25, ],
                                       [1., 3.6539924, 43.68181818, ],
                                       [1., 1.89223058, 8.20652174, ],
                                       [1., 2.31736527, 5.60869565, ],
                                       [1., 19.46666667, 64.88888889, ],
                                       [1., 6.30674847, 18.69090909, ],
                                       [1., 2.92041522, 7.46902655, ],
                                       [1., 1.5144357, 2.00347222, ],
                                       [1., 3.43076923, 9.4893617, ],
                                       [1., 27.11363636, 132.55555556, ],
                                       [1., 10.5, 283.5, ],
                                       [1., 3.65267176, 35.44444444, ],
                                       [1., 2.05277045, 8.74157303, ],
                                       [1., 2.75333333, 6.88333333, ],
                                       [1., 14.59493671, 82.35714286, ],
                                       [1., 6.32926829, 23.59090909, ],
                                       [1., 2.82828283, 7.70642202, ],
                                       [1., 1.43367347, 1.92465753, ],
                                       [1., 3.57429719, 8.31775701, ],
                                       [1., 29.04878049, 85.07142857, ],
                                       [1., 11.31632653, 28.43589744, ],
                                       [1., 7.16083916, 12.96202532, ],
                                       [1., 6.25675676, 5.38372093, ],
                                       [1., 44.66666667, 92.76923077],
                                       ]).to(CONFIG["device"])

CLASS_LOGN_RELATIVE_WEIGHTS = 1 + 2 * torch.log(CLASS_RELATIVE_WEIGHTS)

# !TODO: Refactor
CLASS_RELATIVE_WEIGHTS_MIRROR = CLASS_RELATIVE_WEIGHTS.clone()
CLASS_RELATIVE_WEIGHTS_MIRROR[0:10] += CLASS_RELATIVE_WEIGHTS_MIRROR[10:20]
CLASS_RELATIVE_WEIGHTS_MIRROR[0:10] /= 2
CLASS_RELATIVE_WEIGHTS_MIRROR[10:20] = CLASS_RELATIVE_WEIGHTS_MIRROR[0:10]

CLASS_LOGN_RELATIVE_WEIGHTS_MIRROR = 1 + torch.log(CLASS_RELATIVE_WEIGHTS_MIRROR)

CLASS_NEG_VS_POS = torch.Tensor(
    [3.57439734e-02, 2.93902439e+01, 6.22000000e+02,
     1.02654867e-01, 1.05370370e+01, 1.54750000e+02,
     0.29656608, 3.73764259, 55.63636364,
     0.65033113, 2.12280702, 12.54347826,
     0.60981912, 2.73053892, 8.02898551,
     6.67808219e-02, 1.97666667e+01, 6.82222222e+01,
     0.21206226, 6.64417178, 21.65454545,
     0.47630332, 3.31141869, 10.02654867,
     1.15944541, 2.27034121, 3.32638889,
     0.39686099, 3.79230769, 12.25531915,
     4.44258173e-02, 2.73181818e+01, 1.37444444e+02,
     9.87654321e-02, 1.05370370e+01, 3.10500000e+02,
     0.30198537, 3.75572519, 45.14814815,
     0.60154242, 2.28759894, 13.,
     0.50847458, 3.15333333, 9.38333333,
     8.06591500e-02, 1.47721519e+01, 8.80000000e+01,
     0.20038536, 6.59756098, 27.31818182,
     0.48333333, 3.1952862, 10.43119266,
     1.21708185, 2.17857143, 3.26712329,
     0.4, 4.00401606, 10.64485981,
     4.61796809e-02, 2.93902439e+01, 8.80000000e+01,
     0.12353472, 11.71428571, 30.94871795,
     0.21679688, 7.71328671, 14.7721519,
     0.34557235, 7.41891892, 6.24418605,
     3.31674959e-02, 4.51481481e+01, 9.48461538e+01]
).to(CONFIG["device"])

COMP_WEIGHTS = torch.Tensor([[1, 2, 4] for i in range(25)]).to(CONFIG["device"])

CONFIG["loss_weights"] = CLASS_LOGN_RELATIVE_WEIGHTS_MIRROR

class PointNet_Model_3D_Multihead(nn.Module):
    def __init__(self, in_chans=4, out_classes=5, out_dim=3):
        super(PointNet_Model_3D_Multihead, self).__init__()
        self.out_classes = out_classes

        self.encoder = PointNetEncoder(global_feat=True, feature_transform=True, channel=in_chans)
        head_in_dim = 1024
        self.encoder.classifier = nn.Identity()
        self.heads = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(head_in_dim, 1),
                LogisticCumulativeLink(CONFIG["out_dim"])
            ) for i in range(out_classes)]
        )

        self.ascension_callback = AscensionCallback()

    def forward(self, x):
        feat = self.encoder(x)
        return torch.swapaxes(torch.stack([head(feat) for head in self.heads]), 0, 1)

    def _ascension_callback(self):
        for head in self.heads:
            self.ascension_callback.clip(head[1])

def train_model_pcd(model_label: str):
    (trainloader, valloader, test_loader,
     trainset, valset, testset) = create_subject_level_pcd_datasets_and_loaders(TRAINING_DATA,
                                                                            # transform_3d_train=transform_3d_train,
                                                                            # transform_3d_val=transform_3d_val,
                                                                            base_path=os.path.join(
                                                                                DATA_BASEPATH,
                                                                                "train_images"),
                                                                            num_workers=CONFIG["num_workers"],
                                                                            split_factor=CONFIG["split_rate"],
                                                                            batch_size=CONFIG["batch_size"],
                                                                            downsampling_factor=CONFIG['downsampling_rate'],
                                                                            pin_memory=False
                                                                            )

    NUM_EPOCHS = CONFIG["epochs"]

    model = PointNet_Model_3D_Multihead(out_classes=CONFIG["num_classes"]).to(device)
    optimizers = [
        torch.optim.Adam(model.encoder.parameters(), lr=5e-5),
        torch.optim.Adam(model.heads.parameters(), lr=1e-3),
    ]
    schedulers = [
    ]
    criteria = {
        "train": [
            CumulativeLinkLoss(class_weights=CONFIG["loss_weights"][i]) for i in range(CONFIG["num_classes"])
        ],
        "val": [
            CumulativeLinkLoss() for i in range(CONFIG["num_classes"])
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
                                loss_weights=CONFIG["loss_weights"],
                                callbacks=[model._ascension_callback],
                                gradient_accumulation_per=CONFIG["gradient_accumulation_per"],
                                )

    return model


def train():
    model = train_model_pcd(f"pointnet_downsampled_{CONFIG['downsampling_rate']}")


if __name__ == '__main__':
    train()
