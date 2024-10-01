import os
import gc
import time
import torch
import numpy as np
import torch.cuda.amp as amp
import wandb
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

from config import Config
from loss import criterion
from utils import transform_train, transform_val
from dataset import StudyPerVertebraLevelDataset, get_folds
from model import GradingModel


def train_func(model, loader_train, optimizer, scaler=None):
    model.train()
    train_loss = []
    bar = tqdm(loader_train)
    for images, targets in bar:
        optimizer.zero_grad()
        images = images.cuda()
        targets = targets.cuda()

        with amp.autocast():
            logits = model(images)
            loss = criterion(logits, targets)

        train_loss.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bar.set_description(f'Train Loss:{np.mean(train_loss[-30:]):.4f}')

    return np.mean(train_loss)


def valid_func(model, loader_valid):
    model.eval()
    valid_loss = []
    gts = []
    outputs = []
    bar = tqdm(loader_valid)
    with torch.no_grad():
        for images, targets in bar:
            images = images.to(Config.device)
            targets = targets.to(Config.device)

            logits = model(images)
            loss = criterion(logits, targets)

            gts.append(targets.cpu())
            outputs.append(logits.cpu())
            valid_loss.append(loss.item())

            bar.set_description(f'Valid Loss:{np.mean(valid_loss[-30:]):.4f}')

    outputs = torch.cat(outputs)
    gts = torch.cat(gts)
    valid_loss = criterion(outputs, gts, "valid").item()

    return valid_loss


def run(fold):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(Config.model_dir, exist_ok=True)

    wandb.init(project='Lumbar Spine disease Detection', name=f'CLS_Fold_{fold}', config={
               'batch_size': Config.batch_size, 'learning_rate': Config.lr})

    model_file = os.path.join(
        Config.model_dir, f'{Config.backbone}_fold{fold}_best.pth')

    df = get_folds(pd.read_csv(Config.data_basepath + "processed_studies/preprocessed_train.csv"))

    _train = df[df['fold'] != fold].reset_index(drop=True)
    _valid = df[df['fold'] == fold].reset_index(drop=True)

    dataset_train = StudyPerVertebraLevelDataset(_train, transform=transform_train)
    dataset_valid = StudyPerVertebraLevelDataset(_valid, transform=transform_val)

    loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers, drop_last=True)
    loader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)

    model = GradingModel(
        backbone=Config.backbone,
        
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr, betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler()

    metric_best = np.inf

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    for epoch in range(1, Config.n_epochs+1):
        scheduler.step(epoch-1)

        print(time.ctime(), 'Epoch:', epoch)

        train_loss = train_func(model, loader_train, optimizer, scaler)
        valid_loss = valid_func(model, loader_valid)
        metric = valid_loss

        wandb.log({
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'metric': metric,
            'learning_rate': optimizer.param_groups[0]["lr"]
        })

        if metric < metric_best:
            print(
                f'metric_best ({metric_best:.6f} --> {metric:.6f}). Saving model ...')
            torch.save(model.state_dict(), model_file)
            wandb.save(model_file)
            metric_best = metric

        # Save Last
        if not Config.debug:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'score_best': metric_best,
                },
                model_file.replace('_best', '_last')
            )

    wandb.finish()
    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    for i in range(Config.n_folds):
        run(i)