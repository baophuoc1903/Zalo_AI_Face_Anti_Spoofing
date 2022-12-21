import os
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from dataset import FAS_Dataset, transform_image
from model import AntiSpoofingModel
from utils.metrics import compute_eer
from torchvision import transforms
import numpy as np
from functools import partial

BIGNUM = 1e8


class Training(object):
    def __init__(self, hyps):
        self.hyps = hyps

        self.train_ds = self.valid_ds = self.test_ds = None
        self.train_dl = self.valid_dl = self.test_dl = None

        self.device = None
        self.set_device(hyps)

        self.model = AntiSpoofingModel(eff=hyps.eff).to(self.device).to(self.device)
        if self.hyps.pretrain_state_dict != 'none':
            self.model.load_state_dict(torch.load(self.hyps.pretrain_state_dict, map_location=self.device))

        if hyps.opt.lower() == 'adam':
            self.opt = optim.AdamW(self.model.parameters(), lr=hyps.lr)
        else:
            self.opt = optim.SGD(self.model.parameters(), lr=hyps.lr, nesterov=True, momentum=0.9)

        if hyps.loss.lower() == 'bce':
            self.loss = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss = torch.nn.BCEWithLogitsLoss()

        # self.scheduler = ReduceLROnPlateau(self.opt, mode='min', factor=0.5, patience=4,
        #                                    verbose=True, min_lr=5e-7, threshold_mode='abs', threshold=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, hyps.epoch - 1, verbose=True)
        self.scaler = torch.cuda.amp.GradScaler()

        self.min_metric = 1.0
        self.max_epoch = -1
        self.patience = 0

    def set_device(self, hyps):
        if hyps.gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def next_fold(self, train_df, valid_df, fold):
        self.model = AntiSpoofingModel(eff=self.hyps.eff).to(self.device)
        self.model._initialize_weights()
        self.df = train_df
        self.valid_df = valid_df
        self.fold = fold
        self.data_set_init(self.hyps)
        self.model = self.model.to(self.device)

        if self.hyps.opt.lower() == 'adam':
            self.opt = optim.AdamW(self.model.parameters(), lr=self.hyps.lr)
        else:
            self.opt = optim.SGD(self.model.parameters(), lr=self.hyps.lr, nesterov=True, momentum=0.9)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.hyps.epoch - 1, verbose=True)
        self.scaler = torch.cuda.amp.GradScaler()

    def data_set_init(self, hyps):
        self.train_ds = FAS_Dataset(self.df, hyps.data_path, transforms=transform_image)

        self.valid_ds = FAS_Dataset(self.valid_df, hyps.data_path, transforms=partial(transform_image, task='valid'))

        self.train_dl = DataLoader(self.train_ds, hyps.batch_size, shuffle=True, num_workers=hyps.num_workers,
                                   pin_memory=True, drop_last=True)
        self.valid_dl = DataLoader(self.valid_ds, hyps.batch_size, shuffle=False,
                                   num_workers=hyps.num_workers, pin_memory=True,
                                   drop_last=False)

    def train(self):
        self.min_metric = 1.0
        self.max_epoch = -1
        self.patience = 0

        for epoch in range(self.hyps.epoch):
            self.train_one_epoch(epoch)
            eer = self.validation(self.valid_dl)

            self.update_best(epoch, eer)

            if eer > self.min_metric:
                self.patience += 1
                if self.patience == self.hyps.early_stop:
                    break
            else:
                self.patience = 0
            self.scheduler.step(eer)

    def train_one_epoch(self, epoch):
        self.model.train()

        preds = None
        all_labels = None
        total_loss = 0.0
        num = 0

        t_bar = tqdm(self.train_dl, desc=f"Epoch {epoch}")
        for i, data in enumerate(t_bar):
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device).unsqueeze(1)
            self.opt.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = self.model(images)
                loss = self.loss(output, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyps.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()

            num += labels.shape[0]
            total_loss += loss.item() * labels.shape[0]
            if preds is not None:
                preds = np.concatenate([preds, output.detach().cpu().numpy()], axis=0)
                all_labels = np.concatenate([all_labels, labels.detach().cpu().numpy()], axis=0)
            else:
                preds = output.detach().cpu().numpy()
                all_labels = labels.detach().cpu().numpy()

            t_bar.set_postfix(loss=total_loss / num, eer=compute_eer(all_labels, preds, acc=False))

        print(f"EER: {compute_eer(all_labels, preds):.4f}")

    def validation(self, dl, task='Validation'):
        print(f"=========={task}=========")
        self.model.eval()

        with torch.no_grad():
            preds = None
            all_labels = None
            total_loss = 0.0
            num = 0

            t_bar = tqdm(dl)
            for i, data in enumerate(t_bar):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device).unsqueeze(1)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = self.model(images)
                    loss = self.loss(output, labels)

                num += labels.shape[0]
                total_loss += loss.item() * labels.shape[0]
                if preds is not None:
                    preds = np.concatenate([preds, output.detach().cpu().numpy()], axis=0)
                    all_labels = np.concatenate([all_labels, labels.detach().cpu().numpy()], axis=0)
                else:
                    preds = output.detach().cpu().numpy()
                    all_labels = labels.detach().cpu().numpy()

                t_bar.set_postfix(loss=total_loss / num, eer=compute_eer(all_labels, preds, acc=False))

            eer_metric = compute_eer(all_labels, preds)
            print(f"Validation EER: {eer_metric:.4f}\n")

        return eer_metric

    def update_best(self, epoch, eer):

        if eer < self.min_metric:
            if os.path.exists(os.path.join(self.hyps.output_path, f'best_model_{self.fold}_{self.min_metric:.4f}.pth')):
                os.remove(os.path.join(self.hyps.output_path, f'best_model_{self.fold}_{self.min_metric:.4f}.pth'))

            self.min_metric = eer
            self.max_epoch = epoch

            if not os.path.exists(self.hyps.output_path):
                os.mkdir(self.hyps.output_path)
            torch.save(self.model.state_dict(),
                       os.path.join(self.hyps.output_path, f'best_model_{self.fold}_{self.min_metric:.4f}.pth'))


if __name__ == '__main__':
    pass