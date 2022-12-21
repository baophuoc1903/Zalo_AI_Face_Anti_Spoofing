import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from functools import partial
import torch
from model import AntiSpoofingModel
from dataset import FAS_Dataset, transform_image
from torch.utils.data import DataLoader
from utils import compute_eer
import numpy as np
from main import args_parser
import cv2


def check_eer(model, dl, device):
    model.eval()

    with torch.no_grad():
        preds = None
        all_labels = None
        total_loss = 0.0
        num = 0
        criteria = torch.nn.BCEWithLogitsLoss()
        t_bar = tqdm(dl)
        for i, data in enumerate(t_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = model(images)
                loss = criteria(output, labels)

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


def video_predict(model, device, hyps):
    if hyps.ensemble_dir == 'none':
        models = [model]
    else:
        models = []
        pretrained_paths = os.listdir(hyps.ensemble_dir)
        for path in pretrained_paths:
            print(os.path.join(hyps.ensemble_dir, path))
            model = AntiSpoofingModel(eff=hyps.eff).to(device)
            model.load_state_dict(torch.load(os.path.join(hyps.ensemble_dir, path), map_location=device))
            models.append(model)

    for model in models:
        model.eval()

    fnames = os.listdir(hyps.predict_path)
    test_df = pd.DataFrame(fnames)
    test_df.columns = ['fname']

    vid_names = []
    frame_indices = []
    for i, row in test_df.iterrows():
        # np.random.seed(CFG.seed)
        vid_path = os.path.join(hyps.predict_path, row['fname'])
        cap = cv2.VideoCapture(vid_path)

        frame_counts = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.arange(0, frame_counts, hyps.sampling_frame_rate)
        for ind in indices:
            vid_names.append(row['fname'])
            frame_indices.append(ind)

    ind_df = pd.DataFrame({'fname': vid_names, 'frame_index': frame_indices})
    test_df = ind_df.merge(test_df, on=['fname'])

    test_ds = FAS_Dataset(test_df, hyps.predict_path, transforms=partial(transform_image, task='valid'))

    batch_size = hyps.batch_size
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=hyps.num_workers,
                             shuffle=False, pin_memory=True, drop_last=False)

    test_preds = []
    for images, _ in tqdm(test_loader, total=len(test_loader)):
        with torch.no_grad():
            for i, model in enumerate(models):
                if i == 0:
                    y_prob = model(images.to(device)).sigmoid().view(-1).cpu().numpy()
                else:
                    y_prob += model(images.to(device)).sigmoid().view(-1).cpu().numpy()
            test_preds.append(y_prob/len(models))

    test_preds = np.concatenate(test_preds)

    test_df['prob'] = test_preds

    test_df_grouped = test_df.groupby('fname').mean().reset_index()

    sub = test_df_grouped[['fname', 'prob']]
    sub.columns = ['fname', 'liveness_score']

    os.makedirs(hyps.predict_output_path, exist_ok=True)
    sub.to_csv(os.path.join(hyps.predict_output_path, 'Predict.csv'),
               index=False)


if __name__ == '__main__':
    hyps = args_parser()
    print(hyps)
    # df = pd.read_csv(hyps.df_path)

    # valid_df = df[df['fold'] == 0].reset_index(drop=True)
    # valid_ds = FAS_Dataset(valid_df, hyps.data_path, transforms=partial(transform_image, task='valid'))

    # valid_dl = DataLoader(valid_ds, hyps.batch_size, shuffle=False,
    #                       num_workers=hyps.num_workers, pin_memory=True,
    #                       drop_last=False)
    if hyps.gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = AntiSpoofingModel(eff=hyps.eff).to(device)
    if hyps.pretrain_state_dict != 'none':
        model.load_state_dict(torch.load(hyps.pretrain_state_dict, map_location=device))

    # check_eer(model, valid_dl, device)
    video_predict(model, device, hyps)
    # check_eer(model, valid_dl, device)

