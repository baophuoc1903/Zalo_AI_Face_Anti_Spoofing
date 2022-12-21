from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import pandas as pd
import cv2
from imgaug import augmenters as iaa
import random
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import warnings

warnings.filterwarnings('ignore')
RESIZE_SIZE = int(224 * 1.2)


def random_cropping(image, target_shape=(224, 224, 3), is_random=True):
    image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))
    target_h, target_w, _ = target_shape
    height, width, _ = image.shape
    if is_random:
        start_x = random.randint(0, width - target_w)
        start_y = random.randint(0, height - target_h)
    else:
        start_x = (width - target_w) // 2
        start_y = (height - target_h) // 2

    zeros = image[start_y:start_y + target_h, start_x:start_x + target_w, :]
    return zeros


def random_resize(img, probability=0.5, minRatio=0.2):
    if random.uniform(0, 1) > probability:
        return img

    ratio = random.uniform(minRatio, 1.0)

    h = img.shape[0]
    w = img.shape[1]

    new_h = int(h * ratio)
    new_w = int(w * ratio)

    img = cv2.resize(img, (new_w, new_h))
    img = cv2.resize(img, (w, h))
    return img


def TTA_36_cropps(image, target_shape=(224, 224, 3)):
    image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))

    width, height, d = image.shape
    target_w, target_h, d = target_shape

    start_x = (width - target_w) // 2
    start_y = (height - target_h) // 2

    starts = [[start_x, start_y],

              [start_x - target_w, start_y],
              [start_x, start_y - target_w],
              [start_x + target_w, start_y],
              [start_x, start_y + target_w],

              [start_x + target_w, start_y + target_w],
              [start_x - target_w, start_y - target_w],
              [start_x - target_w, start_y + target_w],
              [start_x + target_w, start_y - target_w],
              ]

    images = []

    for start_index in starts:
        image_ = image.copy()
        x, y = start_index

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        if x + target_w >= RESIZE_SIZE:
            x = RESIZE_SIZE - target_w - 1
        if y + target_h >= RESIZE_SIZE:
            y = RESIZE_SIZE - target_h - 1

        zeros = image_[x:x + target_w, y: y + target_h, :]

        image_ = zeros.copy()

        zeros = np.fliplr(zeros)
        image_flip_lr = zeros.copy()

        zeros = np.flipud(zeros)
        image_flip_lr_up = zeros.copy()

        zeros = np.fliplr(zeros)
        image_flip_up = zeros.copy()

        images.append(image_.reshape([1, target_shape[0], target_shape[1], target_shape[2]]))
        images.append(image_flip_lr.reshape([1, target_shape[0], target_shape[1], target_shape[2]]))
        images.append(image_flip_up.reshape([1, target_shape[0], target_shape[1], target_shape[2]]))
        images.append(image_flip_lr_up.reshape([1, target_shape[0], target_shape[1], target_shape[2]]))

    return images


def transform_image1(image, target_shape=(224, 224, 3), is_infer=False):
    if is_infer:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0),
        ])

        image = augment_img.augment_image(image)
        image = TTA_36_cropps(image, target_shape)
        return image

    else:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-30, 30)),
        ], random_order=True)

        image = augment_img.augment_image(image)
        image = random_resize(image)
        image = random_cropping(image, target_shape, is_random=True)
        return image


def transform_image(image, im_size=224, task='train'):
    if task == 'train':
        tf = A.Compose([A.Downscale(scale_min=0.25, scale_max=0.5, p=0.5),
                        A.Affine(scale=(1.5, 2.0), keep_ratio=True, p=0.5),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.Resize(height=im_size, width=im_size, always_apply=True),
                        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ToTensorV2(always_apply=True),
                        ],
                       p=1.0,
                       )
    else:
        tf = A.Compose([A.Resize(height=im_size, width=im_size, always_apply=True),
                        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ToTensorV2(always_apply=True)],
                       p=1.0,
                       )
    return tf(image=image)['image'].float()


class FAS_Dataset(Dataset):
    def __init__(self, df, video_dir, transforms=transform_image):
        self.df = df.reset_index(drop=True)
        self.video_dir = video_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        vid_name = row['fname']
        vid_path = os.path.join(self.video_dir, vid_name)
        cap = cv2.VideoCapture(vid_path)
        frame_no = row['frame_index']
        cap.set(1, frame_no)  # Where frame_no is the frame you want
        ret, im = cap.read()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        im_ts = self.transforms(im)

        if 'liveness_score' in self.df.columns:
            label = torch.tensor(row['liveness_score']).float()
        else:
            label = -1
        return im_ts, label


if __name__ == '__main__':
    df = pd.read_csv(
        r"/Users/nguyenbaophuoc/Desktop/Studying/My_work/Zalo_AI_Face_Anti_Spoofing/dataset/train/label_3_frame_5folds.csv")
    df = df[df['fold'] == 0]
    from functools import partial
    dataset = FAS_Dataset(df,
                          video_dir=r'/Users/nguyenbaophuoc/Desktop/Studying/My_work/Zalo_AI_Face_Anti_Spoofing/dataset/train/videos',
                          transforms=partial(transform_image, task='train'))

    import matplotlib.pyplot as plt

    train_dl = DataLoader(dataset, batch_size=1, shuffle=True)
    it = iter(train_dl)
    fgx, ax = plt.subplots(nrows=4, ncols=5, figsize=(14, 8))
    ax = ax.flatten()
    for i in range(20):
        image, label = next(it)
        ax[i].imshow(image.squeeze().permute(1, 2, 0))
        ax[i].set_title(label)
        # print(image.max(), image.min())
        # print(label.dtype)
    plt.show()
