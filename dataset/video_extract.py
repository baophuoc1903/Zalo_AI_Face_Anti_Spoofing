import cv2
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def video_to_frame(task='train'):
    data_path = Path(__file__).parents[0].joinpath(task)
    label_df = {}

    if task == 'train':
        label_path = data_path.joinpath('label.csv')
        label_df = pd.read_csv(label_path)

    videos_path = []
    for _, dir_name, filenames in os.walk(data_path.joinpath('videos')):
        videos_path.extend(filenames)
    if not os.path.exists(data_path.joinpath('images')):
        os.mkdir(data_path.joinpath('images'))

    mapping = {}
    if task == 'train':
        mapping = {"images": [], 'label': []}

    for vid_name in tqdm(videos_path):
        vid_path = Path(data_path).joinpath('videos', vid_name)
        vs = cv2.VideoCapture(str(vid_path))
        length = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
        read = 0
        while vs.isOpened():
            grabbed, frame = vs.read()
            if not grabbed:
                break

            read += 1
            if read != length // 2 and read != length*3//4 and read != length // 4:
                continue
            cv2.imwrite(str(data_path.joinpath('images', f"{vid_path.stem}_frame_{read}.jpeg")), frame)

            mapping['images'].append(f"{vid_path.stem}_frame_{read}.jpeg")
            if task == 'train':
                label = label_df[label_df['fname'] == vid_name]['liveness_score'].squeeze()
                mapping['label'].append(label)

            # if read >= 100:
            #     break
        vs.release()

    if task == 'train':
        df = pd.DataFrame(mapping)
        df.to_csv(str(data_path.joinpath('img_label.csv')), index=False)


if __name__ == '__main__':
    video_to_frame()


