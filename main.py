import argparse
from train import Training
import pandas as pd


def args_parser():
    parser = argparse.ArgumentParser(description="Hyperparameter for training process")

    # Model and data hyperparameter
    parser.add_argument('--model_name', type=str, default="Face_Anti_Spoofing",
                        help="Name of model to use")

    parser.add_argument('--output_path', type=str, default='./run', help="Path to save model when training")
    parser.add_argument('--ensemble_dir', type=str, default='none', help="Path to save model when training")
    parser.add_argument('--pretrain_state_dict',
                        default='none',
                        help="Path to pretrain whole model")
    parser.add_argument('--data_path', type=str, default='./dataset/train/videos', help="Path to dataset directory")
    parser.add_argument('--df_path', type=str, default='./dataset/train/label_sr20_frame_10folds.csv', help="Path to dataset directory")
    parser.add_argument('--predict_path', type=str, default='./dataset/public/videos', help="Path to dataset directory")
    parser.add_argument('--sampling_frame_rate', type=int, default=1, help="Sampling frame to predict")
    parser.add_argument('--predict_output_path', type=str, default='./predict', help="Output predict folder")
    parser.add_argument('--image_size', type=int, default=224, help="Input image size")

    # Training hyperparameter
    parser.add_argument('--epoch', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--opt', type=str, default='adam', help="Optimization")
    parser.add_argument('--loss', type=str, default='bce', help="Loss")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--soft_label', type=float, default=0.1, help="Soft label avoid overfitting")
    parser.add_argument('--gpu', action="store_false", help="Not using gpu?")
    parser.add_argument('--eff', action="store_true", help="Using efficientnet")
    parser.add_argument('--kfold', type=int, default=10, help="Number of fold")
    parser.add_argument('--grad_clip', type=float, default=10.0, help="Gradient clipping")

    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=0, help="number of subprocess use to load data")
    parser.add_argument('--early_stop', type=int, default=10,
                        help="Stop the rest epochs when conditions are met aft"
                             "er given epoch")

    hyps = parser.parse_args()
    return hyps


if __name__ == '__main__':
    hyps = args_parser()
    print(hyps)

    df = pd.read_csv(hyps.df_path)
    model_train = Training(hyps)
    for fold in range(hyps.kfold):
        print(f"============== FOLD NUMBER {fold+1} ===============")
        train_df = df[df['fold'] != fold].reset_index(drop=True)
        valid_df = df[df['fold'] == fold].reset_index(drop=True)
        model_train.next_fold(train_df, valid_df, fold)
        model_train.train()
        print(f"=================== DONE =========================")