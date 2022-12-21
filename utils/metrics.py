import torch
import numpy as np
import sklearn.metrics


def compute_eer(label, pred, positive_label=1, acc=True):
    pred = torch.from_numpy(pred)
    pred = torch.sigmoid(pred.float()).squeeze(-1)

    if acc:
        print(f"Prediction min-max-shape: ({pred.min():.2f})-({pred.max():.2f})-{pred.shape}")
        print(f"Accuracy: {sklearn.metrics.accuracy_score(label, (pred>=0.5).int()):.4f}")
    pred = pred.detach().cpu().numpy()
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred, pos_label=positive_label)
    fnr = 1 - tpr

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.argmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.argmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer


if __name__ == '__main__':
    pred = np.random.randint(0, 10, size=(4, 2))
    print(pred)
    idx = np.argmax(pred, axis=1)
    pred[:, 0] = 1 - pred[:, 0]

    pred = pred[np.arange(len(pred)), idx]
    print(pred)