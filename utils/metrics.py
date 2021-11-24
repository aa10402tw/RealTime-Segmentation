from sklearn.metrics import confusion_matrix
import numpy as np

def iou(y_pred, y_true, labels):
    y_pred = y_pred.astype(np.int32).flatten()
    y_true = y_true.astype(np.int32).flatten()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    diag = np.diag(cm)
    sum_row = np.sum(cm, axis=0)
    sum_col = np.sum(cm, axis=1)
    ious = []
    cnt = 0
    for d, label in enumerate(labels):
        if (sum_row[d] + sum_col[d] - diag[d]) > 0:
            iou = diag[d] / (sum_row[d] + sum_col[d] - diag[d])
            cnt += 1
        else:
            iou = 0
        ious.append(iou)
    return np.sum(ious) / cnt

def pixel_accuracy(y_pred, y_true, labels):
    y_pred = y_pred.astype(np.int32).flatten()
    y_true = y_true.astype(np.int32).flatten()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    diag = np.diag(cm)
    return diag.sum() / cm.sum()