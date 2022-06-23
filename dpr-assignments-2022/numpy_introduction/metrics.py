import numpy as np

def confusion_matrix(y_true, y_pred, normalize=None):
    """Computes the confusion matrix from predictions and labels.

    The matrix columns represent the real labels and the rows represent the
    prediction labels. The confusion matrix is always a 2-D array of shape `[n_labels, n_labels]`,
    where `n_labels` is the number of valid labels for a given classification task. Both
    prediction and labels must be 1-D arrays of the same shape in order for this
    function to work.

    Parameters:
        y_true: 1-D array of real labels for the classification task.
        y_pred: 1-D array of predictions for a given classification.
        normalize: One of ['true', 'pred', 'all', None], corresponding to column sum, row sum, matrix sum, or no
                   normalization.

    Returns:
        A 2-D array with shape `[n_labels, n_labels]` representing the confusion
        matrix, where `n` is the number of possible labels in the classification
        task.
    """

    if normalize not in ['true', 'pred', 'all', None]:
        raise ValueError("normalize must be one of {'true', 'pred', 'all', None}")

    n_labels = np.maximum(np.max(y_true), np.max(y_pred)) + 1

    cm = np.zeros((n_labels, n_labels))
    for i, j in zip(y_pred, y_true):
        cm[i, j] += 1

    if normalize == 'true':
        cm = cm / cm.sum(axis=0, keepdims=True)
    elif normalize == 'pred':
        cm = cm / cm.sum(axis=1, keepdims=True)
    elif normalize == 'all':
        cm = cm / cm.sum()

    return cm


def precision(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, normalize='pred')[1, 1]


def recall(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, normalize='true')[1, 1]


def false_alarm_rate(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, normalize='true')[1, 0]
