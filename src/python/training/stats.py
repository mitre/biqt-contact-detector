'''
NOTICE

This software (or technical data) was produced for the U. S. Government under contract, and is subject to the Rights in Data-General Clause 52.227-14, Alt. IV (DEC 2007) 

(C) 2021 The MITRE Corporation. All Rights Reserved.
Approved for Public Release; Distribution Unlimited. Public Release Case Number 18-0812.
'''
# -*- coding: utf-8 -*-
import os
from typing import List, Optional

import numpy as np
from sklearn.metrics import confusion_matrix
from scipy import interp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from itertools import cycle
from sklearn.metrics import roc_curve, auc


def plot_model_history(history, result_subdir, model_name="Model"):
    """saves plots of accuracy and loss during training of the model"""
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f"{model_name} accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(result_subdir, "accuracies.png"))

    # Plot training & validation loss values
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f"{model_name}  loss")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(result_subdir, "losses.png"))


# adopted from https://gist.github.com/zachguo/10296432
def print_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List] = None,
        hide_zeroes: bool = False,
        hide_diagonal: bool = False,
        hide_threshold: Optional[float] = None,
):
    """Print a nicely formatted confusion matrix with labelled rows and columns.

    Predicted labels are in the top horizontal header, true labels on the vertical header.

    Args:
        y_true (np.ndarray): ground truth labels
        y_pred (np.ndarray): predicted labels
        labels (Optional[List], optional): list of all labels. If None, then all labels present in the data are
            displayed. Defaults to None.
        hide_zeroes (bool, optional): replace zero-values with an empty cell. Defaults to False.
        hide_diagonal (bool, optional): replace true positives (diagonal) with empty cells. Defaults to False.
        hide_threshold (Optional[float], optional): replace values below this threshold with empty cells. Set to None
            to display all values. Defaults to None.
    """
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred)
    # find which fixed column width will be used for the matrix
    columnwidth = max(
        [len(str(x)) for x in labels] + [5]
    )  # 5 is the minimum column width, otherwise the longest class name
    empty_cell = ' ' * columnwidth

    # top-left cell of the table that indicates that top headers are predicted classes, left headers are true classes
    padding_fst_cell = (columnwidth - 3) // 2  # double-slash is int division
    fst_empty_cell = padding_fst_cell * ' ' + 't/p' + ' ' * (columnwidth - padding_fst_cell - 3)

    # Print header
    print('    ' + fst_empty_cell, end=' ')
    for label in labels:
        print(f'{label:{columnwidth}}', end=' ')  # right-aligned label padded with spaces to columnwidth

    print()  # newline
    # Print rows
    for i, label in enumerate(labels):
        print(f'    {label:{columnwidth}}', end=' ')  # right-aligned label padded with spaces to columnwidth
        for j in range(len(labels)):
            # cell value padded to columnwidth with spaces and displayed with 1 decimal
            cell = f'{cm[i, j]:{columnwidth}.1f}'
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=' ')
        print()


# adopted from https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/
def multiclass_roc(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List,
        result_subdir: str,
        line_width: int = 2,
        macros: bool = False,
        semilogx: bool = False,
        semilogy: bool = False,
        style: str="seaborn-colorblind"
):
    plt.style.use(style)

    n_classes = len(labels)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # https://designsystem.digital.gov/design-tokens/color/overview/
    # https://stackoverflow.com/questions/15814635/prettier-default-plot-colors-in-matplotlib
    # Plot all ROC curves
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=line_width,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(labels[i], roc_auc[i]))

    if macros:
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 linestyle=':', linewidth=4)

    if not (semilogx or semilogy):
        plt.plot([0, 1], [0, 1], 'k--', lw=line_width)          # diag

    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle='--', color='grey')
    if semilogx:
        plt.xscale('log')
        xax = plt.gca().xaxis

        xax.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        minxval = []
        maxxval = []
        for i in range(n_classes):
            minxval.append(np.min(fpr[i][np.nonzero(fpr[i])]))
            maxxval.append(np.max(fpr[i]))
        plt.xlim([np.min(minxval), np.max(maxxval)])
        xax.set_ticks(np.arange(np.min(minxval), 1, 0.2))
    else:
        plt.xlim([0.0, 1.0])
    if semilogy:
        plt.yscale('log')
        # todo!
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(result_subdir, "roc.png"))

