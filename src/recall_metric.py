from typing import List

import numpy as np
# def recall_at_k(y_true, y_prob, k):
#     true_positive = [0] * 38
#     false_negative = [0] * 38
#     recall = 0

#     for i in range(len(y_true)):
#         true = y_true[i]
#         prob = y_prob[i]
#         sorted_prob = [i[0] for i in sorted(enumerate(prob), key=lambda x:x[1])]

#         if true in sorted_prob[:k]:
#             true_positive[true] += 1
#         else:
#             for j in sorted_prob[:k]:
#                 false_negative[j] += 1

#     for i in range(38):
#         try:
#             r = true_positive[i] / (true_positive[i] + false_negative[i])
#         except ZeroDivisionError:
#             r = 0
#         print(r)
#         recall += r

#     print(recall)


def recall_at_k(y_true: List[int], y_pred: List[List[np.ndarray]], k: int):
    """
    Calculates recall at k ranking metric.
    Args:
        y_true: Labels. Not used in the calculation of the metric.
        y_predicted: Predictions.
            Each prediction contains ranking score of all ranking candidates for the particular data sample.
            It is supposed that the ranking score for the true candidate goes first in the prediction.
    Returns:
        Recall at k
    """
    num_examples = float(len(y_pred))
    predictions = np.array(y_pred)
    predictions = np.flip(np.argsort(predictions, -1), -1)[:, :k]
    num_correct = 0
    for el in predictions:
        if 0 in el:
            num_correct += 1
    return float(num_correct) / num_examples


def r_at_1(y_true, y_pred):
    return recall_at_k(y_true, y_pred, k=1)


def r_at_2(y_true, y_pred):
    return recall_at_k(y_true, y_pred, k=2)


def r_at_5(labels, predictions):
    return recall_at_k(labels, predictions, k=5)


def r_at_10(labels, predictions):
    return recall_at_k(labels, predictions, k=10)
