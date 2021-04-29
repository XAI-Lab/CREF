import numpy as np
import numba as nb


def calculate_precision_at_k(y_pred, y_true, k):
    """
    Calculate the PrecisionAtK of the explanation method. Calculates the precision for top k features according to the method
    and k ground truth features
    Args:
        k (int): the amount of top features to include in the precision calculation
        y_pred (array): values of an explanation method for a specific record
        y_true (array): the ground truth for the record
    """
    top_values = y_pred[0:k]
    equal_values = np.intersect1d(np.sort(np.array(top_values)), np.sort(np.array(y_true)))
    score = len(equal_values) / len(top_values) if len(top_values) > 0 else 0
    return score


def calculate_RPrecision(y_pred, y_true):
    """
    Calculate the R-Precision of the explanation method. Calculates the precision for top k features according to the method
     and k ground truth features where k is the recall point
    Args:
        y_pred (array): values of an explanation method for a specific record
        y_true (array): the ground truth for the record
    """
    explanation_length = len(y_true)
    score = calculate_precision_at_k(y_pred, y_true, explanation_length)
    return score


def calculate_accuracy(y_pred, y_true):
    """
     Calculate the accuracy of the explanation method by comparing top k features according to the method to k
     ground truth features
     Args:
        y_pred (array): values of an explanation method for a specific record
        y_true (array): the ground truth for the record
    """
    explanation_length = len(y_true)
    top_values = y_pred[0:explanation_length]
    if_equals = np.array_equal(np.sort(np.array(top_values)), np.sort(np.array(y_true)))
    return int(if_equals)


def calculate_average_precision(y_pred, y_true):
    """
     Calculate the average precision, meaning precision at the point were relevant features appear.
     Args:
        y_pred (array): values of an explanation method for a specific record
        y_true (array): the ground truth for the record
    """
    intersection = np.intersect1d(y_true, y_pred)
    intersection_idx = [idx+1 for idx, value in enumerate(y_pred) if value in set(intersection)]
    scores = [calculate_precision_at_k(y_pred, y_true, k_i) for k_i in intersection_idx]
    avg_scores = sum(scores) / len(y_true)
    return avg_scores


def calculate_reciprocal_rank(y_pred, y_true):
    """
     Calculate the reciprocal rank, meaning the rank of the first relevant feature (1/location)
     Args:
        y_pred (array): values of an explanation method for a specific record
        y_true (array): the ground truth for the record
    """
    intersection = np.intersect1d(y_true, y_pred)
    if len(intersection) == 0:
        return 0
    for i in range(y_pred.shape[0]):
        if y_pred[nb.int64(i)] in set(intersection):
            return 1.0 / (i + 1)


def calculate_MAP(y_pred, y_true):
    """
     Calculate the mean average precision, meaning precision at the point were relevant features appear
     across all recors.
     Args:
        y_pred (array): values of an explanation method for a specific record
        y_true (array): the ground truth for the record
    """
    if np.array(y_pred).shape[0] == 1:
        MAP = calculate_average_precision(y_pred, y_true)
    else:
        assert y_pred.shape[0] == y_true.shape[0]
        all_ap = list(map(calculate_average_precision, y_pred, y_true))
        MAP = np.mean(all_ap)
    return MAP


def calculate_MRR(y_pred, y_true):
    """
     Calculate the mean reciprocal rank, meaning the mean of the rank of the first relevant feature (1/location)
     across all records
     Args:
        y_pred (array): values of an explanation method for a specific record
        y_true (array): the ground truth for the record
    """
    if np.array(y_pred).shape[0] == 1:
        MRR = calculate_reciprocal_rank(y_pred, y_true)
    else:
        assert y_pred.shape[0] == y_true.shape[0]
        all_rr = list(map(calculate_reciprocal_rank, y_pred, y_true))
        MRR = np.mean(all_rr)
    return MRR


def calculate_MeanRPrecision(y_pred, y_true):
    """
    Calculate the mean R-Precision of the explanation method, meaning the precision for top k features where k is the recall point
    across all records
    Args:
        y_pred (array): values of an explanation method for a specific record
        y_true (array): the ground truth for the record
    """
    if np.array(y_pred).shape[0] == 1:
        meanRPrecision = calculate_RPrecision(y_pred, y_true)
    else:
        assert y_pred.shape[0] == y_true.shape[0]
        all_rp = list(map(calculate_RPrecision, y_pred, y_true))
        meanRPrecision = np.mean(all_rp)
    return meanRPrecision

