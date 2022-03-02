from dataclasses import dataclass
from typing import List

import torch

from detection.metrics.types import EvaluationFrame

torch.multiprocessing.set_sharing_strategy('file_system')
@dataclass
class PRCurve:
    """A precision/recall curve.

    Attributes:
        precision: [N] vector of precision values, where N is the total number of detections.
            The element at index n denotes the precision of the top n detections when ordered by
            decreasing detection scores.
        recall: [N] vector of recall values, where N is the total number of detections.
            The element at index n denotes the recall of the top n detections when ordered by
            decreasing detection scores.
    """

    precision: torch.Tensor
    recall: torch.Tensor


@dataclass
class AveragePrecisionMetric:
    """Stores average precision and its associate precision-recall curve."""

    ap: float
    pr_curve: PRCurve


def compute_precision_recall_curve(
    frames: List[EvaluationFrame], threshold: float
) -> PRCurve:
    """Compute a precision/recall curve over a batch of evaluation frames.

    The PR curve plots the trade-off between precision and recall when sweeping
    across different score thresholds for your detections. To compute precision
    and recall for a score threshold s_i, consider the set of detections with
    scores greater than or equal to s_i. A detection is a true positive if it
    matches a ground truth label; it is a false positive if it does not.

    With this, we define precision = TP / (TP + FP) and recall = TP / (TP + FN),
    where TP is the number of true positive detections, FP is the number of false
    positive detections, and FN is the number of false negative labels (i.e. the
    number of ground truth labels that did not match any detections). By varying
    the score threshold s_i over all detection scores, we have the PR curve.

    What does it mean for a detection to match a ground truth label? In this assignment, we use
    the following definition: A detection matches a ground truth label if: (1) the Euclidean
    distance between their centers is at most `threshold`; and (2) no higher scoring detection
    satisfies condition (1) with respect to the same label.

    Args:
        frames: A batch of evaluation frames, each containing a detection/label pair.
        threshold: Two bounding boxes match if their bird's eye view
            center-to-center distance is strictly less than `threshold`.

    Returns:
        A precision/recall curve.
    """
    # TODO: Replace this stub code.
    # highest detection score & lowest distance
    # for each detection D: closest label L (if not taken yet) ->  else-case
    # sort detections by their scores, for unassigned label etc.

    precisions = []
    recalls = []
    matchings = {}
    total_labels = 0
    for w, frame in enumerate(frames):
        # construct a score and tp, and fn vector
        detections, labels = frame.detections, frame.labels
        N, M = detections.centroids.shape[0], labels.centroids.shape[0]
        total_labels += M
        # sort scores
        scores, idx_for_desc = torch.sort(detections.scores,dim=0, descending=True, stable=True)
        centroids = detections.centroids[idx_for_desc] # detections are now sorted in the descending order of the scores
        tp = torch.tensor([0] * N)
        label_assigned = torch.tensor([0] * M)
        distances = torch.cdist(centroids[None].flatten(2), labels.centroids[None].flatten(2))[0]
        for i in range(N):  # for each detection starting from the detection with the greatest score
            distance_i = distances[i] # distances between detection i and all labels
            j = torch.argmin(distance_i) # get the closest label
            if distances[i][j] <= threshold:
                label_j_distances = distances[:, j]
                detection_scores_j = scores[label_j_distances <= threshold] 
                if scores[i] == torch.max(detection_scores_j): # check max score and have not been assigned
                    tp[i] = 1
                    label_assigned[j] = 1 # change it to have been assigned
        fn = M - torch.sum(tp)
        #print(f"sanity check: {fn + torch.sum(tp) == M}")
        matchings[w] = (scores, tp, fn)

    concat_scores, concat_tp, concat_fn = [], [], []
    for key, (scores, tp, fn) in matchings.items():
        concat_scores.append(scores)
        concat_tp.append(tp)
        concat_fn.append(fn)

    concat_scores = torch.cat(concat_scores)
    concat_tp = torch.cat(concat_tp)
    concat_fn = sum(concat_fn)    

    scores_desc, indices = torch.sort(concat_scores, dim=0, descending=True, stable=True)
    print(indices, scores_desc.shape, concat_scores.shape)
    #print(scores_desc.shape, concat_scores.shape)
    tp_desc = concat_tp[indices]
    fp_desc = 1 - tp_desc
    
    topk_fn = concat_fn
    
    for k in range(1, scores_desc.shape[0]):
        topk_tp = torch.sum(tp_desc[:k])
        topk_fp = torch.sum(fp_desc[:k])
        # topk_tp + topk_fp should be equivalent to k
        # print(topk_tp + topk_fp == k)
        precisions.append(topk_tp/(k))
        # total number of labels it should be torch.sum(tp_desc) + torch.sum(concat_fn)
        #print(torch.sum(tp_desc) == torch.sum(concat_tp))
        #print(torch.sum(tp_desc)+torch.sum(concat_fn) == total_labels)
        recalls.append(topk_tp/(torch.sum(tp_desc)+torch.sum(concat_fn)))
    
    return PRCurve(torch.tensor(precisions), torch.tensor(recalls))


def compute_area_under_curve(curve: PRCurve) -> float:
    """Return the area under the given curve.

    Given a `PRCurve` curve, this function computes the area under the curve as:
        AP = \sum_{i = 1}^{n} (r_i - r_{i - 1}) * p_i
    where r_i (resp. p_i) is the recall (resp. precision) of the top i detections,
    n is the total number of detections, and we set r_0 = 0.0. Intuitively, this
    is computing the integral of the step function defined by the PRCurve.

    Args:
        curve: The precision/recall curve.

    Returns:
        The area under the curve, as defined above.
    """
    # TODO: Replace this stub code.
    recall_minus_1 = torch.cat((torch.tensor([0]),curve.recall[:-1]))
    return torch.sum((curve.recall-recall_minus_1)*curve.precision).item()


def compute_average_precision(
    frames: List[EvaluationFrame], threshold: float
) -> AveragePrecisionMetric:
    """Compute average precision over a batch of evaluation frames.

    Args:
        frames: A batch of evaluation frames, each containing a detection/label pair.
        threshold: Two bounding boxes match if their bird's eye view
            center-to-center distance is strictly less than `threshold`.

    Returns:
        A dataclass consisting of a PRCurve and its average precision.
    """
    # TODO: Replace this stub code.
    pr_curve = compute_precision_recall_curve(frames, threshold)
    ap = compute_area_under_curve(pr_curve)
    return AveragePrecisionMetric(ap, pr_curve)

