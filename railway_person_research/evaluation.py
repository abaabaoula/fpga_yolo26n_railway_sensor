"""Annotation-relative matching; does not equate unknown detections with proven FP."""
import numpy as np


def iou_matrix(boxes, targets):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 4)
    if not len(boxes) or not len(targets):
        return np.zeros((len(boxes), len(targets)), dtype=np.float64)
    low = np.maximum(boxes[:, None, :2], targets[None, :, :2])
    high = np.minimum(boxes[:, None, 2:], targets[None, :, 2:])
    intersection = np.maximum(high - low, 0).prod(axis=2)
    a = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
    b = np.maximum(targets[:, 2:] - targets[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(a[:, None] + b[None, :] - intersection, np.finfo(float).tiny)


def match_predictions(predictions, gt_boxes, score=.25, iou=.5):
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1, 6)
    selected = np.flatnonzero(predictions[:, 4] >= score)
    selected = selected[np.argsort(-predictions[selected, 4], kind='stable')]
    overlaps = iou_matrix(predictions[:, :4], gt_boxes)
    matched_gt, assignments = set(), []
    for prediction in selected:
        available = [target for target in range(overlaps.shape[1]) if target not in matched_gt and overlaps[prediction, target] >= iou]
        if available:
            target = max(available, key=lambda target: (overlaps[prediction, target], -target))
            matched_gt.add(target)
            assignments.append((int(prediction), target))
    return {'matched_gt': matched_gt, 'assignments': assignments,
            'matched': len(assignments), 'predictions': len(selected),
            'unmatched_predictions': len(selected) - len(assignments), 'ious': overlaps}


def height_group(box):
    height = box[3] - box[1]
    return '<32' if height < 32 else '32-64' if height < 64 else '64-128' if height < 128 else '>=128'


def diagnose_misses(predictions, all_classes, gt_boxes, main_match, primary_score=.25):
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1, 6)
    all_classes = np.asarray(all_classes, dtype=np.float64).reshape(-1, 6)
    overlaps = iou_matrix(predictions[:, :4], gt_boxes)
    other = all_classes[(all_classes[:, 5] != 0) & (all_classes[:, 4] >= .05)]
    other_overlaps = iou_matrix(other[:, :4], gt_boxes)
    categories = []
    for index, box in enumerate(gt_boxes):
        if index in main_match['matched_gt']:
            category = 'matched_at_primary'
        else:
            person_max = float(overlaps[:, index].max()) if len(predictions) else 0.0
            well_localized = np.flatnonzero(overlaps[:, index] >= .5)
            if len(well_localized):
                best_score = float(predictions[well_localized, 4].max())
                category = 'score_suppressed' if best_score < primary_score else 'matching_competition'
            elif len(other) and other_overlaps[:, index].max() >= .5:
                category = 'other_class_overlap'
            elif person_max >= .1:
                category = 'person_localization_weak'
            else:
                category = 'no_person_candidate_at_IoU0.1'
        categories.append(category)
    return categories
