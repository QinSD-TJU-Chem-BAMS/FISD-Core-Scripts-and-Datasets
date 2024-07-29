

from math import log
import numpy as np
import torch
from Glycan_Formula import calc_glycan_formula, wrap_formula
from Embedding import GLYCAN_DB, FEATURE_VECT, FEATURE_COUNT


EXCLUDED_FEATURES = (9, 10, 17, 24, 26, 27, 28)
EXCLUDED_FEATURES_SET = set(EXCLUDED_FEATURES)
EXCLUDED_FEATURES_SORTED = tuple(sorted(EXCLUDED_FEATURES))


temp_feature_vect_excluded = []
for i in range(FEATURE_COUNT):
    if i in EXCLUDED_FEATURES_SET: continue
    temp_feature_vect_excluded.append(FEATURE_VECT[i])


FEATURE_VECT_EXCLUDED = tuple(temp_feature_vect_excluded)
FEATURE_COUNT_EXCLUDED = len(FEATURE_VECT_EXCLUDED)


def convert_to_excluded_fv(feature_vector: tuple[bool, ...]) -> tuple[int, ...]:
    feature_vect_excluded: list[int] = []
    for i in range(FEATURE_COUNT):
        if i in EXCLUDED_FEATURES_SET: continue
        if feature_vector[i]:
            feature_vect_excluded.append(1)
        else:
            feature_vect_excluded.append(0)
    return tuple(feature_vect_excluded)


def reduce_to_full_feature_index(feature_index_excluded: int) -> int:
    return FEATURE_VECT.index(FEATURE_VECT_EXCLUDED[feature_index_excluded])


def calc_binary_cross_entropy_loss(pred_fv_excluded: tuple[float, ...], db_fv_excluded: tuple[int, ...],
                                   classified_features_excluded: list[int]) -> float:
    h = 0.0
    for i in classified_features_excluded:
        yi = db_fv_excluded[i]
        pi = pred_fv_excluded[i]
        if pi == 0.0: pi = 1e-8
        if pi == 1.0: pi = 1.0 - 1e-8
        h += yi * log(pi) + (1 - yi) * log(1 - pi)
    return (- 1 / len(classified_features_excluded)) * h


def judge_features_covered(fv_excluded: tuple[int, ...], fv_excluded_target: tuple[int, ...]):
    for i in range(FEATURE_COUNT_EXCLUDED):
        if fv_excluded_target[i] == 0: continue
        if fv_excluded[i] == 0: return False
    return True


def predict_feature_vect_excluded(true_fv_excluded: tuple[int, ...], comp_vect: tuple[int, ...],
                                  pred_fv_excluded_probas: tuple[float, ...], weighted_BCE_scorer = None) -> tuple[int, int, tuple[int, ...], float]:

    formula = wrap_formula(calc_glycan_formula(comp_vect))
    candidate_fvs_excluded_set = {convert_to_excluded_fv(fv) for fv in GLYCAN_DB[formula][comp_vect].keys()}
    candidate_fvs_excluded = sorted(candidate_fvs_excluded_set)
    if not true_fv_excluded in candidate_fvs_excluded_set:
        raise Exception(f'True excluded feature vector {true_fv_excluded} was not presented in candidate feature vectors: {candidate_fvs_excluded}')

    classified_features_excluded = []
    for i in range(FEATURE_COUNT_EXCLUDED):
        feature_positive_sum  = 0
        for fv_excluded in candidate_fvs_excluded:
            feature_positive_sum += fv_excluded[i]
        if feature_positive_sum != 0:
            classified_features_excluded.append(i)
    classified_features_count = len(classified_features_excluded)

    true_fv_positive_sum = 0
    for i in classified_features_excluded:
        true_fv_positive_sum += true_fv_excluded[i]

    true_label = 0 if true_fv_positive_sum < 0.5 * classified_features_count else 1

    matched_fv_excluded = candidate_fvs_excluded[0]
    if weighted_BCE_scorer == None:
        h_min = calc_binary_cross_entropy_loss(pred_fv_excluded_probas, matched_fv_excluded, classified_features_excluded)
        for fv_excluded in candidate_fvs_excluded[1:]:
            h = calc_binary_cross_entropy_loss(pred_fv_excluded_probas, fv_excluded, classified_features_excluded)
            if h < h_min:
                matched_fv_excluded = fv_excluded
                h_min = h
    else:
        pred_probas = torch.tensor([pred_fv_excluded_probas], dtype=torch.float32)
        h_min = weighted_BCE_scorer(pred_probas, torch.tensor([matched_fv_excluded], dtype=torch.float32)).item()
        for fv_excluded in candidate_fvs_excluded[1:]:
            h = weighted_BCE_scorer(pred_probas, torch.tensor([fv_excluded], dtype=torch.float32)).item()
            if h < h_min:
                matched_fv_excluded = fv_excluded
                h_min = h

    pred_label = true_label
    if matched_fv_excluded != true_fv_excluded:
        pred_label = int(not true_label)

    proba_sum = 0.0
    for i in classified_features_excluded:
        proba_sum += pred_fv_excluded_probas[i]
    proba_sum /= classified_features_count

    return true_label, pred_label, matched_fv_excluded, proba_sum


def predict_feature_vect_excluded_nonlabel(comp_vect: tuple[int, ...], pred_fv_excluded_probas: tuple[float, ...]) -> tuple[int, ...]:

    formula = wrap_formula(calc_glycan_formula(comp_vect))
    candidate_fvs_excluded = sorted({convert_to_excluded_fv(fv) for fv in GLYCAN_DB[formula][comp_vect].keys()})

    classified_features_excluded = []
    for i in range(FEATURE_COUNT_EXCLUDED):
        feature_positive_sum  = 0
        for fv_excluded in candidate_fvs_excluded:
            feature_positive_sum += fv_excluded[i]
        if feature_positive_sum != 0:
            classified_features_excluded.append(i)
    if len(classified_features_excluded) == 0:
        raise Exception(f'Composition {comp_vect} does not contain any excluded feature vector that can be predicted.')

    matched_fv_excluded = candidate_fvs_excluded[0]

    h_min = calc_binary_cross_entropy_loss(pred_fv_excluded_probas, matched_fv_excluded, classified_features_excluded)
    for fv_excluded in candidate_fvs_excluded[1:]:
        h = calc_binary_cross_entropy_loss(pred_fv_excluded_probas, fv_excluded, classified_features_excluded)
        if h < h_min:
            matched_fv_excluded = fv_excluded
            h_min = h
    return matched_fv_excluded

