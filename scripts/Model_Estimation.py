

import os
import pickle as pk
import matplotlib.pyplot as plt


KFOLD_RESULT_FILE = 'kfold_results.pkl'
TESTSET_RESULT_FILE = 'testset_results.pkl'


if __name__ == '__main__':

    with open(KFOLD_RESULT_FILE, 'rb') as file:
        accuracies_per_fold, accuracies_per_fold_db, f1_scores_per_fold, f1_scores_per_fold_db,\
        accuracies_total_fold, f1_scores_total_fold, accuracies_total_fold_cov, f1_scores_total_fold_cov = pk.load(file)
    with open(TESTSET_RESULT_FILE, 'rb') as file:
        accuracies_total, accuracies_total_db, f1_scores_total, f1_scores_total_db, roc_data, auc_values,\
        accuracy_total, f1_score_total, accuracy_total_cov, f1_score_total_cov, roc_data_total, auc_value_total = pk.load(file)
    plt.xlim(-0.05, 1.05)
    plt.ylim(0.0, 1.05)
    with open('stdGP_test_result_total.pkl', 'rb') as file:
        accuracy_std, f1_std, accuracy_std_cov, f1_std_cov, roc_data_std, auc_std, feature_roc_data, feature_auc_values = pk.load(file)
    fpr, tpr, thresholds = roc_data_std
    plt.plot(fpr, tpr)

    plt.show()



    test = 0