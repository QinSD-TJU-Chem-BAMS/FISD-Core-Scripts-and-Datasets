

import os
import random
import pickle as pk
import numpy as np
import torch
from math import log
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score
from skmultilearn.model_selection import iterative_train_test_split, IterativeStratification
from Glycan_Sulfated_Peptides_GlycReSoft import get_sulfated_gpsm_meta_infos
from Feature_Prediction import EXCLUDED_FEATURES, FEATURE_COUNT_EXCLUDED, predict_feature_vect_excluded, judge_features_covered


TRAIN_DATA = 'encoded_training_set.pkl'
TRAIN_LABELS = 'encoded_training_set_labels.pkl'
TRAIN_COMPS = 'encoded_training_set_compositions.pkl'


KFOLD_RESULT_FILE = 'kfold_results.pkl'
TESTSET_PRED_FILE = 'testset_predictions.pkl'
TESTSET_RESULT_FILE = 'testset_results.pkl'
TESTSET_RESULT_SULFATE_FILE = 'testset_results_sulfate.pkl'

RUN_KFOLD = False
USE_SAVED_TRAIN_SPLIT_MODEL = True
CONDUCT_SULFATE = True
CREATE_MODEL = False


OVERSAMP_TARGET_RATIO = 0.2
OVERSAMP_MAX_FOLD = 10
OVERSAMP_FEATURE_COUNT = 6
OVERSAMP_RAND_RATIO = 2

SEED = 114


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



class EncodedDataset(Dataset):


    def __init__(self, X: np.array, y: np.array, compositions: np.array):
        if X.shape[0] != y.shape[0] or X.shape[0] != compositions.shape[0]:
            raise Exception(f'Number of samples \"{X.shape[0]}\" should fit with the numbers of their feature vectors \"{y.shape[0]}\" and composition vectors \"{compositions.shape[0]}\".')
        if y.shape[1] != FEATURE_COUNT_EXCLUDED:
            raise Exception(f'Number of feature vector tags \"{y.shape[1]}\" should match with excluded feature count \"{FEATURE_COUNT_EXCLUDED}\".')
        self.X = X
        self.y = y
        self.compositions = compositions
        self.length = X.shape[0]


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.int),\
            torch.tensor(self.compositions[idx], dtype=torch.int)



class MultiLabelNN(nn.Module):


    def __init__(self):
        super(MultiLabelNN, self).__init__()
        self.MLP = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(inplace=False),
        nn.Linear(64, 64),
        nn.ReLU(inplace=False),
        nn.Linear(64, 32),
        nn.ReLU(inplace=False),
        nn.Dropout(0.1),
        nn.Linear(32, 32),
        nn.ReLU(inplace=False),
        nn.Linear(32, FEATURE_COUNT_EXCLUDED),
        )


    def forward(self, x):
        return self.MLP(x)



class WeightedBCELoss(nn.Module):

    def __init__(self, pos_weight):
        super(WeightedBCELoss, self).__init__()
        if len(pos_weight) != FEATURE_COUNT_EXCLUDED:
            raise Exception(f'Number of weights for labels \"{len(pos_weight)}\" should fit with label counts: {FEATURE_COUNT_EXCLUDED}')
        self.pos_weight = pos_weight

    def forward(self, outputs, targets):
        loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(outputs, targets)
        return loss



def train_MLP_model(dataset: EncodedDataset, pos_weight: torch.Tensor, batch_size: int, epochs: int) -> MultiLabelNN:

    if torch.cuda.is_available():
        print('Using GPU')
    else:
        print('GPU not available.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiLabelNN().to(device)
    criterion = WeightedBCELoss(pos_weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

    total_sample = len(dataset)
    epoch_handled_sample = 0
    data_loader = DataLoader(dataset, batch_size=batch_size)

    model.train()
    for epoch in range(epochs):
        for data, feature_vects, comp_vects in data_loader:
            input_data = data.to(device)
            output_data = model(input_data)
            loss = criterion(output_data, feature_vects.to(dtype=torch.float32).to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_handled_sample += batch_size
            print(f'Epoch [{epoch + 1}/{epochs}], Sample [{epoch_handled_sample}/{total_sample}], Loss: {loss.item():.6f}')
        scheduler.step()
        epoch_handled_sample = 0

    return model.cpu().eval()


def save_trained_model(model: MultiLabelNN, file_name):
    while os.path.exists(file_name):
        input(f'Model file already exists: \"{file_name}\", please remove it manually.')
    torch.save(model.state_dict(), file_name)


def load_trained_MLP_model(file_name: str) -> MultiLabelNN:
    if not os.path.exists(file_name):
        raise Exception(f'Model file not found: \"{file_name}\"')
    model = MultiLabelNN()
    model.load_state_dict(torch.load(file_name))
    model.eval()
    return model


def predict_batch_samples(x: torch.Tensor, model: MultiLabelNN) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    with torch.no_grad():
        return torch.sigmoid(model.MLP(x.to(torch.float32).to(device)).cpu())




if __name__ == '__main__':

    file_X = open(TRAIN_DATA, 'rb')
    file_y = open(TRAIN_LABELS, 'rb')
    file_c = open(TRAIN_COMPS, 'rb')

    X = pk.load(file_X)
    y = pk.load(file_y)
    compositions = pk.load(file_c)

    file_X.close()
    file_y.close()
    file_c.close()

    gpsm_meta_path = r'.\Training_GPSMs'
    meta_data_GPSM: list[tuple[str, int, list[tuple[str, int, tuple[str, tuple[int, ...], tuple[int, ...]], tuple[bool, ...]]]]] = []
    gpsm_meta_infos: list[tuple[str, int, tuple[str, tuple[int, ...], tuple[int, ...]], tuple[bool, ...]]] = []
    for root, dirs, files in os.walk(gpsm_meta_path):
        for file_name in files:
            if file_name.endswith('training_GPSMs.pkl'):
                sample = file_name[:file_name.index('_')]
            else:
                continue
            with open(f'{gpsm_meta_path}\\{file_name}', 'rb') as meta:
                meta_data = pk.load(meta)
                GPSMs_count = len(meta_data)
                meta_data_GPSM.append((sample, GPSMs_count, meta_data))
        break
    meta_data_GPSM.sort()
    for sample, gpsm_count, gpsm_infos in meta_data_GPSM:
        gpsm_meta_infos.extend(gpsm_infos)
    if len(gpsm_meta_infos) != X.shape[0]:
        raise Exception(f'Number of total GPSM collected from meta \"{gpsm_meta_path}\" '
                        f'does not match with encoded sample count {X.shape[0]} from \"{TRAIN_DATA}\".')
    for i in range(len(gpsm_meta_infos)):
        sample, spectrum_index, gpid, feature_vector = gpsm_meta_infos[i]
        if feature_vector != tuple(map(bool, y[i])):
            raise Exception(f'Feature vector from meta \"{sample}, {spectrum_index}\" does not match with labels from \"{TRAIN_LABELS}\".')
        if gpid[2] != tuple(compositions[i]):
            raise Exception(f'Composition vector from meta \"{sample}, {spectrum_index}\" does not match with labels from \"{TRAIN_COMPS}\".')

    y = np.delete(y, EXCLUDED_FEATURES, axis=1)

    X_oversamp = []
    y_oversamp = []
    c_oversamp = []
    i_oversamp = []
    label_count_stats = {sum(y[:, i]): i for i in range(FEATURE_COUNT_EXCLUDED)}
    oversamp_target_features = [label_count_stats[i] for i in sorted(label_count_stats.keys())[:OVERSAMP_FEATURE_COUNT]]
    oversamp_original_indices = [i for i in range(y.shape[0]) if sum([y[i, j] for j in oversamp_target_features]) > 0]
    oversamp_original_count = len(oversamp_original_indices)
    oversamp_target_count = X.shape[0] * OVERSAMP_TARGET_RATIO
    if oversamp_target_count > oversamp_original_count * OVERSAMP_MAX_FOLD:
        oversamp_target_count = int(oversamp_original_count * OVERSAMP_MAX_FOLD)
    oversamp_copy_count = int(
        oversamp_target_count // oversamp_original_count + int(oversamp_target_count % oversamp_original_count > 0))
    for j in range(oversamp_copy_count):
        for i in oversamp_original_indices:
            X_oversamp.append(
                [X[i, col_index] * (1.0 + random.randint(-OVERSAMP_RAND_RATIO * 10, OVERSAMP_RAND_RATIO * 10) / 1000.0)
                 for col_index in range(X.shape[1])])
            y_oversamp.append(list(y[i]))
            c_oversamp.append(list(compositions[i]))
            i_oversamp.append(i)
    oversamp_count = len(X_oversamp)
    internal_count = X.shape[0] // oversamp_count
    oversamp_index = 0
    cumulated_count = 0
    X_resamp = []
    y_resamp = []
    c_resamp = []
    i_resamp = []
    for i in range(X.shape[0]):
        X_resamp.append(list(X[i]))
        y_resamp.append(list(y[i]))
        c_resamp.append(list(compositions[i]))
        i_resamp.append(i)
        if cumulated_count == internal_count:
            if oversamp_index < oversamp_count:
                X_resamp.append(list(X_oversamp[oversamp_index]))
                y_resamp.append(list(y_oversamp[oversamp_index]))
                c_resamp.append(list(c_oversamp[oversamp_index]))
                i_resamp.append(i_oversamp[oversamp_index])
                oversamp_index += 1
            cumulated_count = 0
        cumulated_count += 1
    if oversamp_index < oversamp_count:
        X_resamp.extend(X_oversamp[oversamp_index:])
        y_resamp.extend(y_oversamp[oversamp_index:])
        c_resamp.extend(c_oversamp[oversamp_index:])
        i_resamp.extend(i_oversamp[oversamp_index:])

    X: np.array = np.array(X_resamp)
    y: np.array = np.array(y_resamp)
    compositions: np.array = np.array(c_resamp)
    Xci: np.array = np.hstack((X, compositions))
    Xci = np.insert(Xci, Xci.shape[1], i_resamp, axis=1)

    pos_weight = torch.tensor([-log(sum(y[:, i]) / y.shape[0]) for i in range(FEATURE_COUNT_EXCLUDED)],
                              dtype=torch.float32)

    Xci_train, y_train, Xci_test, y_test = iterative_train_test_split(Xci, y, test_size=0.2)
    X_train = Xci_train[:, :X.shape[1]]
    X_test = Xci_test[:, :X.shape[1]]
    c_train = Xci_train[:, X.shape[1]: -1].astype(int)
    c_test = Xci_test[:, X.shape[1]: -1].astype(int)
    i_train = Xci_train[:, -1].astype(int)
    i_test = Xci_test[:, -1].astype(int)


    ### KFold estimation
    if RUN_KFOLD:

        k_fold = IterativeStratification(n_splits=5, order=6)

        accuracies_per_fold = [[] for label in range(FEATURE_COUNT_EXCLUDED)]
        accuracies_per_fold_db = [[] for label in range(FEATURE_COUNT_EXCLUDED)]
        f1_scores_per_fold = [[] for label in range(FEATURE_COUNT_EXCLUDED)]
        f1_scores_per_fold_db = [[] for label in range(FEATURE_COUNT_EXCLUDED)]
        accuracies_total_fold = []
        f1_scores_total_fold = []
        accuracies_total_fold_cov = []
        f1_scores_total_fold_cov = []

        for fold, (train_indices, val_indices) in enumerate(k_fold.split(X_train, y_train)):
            print(f'Fold {fold + 1} training ...')

            X_train_fold, X_val_fold = X_train[train_indices], X_train[val_indices]
            y_train_fold, y_val_fold = y_train[train_indices], y_train[val_indices]
            c_train_fold, c_val_fold = c_train[train_indices], c_train[val_indices]

            dataset = EncodedDataset(X_train_fold, y_train_fold, c_train_fold)

            model = train_MLP_model(dataset, pos_weight, batch_size=64, epochs=150)

            y_pred_fold_probas = predict_batch_samples(torch.tensor(X_val_fold), model).numpy()
            y_pred_fold = (y_pred_fold_probas >= 0.5).astype(int)
            y_pred_fold_db = []
            y_true_total_fold = []
            y_pred_total_fold = []
            y_pred_total_fold_cov = []
            y_pred_total_fold_probas = []
            total_count = 0
            matched_count = 0
            for i in range(y_pred_fold_probas.shape[0]):
                true_fv_excluded = tuple(y_val_fold[i])
                true_label, pred_label, pred_fv_db, pred_proba = predict_feature_vect_excluded(true_fv_excluded,
                                                                                               tuple(c_val_fold[i]),
                                                                                               tuple(y_pred_fold_probas[i]))
                if (true_label == pred_label and true_fv_excluded != pred_fv_db) or (
                        true_label != pred_label and true_fv_excluded == pred_fv_db):
                    raise Exception('Incorrect excluded feature vector match judgement.')
                y_pred_fold_db.append(pred_fv_db)
                y_true_total_fold.append(true_label)
                y_pred_total_fold.append(pred_label)
                if judge_features_covered(pred_fv_db, true_fv_excluded):
                    y_pred_total_fold_cov.append(true_label)
                else:
                    y_pred_total_fold_cov.append(pred_label)
                y_pred_total_fold_probas.append(pred_proba)
                total_count += 1
                if true_label == pred_label: matched_count += 1
            y_pred_fold_db = np.array(y_pred_fold_db)
            print(f'Excluded feature vectors matched [{matched_count}/{total_count}], accuray = {matched_count / total_count}')
            for label in range(FEATURE_COUNT_EXCLUDED):
                y_val = y_val_fold[:, label]
                y_pred = y_pred_fold[:, label]
                y_pred_proba = y_pred_fold_probas[:, label]
                y_pred_db = y_pred_fold_db[:, label]
                accuracy = accuracy_score(y_val, y_pred)
                accuracy_db = accuracy_score(y_val, y_pred_db)
                f1 = f1_score(y_val, y_pred, average='macro')
                f1_db = f1_score(y_val, y_pred_db, average='macro')
                auc = roc_auc_score(y_val, y_pred_proba)
                accuracies_per_fold[label].append(accuracy)
                accuracies_per_fold_db[label].append(accuracy_db)
                f1_scores_per_fold[label].append(f1)
                f1_scores_per_fold_db[label].append(f1_db)
                print(
                    f'Fold {fold + 1}: Feature = {label + 1}, Accuracy = {accuracy}, F1 score = {f1}, Accuracy DB = {accuracy_db}, F1 score DB = {f1_db}, ' +
                    f'AUC = {auc}')
            accuracy_total_fold = accuracy_score(y_true_total_fold, y_pred_total_fold)
            f1_total_fold = f1_score(y_true_total_fold, y_pred_total_fold, average='macro')
            accuracy_total_fold_cov = accuracy_score(y_true_total_fold, y_pred_total_fold_cov)
            f1_total_fold_cov = f1_score(y_true_total_fold, y_pred_total_fold_cov, average='macro')
            accuracies_total_fold.append(accuracy_total_fold)
            f1_scores_total_fold.append(f1_total_fold)
            accuracies_total_fold_cov.append(accuracy_total_fold_cov)
            f1_scores_total_fold_cov.append(f1_total_fold_cov)

        print(accuracies_per_fold)
        print(f1_scores_per_fold)
        print(accuracies_per_fold_db)
        print(f1_scores_per_fold_db)
        print(accuracies_total_fold)
        print(f1_scores_total_fold)
        print(accuracies_total_fold_cov)
        print(f1_scores_total_fold_cov)

        with open(KFOLD_RESULT_FILE, 'wb') as output:
            pk.dump((accuracies_per_fold, accuracies_per_fold_db, f1_scores_per_fold, f1_scores_per_fold_db,
                     accuracies_total_fold, f1_scores_total_fold, accuracies_total_fold_cov, f1_scores_total_fold_cov), output)


    ### Test set split estimation

    dataset = EncodedDataset(X_train, y_train, c_train)

    if USE_SAVED_TRAIN_SPLIT_MODEL:
        model = load_trained_MLP_model('MLP_train_split.pth')
    else:
        model = train_MLP_model(dataset, pos_weight, batch_size=64, epochs=150)

    if CONDUCT_SULFATE:
        gpsm_meta_infos_sulfate = get_sulfated_gpsm_meta_infos()
    else:
        gpsm_meta_infos_sulfate = None

    accuracies_total = []
    accuracies_total_db = []
    f1_scores_total = []
    f1_scores_total_db = []
    roc_data = []
    auc_values = []

    y_test_pred_probas = predict_batch_samples(torch.tensor(X_test), model).numpy()
    y_test_pred = (y_test_pred_probas >= 0.5).astype(int)
    y_test_pred_db = []
    y_true_total = []
    y_pred_total = []
    y_pred_total_cov = []
    y_pred_total_sulfate = []
    y_pred_total_sulfate_cov = []
    y_pred_total_sulfate_fp = []
    y_pred_total_sulfate_cov_fp = []
    y_pred_total_probas = []
    y_pred_total_sulfate_probas = []
    y_pred_total_sulfate_fp_probas = []
    for i in range(y_test_pred_probas.shape[0]):
        true_fv_excluded = tuple(y_test[i])
        true_label, pred_label, pred_fv_db, pred_proba_db = predict_feature_vect_excluded(true_fv_excluded,
                                                                                          tuple(c_test[i]),
                                                                                          tuple(y_test_pred_probas[i]))
        y_true_total.append(true_label)
        y_pred_total.append(pred_label)

        feature_covered = judge_features_covered(pred_fv_db, true_fv_excluded)
        if feature_covered:
            y_pred_total_cov.append(true_label)
        else:
            y_pred_total_cov.append(pred_label)

        if CONDUCT_SULFATE:
            sample_name, spectrum_index, gpid, feature_vector = gpsm_meta_infos[i_test[i]]
            gpids_sulfate = gpsm_meta_infos_sulfate[sample_name]
            if spectrum_index in gpids_sulfate.keys():
                gpid_sulfate = gpids_sulfate[spectrum_index]
                pred_label_sulfate = int(not true_label)
                if gpid[0] == gpid_sulfate[0] and gpid[1] == gpid_sulfate[1]:
                    y_pred_total_sulfate.append(pred_label_sulfate)
                    y_pred_total_sulfate_fp.append(pred_label_sulfate)
                    y_pred_total_sulfate_cov.append(pred_label_sulfate)
                    y_pred_total_sulfate_cov_fp.append(pred_label_sulfate)
                    y_pred_total_sulfate_probas.append(float(pred_label_sulfate))
                    y_pred_total_sulfate_fp_probas.append(float(pred_label_sulfate))
                else:
                    y_pred_total_sulfate.append(pred_label_sulfate)
                    y_pred_total_sulfate_fp.append(pred_label)
                    y_pred_total_sulfate_cov.append(pred_label_sulfate)
                    if feature_covered:
                        y_pred_total_sulfate_cov_fp.append(true_label)
                    else:
                        y_pred_total_sulfate_cov_fp.append(pred_label)
                    y_pred_total_sulfate_probas.append(float(pred_label_sulfate))
                    y_pred_total_sulfate_fp_probas.append(pred_proba_db)
            else:
                y_pred_total_sulfate.append(pred_label)
                y_pred_total_sulfate_fp.append(pred_label)
                if feature_covered:
                    y_pred_total_sulfate_cov.append(true_label)
                    y_pred_total_sulfate_cov_fp.append(true_label)
                else:
                    y_pred_total_sulfate_cov.append(pred_label)
                    y_pred_total_sulfate_cov_fp.append(pred_label)
                y_pred_total_sulfate_probas.append(pred_proba_db)
                y_pred_total_sulfate_fp_probas.append(pred_proba_db)

        y_pred_total_probas.append(pred_proba_db)
        y_test_pred_db.append(list(pred_fv_db))
    y_test_pred_db = np.array(y_test_pred_db)
    for label in range(FEATURE_COUNT_EXCLUDED):
        y_true = y_test[:, label]
        y_pred = y_test_pred[:, label]
        y_pred_db = y_test_pred_db[:, label]
        y_pred_proba = y_test_pred_probas[:, label]
        accuracy = accuracy_score(y_true, y_pred)
        accuracy_db = accuracy_score(y_true, y_pred_db)
        f1 = f1_score(y_true, y_pred, average='macro')
        f1_db = f1_score(y_true, y_pred_db, average='macro')
        roc_curve_data = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        accuracies_total.append(accuracy)
        accuracies_total_db.append(accuracy_db)
        f1_scores_total.append(f1)
        f1_scores_total_db.append(f1_db)
        roc_data.append(roc_curve_data)
        auc_values.append(auc)
        print(
            f'Total: Feature = {label + 1}, Accuracy = {accuracy}, F1 score = {f1}, Accuracy DB = {accuracy_db}, F1 score DB = {f1_db}, ' +
            f'AUC = {auc}')

    accuracy_total = accuracy_score(y_true_total, y_pred_total)
    f1_score_total = f1_score(y_true_total, y_pred_total, average='macro')
    accuracy_total_cov = accuracy_score(y_true_total, y_pred_total_cov)
    f1_score_total_cov = f1_score(y_true_total, y_pred_total_cov, average='macro')
    roc_data_total = roc_curve(y_true_total, y_pred_total_probas)
    auc_value_total = roc_auc_score(y_true_total, y_pred_total_probas)

    print(f'Total accuracy = {accuracy_total}, total F1 score = {f1_score_total}, total accuracy covered = {accuracy_total_cov}, total F1 score covered = {f1_score_total_cov}, total AUC = {auc_value_total}')

    if not USE_SAVED_TRAIN_SPLIT_MODEL:
        save_trained_model(model, 'MLP_train_split.pth')

    if CONDUCT_SULFATE:
        accuracy_total_sulfate = accuracy_score(y_true_total, y_pred_total_sulfate)
        f1_score_total_sulfate = f1_score(y_true_total, y_pred_total_sulfate, average='macro')
        accuracy_total_sulfate_cov = accuracy_score(y_true_total, y_pred_total_sulfate_cov)
        f1_score_total_sulfate_cov = f1_score(y_true_total, y_pred_total_sulfate_cov, average='macro')
        roc_data_total_sulfate = roc_curve(y_true_total, y_pred_total_sulfate_probas)
        auc_value_total_sulfate = roc_auc_score(y_true_total, y_pred_total_sulfate_probas)
        print(f'Total accuracy sulfate = {accuracy_total_sulfate}, total F1 score sulfate = {f1_score_total_sulfate}, ' +
            f'total accuracy sulfate covered = {accuracy_total_sulfate_cov}, total F1 score sulfate covered = {f1_score_total_sulfate_cov}, total sulfate AUC = {auc_value_total_sulfate}')
        accuracy_total_sulfate_fp = accuracy_score(y_true_total, y_pred_total_sulfate_fp)
        f1_score_total_sulfate_fp = f1_score(y_true_total, y_pred_total_sulfate_fp, average='macro')
        accuracy_total_sulfate_fp_cov = accuracy_score(y_true_total, y_pred_total_sulfate_cov_fp)
        f1_score_total_sulfate_fp_cov = f1_score(y_true_total, y_pred_total_sulfate_cov_fp, average='macro')
        roc_data_total_sulfate_fp = roc_curve(y_true_total, y_pred_total_sulfate_fp_probas)
        auc_value_total_sulfate_fp = roc_auc_score(y_true_total, y_pred_total_sulfate_fp_probas)
        print(f'Total accuracy sulfate fp = {accuracy_total_sulfate_fp}, total F1 score sulfate fp = {f1_score_total_sulfate_fp}, ' +
              f'total accuracy sulfate covered fp = {accuracy_total_sulfate_fp_cov}, total F1 score sulfate covered fp = {f1_score_total_sulfate_fp_cov}, total sulfate fp AUC = {auc_value_total_sulfate_fp}')
        with open(TESTSET_RESULT_SULFATE_FILE, 'wb') as output:
            pk.dump((accuracy_total_sulfate, accuracy_total_sulfate_cov, f1_score_total_sulfate, f1_score_total_sulfate_cov, roc_data_total_sulfate, auc_value_total_sulfate,
                     accuracy_total_sulfate_fp, accuracy_total_sulfate_fp_cov, f1_score_total_sulfate_fp, f1_score_total_sulfate_fp_cov, roc_data_total_sulfate_fp, auc_value_total_sulfate_fp,
                     [gpsm_meta_infos[i] for i in i_train], [gpsm_meta_infos[i] for i in i_test]), output)
    else:
        with open(TESTSET_PRED_FILE, 'wb') as output:
            pk.dump((y_test, y_test_pred_db, [gpsm_meta_infos[i] for i in i_test]), output)
        with open(TESTSET_RESULT_FILE, 'wb') as output:
            pk.dump((accuracies_total, accuracies_total_db, f1_scores_total, f1_scores_total_db, roc_data, auc_values,
                     accuracy_total, f1_score_total, accuracy_total_cov, f1_score_total_cov, roc_data_total, auc_value_total), output)


    ### Final model training
    if CREATE_MODEL:
        dataset = EncodedDataset(X, y, compositions)
        model = train_MLP_model(dataset, pos_weight, batch_size=64, epochs=150)
        save_trained_model(model, 'MLP.pth')


