

import sys
import numpy as np
import pandas as pd
import pickle as pk
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score
from spectrum_module import MS2_Spectrum
from Traintest_GPSM_Filtration import GLYCAN_DB, GLYCAN_QUERY_StrucGP, get_gpid_StrucGP
from Glycan_Formula import calc_glycan_formula,wrap_formula
from Embedding import get_embedding_fragments, extract_spectrum_embedding_data, embed_to_nparray
from Feature_Prediction import EXCLUDED_FEATURES, predict_feature_vect_excluded, judge_features_covered
from Dataset_Build import check_predictable
from Auto_Encoder import LargeDataset, ConvAutoEncoder, WeightedMSELoss, load_trained_model, encode_batch_samples, decode_batch_samples
from MLP_Classifier import EncodedDataset, MultiLabelNN, load_trained_MLP_model, predict_batch_samples


DATA_FILE = r'.\Search_Results\StrucGP_StdGP\FourStandardGlycoproteins_result.xlsx'
SHEETS = ('Untreated', 'De-sialylated', 'Cut HexNAc', 'Cut Gal Cut HexNAc', 'Cut Fucose')
MZML_FILE_PATH = r'E:\FIon_DNN_data\Raw_Files_mzML'

DATASET_FILE_TAIL = '-stdGP_test_set'
GPSM_FILE_TAIL = '-stdGP_test_GPSMs'



def extract_struct_gpids_StrucGP(data: pd.DataFrame) -> dict[int, tuple[tuple[str, tuple[int, ...], tuple[int, ...]], tuple[bool, ...], str]]:
    extracted_data: dict[int, tuple[tuple[str, tuple[int, ...], tuple[int, ...]], tuple[bool, ...], str]] = {}
    for i in data.index:
        line_data: pd.Series = data.loc[i]
        glycan = line_data['Structure_codinga']
        if not glycan in GLYCAN_QUERY_StrucGP.keys(): continue
        formula_str, comp_vector, feature_vect, glycan_GPSeeker = GLYCAN_QUERY_StrucGP[glycan]
        if not check_predictable(comp_vector, feature_vect): continue
        spectrum_index = int(line_data['MS2Scan'])
        gpid = get_gpid_StrucGP(line_data, comp_vector)
        if spectrum_index in extracted_data:
            print(f'Repetitive ID detected in StrucGP data. Spectrum={spectrum_index}, GPID=\'{gpid}\'')
            continue
        extracted_data[spectrum_index] = (gpid, feature_vect, glycan_GPSeeker)
    return extracted_data


def build_datasets(target_struct_gpids: dict[str, dict[int, tuple[tuple[str, tuple[int, ...], tuple[int, ...]], tuple[bool, ...], str]]],
                   output_gpsm_path: str, output_dataset_path: str):

    first_mzML_loaded = False
    for sample in sorted(target_struct_gpids.keys()):

        training_data: list[tuple[np.array, tuple[bool, ...]]] = []
        selected_GPSMs: list[tuple[str, int, tuple[str, tuple[int, ...], tuple[int, ...]], tuple[bool, ...], str]] = []

        training_gpids = target_struct_gpids[sample]
        with open('rsRaw_args.rsc', 'w') as raw_arg_file:
            raw_arg_file.write(f'{MZML_FILE_PATH}\\{sample}.mzML##rsRawArgs##0##rsRawArgs##100')
        if first_mzML_loaded:
            del sys.modules['parse_mzML']
        elif not 'parse_mzML' in sys.modules.keys():
            first_mzML_loaded = True

        from parse_mzML import mzML_parser

        spectra_pack = mzML_parser.get_next_spectra_package()
        ms2_spectra = spectra_pack['MS2']

        for spectrum_index in sorted(training_gpids.keys()):

            while not spectrum_index in ms2_spectra.keys():
                spectra_pack = mzML_parser.get_next_spectra_package()
                ms1_spectra = spectra_pack['MS1']
                if len(ms1_spectra) == 0:
                    raise Exception(f'MS2 spectrum \"{spectrum_index}\" not found in sample \"{sample}\".')
                ms2_spectra = spectra_pack['MS2']

            ms2_spectrum: MS2_Spectrum = ms2_spectra[spectrum_index]

            gpid, feature_vector, glycan = training_gpids[spectrum_index]

            total_feature_vector, embed_vector = get_embedding_fragments(gpid[2])
            embedding_data = extract_spectrum_embedding_data(ms2_spectrum.spectrum, gpid, embed_vector)
            if len(embedding_data) == 0: continue
            embedded_matrix = embed_to_nparray(embedding_data, total_feature_vector, block=True)
            training_data.append((embedded_matrix, feature_vector))
            selected_GPSMs.append((sample, spectrum_index, gpid, feature_vector, glycan))
            print(f'Sample={sample}, spectrum={spectrum_index}')

        with open(f'{output_gpsm_path}\\{sample}{GPSM_FILE_TAIL}.pkl', 'wb') as output:
            pk.dump(selected_GPSMs, output)

        with open(f'{output_dataset_path}\\{sample}{DATASET_FILE_TAIL}.pkl', 'wb') as output:
            pk.dump(training_data, output)


def build_datasets_for_stdGP_test(output_gpsm_path: str, output_dataset_path: str):
    for sheet in SHEETS:
        data = pd.read_excel(DATA_FILE, sheet_name=sheet)
        samples = sorted(set(data['FileName']))
        target_sample_gpids: dict[str, dict[int, tuple[tuple[str, tuple[int, ...], tuple[int, ...]], tuple[bool, ...], str]]] = {}
        for sample in samples:
            sample_data = data[data['FileName'] == sample]
            struct_gpids = extract_struct_gpids_StrucGP(sample_data)
            target_sample_gpids[sample] = struct_gpids
        build_datasets(target_sample_gpids, output_gpsm_path, output_dataset_path)


def create_stdGP_test_dataset(gpsm_meta_path: str, dataset_path: str, output_file_name: str, batch_size: int):

    dataset = LargeDataset(gpsm_meta_path, dataset_path, DATASET_FILE_TAIL)

    model: ConvAutoEncoder = load_trained_model('ConvAutoEncoder.pth')

    encoded_features: torch.Tensor = torch.tensor([])
    feature_vectors: torch.Tensor = torch.tensor([])
    composition_vectors: torch.Tensor = torch.tensor([])
    first_batch_added = False
    total_sample = len(dataset)
    encoded_sample = 0
    data_loader = DataLoader(dataset, batch_size=batch_size)
    criterion = WeightedMSELoss(0.8, 0.2)
    for data, feature_vects, comp_vects in data_loader:
        encoded_data = encode_batch_samples(data, model)
        if first_batch_added:
            encoded_features = torch.cat((encoded_features, encoded_data), dim=0)
            feature_vectors = torch.cat((feature_vectors, feature_vects), dim=0)
            composition_vectors = torch.cat((composition_vectors, comp_vects), dim=0)
        else:
            encoded_features = encoded_data
            feature_vectors = feature_vects
            composition_vectors = comp_vects
            first_batch_added = True
        decoded_data = decode_batch_samples(encoded_data, model)
        loss = criterion(decoded_data, data)
        encoded_sample += batch_size
        print(f'Encoded samples [{encoded_sample}/{total_sample}], loss={loss.item():.6f}')
    with open(f'{output_file_name}.pkl', 'wb') as output:
        pk.dump(encoded_features.numpy(), output)
    with open(f'{output_file_name}_labels.pkl', 'wb') as output:
        pk.dump(feature_vectors.int().numpy(), output)
    with open(f'{output_file_name}_compositions.pkl', 'wb') as output:
        pk.dump(composition_vectors.numpy(), output)


def test_on_encoded_dataset(file_names: list[str], batch_size: int, output_file_name: str):

    first_file_name = file_names[0]
    with open(f'{first_file_name}.pkl', 'rb') as file:
        X = pk.load(file)
    with open(f'{first_file_name}_labels.pkl', 'rb') as file:
        y = pk.load(file)
        y = np.delete(y, EXCLUDED_FEATURES, axis=1)
    with open(f'{first_file_name}_compositions.pkl', 'rb') as file:
        compositions = pk.load(file)

    if len(file_names) > 1:
        for file_name in file_names[1:]:
            with open(f'{file_name}.pkl', 'rb') as file:
                X = np.vstack((X, pk.load(file)))
            with open(f'{file_name}_labels.pkl', 'rb') as file:
                _y = pk.load(file)
                _y = np.delete(_y, EXCLUDED_FEATURES, axis=1)
                y = np.vstack((y, _y))
            with open(f'{file_name}_compositions.pkl', 'rb') as file:
                compositions = np.vstack((compositions, pk.load(file)))

    dataset = EncodedDataset(X, y, compositions)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    model = load_trained_MLP_model('MLP.pth')


    first_handled = False
    y_true_labels = []
    y_pred_labels = []
    y_pred_labels_cov = []
    y_pred_probas = []
    y_true_fvs = np.array([])
    y_pred_fv_probas = np.array([])
    feature_roc_data = []
    feature_auc_values = []
    total_count = 0
    matched_count = 0
    for data, feature_vects, comp_vects in data_loader:
        feature_vects = feature_vects.numpy()
        comp_vects = comp_vects.numpy()
        pred_probas = predict_batch_samples(data, model).numpy()
        if first_handled:
            y_true_fvs = np.vstack((y_true_fvs, feature_vects))
            y_pred_fv_probas = np.vstack((y_pred_fv_probas, pred_probas))
        else:
            y_true_fvs = feature_vects
            y_pred_fv_probas = pred_probas
            first_handled = True
        for i in range(pred_probas.shape[0]):
            true_fv_excluded = tuple(feature_vects[i])
            true_label, pred_label, pred_fv_excluded, pred_proba = predict_feature_vect_excluded(true_fv_excluded,
                                                                                           tuple(comp_vects[i]), tuple(pred_probas[i]))
            if (true_label == pred_label and true_fv_excluded != pred_fv_excluded) or (
                    true_label != pred_label and true_fv_excluded == pred_fv_excluded):
                raise Exception('Incorrect excluded feature vector match judgement.')
            y_true_labels.append(true_label)
            y_pred_labels.append(pred_label)
            if judge_features_covered(pred_fv_excluded, true_fv_excluded):
                y_pred_labels_cov.append(true_label)
            else:
                y_pred_labels_cov.append(pred_label)
            y_pred_probas.append(pred_proba)
            total_count += 1
            if true_label == pred_label: matched_count += 1
    for i in range(y_true_fvs.shape[1]):
        y_true = y_true_fvs[:, i]
        if sum(y_true) == 0 or sum(y_true) == y_true_fvs.shape[0]:
            feature_roc_data.append(None)
            feature_auc_values.append(0.0)
            continue
        y_pred = y_pred_fv_probas[:, i]
        roc_data = roc_curve(y_true, y_pred)
        auc_value = roc_auc_score(y_true, y_pred)
        feature_roc_data.append(roc_data)
        feature_auc_values.append(auc_value)
    print(f'Excluded feature vectors matched [{matched_count}/{total_count}], accuray = {matched_count / total_count}')

    accuracy_total = accuracy_score(y_true_labels, y_pred_labels)
    f1_total = f1_score(y_true_labels, y_pred_labels, average='macro')
    accuracy_total_cov = accuracy_score(y_true_labels, y_pred_labels_cov)
    f1_total_cov = f1_score(y_true_labels, y_pred_labels_cov, average='macro')
    roc_data_total = roc_curve(y_true_labels, y_pred_probas)
    auc_total = roc_auc_score(y_true_labels, y_pred_probas)

    print(f'Total accuracy = {accuracy_total}, total accuracy covered = {accuracy_total_cov}, total F1 score = {f1_total}, total F1 score covered = {f1_total_cov}, total AUC = {auc_total}')

    with open(f'{output_file_name}.pkl', 'wb') as output:
        pk.dump((accuracy_total, f1_total, accuracy_total_cov, f1_total_cov, roc_data_total, auc_total, feature_roc_data, feature_auc_values), output)




if __name__ == '__main__':

    # build_datasets_for_stdGP_test()

    gpsm_meta = r'E:\FIon_DNN_data\StdGP_Test\target_GPSMs_StdGP'
    dataset_fold = r'E:\FIon_DNN_data\StdGP_Test\datasets_stdGP'

    # for sheet in SHEETS:
    #     gpsm_meta_path = f'{gpsm_meta}\\{sheet}'
    #     dataset_path = f'{dataset_fold}\\{sheet}'
    #     dataset_file = rf'E:\FIon_DNN_data\StdGP_Test\encoded_stdGP_test_set_{sheet}'
    #     create_stdGP_test_dataset(gpsm_meta_path, dataset_path, dataset_file, batch_size=32)
    #     test_on_encoded_dataset([dataset_file], batch_size=64, output_file_name=f'stdGP_test_result_{sheet}')

    combine_list = []
    for sheet in SHEETS:
        gpsm_meta_path = f'{gpsm_meta}\\{sheet}'
        dataset_path = f'{dataset_fold}\\{sheet}'
        dataset_file = rf'E:\FIon_DNN_data\StdGP_Test\encoded_stdGP_test_set_{sheet}'
        combine_list.append(dataset_file)
    test_on_encoded_dataset(combine_list, batch_size=64, output_file_name=f'stdGP_test_result_total')
