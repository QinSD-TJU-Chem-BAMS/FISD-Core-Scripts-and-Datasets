

import sys
import time
import pickle as pk
import numpy as np
import pandas as pd
import shap
import torch
from torch.utils.data import DataLoader
from analysis_module import Analysis
from Glycan_Formula import calc_glycan_formula, wrap_composition
from Embedding import COMP_COUNT, ION_TYPE_GROUPS, CHARGE_2_THRESHOLD, filter_differential_comps
from Feature_Prediction import EXCLUDED_FEATURES, FEATURE_COUNT_EXCLUDED, convert_to_excluded_fv, reduce_to_full_feature_index
from MLP_Classifier import EncodedDataset, load_trained_MLP_model
from Auto_Encoder import load_trained_model, decode_batch_samples


BATCH_SIZE = 1000

KEY_INPUT_DEGREE_COUNT = 8

OUTPUT_PATH = '.\SHAP_Results'

MZML_FILE_PATH = 'E:\FIon_DNN_data\Raw_Files_mzML'
FIRST_MZML_LOADED = False


def get_feature_importances(encoded_data_file: str, encoded_labels_file: str, feature_index_excluded: int):
    with open(encoded_data_file, 'rb') as file:
        X = pk.load(file)
    with open(encoded_labels_file, 'rb') as file:
        y = pk.load(file)
        y = np.delete(y, EXCLUDED_FEATURES, axis=1)
    if X.shape[0] != y.shape[0]:
        raise Exception(f'Sample count {X.shape[0]} in X does not match with label matrix: {y.shape[0]}')
    indices = []
    for i in range(X.shape[0]):
        if y[i, feature_index_excluded] == 0:
            continue
        indices.append(i)
    target_count = len(indices)
    if target_count > BATCH_SIZE:
        step = target_count / BATCH_SIZE
        X_selected = [list(X[indices[int(round(step * i))]]) for i in range(BATCH_SIZE)]
    else:
        X_selected = [list(X[i]) for i in indices]

    X = torch.tensor(X_selected)
    model = load_trained_MLP_model('MLP.pth')

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU instead.")
    X = X.to(device)
    model.to(device)

    explainer = shap.DeepExplainer(model, X)

    shap_values = explainer.shap_values(X)

    with open(f'{OUTPUT_PATH}\\shap_values_excludedF{feature_index_excluded + 1}.pkl', 'wb') as output:
        pk.dump(shap_values, output)


def stats_feature_importances(feature_index_excluded: int) -> list[int]:
    with open(f'{OUTPUT_PATH}\\shap_values_excludedF{feature_index_excluded + 1}.pkl', 'rb') as shap_file:
        shap_values: np.array = pk.load(shap_file)
    batch_size = shap_values.shape[0]
    input_degree = shap_values.shape[1]
    output_degree = shap_values.shape[2]
    candidate_count = KEY_INPUT_DEGREE_COUNT * 2
    if candidate_count > input_degree:
        candidate_count = input_degree
    if output_degree != FEATURE_COUNT_EXCLUDED:
        raise Exception(f'Output degree \"{output_degree}\" of shape value matrix does not match with excluded feature count: {FEATURE_COUNT_EXCLUDED}.')
    key_input_degrees_stats = [0] * input_degree
    for sample in range(batch_size):
        shap_values_array = shap_values[sample, :, feature_index_excluded]
        threshold = sorted(shap_values_array, reverse=True)[candidate_count - 1]
        for degree in range(input_degree):
            if shap_values_array[degree] < threshold: continue
            key_input_degrees_stats[degree] += 1
    frequency_threshold = sorted(key_input_degrees_stats, reverse=True)[KEY_INPUT_DEGREE_COUNT]
    selected_input_degrees = []
    for degree in range(input_degree):
        if key_input_degrees_stats[degree] > frequency_threshold:
            selected_input_degrees.append(degree)
    return selected_input_degrees


def discover_feature_sources(encoded_data_file: str, encoded_labels_file: str, excluded_feature_indieces: list[int], batch_size: int,
                             impact_const: float = 0.95, differ_threshold: float = 3.0, top_N_hit: int = 5):
    with open(encoded_data_file, 'rb') as file:
        X = pk.load(file)
    with open(encoded_labels_file, 'rb') as file:
        y = pk.load(file)
        y = np.delete(y, EXCLUDED_FEATURES, axis=1)
    if X.shape[0] != y.shape[0]:
        raise Exception(f'Sample count {X.shape[0]} in X does not match with label matrix: {y.shape[0]}')

    model = load_trained_model('ConvAutoEncoder.pth')

    results: list[list[tuple[str, tuple[int, ...], float]]] = []
    for feature_index_excluded in excluded_feature_indieces:
        feature_index = reduce_to_full_feature_index(feature_index_excluded)
        print(f'Current excluded feature = {feature_index_excluded}, original index = {feature_index}')

        X_selected = []
        for i in range(X.shape[0]):
            if y[i, feature_index_excluded] == 0: continue
            X_selected.append(list(X[i]))

        dataset = EncodedDataset(np.array(X_selected), np.zeros((len(X_selected), FEATURE_COUNT_EXCLUDED)), np.zeros((len(X_selected), COMP_COUNT)))
        dataset_size = len(dataset)
        data_loader = DataLoader(dataset, batch_size=batch_size)

        selected_encoded_input_degrees = stats_feature_importances(feature_index_excluded)

        handled_sample = 0
        total_comp_stats: dict[str, dict[tuple[int, ...], int]] = {ion_type_group: {} for ion_type_group in ION_TYPE_GROUPS}
        for data, _y, _c in data_loader:
            batch = data.shape[0]
            handled_sample += batch
            decoded_data_original = decode_batch_samples(data, model).numpy()
            data_mod = data.numpy()
            for i in range(batch):
                for degree in selected_encoded_input_degrees:
                    data_mod[i, degree] *= impact_const
            decoded_data_mod = decode_batch_samples(torch.tensor(data_mod), model).numpy()
            batch_comp_stats = filter_differential_comps(decoded_data_original, decoded_data_mod, feature_index, batch, differ_threshold=differ_threshold)
            for ion_type_group in ION_TYPE_GROUPS:
                compiled_comp_stats = total_comp_stats[ion_type_group]
                comp_stats = batch_comp_stats[ion_type_group]
                for comp_vect, count in comp_stats.items():
                    if comp_vect in compiled_comp_stats.keys():
                        compiled_comp_stats[comp_vect] += count
                    else:
                        compiled_comp_stats[comp_vect] = count
            print(f'Excluded feature {feature_index_excluded}, sample [{handled_sample}/{dataset_size}]')

        total_count = len(X_selected)
        final_featured_ions: list[tuple[str, tuple[int, ...], float]] = []
        for ion_type_group in ION_TYPE_GROUPS:
            comp_stats = total_comp_stats[ion_type_group]
            comp_stats_sorted = [kvp for kvp in comp_stats.items()]
            comp_stats_sorted.sort(key=lambda x: (x[1], x[0]), reverse=True)
            for comp_vect, count in comp_stats_sorted[:top_N_hit]:
                coverage = count / total_count
                final_featured_ions.append((ion_type_group, comp_vect, coverage))
        print(f'Excluded feature {feature_index_excluded}, final featured ions count = {len(final_featured_ions)}')
        print(final_featured_ions)
        results.append(final_featured_ions)

    return results


def search_candidate_feature_signals(total_candidates: list[list[tuple[str, tuple[int, ...], float]]], sample: str, total_spectrum_indices: list[set[int]]) -> list[list[tuple[str, int, tuple[int, ...], bool, float, float, float, float]]]:
    global FIRST_MZML_LOADED
    if FIRST_MZML_LOADED:
        del sys.modules['parse_mzML']
    elif not 'parse_mzML' in sys.modules.keys():
        FIRST_MZML_LOADED = True
    with open('rsRaw_args.rsc', 'w') as raw_arg_file:
        raw_arg_file.write(f'{MZML_FILE_PATH}\\{sample}.mzML##rsRawArgs##0##rsRawArgs##100')
    from parse_mzML import mzML_parser

    total_matched_fragments: list[list[tuple[str, int, tuple[int, ...], bool, float, float, float, float]]] = []
    for candidates, spectrum_indices, excluded_feature_target_index in zip(total_candidates, total_spectrum_indices, range(len(total_candidates))):

        matched_fragments: list[tuple[str, int, tuple[int, ...], bool, float, float, float, float]] = []

        mzML_parser.reset_parser()
        spectra_pack = mzML_parser.get_next_spectra_package()
        ms2_spectra = spectra_pack['MS2']

        for spectrum_index in sorted(spectrum_indices):

            while not spectrum_index in ms2_spectra.keys():
                spectra_pack = mzML_parser.get_next_spectra_package()
                ms1_spectra = spectra_pack['MS1']
                if len(ms1_spectra) == 0:
                    raise Exception(f'MS2 spectrum \"{spectrum_index}\" not found in sample \"{sample}\".')
                ms2_spectra = spectra_pack['MS2']

            spectrum = ms2_spectra[spectrum_index].spectrum

            for ion_type_group, comp_vect, frequency in candidates:
                if not ion_type_group.startswith('B'): continue
                if sum(comp_vect) <= 0: continue
                formula = calc_glycan_formula(comp_vect)
                formula_deH2O = formula.copy()
                formula_deH2O['H'] -= 2
                formula_deH2O['O'] -= 1
                candidate_formula_pair = (formula, formula_deH2O)
                for deH2O in (True, False):
                    iEF1, iMF1 = Analysis.calculate_iEF(candidate_formula_pair[deH2O], 1, is_positive=True)
                    exp_mz1 = Analysis.match_peak(iMF1, spectrum, IPMD=20.0)
                    if exp_mz1 != None:
                        matched_fragments.append((sample, spectrum_index, comp_vect, deH2O, frequency, round(iMF1, 2), exp_mz1, spectrum[exp_mz1]))
                        print(f'Matched: sample={sample}, feature target index={excluded_feature_target_index}, spectrum={spectrum_index}, {ion_type_group}-{wrap_composition(comp_vect)}, m/z={exp_mz1}, z=1, deH2O={deH2O}, freq={frequency}')
                    # elif iMF1 > CHARGE_2_THRESHOLD:
                    #     iMF2 = (iMF1 + 1.0078) / 2
                    #     exp_mz2 = Analysis.match_peak(iMF2, spectrum, IPMD=20.0)
                    #     if exp_mz2 != None:
                    #         print(f'Matched: {ion_type_group}-{wrap_composition(comp_vect)}, m/z={exp_mz2}, z=2, deH2O={deH2O}, freq={frequency}')
        total_matched_fragments.append(matched_fragments)

    with open(f'.\\Feature_Explanation_Results\\{sample}.pkl', 'wb') as output:
        pk.dump(total_matched_fragments, output)
    return total_matched_fragments




if __name__ == '__main__':

    # for i in range(FEATURE_COUNT_EXCLUDED):
    #     print('Start', time.ctime())
    #     get_feature_importances('encoded_training_set.pkl', 'encoded_training_set_labels.pkl', feature_index_excluded=i)
    #     print('End', time.ctime())

    target_excluded_feature_indices = [4, 6, 8, 20, 21]

    feature_candidates = discover_feature_sources('encoded_training_set.pkl', 'encoded_training_set_labels.pkl', target_excluded_feature_indices, impact_const=100.0,
                             batch_size=32, differ_threshold=5.0, top_N_hit=50)

    target_spectra: dict[str, list[set[int]]] = {}
    result_data = pd.read_excel(r'.\Struct_Diag_StdGP_Results\Struct_Diag_StdGP_Results.xlsx', sheet_name=0)
    for i in result_data.index:
        line_data = result_data.loc[i]
        sample = line_data['File']
        if not sample in target_spectra.keys():
            temp = []
            for j in range(len(target_excluded_feature_indices)):
                temp.append(set())
            target_spectra[sample] = temp
        spectrum_index = line_data['MS2 scan']
        fv_full_info = str(line_data['Glycan features (full)']).strip(' ').lstrip('(').rstrip(')').replace(' ', '')
        feature_vect_excluded = convert_to_excluded_fv(tuple(map(lambda x: bool(int(x)), fv_full_info.split(','))))
        for j in range(len(target_excluded_feature_indices)):
            excluded_feature_index = target_excluded_feature_indices[j]
            if feature_vect_excluded[excluded_feature_index] == 0:
                continue
            target_spectra[sample][j].add(spectrum_index)
    result_data = pd.read_excel(r'.\Struct_Diag_TestSplit_Results\Struct_Diag_TestSplit_Results.xlsx', sheet_name=0)
    for i in result_data.index:
        line_data = result_data.loc[i]
        sample = line_data['File']
        if not sample in target_spectra.keys():
            temp = []
            for j in range(len(target_excluded_feature_indices)):
                temp.append(set())
            target_spectra[sample] = temp
        spectrum_index = line_data['MS2 scan']
        fv_excluded_info = str(line_data['Glycan features (reported by StrucGP)']).strip(' ').lstrip('(').rstrip(')').replace(' ', '')
        feature_vect_excluded = tuple(map(int, fv_excluded_info.split(',')))
        for j in range(len(target_excluded_feature_indices)):
            excluded_feature_index = target_excluded_feature_indices[j]
            if feature_vect_excluded[excluded_feature_index] == 0:
                continue
            target_spectra[sample][j].add(spectrum_index)

    with open(f'.\\Feature_Explanation_Results\\target_spectra.pk', 'wb') as target_file:
        pk.dump(target_spectra, target_file)
    print('Target spectra:\n', target_spectra)

    search_results = {}
    for sample in sorted(target_spectra.keys()):
        search_results[sample] = search_candidate_feature_signals(feature_candidates, sample, target_spectra[sample])
    print('Search results:\n', search_results)

    with open(f'.\\Feature_Explanation_Results\\search_results.pk', 'wb') as output:
        pk.dump(search_results, output)

    # stats_result = []
    # for i in range(len(target_excluded_feature_indices)):
    #     excluded_feature_index = target_excluded_feature_indices[i]
    #     total_spectrum_count = 0
    #     current_stats: dict[tuple[tuple[int, ...], bool, float], int] = {}
    #     for sample in sorted(search_results.keys()):
    #         total_spectrum_count += len(target_spectra[sample][i])
    #         matched_fragments = search_results[sample][i]
    #         for fragment in matched_fragments:
    #             if fragment in current_stats.keys():
    #                 current_stats[fragment] += 1
    #             else:
    #                 current_stats[fragment] = 1
    #     stats_result.append((total_spectrum_count, current_stats))
    #
    # with open(f'.\\Feature_Explanation_Results\\stats_result.pk', 'wb') as result_file:
    #     pk.dump(stats_result, result_file)
    #
    # print('Stats results:\n', stats_result)

