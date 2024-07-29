

import os
import sys
from math import log
import pickle as pk
from typing import Union
import numpy as np
import pandas as pd
from analysis_module import Analysis
from spectrum_module import MS2_Spectrum
from Traintest_GPSM_Filtration import GLYCAN_DB, DATA_FOLD, ENGINE_StrucGP, SAMPLE_TITLES, SHEET_SUFFIXES
from Peptide_Backbone import calc_peptide_backbone_formula, wrap_mods
from Glycan_Formula import calc_glycan_formula, wrap_formula, wrap_composition
from Glycan_Fragmentation import get_glycan_theo_fragments, calc_struct_diagnostic_ions_typeDistin
from Embedding import CHARGE_2_THRESHOLD
from Feature_Prediction import FEATURE_COUNT_EXCLUDED, convert_to_excluded_fv
from Model_Test_StdGP import extract_struct_gpids_StrucGP



MZML_FILE_PATH = 'E:\FIon_DNN_data\Raw_Files_mzML'


def get_theo_sdion_assignment(gpid: tuple[str, tuple[int, ...], tuple[int, ...]], feature_vect_full_or_excluded: Union[tuple[int, ...], tuple[bool, ...]])\
        -> dict[str, tuple[list[tuple[str, dict[str, int]]], list[tuple[str, dict[str, int]]]]]:
    comp_vect = gpid[2]
    glycan_formula = wrap_formula(calc_glycan_formula(comp_vect))
    target_glycans = []
    feature_group = GLYCAN_DB[glycan_formula][comp_vect]
    if len(feature_vect_full_or_excluded) == FEATURE_COUNT_EXCLUDED:
        for fv in sorted(feature_group.keys()):
            fv_excluded = convert_to_excluded_fv(fv)
            if fv_excluded != feature_vect_full_or_excluded: continue
            target_glycans.extend(feature_group[fv])
    else:
        target_glycans = feature_group[feature_vect_full_or_excluded]
    if len(target_glycans) == 0:
        raise Exception(f'Excluded feature vector \"{feature_vect_full_or_excluded}\" does not match with any feature vector in the glycan DB.')
    peptide_backbone_formula = calc_peptide_backbone_formula(gpid[0], gpid[1])
    feature_vect_full_or_excluded_str = ''.join(map(lambda x: str(int(x)), feature_vect_full_or_excluded))
    theo_fragments = get_glycan_theo_fragments(target_glycans, output_formula=True, calc_internal_ions=False,
                                               cache_name=f'{wrap_composition(comp_vect)}-{feature_vect_full_or_excluded_str}_1_0')
    sdions = calc_struct_diagnostic_ions_typeDistin(theo_fragments)
    sdions_assigned: dict[str, tuple[list[tuple[str, dict[str, int]]], list[tuple[str, dict[str, int]]]]] = {}
    for glycan, (sdion_array_Btype, sdion_array_Ytype) in sdions.items():
        if len(sdion_array_Ytype) == 0:
            sdions_assigned[glycan] = (sdion_array_Btype, [])
            continue
        sdions_assigned[glycan] = (sdion_array_Btype,
                                   [(ion_info[0], Analysis.combine_formula(ion_info[1], peptide_backbone_formula)) for ion_info in sdion_array_Ytype])
    return sdions_assigned


def match_glycan_by_struct_diagnosis(sdions_assigned: dict[str, tuple[list[tuple[str, dict[str, int]]], list[tuple[str, dict[str, int]]]]],
                                     spectrum: dict[float, float]) -> tuple[dict[float, tuple[str, list[tuple[str, float]]]], list[float], list[str], list[str]]:
    match_results: dict[float, tuple[str, list[tuple[str, float]]]] = {}
    glycans_with_no_sdion_matched: list[str] = []
    glycans_with_no_theo_sdion: list[str] = []
    for glycan, (sdion_array_Btype, sdion_array_Ytype) in sdions_assigned.items():
        theo_sdion_count = len(sdion_array_Btype) + len(sdion_array_Ytype)
        if theo_sdion_count == 0:
            glycans_with_no_theo_sdion.append(glycan)
            print(f'Glycan with no theoretical structure-diagnostic ion: {glycan}')
            continue
        matched_sdion_count = 0
        matched_sdion_intensity_sum = 0.0
        current_match_result = []
        for ion_name, formula in sdion_array_Btype:
            matched = False
            iEF1, iMF1 = Analysis.calculate_iEF(formula, 1, is_positive=True)
            exp_mz1 = Analysis.match_peak(iMF1, spectrum, IPMD=20)
            if exp_mz1 != None:
                current_match_result.append((f'{ion_name}_{1}', exp_mz1))
                matched_sdion_intensity_sum += spectrum[exp_mz1]
                matched = True
            if iMF1 > CHARGE_2_THRESHOLD:
                iMF2 = (iMF1 + 1.0078250321) / 2
                exp_mz2 = Analysis.match_peak(iMF2, spectrum, IPMD=20)
                if exp_mz2 != None:
                    current_match_result.append((f'{ion_name}_{2}', exp_mz2))
                    matched_sdion_intensity_sum += spectrum[exp_mz2]
                    matched = True
            if matched: matched_sdion_count += 1
        for ion_name, formula in sdion_array_Ytype:
            matched = False
            iEF1, iMF1 = Analysis.calculate_iEF(formula, 1, is_positive=True)
            iMF2 = (iMF1 + 1.0078250321) / 2
            exp_mz1 = Analysis.match_peak(iMF1, spectrum, IPMD=20)
            exp_mz2 = Analysis.match_peak(iMF2, spectrum, IPMD=20)
            if exp_mz1 != None:
                current_match_result.append((f'{ion_name}_{1}', exp_mz1))
                matched_sdion_intensity_sum += spectrum[exp_mz1]
                matched = True
            if exp_mz2 != None:
                current_match_result.append((f'{ion_name}_{2}', exp_mz2))
                matched_sdion_intensity_sum += spectrum[exp_mz2]
                matched = True
            if matched: matched_sdion_count += 1
        score = matched_sdion_intensity_sum * log(matched_sdion_count / theo_sdion_count + 1)
        if score == 0.0:
            glycans_with_no_sdion_matched.append(glycan)
            continue
        if score in match_results:
            raise Exception(f'Glycan \"{glycan}\" got the same structure-diagnostic score with \"{match_results[score][0]}\".')
        match_results[score] = (glycan, current_match_result)

    matched_glycan_count = len(match_results)
    if matched_glycan_count < 2:
        return match_results, list(match_results.keys()), glycans_with_no_sdion_matched, glycans_with_no_theo_sdion
    if matched_glycan_count == 2:
        score_array = sorted(match_results.keys(), reverse=True)
        if score_array[0]  > 3 * score_array[1]:
            return match_results, [score_array[0]], glycans_with_no_sdion_matched, glycans_with_no_theo_sdion
        else:
            return match_results, score_array, glycans_with_no_sdion_matched, glycans_with_no_theo_sdion
    else:
        score_array = sorted(match_results.keys(), reverse=True)
        cutoff = 0
        for bound in range(1, matched_glycan_count):
            upper_mean = sum(score_array[:bound]) / bound
            if upper_mean > 3 * score_array[bound]:
                cutoff = bound
        if cutoff == 0:
            return match_results, score_array, glycans_with_no_sdion_matched, glycans_with_no_theo_sdion
        else:
            return match_results, score_array[:cutoff], glycans_with_no_sdion_matched, glycans_with_no_theo_sdion


def validate_structure_diagnosis_on_stdGP():
    gpsm_meta_path = r'E:\FIon_DNN_data\StdGP_Test\target_GPSMs_StdGP'
    gpsm_meta_tail = '-stdGP_test_GPSMs'
    folds = ('Untreated', 'De-sialylated', 'Cut HexNAc', 'Cut Gal Cut HexNAc', 'Cut Fucose')
    output_path = r'.\Struct_Diag_StdGP_Results'
    covered_count = 0
    differed_count = 0
    untargeted_count = 0
    single_count = 0
    multi_count = 0
    unmatch_count = 0
    first_mzML_loaded = False
    for fold in folds:
        meta_path = f'{gpsm_meta_path}\\{fold}'
        for root, dirs, file_names in os.walk(meta_path):
            for file_name in file_names:
                if not file_name.endswith(f'{gpsm_meta_tail}.pkl'): continue
                sample = file_name[:file_name.index(gpsm_meta_tail)]
                with open(f'{meta_path}\\{file_name}', 'rb') as file:
                    meta_data: list[tuple[str, int, tuple[str, tuple[int, ...], tuple[int, ...]], tuple[bool, ...], str]] = pk.load(file)
                for gpid_info in meta_data:
                    if gpid_info[0] != sample:
                        raise Exception(f'Sample \"{gpid_info[0]}\" in meta file \"{file_name}\" does not match with its file name.')

                meta_data.sort(key=lambda x: x[1])
                total_spectrum = len(meta_data)

                print(f'Current sample: {sample}')

                results = []

                with open('rsRaw_args.rsc', 'w') as raw_arg_file:
                    raw_arg_file.write(f'{MZML_FILE_PATH}\\{sample}.mzML##rsRawArgs##0##rsRawArgs##100')
                if first_mzML_loaded:
                    del sys.modules['parse_mzML']
                elif not 'parse_mzML' in sys.modules.keys():
                    first_mzML_loaded = True

                from parse_mzML import mzML_parser

                spectra_pack = mzML_parser.get_next_spectra_package()
                ms2_spectra = spectra_pack['MS2']

                handled_spectrum = 0

                for sample_name, spectrum_index, gpid, feature_vector, glycan in meta_data:

                    handled_spectrum += 1
                    print(f'Spectrum = {spectrum_index} [{handled_spectrum}/{total_spectrum}], sample = {sample_name}')

                    while not spectrum_index in ms2_spectra.keys():
                        spectra_pack = mzML_parser.get_next_spectra_package()
                        ms1_spectra = spectra_pack['MS1']
                        if len(ms1_spectra) == 0:
                            raise Exception(f'MS2 spectrum \"{spectrum_index}\" not found in sample \"{sample}\".')
                        ms2_spectra = spectra_pack['MS2']

                    ms2_spectrum: MS2_Spectrum = ms2_spectra[spectrum_index]

                    # feature_vector_excluded = convert_to_excluded_fv(feature_vector)
                    sdion_assigment = get_theo_sdion_assignment(gpid, feature_vector)
                    match_info, selected_scores, glycans_no_match, glycans_no_theo = match_glycan_by_struct_diagnosis(sdion_assigment, ms2_spectrum.spectrum)
                    matched_count = len(selected_scores)
                    if matched_count == 0:
                        unmatch_count += 1
                    else:
                        if matched_count > 1:
                            multi_count += 1
                        else:
                            single_count += 1
                        covered = False
                        for score in selected_scores:
                            if match_info[score][0] == glycan:
                                covered = True
                                break
                        if covered: covered_count += 1
                        else:
                            if glycan in glycans_no_theo:
                                untargeted_count += 1
                            else:
                                differed_count += 1
                    results.append((sample_name, spectrum_index, gpid, feature_vector, glycan, selected_scores, match_info, glycans_no_match, glycans_no_theo))

                print(f'Current sample: {sample}')
                print(f'Current covered = {covered_count}')
                print(f'Current differed = {differed_count}')
                print(f'Current untargeted = {untargeted_count}')
                print(f'Current single matched = {single_count}')
                print(f'Current multiple matched = {multi_count}')
                print(f'Current unmatched = {unmatch_count}')

                with open(f'{output_path}\\{sample}-struct_diag_results.txt', 'w') as output:
                    output.write('File\tMS2 scan\tPeptide seq\tModifications\tGlycan composition\tGlycan features (full)\t' +
                                 'Glycan structure (reported by StrucGP)\tGlycan structures selected\tGlycan structures with no theoretical structure-diagnostic fragments\t' +
                                 'Glycan structures with structure-diagnostic fragments\tGlycan structures with no matched structure-diagnostic fragments\n')
                    for sample_name, spectrum_index, gpid, feature_vector, glycan, selected_scores, match_info, glycans_no_match, glycans_no_theo in results:
                        peptide, mod_vect, comp_vect = gpid
                        mods = wrap_mods(mod_vect)
                        mods = 'Nglyco' if mods == '' else f'{mods}+Nglyco'
                        composition = wrap_composition(comp_vect)
                        selected_glycans = '; '.join([match_info[score][0] for score in selected_scores]) if len(selected_scores) > 0 else ''
                        match_info_list = []
                        for score in sorted(match_info.keys(), reverse=True):
                            linkage, matched_sdions = match_info[score]
                            matched_sdions_str = '; '.join([f'{ion}[m/z={round(mz, 4)}]' for ion, mz in matched_sdions])
                            match_info_list.append(f'{linkage}: {matched_sdions_str}')
                        match_info_str = ' | '.join(match_info_list) if len(match_info_list) > 0 else ''
                        glycans_no_match_str = '; '.join(glycans_no_match) if len(glycans_no_match) > 0 else ''
                        glycans_no_theo_str = '; '.join(glycans_no_theo) if len(glycans_no_theo) > 0 else ''
                        output.write(f'{sample_name}\t{spectrum_index}\t{peptide}\t{mods}\t{composition}\t{tuple(map(int, feature_vector))}\t{glycan}\t{selected_glycans}\t{glycans_no_theo_str}\t{match_info_str}\t{glycans_no_match_str}\n')

            break


def run_structure_diagnosis_on_testset(target_gpsms: list[tuple[str, int, tuple[str, tuple[int, ...], tuple[int, ...]]]], pred_fvs_excluded: list[tuple[int, ...]],
                                       output_path: str, reported_glycan_infos: Union[list[tuple[str, tuple[bool, ...]]], None] = None):
    total_spectrum = len(target_gpsms)
    if total_spectrum != len(pred_fvs_excluded) or (reported_glycan_infos != None and total_spectrum != len(reported_glycan_infos)):
        raise Exception(f'Number of GPSM does not match with predicted feature vectors or reported glycans.')
    target_gpsms_assigned: dict[str, list[tuple[int, tuple[str, tuple[int, ...], tuple[int, ...]], tuple[int, ...], Union[tuple[str, tuple[bool, ...]], None]]]] = {}
    for i in range(total_spectrum):
        gpsm_info = target_gpsms[i]
        pred_fv_excluded = pred_fvs_excluded[i]
        sample = gpsm_info[0]
        if sample in target_gpsms_assigned.keys():
            if reported_glycan_infos == None:
                target_gpsms_assigned[sample].append(
                    (gpsm_info[1], gpsm_info[2], pred_fv_excluded, None))
            else:
                target_gpsms_assigned[sample].append(
                    (gpsm_info[1], gpsm_info[2], pred_fv_excluded, reported_glycan_infos[i]))
        else:
            if reported_glycan_infos == None:
                target_gpsms_assigned[sample] = [
                    (gpsm_info[1], gpsm_info[2], pred_fv_excluded, None)]
            else:
                target_gpsms_assigned[sample] = [
                    (gpsm_info[1], gpsm_info[2], pred_fv_excluded, reported_glycan_infos[i])]
    for gpsm_array in target_gpsms_assigned.values():
        gpsm_array.sort(key=lambda x: x[0])

    first_mzML_loaded = False
    handled_spectrum = 0
    for sample in sorted(target_gpsms_assigned.keys()):

        print(f'Current sample: {sample}')
        results = []

        with open('rsRaw_args.rsc', 'w') as raw_arg_file:
            raw_arg_file.write(f'{MZML_FILE_PATH}\\{sample}.mzML##rsRawArgs##0##rsRawArgs##100')
        if first_mzML_loaded:
            del sys.modules['parse_mzML']
        elif not 'parse_mzML' in sys.modules.keys():
            first_mzML_loaded = True

        from parse_mzML import mzML_parser

        spectra_pack = mzML_parser.get_next_spectra_package()
        ms2_spectra = spectra_pack['MS2']

        for spectrum_index, gpid, pred_fv_excluded, reported_glycan_info in target_gpsms_assigned[sample]:

            handled_spectrum += 1
            print(f'Spectrum = {spectrum_index} [{handled_spectrum}/{total_spectrum}], sample = {sample}')

            while not spectrum_index in ms2_spectra.keys():
                spectra_pack = mzML_parser.get_next_spectra_package()
                ms1_spectra = spectra_pack['MS1']
                if len(ms1_spectra) == 0:
                    raise Exception(f'MS2 spectrum \"{spectrum_index}\" not found in sample \"{sample}\".')
                ms2_spectra = spectra_pack['MS2']

            ms2_spectrum: MS2_Spectrum = ms2_spectra[spectrum_index]

            sdion_assigment = get_theo_sdion_assignment(gpid, pred_fv_excluded)
            match_info, selected_scores, glycans_no_match, glycans_no_theo = match_glycan_by_struct_diagnosis(sdion_assigment, ms2_spectrum.spectrum)

            results.append((spectrum_index, gpid, reported_glycan_info, pred_fv_excluded, selected_scores, match_info, glycans_no_match, glycans_no_theo))

        with open(f'{output_path}\\{sample}-struct_diag_results.txt', 'w') as output:
            if reported_glycan_infos == None:
                output.write(
                    'File\tMS2 scan\tPeptide seq\tModifications\tGlycan composition\tGlycan features (predicted)\t' +
                    'Glycan structures selected\tGlycan structures with no theoretical structure-diagnostic fragments\t' +
                    'Glycan structures with structure-diagnostic fragments\tGlycan structures with no matched structure-diagnostic fragments\n')
            else:
                output.write('File\tMS2 scan\tPeptide seq\tModifications\tGlycan composition\tGlycan features (reported by StrucGP)\tGlycan features (predicted)\t' +
                         'Glycan structure (reported by StrucGP)\tGlycan structures selected\tGlycan structures with no theoretical structure-diagnostic fragments\t' +
                         'Glycan structures with structure-diagnostic fragments\tGlycan structures with no matched structure-diagnostic fragments\n')
            for spectrum_index, gpid, reported_glycan_info, pred_fv_excluded, selected_scores, match_info, glycans_no_match, glycans_no_theo in results:
                peptide, mod_vect, comp_vect = gpid
                mods = wrap_mods(mod_vect)
                mods = 'Nglyco' if mods == '' else f'{mods}+Nglyco'
                composition = wrap_composition(comp_vect)
                selected_glycans = '; '.join([match_info[score][0] for score in selected_scores]) if len(selected_scores) > 0 else ''
                match_info_list = []
                for score in sorted(match_info.keys(), reverse=True):
                    linkage, matched_sdions = match_info[score]
                    matched_sdions_str = '; '.join([f'{ion}[m/z={round(mz, 4)}]' for ion, mz in matched_sdions])
                    match_info_list.append(f'{linkage}: {matched_sdions_str}')
                match_info_str = ' | '.join(match_info_list) if len(match_info_list) > 0 else ''
                glycans_no_match_str = '; '.join(glycans_no_match) if len(glycans_no_match) > 0 else ''
                glycans_no_theo_str = '; '.join(glycans_no_theo) if len(glycans_no_theo) > 0 else ''
                if reported_glycan_info == None:
                    output.write(
                        f'{sample}\t{spectrum_index}\t{peptide}\t{mods}\t{composition}\t{pred_fv_excluded}\t{selected_glycans}\t{glycans_no_theo_str}\t{match_info_str}\t{glycans_no_match_str}\n')
                else:
                    reported_fv_excluded = convert_to_excluded_fv(reported_glycan_info[1])
                    output.write(
                        f'{sample}\t{spectrum_index}\t{peptide}\t{mods}\t{composition}\t{reported_fv_excluded}\t{pred_fv_excluded}\t{reported_glycan_info[0]}\t{selected_glycans}\t{glycans_no_theo_str}\t{match_info_str}\t{glycans_no_match_str}\n')


def validate_structure_diagnosis_on_testset():
    query_path = r'.\training_set'
    query_file_tail = '_overlap_total'
    query_glycan_infos: dict[str, dict[int, tuple[tuple[str, tuple[int, ...], tuple[int, ...]], tuple[bool, ...]]]] = {}
    query_glycan_infos_StrucGP: dict[str, dict[int, tuple[tuple[str, tuple[int, ...], tuple[int, ...]], tuple[bool, ...], str]]] = {}
    for root, dirs, file_names in os.walk(query_path):
        for file_name in file_names:
            if not file_name.endswith(f'{query_file_tail}.pkl'): continue
            sample = file_name[:file_name.index(query_file_tail)]
            with open(f'{query_path}\\{file_name}', 'rb') as file:
                query_glycan_infos[sample] = pk.load(file)
    for sample_title in SAMPLE_TITLES:
        for sheet_suffix in SHEET_SUFFIXES:
            sample_name = f'{sample_title}{sheet_suffix}'
            query_glycan_infos_StrucGP[sample_name] = extract_struct_gpids_StrucGP(
                pd.read_excel(f'{DATA_FOLD}\\{ENGINE_StrucGP}\\{sample_title}.xlsx', sheet_name=sample_name))

    with open('testset_predictions.pkl', 'rb') as file:
        y_test, y_test_pred_db, gpsm_array = pk.load(file)
    target_gpsms: list[tuple[str, int, tuple[str, tuple[int, ...], tuple[int, ...]]]] = []
    pred_fvs_excluded: list[tuple[int, ...]] = []
    reported_glycan_infos: list[tuple[str, tuple[bool, ...]]]= []
    for i in range(len(gpsm_array)):
        sample, spectrum_index, gpid, feature_vector = gpsm_array[i]
        y_true_fv_excluded = tuple(y_test[i])
        y_pred_fv_excluded = tuple(y_test_pred_db[i])
        if y_true_fv_excluded != convert_to_excluded_fv(feature_vector):
            raise Exception(f'The true label of excluded feature vector \"{y_true_fv_excluded}\" does not match with meta GPSM: {sample}, {gpid}, {feature_vector}')
        queried_sample_info = query_glycan_infos[sample]
        if spectrum_index in queried_sample_info.keys():
            queried_gpid, queried_fv = queried_sample_info[spectrum_index]
            if gpid != queried_gpid and feature_vector != queried_fv:
                raise Exception(f'Testing GPSM info does not match with meta GPSM: {sample}, {gpid}, {feature_vector}')
            target_gpsms.append((sample, spectrum_index, gpid))
            pred_fvs_excluded.append(y_pred_fv_excluded)
            queried_sample_info_struct = query_glycan_infos_StrucGP[sample]
            if spectrum_index in queried_sample_info_struct.keys():
                queried_gpid_struct, queried_fv_struct, queried_glycan = queried_sample_info_struct[spectrum_index]
                if gpid != queried_gpid_struct and feature_vector != queried_fv_struct:
                    raise Exception(f'Testing GPSM info does not match with StrucGP meta GPSM: {sample}, {gpid}, {feature_vector}')
                reported_glycan_infos.append((queried_glycan, queried_fv_struct))
            else:
                reported_glycan_infos.append(('', queried_fv))
        else:
            continue

    run_structure_diagnosis_on_testset(target_gpsms, pred_fvs_excluded, r'.\Struct_Diag_TestSplit_Results', reported_glycan_infos)




if __name__ == '__main__':

    # validate_structure_diagnosis_on_stdGP()

    validate_structure_diagnosis_on_testset()
