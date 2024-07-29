

import pandas as pd
from Peptide_Backbone import wrap_mods
from Glycan_Sulfated_Peptides_GlycReSoft import get_sulfated_gpsm_meta_infos



def stats_comparison(data_file, conduct_sulfate=False, stat_mass_subst_only=True):

    data: pd.DataFrame = pd.read_excel(data_file, sheet_name=0)

    total_count = 0
    covered_count = 0
    ambiguous_count = 0
    differed_count = 0
    single_matched = 0
    multi_matched = 0
    no_theo = 0
    no_match = 0

    covered_count_sulfate = 0
    ambiguous_count_sulfate = 0
    differed_count_sulfate = 0
    single_matched_sulfate = 0
    multi_matched_sulfate = 0
    no_theo_sulfate = 0
    no_match_sulfate = 0

    if conduct_sulfate:
        gpsm_meta_infos_sulfate = get_sulfated_gpsm_meta_infos()
    else:
        gpsm_meta_infos_sulfate = None

    for i in data.index:
        line_data = data.loc[i]
        true_glycan = line_data['Glycan structure (reported by StrucGP)']
        pred_glycans = str(line_data['Glycan structures selected'])
        no_theo_glycans = str(line_data['Glycan structures with no theoretical structure-diagnostic fragments'])
        no_match_glycans = str(line_data['Glycan structures with no matched structure-diagnostic fragments'])
        total_count += 1
        if pred_glycans == '' or pred_glycans == 'nan':
            if no_theo_glycans == '' or no_theo_glycans == 'nan':
                differed_count += 1
                no_match += 1
            else:
                if true_glycan in no_theo_glycans:
                    ambiguous_count += 1
                else:
                    differed_count += 1
                if no_match_glycans == '' or no_match_glycans == 'nan':
                    no_theo += 1
                else:
                    no_match += 1
        else:
            if true_glycan in pred_glycans:
                covered_count += 1
            else:
                if true_glycan in no_theo_glycans:
                    ambiguous_count += 1
                else:
                    differed_count += 1
            if '; ' in pred_glycans:
                multi_matched += 1
            else:
                single_matched += 1

        if conduct_sulfate:
            sample_name = line_data['File']
            spectrum_index = line_data['MS2 scan']
            peptide = line_data['Peptide seq']
            mod_info: str = line_data['Modifications'].replace('Nglyco', '')
            if mod_info.endswith('+'): mod_info = mod_info.rstrip('+')
            gpsm_meta_info = gpsm_meta_infos_sulfate[sample_name]
            if not spectrum_index in gpsm_meta_info.keys():
                continue
            gpid_sulfate = gpsm_meta_info[spectrum_index]
            if stat_mass_subst_only and (peptide != gpid_sulfate[0] or mod_info != wrap_mods(gpid_sulfate[1])):
                continue
            if pred_glycans == '' or pred_glycans == 'nan':
                if no_theo_glycans == '' or no_theo_glycans == 'nan':
                    differed_count_sulfate += 1
                    no_match_sulfate += 1
                else:
                    if true_glycan in no_theo_glycans:
                        ambiguous_count_sulfate += 1
                    else:
                        differed_count_sulfate += 1
                    if no_match_glycans == '' or no_match_glycans == 'nan':
                        no_theo_sulfate += 1
                    else:
                        no_match_sulfate += 1
            else:
                if true_glycan in pred_glycans:
                    covered_count_sulfate += 1
                else:
                    if true_glycan in no_theo_glycans:
                        ambiguous_count_sulfate += 1
                    else:
                        differed_count_sulfate += 1
                if '; ' in pred_glycans:
                    multi_matched_sulfate += 1
                else:
                    single_matched_sulfate += 1

    print('Total =', total_count)
    print('Covered =', covered_count)
    print('Amb =', ambiguous_count)
    print('Diff =', differed_count)
    print('Single =', single_matched)
    print('Multi =', multi_matched)
    print('No theo =', no_theo)
    print('No match =', no_match)

    if conduct_sulfate:
        print('Covered sulfate =', covered_count_sulfate)
        print('Amb sulfate =', ambiguous_count_sulfate)
        print('Diff sulfate =', differed_count_sulfate)
        print('Single sulfate =', single_matched_sulfate)
        print('Multi sulfate =', multi_matched_sulfate)
        print('No theo sulfate =', no_theo_sulfate)
        print('No match sulfate =', no_match_sulfate)


def stats_new(data_file: str):

    data: pd.DataFrame = pd.read_excel(data_file, sheet_name=0)

    total_count = 0
    single_matched = 0
    multi_matched = 0
    no_theo = 0
    no_match = 0

    for i in data.index:
        line_data = data.loc[i]
        pred_glycans = str(line_data['Glycan structures selected'])
        no_theo_glycans = str(line_data['Glycan structures with no theoretical structure-diagnostic fragments'])
        no_match_glycans = str(line_data['Glycan structures with no matched structure-diagnostic fragments'])
        total_count += 1
        if pred_glycans == '' or pred_glycans == 'nan':
            if no_theo_glycans == '' or no_theo_glycans == 'nan':
                no_match += 1
            else:
                if no_match_glycans == '' or no_match_glycans == 'nan':
                    no_theo += 1
                else:
                    no_match += 1
        else:
            if '; ' in pred_glycans:
                multi_matched += 1
            else:
                single_matched += 1

    print('Total =', total_count)
    print('Single =', single_matched)
    print('Multi =', multi_matched)
    print('No theo =', no_theo)
    print('No match =', no_match)


def select_stdGP_GPSMs_for_annotation(data_file: str, fv_predicted: bool=False) -> dict[str, list[tuple[str, int, tuple[str, str], tuple[str, tuple[int, ...], tuple[int, ...]], tuple[int, ...], list[str], dict[str, str]]]]:
    from Glycan_Formula import parse_composition
    from Feature_Prediction import convert_to_excluded_fv
    results = {}
    data: pd.DataFrame = pd.read_excel(data_file, sheet_name=0)
    for i in data.index:
        line_data = data.loc[i]
        glycan_pred = str(line_data['Glycan structures selected'])
        if glycan_pred == '' or glycan_pred == 'nan': continue
        glycan_true = line_data['Glycan structure (reported by StrucGP)']
        if glycan_true == glycan_pred: continue
        if glycan_true in str(line_data['Glycan structures with no theoretical structure-diagnostic fragments']).split('; '):
            continue
        if fv_predicted:
            true_feature_vect_excluded: tuple[int, ...] = eval(line_data['Glycan features (reported by StrucGP)'])
            feature_vect_excluded: tuple[int, ...] = eval(line_data['Glycan features (predicted)'])
            if feature_vect_excluded != true_feature_vect_excluded:
                continue
        else:
            feature_vect_excluded = convert_to_excluded_fv(eval(line_data['Glycan features (full)']))
        candidate_glycans = []
        if '; ' in glycan_pred:
            pred_glycans = glycan_pred.split('; ')
            if glycan_true in pred_glycans: continue
            candidate_glycans.extend(pred_glycans)
        else:
            candidate_glycans.append(glycan_pred)
        sample = line_data['File']
        scan = line_data['MS2 scan']
        peptide = line_data['Peptide seq']
        mod_info = line_data['Modifications'].split('+')
        if len(mod_info) == 1:
            mod_vect = (0, 0)
        else:
            c_count =0
            o_count = 0
            for mod in mod_info:
                if '[C]' in mod:
                    c_count = int(mod[:mod.index('x')])
                elif '[M]' in mod:
                    o_count = int(mod[:mod.index('x')])
            mod_vect = (c_count, o_count)
        comp_vect = parse_composition(line_data['Glycan composition'])
        sdion_match_info = str(line_data['Glycan structures with structure-diagnostic fragments']).split(' | ')
        sdion_match_query = {}
        for info in sdion_match_info:
            pair = info.split(': ')
            sdion_match_query[pair[0]] = pair[1].split('; ')
        for candidate_glycan in candidate_glycans:
            matched_sdions = sdion_match_query[candidate_glycan]
            matched_sdion_names = {ion[:ion.index('_')] for ion in matched_sdions}
            matched_sdion_info = {ion[:ion.index('[')]: ion[ion.index('['):] for ion in matched_sdions}
            pack = (sample, scan, (glycan_true, candidate_glycan),
                    (peptide, mod_vect, comp_vect), feature_vect_excluded, sorted(matched_sdion_names), matched_sdion_info)
            if candidate_glycan in results.keys():
                results[candidate_glycan].append(pack)
            else:
                results[candidate_glycan] = [pack]
    return results


def select_testSplit_GPSMs_for_fvDivergence_annotation(data_file: str, use_cover: bool = True) -> dict[str, list[tuple[str, int, tuple[str, str], tuple[str, tuple[int, ...], tuple[int, ...]], tuple[int, ...], list[str], dict[str, str]]]]:
    from Glycan_Formula import parse_composition
    from Feature_Prediction import judge_features_covered
    results = {}
    data: pd.DataFrame = pd.read_excel(data_file, sheet_name=0)
    for i in data.index:
        line_data = data.loc[i]
        glycan_pred = str(line_data['Glycan structures selected'])
        if glycan_pred == '' or glycan_pred == 'nan': continue
        true_feature_vect_excluded: tuple[int, ...] = eval(line_data['Glycan features (reported by StrucGP)'])
        pred_feature_vect_excluded: type[int, ...] = eval(line_data['Glycan features (predicted)'])
        if pred_feature_vect_excluded == true_feature_vect_excluded:
            continue
        if use_cover and not judge_features_covered(pred_feature_vect_excluded, true_feature_vect_excluded):
            continue
        glycan_true = line_data['Glycan structure (reported by StrucGP)']
        candidate_glycans = []
        if '; ' in glycan_pred:
            pred_glycans = glycan_pred.split('; ')
            candidate_glycans.extend(pred_glycans)
        else:
            candidate_glycans.append(glycan_pred)
        sample = line_data['File']
        scan = line_data['MS2 scan']
        peptide = line_data['Peptide seq']
        mod_info = line_data['Modifications'].split('+')
        if len(mod_info) == 1:
            mod_vect = (0, 0)
        else:
            c_count = 0
            o_count = 0
            for mod in mod_info:
                if '[C]' in mod:
                    c_count = int(mod[:mod.index('x')])
                elif '[M]' in mod:
                    o_count = int(mod[:mod.index('x')])
            mod_vect = (c_count, o_count)
        comp_vect = parse_composition(line_data['Glycan composition'])

        sdion_match_info = str(line_data['Glycan structures with structure-diagnostic fragments']).split(' | ')
        sdion_match_query = {}
        for info in sdion_match_info:
            pair = info.split(': ')
            sdion_match_query[pair[0]] = pair[1].split('; ')
        for candidate_glycan in candidate_glycans:
            matched_sdions = sdion_match_query[candidate_glycan]
            matched_sdion_names = {ion[:ion.index('_')] for ion in matched_sdions}
            matched_sdion_info = {ion[:ion.index('[')]: ion[ion.index('['):] for ion in matched_sdions}
            pack = (sample, scan, (glycan_true, candidate_glycan),
                    (peptide, mod_vect, comp_vect), pred_feature_vect_excluded, sorted(matched_sdion_names),
                    matched_sdion_info)
            if candidate_glycan in results.keys():
                results[candidate_glycan].append(pack)
            else:
                results[candidate_glycan] = [pack]
    return results





if __name__ == '__main__':

    # stats_comparison(r'.\Struct_Diag_StdGP_Results\Struct_Diag_StdGP_Results.xlsx')
    # stats_comparison(r'.\Struct_Diag_TestSplit_Results\Struct_Diag_TestSplit_Results.xlsx')
    # stats_new(r'.\Struct_Diag_SupTest_Results\Struct_Diag_SupTest_Results.xlsx')

    stats_comparison(r'.\Struct_Diag_TestSplit_Results\Struct_Diag_TestSplit_Results.xlsx',
                     conduct_sulfate=True, stat_mass_subst_only=True)

    # a = select_stdGP_GPSMs_for_annotation(r'.\Struct_Diag_StdGP_Results\Struct_Diag_StdGP_Results.xlsx', fv_predicted=False)
    # a = select_testSplit_GPSMs_for_fvDivergence_annotation(r'.\Struct_Diag_TestSplit_Results\Struct_Diag_TestSplit_Results.xlsx', use_cover=True)
    # a = select_stdGP_GPSMs_for_annotation(r'.\Struct_Diag_TestSplit_Results\Struct_Diag_TestSplit_Results.xlsx', fv_predicted=True)

    test = 0