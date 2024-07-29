

import os
import sys
import math
import pickle as pk
import pandas as pd
from Peptide_Backbone import calc_peptide_backbone_formula
from Glycan_Formula import calc_glycan_formula, calc_composition_GPSeeker, wrap_composition, wrap_formula
from Glycan_Fragmentation import calc_ion_formula, get_glycan_theo_fragments, calc_struct_diagnostic_ions
from spectrum_module import MS1_Spectrum, MS2_Spectrum
from analysis_module import Analysis


CREATE_NEW_SUP_FRAGMENTS = False

TRAINING_SET_SOURCE = r'.\training_set_candidates'
TRAINING_SET_TARGET = r'.\training_set'
MZML_FILE_PATH = r'E:\FIon_DNN_data\Raw_Files_mzML'

BRANCH_DATA = 'glycan_branches.xlsx'

FEATURE_TABLE_DF = pd.read_excel(BRANCH_DATA, sheet_name='feature_structs')
SUP_FEATURES_DF: pd.DataFrame = pd.read_excel(BRANCH_DATA, sheet_name='feature_validation_fragments')

FEATURE_VECT = tuple(FEATURE_TABLE_DF['Abbreviation'].values)
SUP_FEATURES: tuple[str] = tuple(SUP_FEATURES_DF['Abbreviation'].values)
SUP_FEATURE_FRAG_INFOS: tuple[str] = tuple(SUP_FEATURES_DF['Fragments'].values)
SUP_FEATURE_COUNT = len(SUP_FEATURES)
SUP_FEATURE_INDICES = tuple(map(lambda x: FEATURE_VECT.index(x), SUP_FEATURES))

SUP_FEATURE_FRAGMENTS: dict[str, tuple[tuple[tuple[float, ...], ...], tuple[str, ...]]]
if CREATE_NEW_SUP_FRAGMENTS:
    SUP_FEATURE_FRAGMENTS = {}
    for i in range(SUP_FEATURE_COUNT):
        feature_abbr = SUP_FEATURES[i]
        feature_frag_info: list[str] = SUP_FEATURE_FRAG_INFOS[i].split(', ')
        mass_list: set[tuple[float, ...]] = set()
        fragment_list: list[str] = []
        for fragment in feature_frag_info:
            info = fragment.split('-')
            if len(info) == 1:
                info = info[0].split(': ')
                mass = float(info[1])
                mass_list.add((mass, ))
                fragment_list.append(f'{info[0]}-ion (m/z={round(mass, 4)})')
                continue
            ion_types = info[0].split('/')
            fragment_struct = info[1]
            comp_vector = calc_composition_GPSeeker(fragment_struct)
            composition = wrap_composition(comp_vector)
            frag_formula = calc_glycan_formula(comp_vector)
            frag_formula['H'] += 1
            current_masses: set[float] = set()
            for ion_type in ion_types:
                formula = calc_ion_formula(frag_formula, ion_type)
                mass = Analysis.calc_monoIsotopic_mass(formula)
                current_masses.add(mass)
                fragment_list.append(f'{ion_type}-ion ({composition}, m/z={round(mass, 4)})')
            mass_list.add(tuple(sorted(current_masses)))
        SUP_FEATURE_FRAGMENTS[feature_abbr] = (tuple(sorted(mass_list)),tuple(fragment_list))
    with open('sup_feature_fragments.pkl', 'wb') as output:
        pk.dump(SUP_FEATURE_FRAGMENTS, output)
else:
    with open('sup_feature_fragments.pkl', 'rb') as frag_file:
        SUP_FEATURE_FRAGMENTS = pk.load(frag_file)


GLYCAN_DB: dict[str, dict[tuple[int, ...], dict[tuple[bool, ...], tuple[str, ...]]]]
with open('glycan_theo_db.pkl', 'rb') as db_file:
    GLYCAN_DB= pk.load(db_file)


def check_sup_features(feature_vector: tuple[bool, ...]) -> tuple[bool, set[tuple[float, ...]]]:
    covered = False
    target_mass_groups: set[tuple[float, ...]] = set()
    for index in SUP_FEATURE_INDICES:
        if feature_vector[index]:
            covered = True
            feature = FEATURE_VECT[index]
            for mass_group in SUP_FEATURE_FRAGMENTS[feature][0]:
                target_mass_groups.add(mass_group)
    return covered, target_mass_groups


def select_training_set_supplement(use_sd_ions: bool = False):

    target_files = []
    for root, dirs, files in os.walk(TRAINING_SET_SOURCE):
        for file_name in files:
            if file_name.endswith('.pkl'):
                target_files.append(file_name)
        break

    first_mzML_loaded = False
    for file_name in target_files:

        sample_name = file_name[:file_name.index('_')]

        file = open(f'{TRAINING_SET_SOURCE}\\{file_name}', 'rb')
        candidate_gpids: dict[int, tuple[str, tuple[int, ...], tuple[int, ...]]]= pk.load(file)
        file.close()

        selected_gpids: dict[int, tuple[tuple[str, tuple[int, ...], tuple[int, ...]], tuple[bool, ...]]] = {}

        with open('rsRaw_args.rsc', 'w') as raw_arg_file:
            raw_arg_file.write(f'{MZML_FILE_PATH}\\{sample_name}.mzML##rsRawArgs##0##rsRawArgs##100')
        if first_mzML_loaded:
            del sys.modules['parse_mzML']
        elif not 'parse_mzML' in sys.modules.keys():
            first_mzML_loaded = True

        from parse_mzML import mzML_parser

        spectra_pack = mzML_parser.get_next_spectra_package()
        ms2_spectra = spectra_pack['MS2']

        for spectrum_index in sorted(candidate_gpids.keys()):

            while not spectrum_index in ms2_spectra.keys():
                spectra_pack = mzML_parser.get_next_spectra_package()
                ms1_spectra = spectra_pack['MS1']
                if len(ms1_spectra) == 0:
                    raise Exception(f'MS2 spectrum \"{spectrum_index}\" not found in sample \"{sample_name}\".')
                ms2_spectra = spectra_pack['MS2']

            ms2_spectrum: MS2_Spectrum = ms2_spectra[spectrum_index]

            peptide, mod_vector, comp_vector = candidate_gpids[spectrum_index]
            formula = wrap_formula(calc_glycan_formula(comp_vector))
            theo_comp_group = GLYCAN_DB[formula]
            theo_feature_group = theo_comp_group[comp_vector]

            candidate_feature_vectors: dict[tuple[bool, ...], set[tuple[float, ...]]] = {}
            for feature_vector in theo_feature_group.keys():
                covered, target_mass_groups = check_sup_features(feature_vector)
                if not covered: continue
                candidate_feature_vectors[feature_vector] = target_mass_groups
            if len(candidate_feature_vectors) == 0: continue

            ### Checking feature ions

            selected_feature_vectors: list[tuple[bool, ...]] = []
            for feature_vector in sorted(candidate_feature_vectors.keys()):
                checked = True
                for mass_group in candidate_feature_vectors[feature_vector]:
                    matched = False
                    for mz in mass_group:
                        if Analysis.match_peak(mz, ms2_spectrum.spectrum, IPMD=20) != None:
                            matched = True
                            break
                    if matched: continue
                    checked = False
                    break
                if not checked: continue
                selected_feature_vectors.append(feature_vector)
            if len(selected_feature_vectors) == 0: continue

            if use_sd_ions:
                ### Checking structure-diagnostic ions
                candidate_glycans: list[tuple[str, tuple[bool, ...]]] = []
                for feature_vector in selected_feature_vectors:
                    glycans = theo_feature_group[feature_vector]
                    for glycan in glycans:
                        candidate_glycans.append((glycan, feature_vector))
                target_glycans = []
                for comp_v in sorted(theo_comp_group.keys()):
                    feature_group = theo_comp_group[comp_v]
                    for feature_v in sorted(feature_group.keys()):
                        target_glycans.extend(feature_group[feature_v])
                theo_fragments = get_glycan_theo_fragments(target_glycans, True, False, f'{formula}_1_0')
                struct_diag_ions = calc_struct_diagnostic_ions(theo_fragments)
                peptide_backbone_formula = calc_peptide_backbone_formula(peptide, mod_vector)
                matched_glycans: list[tuple[str, float]] = []
                for glycan, feature_vector in candidate_glycans:
                    sd_ions = struct_diag_ions[glycan]
                    sd_ion_count = len(sd_ions)
                    if sd_ion_count == 0: continue
                    matched_sd_ions: list[float] = []
                    for ion_name, ion_formula in sd_ions:
                        matched_peak = None
                        if ion_name.startswith('B') or ion_name.startswith('C') or 'A' in ion_name:
                            iEF, iMF = Analysis.calculate_iEF(ion_formula, 1, is_positive=True)
                            matched_peak = Analysis.match_peak(iMF, ms2_spectrum.spectrum, IPMD=20)
                        elif ion_name.startswith('Y') or ion_name.startswith('Z') or 'X' in ion_name:
                            ion_formula_total = Analysis.combine_formula(ion_formula, peptide_backbone_formula)
                            iEF, iMF = Analysis.calculate_iEF(ion_formula_total, 1, is_positive=True)
                            matched_peak = Analysis.match_peak(iMF, ms2_spectrum.spectrum, IPMD=20)
                        else:
                            raise Exception(f'Unsupported fragment ion name: \"{ion_name}\"')
                        if matched_peak == None: continue
                        matched_sd_ions.append(ms2_spectrum.spectrum[matched_peak])
                    matched_sd_ion_count = len(matched_sd_ions)
                    if matched_sd_ion_count == 0: continue
                    score = sum(matched_sd_ions) * math.log(matched_sd_ion_count / sd_ion_count + 1)
                    matched_glycans.append((glycan, score))
                if len(matched_glycans) == 0: continue
                matched_glycans.sort(key=lambda x: x[1], reverse=True)
                matched_glycan = matched_glycans[0]
                for glycan, feature_vector in candidate_glycans:
                    if glycan == matched_glycan:
                        selected_gpids[spectrum_index] = ((peptide, mod_vector, comp_vector), feature_vector)
                        break
            else:
                selected_feature_vectors.sort(key=lambda v: len(candidate_feature_vectors[v]), reverse=True)
                selected_gpids[spectrum_index] = ((peptide, mod_vector, comp_vector), selected_feature_vectors[0])
        if len(selected_gpids) > 0:
            with open(f'{TRAINING_SET_TARGET}\\{sample_name}_sup_features.pkl', 'wb') as output:
                pk.dump(selected_gpids, output)

        print(f'Sample = {sample_name}, selected GPSMs: {selected_gpids}')








if __name__ == '__main__':

    select_training_set_supplement(use_sd_ions=False)