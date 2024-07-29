

import os
import pickle as pk
import numpy as np
import pandas as pd
from typing import Union
from analysis_module import Analysis
from Peptide_Backbone import calc_peptide_backbone_formula
from Glycan_Formula import *
from Traintest_GPSM_Filtration import GLYCAN_DB
from Glycan_Fragmentation import calc_ion_formula, get_glycan_theo_fragments, get_fragment_union
from sklearn.preprocessing import MaxAbsScaler


CHARGE_2_THRESHOLD = 900.0

COMP_BASE = -2
COMP_LIM = {'N': 6, 'H': 6, 'F': 5, 'A': 4, 'G': 4}
COMP_LIM_VECT: tuple[int, ...] = tuple([COMP_LIM[comp] for comp in COMP_VECT])

ION_TYPE_LOCS = {'B': 1, 'C': 1, 'Y': 3, 'Z': 3, 'BY': 2, 'BZ': 2, 'CY': 2, 'YY': 4, 'YZ': 4, 'ZZ': 4, 'O': 0}
ION_TYPE_LOC_COUNT = len(set(ION_TYPE_LOCS.values()))
ION_TYPE_GROUPS = [''] * ION_TYPE_LOC_COUNT
for ion_type, ion_type_group_index in sorted(ION_TYPE_LOCS.items()):
    ION_TYPE_GROUPS[ion_type_group_index] = ion_type if ION_TYPE_GROUPS[ion_type_group_index] == '' else f'{ION_TYPE_GROUPS[ion_type_group_index]}/{ion_type}'
ION_TYPE_GROUPS = tuple(ION_TYPE_GROUPS)


CACHE_PATH = r'.\embedding_cache'

BRANCH_DATA = 'glycan_branches.xlsx'
FEATURE_TABLE_DF = pd.read_excel(BRANCH_DATA, sheet_name='feature_structs')
FEATURE_VECT = tuple(FEATURE_TABLE_DF['Abbreviation'].values)
FEATURE_STRUCTS = tuple(FEATURE_TABLE_DF['Structure'].values)
FEATURE_COMPS = tuple(map(calc_composition_GPSeeker, FEATURE_STRUCTS))
FEATURE_COUNT = len(FEATURE_VECT)
FIXED_ION_DF: pd.DataFrame = pd.read_excel(BRANCH_DATA, sheet_name='feature_fixed_fragments')

HEXNAC_FORMULA: dict[str, int] = COMP_FORMULA_QUERY[N_INDEX]

temp_fixed_ion_vector: list[tuple[tuple[int, ...], str, Union[float, dict[str, int]]]] = []
for i in FIXED_ION_DF.index:
    structure = FIXED_ION_DF.loc[i, 'Structure']
    info = FIXED_ION_DF.loc[i, 'Fragments'].split('-')
    if len(info) == 1:
        info = info[0].split(': ')
        comp_vector = calc_composition_GPSeeker(structure)
        mass = float(info[1])
        temp_fixed_ion_vector.append((comp_vector, 'O', mass))
        continue
    ion_types = info[0].split('/')
    fragment_struct = info[1]
    comp_vector = calc_composition_GPSeeker(fragment_struct)
    frag_formula = calc_glycan_formula(comp_vector)
    for ion_type in ion_types:
        formula = calc_ion_formula(frag_formula, ion_type)
        if ion_type.startswith('B') or ion_type.startswith('C'):
            iEF, iMF = Analysis.calculate_iEF(formula, 1, is_positive=True)
            temp_fixed_ion_vector.append((comp_vector, ion_type, iMF))
        elif ion_type.startswith('Y') or ion_type.startswith('Z'):
            temp_fixed_ion_vector.append((comp_vector, ion_type, formula))
        else:
            raise Exception(f'Unspported ion type \"{ion_type}\" for embedding.')

FIXED_ION_VECTOR: tuple[tuple[tuple[int, ...], str, Union[float, dict[str, int]]], ...] = tuple(temp_fixed_ion_vector)


def combine_feature_vectors(feature_vectors) -> tuple[bool, ...]:
    result = [False] * FEATURE_COUNT
    for feature_vector in feature_vectors:
        for i in range(FEATURE_COUNT):
            if feature_vector[i]: result[i] = True
    return tuple(result)


def get_embedding_fragments(comp_vector: tuple[int, ...]) -> tuple[tuple[bool, ...], tuple[tuple[tuple[int, ...], str, Union[float, dict[str, int]]], ...]]:
    cache_name = wrap_composition(comp_vector)
    cache_file = f'{CACHE_PATH}\\{cache_name}.pkl'
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as cache:
            total_feature_vector = pk.load(cache)
            return total_feature_vector, pk.load(cache)
    else:
        formula = wrap_formula(calc_glycan_formula(comp_vector))
        feature_group = GLYCAN_DB[formula][comp_vector]
        feature_vectors = sorted(feature_group.keys())
        total_feature_vector = combine_feature_vectors(feature_vectors)
        target_glycans: list[str] = []
        for feature_vector in feature_vectors:
            target_glycans.extend(feature_group[feature_vector])
        theo_fragments = get_glycan_theo_fragments(target_glycans, False, True, f'{cache_name}_0_1')
        fragment_union: dict[str, tuple[tuple[str, tuple[int, ...]], ...]] = get_fragment_union(theo_fragments)
        embedding_array: list[tuple[tuple[int, ...], str, Union[float, dict[str, int]]]] = []
        for formula_str in sorted(fragment_union.keys()):
            fragment_formula = Analysis.parse_formula(formula_str)
            fragments = fragment_union[formula_str]
            for ion_type, comp_v in fragments:
                if ion_type.startswith('B') or ion_type.startswith('C'):
                    iEF, iMF = Analysis.calculate_iEF(fragment_formula, 1, is_positive=True)
                    embedding_array.append((comp_v, ion_type, iMF))
                elif ion_type.startswith('Y') or ion_type.startswith('Z'):
                    embedding_array.append((comp_v, ion_type, fragment_formula))
                else:
                    raise Exception(f'Unsupported ion type \"{ion_type}\" for creating embedding array.')
        embedding_array.extend(FIXED_ION_VECTOR)
        embedding_vector = tuple(embedding_array)
        with open(cache_file, 'wb') as cache:
            pk.dump(total_feature_vector, cache)
            pk.dump(embedding_vector, cache)
        return total_feature_vector, embedding_vector


def extract_spectrum_embedding_data(spectrum: dict[float, float], gpid: tuple[str, tuple[int, ...], tuple[int, ...]],
                          embed_vector: tuple[tuple[tuple[int, ...], str, Union[float, dict[str, int]]], ...]) -> tuple[tuple[tuple[int, ...], str, float], ...]:
    temp_result: list[tuple[tuple[int, ...], str, float]] = []
    peptide_backbone_formula = calc_peptide_backbone_formula(gpid[0], gpid[1])
    glycan_formula = calc_glycan_formula(gpid[2])
    glycopeptide_formula = Analysis.combine_formula(peptide_backbone_formula, glycan_formula)
    Y1_formula = Analysis.combine_formula(peptide_backbone_formula, HEXNAC_FORMULA)

    precursor_iEF1, precursor_iMF1 = Analysis.calculate_iEF(glycopeptide_formula, 1, is_positive=True)
    precursor_iEF2, precursor_iMF2 = Analysis.calculate_iEF(glycopeptide_formula, 2, is_positive=True)
    Y1_iEF1, Y1_iMF1 = Analysis.calculate_iEF(Y1_formula, 1, is_positive=True)
    Y1_iEF2, Y1_iMF2 = Analysis.calculate_iEF(Y1_formula, 2, is_positive=True)
    precursor_mz1 = Analysis.match_peak(precursor_iMF1, spectrum, IPMD=20)
    precursor_mz2 = Analysis.match_peak(precursor_iMF2, spectrum, IPMD=20)
    scale_intensity = 0.0
    if precursor_mz1 != None: scale_intensity += spectrum[precursor_mz1]
    if precursor_mz2 != None: scale_intensity += spectrum[precursor_mz2]
    Y1_mz1 = Analysis.match_peak(Y1_iMF1, spectrum, IPMD=20)
    Y1_mz2 = Analysis.match_peak(Y1_iMF2, spectrum, IPMD=20)
    if Y1_mz1 != None: scale_intensity += spectrum[Y1_mz1]
    if Y1_mz2 != None: scale_intensity += spectrum[Y1_mz2]
    if scale_intensity == 0.0:
        print(f'Precursor and Y1 ion not matched. GPID = {gpid}')
        return ()
    for comp_vector, ion_type, mz_or_formula in embed_vector:
        intensity = 0.0
        if type(mz_or_formula) == dict:
            formula = Analysis.combine_formula(peptide_backbone_formula, mz_or_formula)
            iEF1, iMF1 = Analysis.calculate_iEF(formula, 1, is_positive=True)
            iEF2, iMF2 = Analysis.calculate_iEF(formula, 2, is_positive=True)
            exp_mz1 = Analysis.match_peak(iMF1, spectrum, IPMD=20)
            exp_mz2 = Analysis.match_peak(iMF2, spectrum, IPMD=20)
            if exp_mz1 != None: intensity = spectrum[exp_mz1]
            if exp_mz2 != None: intensity += spectrum[exp_mz2]
        else:
            exp_mz = Analysis.match_peak(mz_or_formula, spectrum, IPMD=20)
            if exp_mz != None: intensity = spectrum[exp_mz]
            if mz_or_formula > CHARGE_2_THRESHOLD:
                mz2 = (mz_or_formula + 1.0078250321) / 2
                exp_mz2 = Analysis.match_peak(mz2, spectrum, IPMD=20)
                if exp_mz2 != None: intensity += spectrum[exp_mz2]
        temp_result.append((comp_vector, ion_type, intensity / (scale_intensity + intensity) * 100.0))

    return tuple(temp_result)


def embed_to_nparray(embed_data: tuple[tuple[tuple[int, ...], str, float], ...], total_feature_vector: tuple[bool, ...], block: bool=True) -> np.array:
    matrix = np.zeros((FEATURE_COUNT,
                       ION_TYPE_LOC_COUNT,
                       COMP_LIM_VECT[N_INDEX],
                       COMP_LIM_VECT[H_INDEX],
                       COMP_LIM_VECT[F_INDEX],
                       COMP_LIM_VECT[A_INDEX],
                       COMP_LIM_VECT[G_INDEX]))
    for feature_index in range(FEATURE_COUNT):
        if block and not total_feature_vector[feature_index]: continue
        feature_comp = FEATURE_COMPS[feature_index]
        for comp_vector, ion_type, value in embed_data:
            if value == 0.0: continue
            subtracted_cv = subtract_composition_vector(comp_vector, feature_comp)
            locate_failed = False
            for comp_index in range(COMP_COUNT):
                location = subtracted_cv[comp_index] - COMP_BASE
                if 0 <= location < COMP_LIM_VECT[comp_index]: continue
                locate_failed = True
                break
            if locate_failed: continue
            ion_loc = ION_TYPE_LOCS[ion_type]

            matrix[feature_index,
            ion_loc,
            subtracted_cv[N_INDEX] - COMP_BASE,
            subtracted_cv[H_INDEX] - COMP_BASE,
            subtracted_cv[F_INDEX] - COMP_BASE,
            subtracted_cv[A_INDEX] - COMP_BASE,
            subtracted_cv[G_INDEX] - COMP_BASE] += value

    shape = matrix.shape
    matrix = matrix.reshape(FEATURE_COUNT, -1)
    scaler = MaxAbsScaler()
    matrix = scaler.fit_transform(matrix)
    matrix = matrix.reshape(shape)

    return matrix


def reduce_matrix_to_original_shape(decoded_data: np.array, batch_size: int) -> np.array:
    if decoded_data.shape[0] != batch_size:
        raise Exception(f'Sample count \"{decoded_data.shape[0]}\" in decoded data does not match with batch size: {batch_size}.')
    return decoded_data.reshape((
                        batch_size,
                        FEATURE_COUNT,
                        ION_TYPE_LOC_COUNT,
                        COMP_LIM_VECT[N_INDEX],
                        COMP_LIM_VECT[H_INDEX],
                        COMP_LIM_VECT[F_INDEX],
                        COMP_LIM_VECT[A_INDEX],
                        COMP_LIM_VECT[G_INDEX]
    ))


def reduce_comp_vect_by_location(N_loc: int, H_loc: int, F_loc: int, A_loc: int, G_loc: int, feature_comp: tuple[int, ...]) -> tuple[int, ...]:
    subtracted_comp = (N_loc + COMP_BASE,
                       H_loc + COMP_BASE,
                       F_loc + COMP_BASE,
                       A_loc + COMP_BASE,
                       G_loc + COMP_BASE)
    return combine_composition_vector(subtracted_comp, feature_comp)


def filter_differential_comps(decoded_matrix_original: np.array, decoded_matrix_mod: np.array, feature_index: int,
                              batch_size: int, differ_threshold: float = 3.0) -> dict[str, dict[tuple[int, ...], int]]:
    decoded_matrix_original = reduce_matrix_to_original_shape(decoded_matrix_original, batch_size)
    decoded_matrix_mod = reduce_matrix_to_original_shape(decoded_matrix_mod, batch_size)
    feature_comp = FEATURE_COMPS[feature_index]
    N_start =  -feature_comp[N_INDEX] - COMP_BASE
    H_start = -feature_comp[H_INDEX] - COMP_BASE
    F_start = -feature_comp[F_INDEX] - COMP_BASE
    A_start = -feature_comp[A_INDEX] - COMP_BASE
    G_start = -feature_comp[G_INDEX] - COMP_BASE
    grouped_comp_stats: list[dict[tuple[int, ...], int]] = []
    for ion_type_group in range(ION_TYPE_LOC_COUNT):
        comp_stats: dict[tuple[int, ...], int] = {}
        for sample in range(batch_size):
            matrix_original = decoded_matrix_original[sample, feature_index, ion_type_group]
            matrix_mod = decoded_matrix_mod[sample, feature_index, ion_type_group]
            for N_loc in range(N_start, COMP_LIM_VECT[N_INDEX]):
                for H_loc in range(H_start, COMP_LIM_VECT[H_INDEX]):
                    for F_loc in range(F_start, COMP_LIM_VECT[F_INDEX]):
                        for A_loc in range(A_start, COMP_LIM_VECT[A_INDEX]):
                            for G_loc in range(G_start, COMP_LIM_VECT[G_INDEX]):
                                value_original = matrix_original[N_loc, H_loc, F_loc, A_loc, G_loc]
                                if value_original <= 1e-4: continue
                                value_mod = abs(matrix_mod[N_loc, H_loc, F_loc, A_loc, G_loc])
                                if 1.0 / differ_threshold < abs(value_mod  / value_original) < differ_threshold:
                                    continue
                                comp_vect = reduce_comp_vect_by_location(N_loc, H_loc, F_loc, A_loc, G_loc, feature_comp)
                                if comp_vect in comp_stats.keys():
                                    comp_stats[comp_vect] += 1
                                else:
                                    comp_stats[comp_vect] = 1
        grouped_comp_stats.append(comp_stats)

    return dict(zip(ION_TYPE_GROUPS, grouped_comp_stats))
