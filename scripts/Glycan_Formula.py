

from analysis_module import Analysis



COMP_VECT = ('N', 'H', 'F', 'A', 'G')
COMP_FORMULAs = ('C8H15N1O6', 'C6H12O6', 'C6H12O5', 'C11H19N1O9', 'C11H19N1O10')
COMP_FORMULA_QUERY: tuple[dict[str, int], ...] = tuple(map(Analysis.parse_formula, COMP_FORMULAs))
COMP_COUNT = len(COMP_VECT)

N_INDEX = COMP_VECT.index('N')
H_INDEX = COMP_VECT.index('H')
F_INDEX = COMP_VECT.index('F')
A_INDEX = COMP_VECT.index('A')
G_INDEX = COMP_VECT.index('G')


def parse_composition(composition: str) -> tuple[int, int, int, int, int]:
    comp_dict = Analysis.parse_formula(composition)
    return (comp_dict['N'], comp_dict['H'], comp_dict['F'], comp_dict['A'], comp_dict['G'])


def parse_composition_GPSeeker(composition: str) -> tuple[int, int, int, int, int]:
    comp_dict = Analysis.parse_formula(composition[:composition.index('X')])
    return (comp_dict['N'], comp_dict['H'], comp_dict['F'], comp_dict['S'], comp_dict['T'])


def calc_composition_GPSeeker(glycan: str) -> tuple[int, int, int, int, int]:
    N_number = 0
    H_number = 0
    F_number = 0
    A_number = 0
    G_number = 0
    for code in glycan:
        if code == 'M' or code == 'L' or code == 'G':
            H_number += 1
        elif code == 'Y' or code == 'V':
            N_number += 1
        elif code == 'F':
            F_number += 1
        elif code == 'S':
            A_number += 1
        elif code == 'T':
            G_number += 1
    return (N_number, H_number, F_number, A_number, G_number)


def calc_composition_pGlyco(glycan: str) -> tuple[int, int, int, int, int]:
    N_number = 0
    H_number = 0
    F_number = 0
    A_number = 0
    G_number = 0
    for code in glycan:
        if code == 'H':
            H_number += 1
        elif code == 'N':
            N_number += 1
        elif code == 'F':
            F_number += 1
        elif code == 'A':
            A_number += 1
        elif code == 'G':
            G_number += 1
    return (N_number, H_number, F_number, A_number, G_number)


def calc_composition_Byonic(composition: str) -> tuple[int, int, int, int, int]:
    N_number = 0
    H_number = 0
    F_number = 0
    A_number = 0
    G_number = 0
    raw_composition: list[str] = composition.split(')')[:-1]
    for pair in raw_composition:
        bound_index = pair.index('(')
        monosac_type = pair[:bound_index]
        count = int(pair[bound_index + 1:])
        if monosac_type == 'HexNAc':
            N_number += count
        elif monosac_type == 'Hex':
            H_number += count
        elif monosac_type == 'Fuc':
            F_number += count
        elif monosac_type == 'NeuAc':
            A_number += count
        elif monosac_type == 'NeuGc':
            G_number += count
        else:
            raise Exception(f'Undefined monosaccharide type \"{monosac_type}\" detected in Byonic style glycan composition \"{composition}\".')
    return (N_number, H_number, F_number, A_number, G_number)


def calc_composition_StrucGP(glycan: str) -> tuple[int, ...]:
    N_number = 0
    H_number = 0
    F_number = 0
    A_number = 0
    G_number = 0
    for code in glycan:
        order = ord(code)
        if 64 < order < 91:
            continue
        elif 96 < order < 123:
            continue
        num = int(code)
        if num == 1:
            H_number += 1
        elif num == 2:
            N_number += 1
        elif num == 3:
            A_number += 1
        elif num == 4:
            G_number += 1
        elif num == 5:
            F_number += 1
    return (N_number, H_number, F_number, A_number, G_number)


def combine_composition_vector(comp_v1: tuple[int, ...], comp_v2: tuple[int, ...]) -> tuple[int, ...]:
    comp_v = [0] * COMP_COUNT
    for i in range(COMP_COUNT):
        comp_v[i] = comp_v1[i] + comp_v2[i]
    return tuple(comp_v)


def combine_composition_vectors(*comp_vectors: tuple[int, ...]) -> tuple[int, ...]:
    comp_v = [0] * COMP_COUNT
    for v in comp_vectors:
        for i in range(COMP_COUNT):
            comp_v[i] += v[i]
    return tuple(comp_v)


def subtract_composition_vector(base_comp_vector: tuple[int, ...], comp_vector_to_subtract: tuple[int, ...]) -> tuple[int, ...]:
    result = list(base_comp_vector)
    for i in range(COMP_COUNT):
        count = comp_vector_to_subtract[i]
        if count == 0: continue
        result[i] -= count
    return tuple(result)


def wrap_composition(composition_vector: tuple[int, ...]) -> str:
    wrapped_comp = ''
    for i in range(COMP_COUNT):
        wrapped_comp += f'{COMP_VECT[i]}{composition_vector[i]}'
    return wrapped_comp



def jude_high_mannose_and_oligo(composition_vector: tuple[int, ...]) -> bool:
    return composition_vector[N_INDEX] == 2 and composition_vector[H_INDEX] + 2 == sum(composition_vector)


def wrap_formula(formula: dict[str, int]) -> str:
    return Analysis.wrap_formula(formula)


def calc_glycan_formula(composition_vector: tuple[int, ...]) -> dict[str, int]:
    formula = {}
    monosac_count = 0
    for i in range(COMP_COUNT):
        count = composition_vector[i]
        if count == 0: continue
        Analysis.modify_repetitive_formula(formula, COMP_FORMULA_QUERY[i], count)
        monosac_count += count
    monosac_count -= 1
    formula['H'] -= monosac_count * 2
    formula['O'] -= monosac_count
    return formula