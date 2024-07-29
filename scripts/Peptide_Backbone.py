

from analysis_module import Analysis

AA_TABLE: dict[str, dict[str, int]] = {}
with open('Amino Acid Table.txt', 'r') as table_file:
    for line in table_file:
        raw_data = line.rstrip('\n').strip(' ').replace('-', '').split('>')
        AA_TABLE[raw_data[0]] = Analysis.parse_formula(raw_data[1])

MOD_VECTOR = ('Carbamidomethyl[C]', 'Oxidation[M]')
MOD_VECTOR_LEN = len(MOD_VECTOR)
MOD_FORMULAS = (Analysis.parse_formula('C2H3NO'), {'O': 1})
MOD_TABLE = dict(zip(MOD_VECTOR, MOD_FORMULAS))
MOD_LAST_INDEX = MOD_VECTOR_LEN - 1


def calc_peptide_backbone_formula(peptide: str, mod_vector: tuple[int, ...]) -> dict[str, int]:
    length = len(peptide)
    formula: dict[str, int] = {}
    for aa in peptide:
        Analysis.modify_formula(formula, AA_TABLE[aa])
    for i in range(MOD_VECTOR_LEN):
        count = mod_vector[i]
        if count == 0: continue
        Analysis.modify_repetitive_formula(formula, MOD_FORMULAS[i], count)
    formula['H'] -= length * 2
    formula['O'] -= length
    return formula


def wrap_mods(mod_vector: tuple[int, ...]) -> str:
    mod = ''
    for i in range(MOD_LAST_INDEX):
        count = mod_vector[i]
        if count == 0: continue
        mod = f'{mod}{count}x{MOD_VECTOR[i]}+'
    last_mod_count = mod_vector[MOD_LAST_INDEX]
    if last_mod_count == 0:
        return mod.rstrip('+')
    else:
        return f'{mod}{last_mod_count}x{MOD_VECTOR[MOD_LAST_INDEX]}'
