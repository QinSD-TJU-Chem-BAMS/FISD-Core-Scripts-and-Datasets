

import os
import pickle as pk
import pandas as pd
from Glycan_Formula import calc_composition_GPSeeker, combine_composition_vector, combine_composition_vectors, calc_glycan_formula, wrap_formula



BRANCH_DATA = 'glycan_branches.xlsx'
BRANCH_SHEET = 'branch_tree-network'
FEATURE_SHEET = 'feature_structs'

BRANCH_TABLE_DF = pd.read_excel(BRANCH_DATA, sheet_name=BRANCH_SHEET, index_col='Group')
FEATURE_TABLE_DF = pd.read_excel(BRANCH_DATA, sheet_name=FEATURE_SHEET)

BRANCH_GROUP: dict[str, list[str]] = {}
BRANCH_INFOS: dict[str, tuple[tuple[bool, ...], tuple[int, ...]]] = {}
FEATURE_VECT = tuple(FEATURE_TABLE_DF['Abbreviation'].values)
FEATURE_STRUCTS = tuple(FEATURE_TABLE_DF['Structure'].values)
FEATURE_COUNT = len(FEATURE_VECT)
CRFUC_FEATURE_INDEX = FEATURE_VECT.index('CrFuc')
BISECT_FEATURE_INDEX = FEATURE_VECT.index('Bisect')
HYBRID_FEATURE_INDEX = FEATURE_VECT.index('Hybrid')

for group in BRANCH_TABLE_DF.index:
    base_struct: str = BRANCH_TABLE_DF.loc[group, 'Base struct']
    sub_structs: list[str] = str(BRANCH_TABLE_DF.loc[group, 'Sub-structs']).split('; ')
    if 'nan' in sub_structs: sub_structs = []
    sub_structs.insert(0, base_struct)
    BRANCH_GROUP[base_struct] = sub_structs
    for struct in sub_structs:
        if struct in BRANCH_INFOS.keys(): continue
        struct_fuc = struct.replace('31F', '(31F)')
        struct_defuc = struct.replace('(31F)', '')
        feature_vector = []
        for i in range(len(FEATURE_VECT)):
            feature_struct = FEATURE_STRUCTS[i]
            if feature_struct in struct:
                feature_vector.append(True)
            elif feature_struct in struct_fuc:
                feature_vector.append(True)
            elif feature_struct in struct_defuc:
                feature_vector.append(True)
            else:
                feature_vector.append(False)
        composition_vector = calc_composition_GPSeeker(struct)
        BRANCH_INFOS[struct] = (tuple(feature_vector), composition_vector)

TOTAL_BRANCHES = tuple(sorted(BRANCH_INFOS.keys(), key=lambda struct: BRANCH_INFOS[struct][1]))
TOTAL_BRANCH_COUNT = len(TOTAL_BRANCHES)
BASE_BRANCHES = tuple(sorted(BRANCH_GROUP.keys()))
BASE_BRANCH_COUNT = len(BASE_BRANCHES)
HYBRID_TAILS = ('61M', '(31M)61M', '(31M)61M21M', '(31M21M)61M21M')
HYBRID_TAIL_COUNT = len(HYBRID_TAILS)
HYBRID_COMP_VECTS = tuple(map(calc_composition_GPSeeker, HYBRID_TAILS))
TOTAL_BRANCH_COUNT_HYBRID = TOTAL_BRANCH_COUNT + len(HYBRID_TAILS)
CORE_STRUCTS = ('01Y41Y41M(31M)61M', '01Y(61F)41Y41M(31M)61M', '01Y41Y41M(31M)(41Y)61M', '01Y(61F)41Y41M(31M)(41Y)61M')
CORE_COMP_VECTORS = tuple(map(calc_composition_GPSeeker, CORE_STRUCTS))
CORE_FEATURE_PAIRS = ((False, False), (True, False), (False, True), (True, True))
BRANCH_LOCATION_COUNT = 4
BRANCH_LOCATIONS = tuple(range(BRANCH_LOCATION_COUNT))
GLYCAN_DB_DYNAMIC: dict[str, dict[tuple[int, ...], dict[tuple[bool, ...], set[str]]]] = {}


def combine_feature_vectors(*feature_vectors: tuple[bool, ...], add_corefuc: bool = False, add_bisect: bool = False, add_hybrid: bool = False) -> tuple[bool, ...]:
    combined_vector = [False] * FEATURE_COUNT
    for feature_vector in feature_vectors:
        for i in range(FEATURE_COUNT):
            if combined_vector[i]: continue
            if feature_vector[i]: combined_vector[i] = True
    if add_corefuc: combined_vector[CRFUC_FEATURE_INDEX] = True
    if add_bisect: combined_vector[BISECT_FEATURE_INDEX] = True
    if add_hybrid: combined_vector[HYBRID_FEATURE_INDEX] = True
    return tuple(combined_vector)


def build_complex_glycans():

    # single antenna
    for branch_struct in TOTAL_BRANCHES:
        feature_vector, composition_vector = BRANCH_INFOS[branch_struct]
        for crfuc, bisect in CORE_FEATURE_PAIRS:
            fv = combine_feature_vectors(feature_vector, add_corefuc=crfuc, add_bisect=bisect, add_hybrid=False)
            comp_vector = combine_composition_vector(composition_vector, CORE_COMP_VECTORS[crfuc + bisect * 2])
            formula = wrap_formula(calc_glycan_formula(comp_vector))
            core_start = '01Y(61F)41Y41M(31M' if crfuc else '01Y41Y41M(31M'
            for location in BRANCH_LOCATIONS:
                if location % 2 > 0: continue
                linkage = ''
                if location == 0:
                    linkage = f'{core_start}21{branch_struct})(41Y)61M' if bisect else f'{core_start}21{branch_struct})61M'
                else:
                    linkage = f'{core_start})(41Y)61M21{branch_struct}' if bisect else f'{core_start})61M21{branch_struct}'
                if formula in GLYCAN_DB_DYNAMIC.keys():
                    comp_group = GLYCAN_DB_DYNAMIC[formula]
                    if comp_vector in comp_group.keys():
                        feature_group = comp_group[comp_vector]
                        if fv in feature_group.keys():
                            feature_group[fv].add(linkage)
                        else:
                            feature_group[fv] = {linkage}
                    else:
                        comp_group[comp_vector] = {fv: {linkage}}
                else:
                    GLYCAN_DB_DYNAMIC[formula] = {comp_vector: {fv: {linkage}}}

    # double antennae
    for branch_index1 in range(TOTAL_BRANCH_COUNT):
        branch_struct1 = TOTAL_BRANCHES[branch_index1]
        feature_v1, composition_v1 = BRANCH_INFOS[branch_struct1]
        for branch_index2 in range(branch_index1, TOTAL_BRANCH_COUNT):
            branch_struct2 = TOTAL_BRANCHES[branch_index2]
            feature_v2, composition_v2 = BRANCH_INFOS[branch_struct2]
            branch_combs = ((branch_struct1, branch_struct2), (branch_struct2, branch_struct1))
            combined_fv = combine_feature_vectors(feature_v1, feature_v2)
            for crfuc, bisect in CORE_FEATURE_PAIRS:
                fv = combine_feature_vectors(combined_fv, add_corefuc=crfuc, add_bisect=bisect, add_hybrid=False)
                comp_vector = combine_composition_vectors(composition_v1, composition_v2, CORE_COMP_VECTORS[crfuc + bisect * 2])
                formula = wrap_formula(calc_glycan_formula(comp_vector))
                core_start = '01Y(61F)41Y41M(31M' if crfuc else '01Y41Y41M(31M'
                for b1, b2 in branch_combs:
                    linkage = f'{core_start}21{b1})(41Y)61M21{b2}' if bisect else f'{core_start}21{b1})61M21{b2}'
                    if formula in GLYCAN_DB_DYNAMIC.keys():
                        comp_group = GLYCAN_DB_DYNAMIC[formula]
                        if comp_vector in comp_group.keys():
                            feature_group = comp_group[comp_vector]
                            if fv in feature_group.keys():
                                feature_group[fv].add(linkage)
                            else:
                                feature_group[fv] = {linkage}
                        else:
                            comp_group[comp_vector] = {fv: {linkage}}
                    else:
                        GLYCAN_DB_DYNAMIC[formula] = {comp_vector: {fv: {linkage}}}

    # triple antennae
    for branch_struct1 in TOTAL_BRANCHES:
        feature_v1, composition_v1 = BRANCH_INFOS[branch_struct1]
        for base_struct in BASE_BRANCHES:
            sub_structs = BRANCH_GROUP[base_struct]
            sub_struct_count = len(sub_structs)
            for i in range(sub_struct_count):
                branch_struct2 = sub_structs[i]
                feature_v2, composition_v2 = BRANCH_INFOS[branch_struct2]
                combined_comp_v12 = combine_composition_vector(composition_v1, composition_v2)
                combined_fv12 = combine_feature_vectors(feature_v1, feature_v2)
                for j in range(i, sub_struct_count):
                    branch_struct3 = sub_structs[j]
                    branch_combs = ((branch_struct2, branch_struct3), (branch_struct3, branch_struct2))
                    feature_v3, composition_v3 = BRANCH_INFOS[branch_struct3]
                    combined_comp_v123 = combine_composition_vector(combined_comp_v12, composition_v3)
                    combined_fv123 = combine_feature_vectors(combined_fv12, feature_v3)
                    for crfuc, bisect in CORE_FEATURE_PAIRS:
                        fv = combine_feature_vectors(combined_fv123, add_corefuc=crfuc, add_bisect=bisect, add_hybrid=False)
                        comp_vector = combine_composition_vector(combined_comp_v123, CORE_COMP_VECTORS[crfuc + bisect * 2])
                        formula = wrap_formula(calc_glycan_formula(comp_vector))
                        core_start = f'01Y(61F)41Y41M(31M21{branch_struct1})' if crfuc else f'01Y41Y41M(31M21{branch_struct1})'
                        for b1, b2 in branch_combs:
                            linkage = f'{core_start}(41Y)61M(21{b1})61{b2}' if bisect else f'{core_start}61M(21{b1})61{b2}'
                            if formula in GLYCAN_DB_DYNAMIC.keys():
                                comp_group = GLYCAN_DB_DYNAMIC[formula]
                                if comp_vector in comp_group.keys():
                                    feature_group = comp_group[comp_vector]
                                    if fv in feature_group.keys():
                                        feature_group[fv].add(linkage)
                                    else:
                                        feature_group[fv] = {linkage}
                                else:
                                    comp_group[comp_vector] = {fv: {linkage}}
                            else:
                                GLYCAN_DB_DYNAMIC[formula] = {comp_vector: {fv: {linkage}}}
                        for b1, b2 in branch_combs:
                            core_start = f'01Y(61F)41Y41M(31M(21{b1})41{b2})' if crfuc else f'01Y41Y41M(31M(21{b1})41{b2})'
                            linkage = f'{core_start}(41Y)61M21{branch_struct1}' if bisect else f'{core_start}61M21{branch_struct1}'
                            if formula in GLYCAN_DB_DYNAMIC.keys():
                                comp_group = GLYCAN_DB_DYNAMIC[formula]
                                if comp_vector in comp_group.keys():
                                    feature_group = comp_group[comp_vector]
                                    if fv in feature_group.keys():
                                        feature_group[fv].add(linkage)
                                    else:
                                        feature_group[fv] = {linkage}
                                else:
                                    comp_group[comp_vector] = {fv: {linkage}}
                            else:
                                GLYCAN_DB_DYNAMIC[formula] = {comp_vector: {fv: {linkage}}}

    # tetra- antennae
    for base_struct_index1 in range(BASE_BRANCH_COUNT):
        base_struct1 = BASE_BRANCHES[base_struct_index1]
        sub_structs1 = BRANCH_GROUP[base_struct1]
        sub_struct_count1 = len(sub_structs1)
        for i in range(sub_struct_count1):
            branch_struct1 = sub_structs1[i]
            feature_v1, composition_v1 = BRANCH_INFOS[branch_struct1]
            for j in range(i, sub_struct_count1):
                branch_struct2 = sub_structs1[j]
                branch_combs12 = ((branch_struct1, branch_struct2), (branch_struct2, branch_struct1))
                feature_v2, composition_v2 = BRANCH_INFOS[branch_struct2]
                combined_comp_v12 = combine_composition_vector(composition_v1, composition_v2)
                combined_fv12 = combine_feature_vectors(feature_v1, feature_v2)
                for crfuc, bisect in CORE_FEATURE_PAIRS:
                    combined_fv12core = combine_feature_vectors(combined_fv12, add_corefuc=crfuc, add_bisect=bisect, add_hybrid=False)
                    combined_comp_v12core = combine_composition_vector(combined_comp_v12, CORE_COMP_VECTORS[crfuc + bisect * 2])
                    for b1, b2 in branch_combs12:
                        core_start = f'01Y(61F)41Y41M(31M(21{b1})41{b2})' if crfuc else f'01Y41Y41M(31M(21{b1})41{b2})'
                        if bisect: core_start += '(41Y)'
                        for base_struct_index2 in range(BASE_BRANCH_COUNT):
                            base_struct2 = BASE_BRANCHES[base_struct_index2]
                            sub_structs2 = BRANCH_GROUP[base_struct2]
                            sub_struct_count2 = len(sub_structs2)
                            for m in range(sub_struct_count2):
                                branch_struct3 = sub_structs2[m]
                                feature_v3, composition_v3 = BRANCH_INFOS[branch_struct3]
                                combined_fv12core3 = combine_feature_vectors(combined_fv12core, feature_v3)
                                combined_comp_v12core3 = combine_composition_vector(combined_comp_v12core, composition_v3)
                                for n in range(m, sub_struct_count2):
                                    branch_struct4 = sub_structs2[n]
                                    branch_combs34 = ((branch_struct3, branch_struct4), (branch_struct4, branch_struct3))
                                    feature_v4, composition_v4 = BRANCH_INFOS[branch_struct4]
                                    fv = combine_feature_vectors(combined_fv12core3, feature_v4)
                                    comp_vector = combine_composition_vector(combined_comp_v12core3, composition_v4)
                                    formula = wrap_formula(calc_glycan_formula(comp_vector))
                                    for b3, b4 in branch_combs34:
                                        linkage = f'{core_start}61M(21{b3})61{b4}'
                                        if formula in GLYCAN_DB_DYNAMIC.keys():
                                            comp_group = GLYCAN_DB_DYNAMIC[formula]
                                            if comp_vector in comp_group.keys():
                                                feature_group = comp_group[comp_vector]
                                                if fv in feature_group.keys():
                                                    feature_group[fv].add(linkage)
                                                else:
                                                    feature_group[fv] = {linkage}
                                            else:
                                                comp_group[comp_vector] = {fv: {linkage}}
                                        else:
                                            GLYCAN_DB_DYNAMIC[formula] = {comp_vector: {fv: {linkage}}}


def build_hybrid_glycans():

    # single antenna
    for branch_struct in TOTAL_BRANCHES:
        feature_vector, composition_vector = BRANCH_INFOS[branch_struct]
        for hybrid_index in range(HYBRID_TAIL_COUNT):
            hybrid_struct = HYBRID_TAILS[hybrid_index]
            hybrid_comp_vector = HYBRID_COMP_VECTS[hybrid_index]
            composition_v1 = combine_composition_vector(composition_vector, hybrid_comp_vector)
            for crfuc, bisect in CORE_FEATURE_PAIRS:
                fv = combine_feature_vectors(feature_vector, add_corefuc=crfuc, add_bisect=bisect, add_hybrid=True)
                comp_vector = combine_composition_vector(composition_v1, CORE_COMP_VECTORS[crfuc + bisect * 2])
                formula = wrap_formula(calc_glycan_formula(comp_vector))
                core_start = f'01Y(61F)41Y41M(31M21{branch_struct})' if crfuc else f'01Y41Y41M(31M21{branch_struct})'
                linkage = f'{core_start}(41Y)61M{hybrid_struct}' if bisect else f'{core_start}61M{hybrid_struct}'
                if formula in GLYCAN_DB_DYNAMIC.keys():
                    comp_group = GLYCAN_DB_DYNAMIC[formula]
                    if comp_vector in comp_group.keys():
                        feature_group = comp_group[comp_vector]
                        if fv in feature_group.keys():
                            feature_group[fv].add(linkage)
                        else:
                            feature_group[fv] = {linkage}
                    else:
                        comp_group[comp_vector] = {fv: {linkage}}
                else:
                    GLYCAN_DB_DYNAMIC[formula] = {comp_vector: {fv: {linkage}}}

    #double antennae
    for base_struct in BASE_BRANCHES:
        sub_structs = BRANCH_GROUP[base_struct]
        sub_struct_count = len(sub_structs)
        for i in range(sub_struct_count):
            branch_struct1 = sub_structs[i]
            feature_v1, composition_v1 = BRANCH_INFOS[branch_struct1]
            for j in range(i, sub_struct_count):
                branch_struct2 = sub_structs[j]
                branch_combs = ((branch_struct1, branch_struct2), (branch_struct2, branch_struct1))
                feature_v2, composition_v2 = BRANCH_INFOS[branch_struct2]
                combined_comp_v12 = combine_composition_vector(composition_v1, composition_v2)
                combined_fv12 = combine_feature_vectors(feature_v1, feature_v2)
                for hybrid_index in range(HYBRID_TAIL_COUNT):
                    hybrid_struct = HYBRID_TAILS[hybrid_index]
                    hybrid_comp_vector = HYBRID_COMP_VECTS[hybrid_index]
                    combined_comp_v12hybrid = combine_composition_vector(combined_comp_v12, hybrid_comp_vector)
                    for crfuc, bisect in CORE_FEATURE_PAIRS:
                        fv = combine_feature_vectors(combined_fv12, add_corefuc=crfuc, add_bisect=bisect, add_hybrid=True)
                        comp_vector = combine_composition_vector(combined_comp_v12hybrid, CORE_COMP_VECTORS[crfuc + bisect * 2])
                        formula = wrap_formula(calc_glycan_formula(comp_vector))
                        for b1, b2 in branch_combs:
                            core_start = f'01Y(61F)41Y41M(31M(21{b1})41{b2})' if crfuc else f'01Y41Y41M(31M(21{b1})41{b2})'
                            linkage = f'{core_start}(41Y)61M{hybrid_struct}' if bisect else f'{core_start}61M{hybrid_struct}'
                            if formula in GLYCAN_DB_DYNAMIC.keys():
                                comp_group = GLYCAN_DB_DYNAMIC[formula]
                                if comp_vector in comp_group.keys():
                                    feature_group = comp_group[comp_vector]
                                    if fv in feature_group.keys():
                                        feature_group[fv].add(linkage)
                                    else:
                                        feature_group[fv] = {linkage}
                                else:
                                    comp_group[comp_vector] = {fv: {linkage}}
                            else:
                                GLYCAN_DB_DYNAMIC[formula] = {comp_vector: {fv: {linkage}}}


def get_packed_glycan_db() -> dict[str, dict[tuple[int, ...], dict[tuple[bool, ...], tuple[str, ...]]]]:
    glycan_db: dict[str, dict[tuple[int, ...], dict[tuple[bool, ...], tuple[str, ...]]]] = {}
    for formula in GLYCAN_DB_DYNAMIC.keys():
        comp_group_dynamic = GLYCAN_DB_DYNAMIC[formula]
        comp_group: dict[tuple[int, ...], dict[tuple[bool, ...], tuple[str, ...]]] = {}
        for comp_v in comp_group_dynamic.keys():
            feature_group_dynamic = comp_group_dynamic[comp_v]
            feature_group = {feature_v: tuple(sorted(feature_group_dynamic[feature_v])) for feature_v in feature_group_dynamic}
            comp_group[comp_v] = feature_group
        glycan_db[formula] = comp_group
    return glycan_db





if __name__ == '__main__':

    build_complex_glycans()
    build_hybrid_glycans()
    glycan_db = get_packed_glycan_db()
    file_name = 'glycan_theo_db.pkl'
    if os.path.exists(file_name):
        not_deleted = True
        while not_deleted:
            input(f'{file_name} already exists. Please remove it manually before creating a new database.')
            not_deleted = os.path.exists(file_name)
    with open(file_name, 'wb') as output:
        pk.dump(glycan_db, output)
