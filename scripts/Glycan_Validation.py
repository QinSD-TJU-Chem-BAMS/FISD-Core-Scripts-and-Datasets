


import pickle as pk
import pandas as pd
from struct_conv import parse_GPSeeker_linkage, convert_to_pGlyco, convert_to_StrucGP, convert_to_GWB
from Glycan_Formula import calc_composition_pGlyco, calc_composition_Byonic, calc_glycan_formula, wrap_composition, jude_high_mannose_and_oligo, wrap_formula


pGlyco_DB = 'pGlyco-N-Mouse-large.gdb'
MSFragger_DB = 'Mouse_N-glycans-1670-pGlyco.glyc'

pGlyco_compositions: set[tuple[int, ...]] = set()
pGlyco_formulas: set[str] = set()
MSFragger_compositions: set[tuple[int, ...]] = set()
MSFragger_formulas: set[str] = set()


glycan_db_file = open('glycan_theo_db.pkl', 'rb')
glycan_db: dict[str, dict[tuple[int, ...], dict[tuple[bool, ...], tuple[str, ...]]]] = pk.load(glycan_db_file)
glycan_db_file.close()

glycan_query_pGlyco: dict[str, str] = {}
glycan_query_StrucGP: dict[str, str] = {}
glycan_query_GWB: dict[str, str] = {}


def validate_compositions(write_to_file: bool = False) -> tuple[set[tuple[int, ...]], set[tuple[int, ...]], set[tuple[int, ...]]]:

    with open(pGlyco_DB, 'r') as pGlyco_file:
        pGlyco_file.readline()
        for line in pGlyco_file:
            comp_vector = calc_composition_pGlyco(line.rstrip('\n'))
            if comp_vector in pGlyco_compositions: continue
            if jude_high_mannose_and_oligo(comp_vector): continue
            pGlyco_compositions.add(comp_vector)
            pGlyco_formulas.add(wrap_formula(calc_glycan_formula(comp_vector)))

    with open(MSFragger_DB, 'r') as MSF_file:
        for line in MSF_file:
            comp_vector = calc_composition_Byonic(line.rstrip('\n'))
            if comp_vector in MSFragger_compositions: continue
            if jude_high_mannose_and_oligo(comp_vector): continue
            MSFragger_compositions.add(comp_vector)
            MSFragger_formulas.add(wrap_formula(calc_glycan_formula(comp_vector)))

    theo_compositions: set[tuple[int, ...]] = set()
    theo_formulas: set[str] = set(glycan_db.keys())
    for comp_group in glycan_db.values(): theo_compositions |= comp_group.keys()

    pGlyco_MSFragger_overlap = pGlyco_compositions & MSFragger_compositions
    total_overlap = pGlyco_MSFragger_overlap & theo_compositions
    uncovered = pGlyco_MSFragger_overlap - theo_compositions

    pGlyco_MSFragger_overlap_formulas = pGlyco_formulas & MSFragger_formulas
    total_overlap_formulas = pGlyco_MSFragger_overlap_formulas & theo_formulas
    uncovered_formulas = pGlyco_MSFragger_overlap_formulas - theo_formulas

    for formula in glycan_db.keys():
        comp_group = glycan_db[formula]
        for comp_vector in comp_group.keys():
            feature_group = comp_group[comp_vector]
            for glycans in feature_group.values():
                for glycan in glycans:
                    glycan_query_StrucGP[glycan] = convert_to_StrucGP(parse_GPSeeker_linkage(glycan.replace('31F', '61F')))
                    decon_struct = parse_GPSeeker_linkage(glycan)
                    glycan_query_pGlyco[glycan] = convert_to_pGlyco(decon_struct)
                    glycan_query_GWB[glycan] = convert_to_GWB(decon_struct)

    if write_to_file:

        with open('overlap_compositions_pGlyco+MSFragger.txt', 'w', encoding='utf-8') as output_overlap1:
            for comp_vector in sorted(pGlyco_MSFragger_overlap): output_overlap1.write(wrap_composition(comp_vector) + '\n')

        with open('overlap_formulas_pGlyco+MSFragger.txt', 'w', encoding='utf-8') as output_overlap1:
            for formula in sorted(pGlyco_MSFragger_overlap_formulas): output_overlap1.write(formula + '\n')

        with open('total_overlap_compositions.txt', 'w', encoding='utf-8') as output_overlap2:
            for comp_vector in sorted(total_overlap): output_overlap2.write(wrap_composition(comp_vector) + '\n')

        with open('total_overlap_formulas.txt', 'w', encoding='utf-8') as output_overlap2:
            for formula in sorted(total_overlap_formulas): output_overlap2.write(formula + '\n')

        with open('uncovered_compositions.txt', 'w', encoding='utf-8') as output_uncovered:
            for comp_vector in sorted(uncovered): output_uncovered.write(wrap_composition(comp_vector) + '\n')

        with open('uncovered_formulas.txt', 'w', encoding='utf-8') as output_uncovered:
            for formula in sorted(uncovered_formulas): output_uncovered.write(formula + '\n')

        with open('glycan_theo_db.txt', 'w', encoding='utf-8') as output_db:
            output_db.write(f'Formula\tComposition\tFeature vector\tGlycan\tGlycan (StrucGP)\tGlycan (pGlyco3)\tGlycan (GlycoWorkBench .gws)\n')
            for comp_vector in sorted(total_overlap):
                formula = wrap_formula(calc_glycan_formula(comp_vector))
                composition = wrap_composition(comp_vector)
                feature_group = glycan_db[formula][comp_vector]
                for fv in sorted(feature_group.keys()):
                    glycans = feature_group[fv]
                    for glycan in glycans:
                        glycan_StrucGP = glycan_query_StrucGP[glycan]
                        glycan_pGlyco = glycan_query_pGlyco[glycan]
                        glycan_GWB = glycan_query_GWB[glycan]
                        output_db.write(f'{formula}\t{composition}\t{tuple(map(int, fv))}\t{glycan}\t{glycan_StrucGP}\t{glycan_pGlyco}\t{glycan_GWB}\n')

    return pGlyco_MSFragger_overlap, total_overlap, uncovered


def get_StrucGP_unconverage(data_file: str, write_to_file: bool = False) -> tuple[set[str], set[str]]:
    glycans_StrucGP =  set()
    with open(data_file, 'r') as data:
        for line in data: glycans_StrucGP.add(line.rstrip('\n').strip(' '))
    uncovered: set[str] =  glycans_StrucGP - set(glycan_query_StrucGP.values())
    covered = glycans_StrucGP - uncovered

    if write_to_file:
        with open('covered_glycans_StrucGP.txt', 'w', encoding='utf-8') as output:
            for glycan in sorted(covered): output.write(f'{glycan}\n')
        with open('uncovered_glycans_StrucGP.txt', 'w', encoding='utf-8') as output:
            for glycan in sorted(uncovered): output.write(f'{glycan}\n')

    return covered, uncovered




if __name__ == '__main__':


    pGlyco_MSFragger_overlap, total_overlap, uncovered = validate_compositions(write_to_file=True)
    uncovered_StrucGP = get_StrucGP_unconverage('MouseFiveTissues_result_glycans_StrucGP.txt', write_to_file=True)

