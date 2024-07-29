

import pickle as pk
import pandas as pd
from Glycan_Formula import parse_composition, calc_composition_Byonic, calc_composition_pGlyco, calc_composition_StrucGP, calc_glycan_formula


DATA_FOLD = r'.\Search_Results'
ENGINE_MSFragger = 'MSFragger-Glyco'
ENGINE_pGlyco3 = 'pGlyco3'
ENGINE_StrucGP = 'StrucGP'
ENGINES = (ENGINE_MSFragger, ENGINE_pGlyco3, ENGINE_StrucGP)
SAMPLE_TITLES = ('MouseBrain', 'MouseHeart', 'MouseKidney', 'MouseLiver', 'MouseLung')
SHEET_SUFFIXES = ('-Z-T-1', '-Z-T-2', '-Z-T-3', '-Z-T-4', '-Z-T-5')

TEST_SET_PATH = r'.\test_set'
TRAIN_SET_PATH = r'.\training_set'
TRAIN_SET_CAND_PATH = r'.\training_set_candidates'



GLYCAN_DB: dict[str, dict[tuple[int, ...], dict[tuple[bool, ...], tuple[str, ...]]]]
GLYCAN_QUERY_StrucGP: dict[str, tuple[str, tuple[int, ...], tuple[bool, ...], str]] = {}
COVERED_COMPS: set[tuple[int, ...]] = set()
with open('glycan_theo_db.pkl', 'rb') as db_file:
    GLYCAN_DB = pk.load(db_file)
with open('glycan_theo_db.txt', 'r', encoding='utf-8') as db_file:
    db_file.readline()
    for line in db_file:
        data = line.rstrip('\n').split('\t')
        GLYCAN_QUERY_StrucGP[data[4]] = (data[0], parse_composition(data[1]), tuple(map(bool, eval(data[2]))), data[3])
with open('total_overlap_compositions.txt', 'r', encoding='utf-8') as comp_file:
    for line in comp_file: COVERED_COMPS.add(parse_composition(line.rstrip('\n')))


def parse_MSFragger_mods(mods: str) -> tuple[int, ...]:
    return  (mods.count('C'), mods.count('M'))


def parse_pGlyco_mods(mods: str) -> tuple[int, ...]:
    return (mods.count('[C]'), mods.count('O'))


def parse_StrucGP_mods(mods: str) -> tuple[int, ...]:
    return (mods.count('[Car'), mods.count('[Oxi'))



def get_gpid_MSFragger(line_data: pd.Series, comp_vector: tuple[int, ...]) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    return (line_data['Peptide'], parse_MSFragger_mods(str(line_data['Assigned Modifications'])), comp_vector)


def get_gpid_pGlyco(line_data: pd.Series, comp_vector: tuple[int, ...]) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    return (line_data['Peptide'].replace('J', 'N'), parse_pGlyco_mods(str(line_data['Mod'])), comp_vector)


def get_gpid_StrucGP(line_data: pd.Series, comp_vector: tuple[int, ...]) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    return (line_data['PeptideSequence'], parse_StrucGP_mods(line_data['Peptide']), comp_vector)


def extract_gpids_MSFragger(data: pd.DataFrame) -> dict[int, tuple[str, tuple[int, ...], tuple[int, ...]]]:
    extracted_data: dict[int, tuple[str, tuple[int, ...], tuple[int, ...]]] = {}
    for i in data.index:
        line_data: pd.Series = data.loc[i]
        glyco_info = str(line_data['Total Glycan Composition'])
        if glyco_info == 'nan' or glyco_info == '' or glyco_info.startswith('Mod') or not ' %' in glyco_info: continue
        comp_vector = ()
        try:
            comp_vector = calc_composition_Byonic(glyco_info[:glyco_info.index(' %')])
            if not comp_vector in COVERED_COMPS: continue
        except:
            continue
        if line_data['Glycan q-value'] >= 0.01: continue
        spectrum_index = int(line_data['Spectrum'].split('.')[1])
        gpid = get_gpid_MSFragger(line_data, comp_vector)
        if spectrum_index in extracted_data:
            print(f'Repetitive ID detected in MSFragger data. Spectrum={spectrum_index}, GPID=\'{gpid}\'')
            continue
        extracted_data[spectrum_index] = gpid
    return extracted_data


def extract_gpids_pGlyco(data: pd.DataFrame) -> dict[int, tuple[str, tuple[int, ...], tuple[int, ...]]]:
    extracted_data: dict[int, tuple[str, tuple[int, ...], tuple[int, ...]]] = {}
    data.sort_values(by='TotalFDR', ascending=True, inplace=True)
    for i in data.index:
        line_data: pd.Series = data.loc[i]
        comp_vector = calc_composition_pGlyco(line_data['PlausibleStruct'])
        if not comp_vector in COVERED_COMPS: continue
        spectrum_index = int(line_data['GlySpec'].split('.')[1])
        gpid = get_gpid_pGlyco(line_data, comp_vector)
        if spectrum_index in extracted_data:
            print(f'Repetitive ID detected in pGlyco3 data. Spectrum={spectrum_index}, GPID=\'{gpid}\'')
            continue
        extracted_data[spectrum_index] = gpid
    return extracted_data


def extract_gpids_StrucGP(data: pd.DataFrame) -> dict[int, tuple[tuple[str, tuple[int, ...], tuple[int, ...]], tuple[bool, ...]]]:
    extracted_data: dict[int, tuple[tuple[str, tuple[int, ...], tuple[int, ...]], tuple[bool, ...]]] = {}
    for i in data.index:
        line_data: pd.Series = data.loc[i]
        glycan = line_data['Structure_codinga']
        if not glycan in GLYCAN_QUERY_StrucGP.keys(): continue
        glycan_info = GLYCAN_QUERY_StrucGP[glycan]
        comp_vector = glycan_info[1]
        spectrum_index = int(line_data['MS2Scan'])
        gpid = get_gpid_StrucGP(line_data, comp_vector)
        if spectrum_index in extracted_data:
            print(f'Repetitive ID detected in StrucGP data. Spectrum={spectrum_index}, GPID=\'{gpid}\'')
            continue
        extracted_data[spectrum_index] = (gpid, glycan_info[2])
    return extracted_data




if __name__ == '__main__':


    for sample_title in SAMPLE_TITLES:
        for sheet_suffix in SHEET_SUFFIXES:

            sample_name = f'{sample_title}{sheet_suffix}'

            extracted_data_MSFragger = extract_gpids_MSFragger(
                pd.read_excel(f'{DATA_FOLD}\\{ENGINE_MSFragger}\\{sample_title}.xlsx', sheet_name=sample_name))
            extracted_data_pGlyco = extract_gpids_pGlyco(
                pd.read_excel(f'{DATA_FOLD}\\{ENGINE_pGlyco3}\\{sample_title}.xlsx', sheet_name=sample_name))
            extracted_data_StrucGP = extract_gpids_StrucGP(
                pd.read_excel(f'{DATA_FOLD}\\{ENGINE_StrucGP}\\{sample_title}.xlsx', sheet_name=sample_name))

            overlap_MSF_pGlyco = {}
            for spectrum_index in extracted_data_MSFragger.keys() & extracted_data_pGlyco.keys():
                gpid = extracted_data_MSFragger[spectrum_index]
                if gpid == extracted_data_pGlyco[spectrum_index]:
                    overlap_MSF_pGlyco[spectrum_index] = gpid
            print(f'{sample_name} : overlap IDs MSFragger & pGlyco3 = {len(overlap_MSF_pGlyco)}')

            overlap_total = {}
            for spectrum_index in overlap_MSF_pGlyco.keys() & extracted_data_StrucGP.keys():
                gpid_info = extracted_data_StrucGP[spectrum_index]
                if gpid_info[0] == overlap_MSF_pGlyco[spectrum_index]:
                    overlap_total[spectrum_index] = gpid_info
            with open(f'{TRAIN_SET_PATH}\\{sample_name}_overlap_total.pkl', 'wb') as output:
                pk.dump(overlap_total, output)
            print(f'{sample_name} : overlap IDs total = {len(overlap_total)}')

            extra_MSF_pGlyco = {}
            for spectrum_index in overlap_MSF_pGlyco.keys() - overlap_total.keys():
                extra_MSF_pGlyco[spectrum_index] = overlap_MSF_pGlyco[spectrum_index]
            with open(f'{TRAIN_SET_CAND_PATH}\\{sample_name}_extra_MSFragger&pGlyco3.pkl', 'wb') as output:
                pk.dump(extra_MSF_pGlyco, output)
            print(f'{sample_name} : extra IDs MSFragger & pGlyco3 = {len(extra_MSF_pGlyco)}')

            extra_StrucGP = {}
            for spectrum_index in extracted_data_StrucGP.keys() - overlap_total.keys():
                extra_StrucGP[spectrum_index] = extracted_data_StrucGP[spectrum_index]
            with open(f'{TEST_SET_PATH}\\{sample_name}_extra_StrucGP.pkl', 'wb') as output:
                pk.dump(extra_StrucGP, output)
            print(f'{sample_name} : extra IDs StrucGP = {len(extra_StrucGP)}')





