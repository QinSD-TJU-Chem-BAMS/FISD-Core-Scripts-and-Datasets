

import os
import pickle as pk

import pandas as pd

from Peptide_Backbone import wrap_mods



GPSM_META_PATH = r'.\Training_GPSMs'
OUTPUT_FASTA_NAME = 'peptides_fasta_for_GlycReSoft'

GPSM_RESULT_PATH = r'.\Search_Results\GlycReSoft'



def get_sp_line(peptide: str, mod_vector: tuple[int, ...]) -> str:
    name = f'{peptide}_{wrap_mods(mod_vector)}'
    return f'>sp|{name}|{name.upper()} Peptide used in training set'


def extract_peptide_backbones():
    fasta_data: dict[str, str] = {}
    for root, dirs, files in os.walk(GPSM_META_PATH):
        for file_name in files:
            sample = ''
            if file_name.endswith('training_GPSMs.pkl'):
                sample = file_name[:file_name.index('_')]
            elif file_name.endswith('stdGP_test_GPSMs.pkl'):
                sample = file_name[:file_name.index('-')]
            else:
                continue
            with open(f'{GPSM_META_PATH}\\{file_name}', 'rb') as meta:
                meta_data = pk.load(meta)
                for sample_name, spectrum_index, gpid, feature_vector in meta_data:
                    sp_line = get_sp_line(gpid[0], gpid[1])
                    if sp_line in fasta_data.keys(): continue
                    fasta_data[sp_line] = gpid[0]
        break
    with open(f'{OUTPUT_FASTA_NAME}.fasta', 'w') as output:
        for sp_line in sorted(fasta_data.keys()):
            output.write(f'{sp_line}\n')
            output.write(f'{fasta_data[sp_line]}\n')


def parse_peptide_backbone_GlycReSoft(peptide_info: str) -> tuple[str, tuple[int, int]]:
    carba_count = peptide_info.count('(Carbamidomethyl)')
    oxida_count = peptide_info.count('(Oxidation)')
    return peptide_info.replace('(N-Glycosylation)', '').replace('(Carbamidomethyl)', '').replace('(Oxidation)', ''), (carba_count, oxida_count)


def extract_gpids_GlycReSoft(data: pd.DataFrame) -> dict[int, tuple[str, tuple[int, int], str]]:
    extracted_gpids: dict[int, tuple[str, tuple[int, int], str]] = {}
    data.sort_values(by='ms2_score', ascending=False, inplace=True)
    for i in data.index:
        line_data = data.loc[i]
        qvalue: float = line_data['q_value']
        if qvalue >= 0.05: continue
        if not line_data['is_best_match']: continue
        scan_info: str = line_data['scan_id']
        spectrum_index = int(scan_info[scan_info.index('scan=') + 5:])
        if spectrum_index in extracted_gpids.keys():
            # raise Exception(f'Repetitive spectrum detected in GlycReSoft results: {spectrum_index}')
            continue
        gpid_info: str = line_data['glycopeptide']
        glycan = gpid_info[gpid_info.index('{'):]
        peptide_info = gpid_info[:gpid_info.index('{')]
        peptide, mod_vector = parse_peptide_backbone_GlycReSoft(peptide_info)
        extracted_gpids[spectrum_index] = (peptide, mod_vector, glycan)
    return extracted_gpids


def get_sulfated_gpsm_meta_infos() -> dict[str, dict[int, tuple[str, tuple[int, int], str]]]:
    gpsm_meta_infos: dict[str, dict[int, tuple[str, tuple[int, int], str]]] = {}
    for root, dirs, files in os.walk(GPSM_RESULT_PATH):
        for file_name in files:
            if file_name.endswith('.csv'):
                sample = file_name[:file_name.rindex('.')]
            else:
                continue
            data: pd.DataFrame = pd.read_csv(f'{GPSM_RESULT_PATH}\\{file_name}', index_col=None)
            extracted_gpids = extract_gpids_GlycReSoft(data)
            gpsm_meta_infos[sample] = extracted_gpids
        break
    return gpsm_meta_infos




if __name__ == '__main__':

    extract_peptide_backbones()