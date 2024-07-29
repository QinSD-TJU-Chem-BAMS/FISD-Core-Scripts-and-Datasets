

import os
import sys
import pickle as pk
import numpy as np
import pandas as pd
from spectrum_module import MS1_Spectrum, MS2_Spectrum
from Traintest_GPSM_Filtration import GLYCAN_DB
from Glycan_Formula import calc_glycan_formula,wrap_formula
from Embedding import get_embedding_fragments, extract_spectrum_embedding_data, embed_to_nparray


TRAIN_SET_PATH = r'.\training_set'
TRAIN_GPSM_PATH = r'Training_GPSMs'
DATA_SET_PATH = r'.\datasets'
MZML_FILE_PATH = r'E:\FIon_DNN_data\Raw_Files_mzML'



def check_predictable(comp_vector: tuple[int, ...], feature_vector: tuple[bool, ...]) -> bool:
    if sum(feature_vector) > 0: return True
    formula = wrap_formula(calc_glycan_formula(comp_vector))
    return len(GLYCAN_DB[formula][comp_vector]) > 1


def build_dataset_for_model_training():

    target_gpids: dict[str, dict[int, tuple[tuple[str, tuple[int, ...], tuple[int, ...]], tuple[bool, ...]]]] = {}

    for root, dirs, files in os.walk(TRAIN_SET_PATH):
        for file_name in files:
            if not file_name.endswith('.pkl'): continue
            file = open(f'{root}\\{file_name}', 'rb')
            gpids: dict[int, tuple[tuple[str, tuple[int, ...], tuple[int, ...]], tuple[bool, ...]]] = pk.load(file)
            file.close()
            removals = []
            sample_name = file_name[:file_name.index('_')]
            for spectrum_index, (gpid, feature_vector) in gpids.items():
                if check_predictable(gpid[2], feature_vector): continue
                removals.append(spectrum_index)
                print(f'{sample_name}: filtered {gpid}')
            for spectrum_index in removals: del gpids[spectrum_index]
            if sample_name in target_gpids.keys():
                added_gpids = target_gpids[sample_name]
                if len(gpids.keys() & added_gpids.keys()) > 0:
                    raise Exception(f'MS2 spectra were repetitively selected for training in sample \"{sample_name}\".')
                added_gpids.update(gpids)
            else:
                target_gpids[sample_name] = gpids


    first_mzML_loaded = False
    for sample in sorted(target_gpids.keys()):

        training_data: list[tuple[np.array, tuple[bool, ...]]] = []
        selected_GPSMs: list[tuple[str, int, tuple[str, tuple[int, ...], tuple[int, ...]], tuple[bool, ...]]] = []

        training_gpids = target_gpids[sample]
        with open('rsRaw_args.rsc', 'w') as raw_arg_file:
            raw_arg_file.write(f'{MZML_FILE_PATH}\\{sample}.mzML##rsRawArgs##0##rsRawArgs##100')
        if first_mzML_loaded:
            del sys.modules['parse_mzML']
        elif not 'parse_mzML' in sys.modules.keys():
            first_mzML_loaded = True

        from parse_mzML import mzML_parser

        spectra_pack = mzML_parser.get_next_spectra_package()
        ms2_spectra = spectra_pack['MS2']

        for spectrum_index in sorted(training_gpids.keys()):

            while not spectrum_index in ms2_spectra.keys():
                spectra_pack = mzML_parser.get_next_spectra_package()
                ms1_spectra = spectra_pack['MS1']
                if len(ms1_spectra) == 0:
                    raise Exception(f'MS2 spectrum \"{spectrum_index}\" not found in sample \"{sample}\".')
                ms2_spectra = spectra_pack['MS2']

            ms2_spectrum: MS2_Spectrum = ms2_spectra[spectrum_index]

            gpid, feature_vector = training_gpids[spectrum_index]

            total_feature_vector, embed_vector = get_embedding_fragments(gpid[2])
            embedding_data = extract_spectrum_embedding_data(ms2_spectrum.spectrum, gpid, embed_vector)
            if len(embedding_data) == 0: continue
            embedded_matrix = embed_to_nparray(embedding_data, total_feature_vector, block=True)
            training_data.append((embedded_matrix, feature_vector))
            selected_GPSMs.append((sample, spectrum_index, gpid, feature_vector))
            print(f'Sample={sample}, spectrum={spectrum_index}')

        with open(f'{DATA_SET_PATH}\\{sample}_training_set.pkl', 'wb') as output:
            pk.dump(training_data, output)

        with open(f'{TRAIN_GPSM_PATH}\\{sample}_training_GPSMs.pkl', 'wb') as output:
            pk.dump(selected_GPSMs, output)


def build_dataset_for_testing():

    pass




if __name__ == '__main__':

    build_dataset_for_model_training()

