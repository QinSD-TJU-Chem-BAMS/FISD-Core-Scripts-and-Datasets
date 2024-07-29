

import os
import sys
import pickle as pk
import numpy as np
from spectrum_module import MS2_Spectrum
from Embedding import get_embedding_fragments, extract_spectrum_embedding_data, embed_to_nparray
from Feature_Prediction import FEATURE_COUNT_EXCLUDED, predict_feature_vect_excluded_nonlabel
import torch
from torch.utils.data import Dataset, DataLoader
from Auto_Encoder import LargeDataset, ConvAutoEncoder, WeightedMSELoss, load_trained_model, encode_batch_samples, decode_batch_samples
from MLP_Classifier import EncodedDataset, load_trained_MLP_model, predict_batch_samples
from Structure_Diagnosis import run_structure_diagnosis_on_testset




GPSM_META_PATH = r'.\training_set_candidates'
GPSM_META_FILE_TAIL = '_extra_MSFragger&pGlyco3'
GPSM_EXCLUSION_PATH = r'.\training_set'
GPSM_EXCLUSION_FILE_TAIL = '_sup_features'
MZML_FILE_PATH = r'E:\FIon_DNN_data\Raw_Files_mzML'

DATASET_FILE_TAIL = '-sup_test_set'
GPSM_FILE_TAIL = '-sup_test_GPSMs'


class SupDataset(Dataset):

    def __init__(self, gpsm_meta_path: str, dataset_path: str):

        self.__gpsm_meta_path = gpsm_meta_path
        self.__dataset_path = dataset_path
        self.__sample_count = 0
        self.__GPSM_count = 0
        self.__meta_data_GPSM: list[tuple[str, int, list[tuple[str, int, tuple[str, tuple[int, ...], tuple[int, ...]]]]]] = []
        self.__current_sample_index: int = 0
        self.__current_sample_data: tuple[tuple[np.array, tuple[bool, ...]]]

        for root, dirs, files in os.walk(gpsm_meta_path):
            for file_name in files:
                if not file_name.endswith(f'{GPSM_FILE_TAIL}.pkl'):
                    continue
                sample = file_name[:file_name.index(GPSM_FILE_TAIL)]
                with open(f'{gpsm_meta_path}\\{file_name}', 'rb') as meta:
                    meta_data = pk.load(meta)
                    GPSMs_count = len(meta_data)
                    self.__meta_data_GPSM.append((sample, GPSMs_count, meta_data))
                    self.__sample_count += 1
                    self.__GPSM_count += GPSMs_count
            break
        self.__meta_data_GPSM.sort()
        with open(f'{dataset_path}\\{self.__meta_data_GPSM[0][0]}{DATASET_FILE_TAIL}.pkl', 'rb') as data:
            self.__current_sample_data = pk.load(data)


    def __len__(self):
        return self.__GPSM_count


    def __getitem__(self, idx):
        covered_gpsm_count = 0
        for sample_index in range(self.__sample_count):
            sample, gpsm_count, gpsm_list = self.__meta_data_GPSM[sample_index]
            if covered_gpsm_count + gpsm_count <= idx:
                covered_gpsm_count += gpsm_count
                continue
            gpsm_index = idx - covered_gpsm_count
            if sample_index == self.__current_sample_index:
                matrix, spectrum_index = self.__current_sample_data[gpsm_index]
            else:
                del self.__current_sample_data
                with open(f'{self.__dataset_path}\\{self.__meta_data_GPSM[sample_index][0]}{DATASET_FILE_TAIL}.pkl', 'rb') as data:
                    self.__current_sample_data = pk.load(data)
                self.__current_sample_index = sample_index
                matrix, spectrum_index = self.__current_sample_data[gpsm_index]
            gpsm_info: tuple[str, int, tuple[str, tuple[int, ...], tuple[int, ...]]] = gpsm_list[gpsm_index]
            if spectrum_index != gpsm_info[1]:
                raise Exception(f'Spectrum index {spectrum_index} of GPSM from sample \"{sample}\" does not match with its meta data: {gpsm_info} .')
            return torch.tensor(matrix.reshape((30, 5, 36, 80)), dtype=torch.float32), pk.dumps(gpsm_info)
        raise Exception(f'Index out of range: \"{idx}\", total GPSM = {self.__GPSM_count}')


def build_datasets(target_gpids: dict[str, dict[int, tuple[str, tuple[int, ...], tuple[int, ...]]]],
                   output_gpsm_path: str, output_dataset_path: str):

    first_mzML_loaded = False
    for sample in sorted(target_gpids.keys()):

        training_data: list[tuple[np.array, int]] = []
        selected_GPSMs: list[tuple[str, int, tuple[str, tuple[int, ...], tuple[int, ...]]]] = []

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

            gpid = training_gpids[spectrum_index]

            total_feature_vector, embed_vector = get_embedding_fragments(gpid[2])
            embedding_data = extract_spectrum_embedding_data(ms2_spectrum.spectrum, gpid, embed_vector)
            if len(embedding_data) == 0: continue
            embedded_matrix = embed_to_nparray(embedding_data, total_feature_vector, block=True)
            training_data.append((embedded_matrix, spectrum_index))
            selected_GPSMs.append((sample, spectrum_index, gpid))
            print(f'Sample={sample}, spectrum={spectrum_index}')

        with open(f'{output_gpsm_path}\\{sample}{GPSM_FILE_TAIL}.pkl', 'wb') as output:
            pk.dump(selected_GPSMs, output)

        with open(f'{output_dataset_path}\\{sample}{DATASET_FILE_TAIL}.pkl', 'wb') as output:
            pk.dump(training_data, output)


def build_datasets_for_sup_test():
    target_gpsms: dict[str, dict[int, tuple[str, tuple[int, ...], tuple[int, ...]]]] = {}
    for root, dirs, file_names in os.walk(GPSM_META_PATH):
        for file_name in file_names:
            if not file_name.endswith(f'{GPSM_META_FILE_TAIL}.pkl'):
                continue
            sample = file_name[:file_name.index('_')]
            if sample in target_gpsms.keys():
                raise Exception(f'Sample \"{sample}\" repetitively detected in candidate GPSM files.')
            with open(f'{root}\\{file_name}', 'rb') as file:
                target_gpsms[sample] = pk.load(file)
    for root, dirs, file_names in os.walk(GPSM_EXCLUSION_PATH):
        for file_name in file_names:
            if not file_name.endswith(f'{GPSM_EXCLUSION_FILE_TAIL}.pkl'):
                continue
            sample = file_name[:file_name.index('_')]
            if not sample in target_gpsms.keys(): continue
            with open(f'{root}\\{file_name}', 'rb') as file:
                sample_exclusion_gpsms = pk.load(file)
                sample_candidate_gpsms = target_gpsms[sample]
                for spectrum_index in sorted(sample_exclusion_gpsms.keys()):
                    del sample_candidate_gpsms[spectrum_index]
    build_datasets(target_gpsms, r'.\Sup_Test_GPSMs', r'.\datasets_sup_test')


def create_sup_test_dataset(gpsm_meta_path: str, dataset_path: str, output_file_name: str, batch_size: int):

    dataset = SupDataset(gpsm_meta_path, dataset_path)

    model: ConvAutoEncoder = load_trained_model('ConvAutoEncoder.pth')

    encoded_features: torch.Tensor = torch.tensor([])
    total_gpsm_infos: list[tuple[str, int, tuple[str, tuple[int, ...], tuple[int, ...]]]] = []
    first_batch_added = False
    total_sample = len(dataset)
    encoded_sample = 0
    data_loader = DataLoader(dataset, batch_size=batch_size)
    criterion = WeightedMSELoss(0.8, 0.2)
    for data, batch_gpsm_infos in data_loader:
        encoded_data = encode_batch_samples(data, model)
        if first_batch_added:
            encoded_features = torch.cat((encoded_features, encoded_data), dim=0)
        else:
            encoded_features = encoded_data
            first_batch_added = True
        total_gpsm_infos.extend([pk.loads(bytes) for bytes in batch_gpsm_infos])
        decoded_data = decode_batch_samples(encoded_data, model)
        loss = criterion(decoded_data, data)
        encoded_sample += batch_size
        print(f'Encoded samples [{encoded_sample}/{total_sample}], loss={loss.item():.6f}')
    with open(f'{output_file_name}.pkl', 'wb') as output:
        pk.dump(encoded_features.numpy(), output)
    with open(f'{output_file_name}_GPSMs.pkl', 'wb') as output:
        pk.dump(total_gpsm_infos, output)


def predict_feature_vectors_excluded(encoded_dataset_file_name: str, batch_size: int) -> tuple[list[tuple[str, int, tuple[str, tuple[int, ...], tuple[int, ...]]]], list[tuple[int, ...]]]:
    with open(f'{encoded_dataset_file_name}.pkl', 'rb') as file:
        X: np.array = pk.load(file)
    with open(f'{encoded_dataset_file_name}_GPSMs.pkl', 'rb') as file:
        gpsm_infos: list[tuple[str, int, tuple[str, tuple[int, ...], tuple[int, ...]]]] = pk.load(file)
    if X.shape[0] != len(gpsm_infos):
        raise Exception(f'Number of GPSM presented in encoded dataset \"{encoded_dataset_file_name}\" does not match with its GPSM source list.')
    y = []
    compositions = []
    for sample, spectrum_index, gpid in gpsm_infos:
        y.append([0] * FEATURE_COUNT_EXCLUDED)
        compositions.append(list(gpid[2]))

    dataset = EncodedDataset(X, np.array(y), np.array(compositions))
    data_loader = DataLoader(dataset, batch_size=batch_size)

    model = load_trained_MLP_model('MLP.pth')

    gpsm_index = 0
    gpsms_selected = []
    y_pred_fvs = []
    for data, _, batch_comp_vects in data_loader:
        batch_comp_vects = batch_comp_vects.numpy()
        pred_fv_probas = predict_batch_samples(data, model).numpy()
        for i in range(pred_fv_probas.shape[0]):
            gpsm_info = gpsm_infos[gpsm_index]
            comp_vect = tuple(batch_comp_vects[i])
            if comp_vect != gpsm_info[2][2]:
                raise Exception(f'Composition vector {comp_vect} does not match with GPSM at {gpsm_index} from meta \"{encoded_dataset_file_name}_GPSMs\".')
            gpsm_index += 1
            try:
                pred_fv = predict_feature_vect_excluded_nonlabel(comp_vect, tuple(pred_fv_probas[i]))
                y_pred_fvs.append(pred_fv)
            except:
                continue
            gpsms_selected.append(gpsm_info)

    return gpsms_selected, y_pred_fvs





if __name__ == '__main__':

    # build_datasets_for_sup_test()

    # create_sup_test_dataset(r'E:\FIon_DNN_data\Sup_Test\Sup_Test_GPSMs',
    #                         r'E:\FIon_DNN_data\Sup_Test\datasets_sup_test', 'encoded_sup_test_set', batch_size=32)

    gpsm_infos, y_pred_fvs = predict_feature_vectors_excluded('encoded_sup_test_set', batch_size=64)
    run_structure_diagnosis_on_testset(gpsm_infos, y_pred_fvs, r'.\Struct_Diag_SupTest_Results', None)


