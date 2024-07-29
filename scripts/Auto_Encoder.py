

import os
import pickle as pk
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR



GPSM_META_PATH = r'.\Training_GPSMs'
DATASET_PATH = r'.\datasets'


SEED = 114



random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



class LargeDataset(Dataset):

    def __init__(self, meta_path: str, dataset_path: str, dataset_file_tail: str):

        self.__dataset_path = dataset_path
        self._dataset_file_tail = dataset_file_tail
        self.__sample_count = 0
        self.__GPSM_count = 0
        self.__meta_data_GPSM: list[tuple[str, int, list[tuple[str, int, tuple[str, tuple[int, ...], tuple[int, ...]], tuple[bool, ...]]]]] = []
        self.__current_sample_index: int = 0
        self.__current_sample_data: tuple[tuple[np.array, tuple[bool, ...]]]

        for root, dirs, files in os.walk(meta_path):
            for file_name in files:
                sample = ''
                if file_name.endswith('training_GPSMs.pkl'):
                    sample = file_name[:file_name.index('_')]
                elif file_name.endswith('stdGP_test_GPSMs.pkl'):
                    sample = file_name[:file_name.index('-')]
                else:
                    continue
                with open(f'{meta_path}\\{file_name}', 'rb') as meta:
                    meta_data = pk.load(meta)
                    GPSMs_count = len(meta_data)
                    self.__meta_data_GPSM.append((sample, GPSMs_count, meta_data))
                    self.__sample_count += 1
                    self.__GPSM_count += GPSMs_count
            break
        self.__meta_data_GPSM.sort()
        with open(f'{dataset_path}\\{self.__meta_data_GPSM[0][0]}{dataset_file_tail}.pkl', 'rb') as data:
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
            matrix: np.array = None
            feature_vector: tuple[bool, ...]
            if sample_index == self.__current_sample_index:
                matrix, feature_vector = self.__current_sample_data[gpsm_index]
            else:
                del self.__current_sample_data
                with open(f'{self.__dataset_path}\\{self.__meta_data_GPSM[sample_index][0]}{self._dataset_file_tail}.pkl', 'rb') as data:
                    self.__current_sample_data = pk.load(data)
                self.__current_sample_index = sample_index
                matrix, feature_vector = self.__current_sample_data[gpsm_index]
            gpsm_info: tuple[str, int, tuple[str, tuple[int, ...], tuple[int, ...]], tuple[bool, ...]] = gpsm_list[gpsm_index]
            if feature_vector != gpsm_info[3]:
                raise Exception(f'Feature vector {feature_vector} of GPSM from sample \"{sample}\" does not match with its meta data: {gpsm_info} .')
            return torch.tensor(matrix.reshape((30, 5, 36, 80)), dtype=torch.float32),\
                torch.tensor(list(map(int, feature_vector)), dtype=torch.float32),\
                torch.tensor(gpsm_info[2][2], dtype=torch.int)
        raise Exception(f'Index out of range: \"{idx}\", total GPSM = {self.__GPSM_count}')



class ConvAutoEncoder(nn.Module):

    def __init__(self):

        super(ConvAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=30, out_channels=32, kernel_size=(2, 3, 4), stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(2, 3, 4), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(2, 3, 4), stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(64 * 4 * 18 * 39, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64 * 4 * 18 * 39),
            nn.ReLU(inplace=True),
            nn.Unflatten(dim=1, unflattened_size=(64, 4, 18, 39)),
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(2, 3, 4), stride=2, padding=1, output_padding=(1, 1, 0)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=(2, 3, 4), stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(in_channels=32, out_channels=30, kernel_size=(2, 3, 4), stride=1, padding=1),
            nn.BatchNorm3d(30),
            nn.ReLU(inplace=True)
            )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class WeightedMSELoss(nn.Module):

    def __init__(self, non_zero_weight=0.8, zero_weight=0.2):
        super(WeightedMSELoss, self).__init__()
        self.non_zero_weight = non_zero_weight
        self.zero_weight = zero_weight

    def forward(self, output, target):
        total_mse = F.mse_loss(output, target, reduction='mean')
        mask = target != 0
        non_zero_output = output[mask]
        non_zero_target = target[mask]
        if non_zero_target.nelement() > 0:
            non_zero_mse = F.mse_loss(non_zero_output, non_zero_target, reduction='mean')
        else:
            non_zero_mse = torch.tensor(0.0, device=output.device, dtype=output.dtype)
        combined_mse = self.non_zero_weight * non_zero_mse + self.zero_weight * total_mse
        return combined_mse



def train_self_encoder_model(dataset: LargeDataset, batch_size: int, epochs: int) -> tuple[torch.device, ConvAutoEncoder]:

    if torch.cuda.is_available():
        print('Using GPU')
    else:
        print('GPU not available.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoEncoder().to(device)
    criterion = WeightedMSELoss(0.8, 0.2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

    total_sample = len(dataset)
    epoch_handled_sample = 0
    data_loader = DataLoader(dataset, batch_size=batch_size)

    model.train()
    for epoch in range(epochs):
        for data, feature_vects, comp_vects in data_loader:
            input_data = data.to(device)
            output_data = model(input_data)
            loss = criterion(output_data, input_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_handled_sample += batch_size
            print(f'Epoch [{epoch + 1}/{epochs}], Sample [{epoch_handled_sample}/{total_sample}], Loss: {loss.item():.6f}')
        scheduler.step()
        epoch_handled_sample = 0

    return device, model.cpu().eval()


def save_trained_model(model: ConvAutoEncoder, file_name):
    while os.path.exists(file_name):
        input(f'Model file already exists: \"{file_name}\", please remove it manually.')
    torch.save(model.state_dict(), file_name)


def load_trained_model(file_name: str) -> ConvAutoEncoder:
    if not os.path.exists(file_name):
        raise Exception(f'Model file not found: \"{file_name}\"')
    model = ConvAutoEncoder()
    model.load_state_dict(torch.load(file_name))
    model.eval()
    return model


def encode_batch_samples(x: torch.Tensor, model: ConvAutoEncoder) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    with torch.no_grad():
        return model.encoder(x.to(device)).cpu()


def decode_batch_samples(x: torch.Tensor, model: ConvAutoEncoder) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    with torch.no_grad():
        return model.decoder(x.to(device)).cpu()


def create_training_set(dataset: LargeDataset, model: ConvAutoEncoder, batch_size: int, file_name: str):
    encoded_features: torch.Tensor = torch.tensor([])
    feature_vectors: torch.Tensor = torch.tensor([])
    composition_vectors: torch.Tensor = torch.tensor([])
    first_batch_added = False
    total_sample = len(dataset)
    encoded_sample = 0
    data_loader = DataLoader(dataset, batch_size=batch_size)
    criterion = WeightedMSELoss(0.8, 0.2)
    for data, feature_vects, comp_vects in data_loader:
        encoded_data = encode_batch_samples(data, model)
        if first_batch_added:
            encoded_features = torch.cat((encoded_features, encoded_data), dim=0)
            feature_vectors = torch.cat((feature_vectors, feature_vects), dim=0)
            composition_vectors = torch.cat((composition_vectors, comp_vects), dim=0)
        else:
            encoded_features = encoded_data
            feature_vectors = feature_vects
            composition_vectors = comp_vects
            first_batch_added = True
        decoded_data = decode_batch_samples(encoded_data, model)
        loss = criterion(decoded_data, data)
        encoded_sample += batch_size
        print(f'Encoded samples [{encoded_sample}/{total_sample}], loss={loss.item():.6f}')
    with open(f'{file_name}.pkl', 'wb') as output:
        pk.dump(encoded_features.numpy(), output)
    with open(f'{file_name}_labels.pkl', 'wb') as output:
        pk.dump(feature_vectors.int().numpy(), output)
    with open(f'{file_name}_compositions.pkl', 'wb') as output:
        pk.dump(composition_vectors.numpy(), output)




if __name__ == '__main__':

    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())

    dataset = LargeDataset(GPSM_META_PATH, DATASET_PATH, '_training_set')

    # device, model = train_self_encoder_model(dataset, batch_size=32, epochs=20)

    # save_trained_model(model, 'ConvAutoEncoder.pth')

    model = load_trained_model('ConvAutoEncoder.pth')

    test_data = torch.stack([dataset[i][0] for i in (96, 114, 162, 221, 3121)], dim=0)
    test_data_encoded = encode_batch_samples(test_data, model)
    test_data_decoded = decode_batch_samples(test_data_encoded, model)
    mae_loss = F.l1_loss(test_data_decoded, test_data)
    print("MAE Loss:", mae_loss.item())

    create_training_set(dataset, model, 32, 'encoded_training_set')




