
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import librosa
import json

class CustomDataset(Dataset):
    def __init__(self, filepath, batch_size=128, device="cuda"):
        self.filepath = filepath
        self.batch_size = batch_size
        self.length = int(sum(1 for line in open(self.filepath)) / self.batch_size) - 1
        self.device = device

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx): 
        # The following condition is actually needed in Pytorch. 
        # Otherwise the iterator will be an infinite loop.
        if idx == self.__len__():  
          raise IndexError
       
        df = pd.read_csv(self.filepath, skiprows = self.batch_size * idx, nrows=self.batch_size)
        df.columns = ['track_id', 'timbre', 'pitch','segment time', 'matlab reconstruction']

        def parse_string(str_in):
            return np.array(json.loads(str_in)).squeeze()
        
        def create_spectrogram(y):
            current_length = len(y)
            target_length = 0.29955542442579697 * 22050
            ratio = target_length / current_length
            target_sr = 22050 * ratio
            y = librosa.resample(y, orig_sr=22050, target_sr=target_sr)
            mel = librosa.feature.melspectrogram(y = y, sr = target_sr, n_mels = 128)
            return mel

        
        timbre = df['timbre'].apply(parse_string)
        y = df['matlab reconstruction'].apply(parse_string)
       
        mel = y.apply(create_spectrogram)
        timbre = np.stack(timbre.values)
        mel = np.stack(mel.values)

        return torch.tensor(mel).to(self.device), torch.tensor(timbre).to(self.device)
       

def load_MSD(train_file="C:\\Users\\So\\Documents\\code\\shazam\\New Aproximating Timbre\\train.csv", test_file="C:\\Users\\So\\Documents\\code\\shazam\\New Aproximating Timbre\\test.csv", validation_file="C:\\Users\\So\\Documents\\code\\shazam\\New Aproximating Timbre\\validation.csv", batch_size=128, device="cuda"):
    
    train_dataset = CustomDataset(train_file, batch_size, device=device)
    test_dataset = CustomDataset(test_file, batch_size, device=device)
    validation_dataset = CustomDataset(validation_file, batch_size, device=device)

    train_loader = DataLoader(train_dataset, batch_size = None, shuffle = True, num_workers = 10)
    test_loader = DataLoader(test_dataset, batch_size = None, shuffle = False, num_workers = 10)
    validation_loader = DataLoader(validation_dataset, batch_size = None, shuffle = False, num_workers = 10)

    return train_loader, test_loader, validation_loader
