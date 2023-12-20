import ast
import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import re   
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    
    #filenames : list of filenames to contain in this dataset
    #num_segments : not used
    #batch_size : size of one batch to be find into model
    def __init__(self, filenames, num_segments, batch_size):
        self.filenames= filenames
        self.num_segments = num_segments
        self.classes_dict = {'Rock' : 0, 'Electronic' : 1, 'Pop': 2, 'Blues' : 3, 'Jazz' : 4, 'Rap': 5, \
                             'Metal' : 6, 'RnB' : 7, 'Country' : 8, 'Reggae' : 9,  'Folk' : 10,  'World' : 11, \
                             'Punk' : 12, 'New' : 13, 'Latin' : 14
                            }
        self.batch_size = batch_size
        self.index_list = [];
        
        
        length = 0
        for filepath in self.filenames:
            for curr_chunk, chunk in enumerate(pd.read_csv(filepath, chunksize=self.batch_size)):
                self.index_list.append((filepath, curr_chunk));
                length += 1
                
      
        self.length = length
        print(f"initialized loader with {len(self.filenames)} files and {self.length} chunks")
        

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx): 
    
        # The following condition is actually needed in Pytorch. 
        # Otherwise the iterator will be an infinite loop.
        if idx == self.__len__():  
          raise IndexError
       
        filepath, chunk_index = self.index_list[idx]

        df = pd.read_csv(filepath, skiprows = self.batch_size * chunk_index, nrows=self.batch_size)


        df.iloc[:,0] = df.iloc[:,0].transform(lambda x: np.array(json.loads(x)))
        df.iloc[:,1] = df.iloc[:,1].transform(lambda x: np.array(json.loads(x)))

        timbre = np.array([value for value in df.iloc[:,0].values])
        pitch = np.array([value for value in df.iloc[:,1].values])

        data = np.array([timbre, pitch])
        data = np.transpose(data, (1, 0, 2, 3))
        labels = [self.classes_dict[key] for key in list(df.iloc[:,3])]

        data = torch.tensor(data)
        labels = torch.tensor(labels)
        
        return data, labels

def load_MSD(num_segments, test_split=0.2, validation_split=0.1, batch_size=128):
    train_dir = "../public/train_test/train"
    test_dir = "../public/train_test/test"

    def get_csv_files(path):
        csv_files = []
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith('.csv'):
                    csv_files.append(os.path.join(dirpath, filename))
        return csv_files
    
    train_files = get_csv_files(train_dir)
    all_test_files = get_csv_files(test_dir)

    all_test_length = len(all_test_files)
    
    test_length = int(all_test_length * test_split)
    validation_length = int(all_test_length * validation_split)

    all_test_files, test_files = train_test_split(train_files, test_size = test_length)
    all_test_files, validation_files = train_test_split(train_files, test_size = validation_length)
    
    train_dataset = CustomDataset(train_files, num_segments, batch_size)
    test_dataset = CustomDataset(test_files, num_segments, batch_size)
    validation_dataset = CustomDataset(validation_files, num_segments, batch_size)

    train_loader = DataLoader(train_dataset, batch_size = None, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = None, shuffle = False)
    validation_loader = DataLoader(validation_dataset, batch_size = None, shuffle = False)
    
    return train_loader, test_loader, validation_loader
