from json import encoder
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

class PretrainDataset(Dataset) :
    def __init__(self, csv_file : str, skiprows : int, chunksize : int, num_data : int, tokenizer, args) :
        self.chunksize = chunksize
        self.csv_file = csv_file
        self.len = num_data # 35091902

        self.chunk_idx = 0
        self.data = pd.read_csv(
            self.csv_file,
            skiprows=1 + skiprows,
            chunksize=self.chunksize,
            names=['idx', 'encoder_input', 'decoder_input'],
            encoding='utf-8',
            encoding_errors='ignore'
        )

        self.tokenizer = tokenizer
        self.max_length = args.max_length

        self.chunk = next(self.data)

    
    def __getitem__(self, idx) :
        chunk_idx = idx % self.chunksize

        if chunk_idx == 0 :
            self.chunk = next(self.data)

        encoder_input = self.chunk['encoder_input'].iloc[chunk_idx]
        decoder_input = self.chunk['decoder_input'].iloc[chunk_idx]

        encoder_inputs = self.tokenizer(
            encoder_input,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        decoder_inputs = self.tokenizer(
            decoder_input,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        label_tokens = self.tokenizer(decoder_input)['input_ids'][1:512] + [1]
        labels = torch.tensor(label_tokens + [-100] * (self.max_length - len(label_tokens)))

        return {
            'input_ids' : encoder_inputs['input_ids'][0],
            'attention_mask' : encoder_inputs['attention_mask'][0],
            'decoder_input_ids' : decoder_inputs['input_ids'][0],
            'decoder_attention_mask' : decoder_inputs['attention_mask'][0],
            'labels' : labels
        }
    
    def __len__(self) :
        return self.len
    
class PretrainDataModule(pl.LightningDataModule) :
    def __init__(self, path : str, skiprows : int, chunksize : int, num_data : int, tokenizer, args) :
        '''
        path : path of csv
        skiprows : nums of valid set # valid from 0, train from end of validation
        chunksize : chunksize
        num_data : num of entire data
        tokenizer : BartTokenizer
        args : args

        '''
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer

        self.path = path
        self.skiprows = skiprows
        self.chunksize = chunksize
        self.num_data = num_data

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

    def setup(self, stage=None) :
        self.set_train = PretrainDataset(
            self.path, 
            self.skiprows, 
            self.chunksize, 
            self.num_data - self.skiprows, 
            self.tokenizer, 
            self.args
        )

        self.set_valid = PretrainDataset(
            self.path,
            0,
            self.chunksize,
            self.skiprows,
            self.tokenizer,
            self.args
        )

    def train_dataloader(self) :
        train = DataLoader(self.set_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return train
    
    def val_dataloader(self) :
        valid = DataLoader(self.set_valid, batch_size=self.batch_size, num_workers=self.num_workers)
        return valid