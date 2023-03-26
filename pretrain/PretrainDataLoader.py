from json import encoder
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

class PretrainDataset(Dataset) :
    def __init__(self, csv_file : str, chunksize : int, num_data : int, tokenizer, args) :
        self.chunksize = chunksize
        self.csv_file = csv_file
        self.len = num_data

        self.chunk_idx = 0
        self.data = next(
            pd.read_csv(
                self.csv_file,
                skiprows=1,
                chunksize=self.chunksize,
                names=['idx', 'encoder_input', 'decoder_input']
            )
        )

        self.tokenizer = tokenizer
        self.max_length = args.max_length

    
    def __getitem__(self, idx) :
        chunk_idx = idx % self.chunksize

        if chunk_idx == 0 :
            self.data = next(
                pd.read_csv(
                    self.csv_file,
                    skiprows=(idx//self.chunksize) * self.chunksize + 1,
                    chunksize=self.chunksize,
                    names=['idx', 'encoder_input', 'decoder_input']
                )
            )

        encoder_input = self.data['encoder_input'].iloc[chunk_idx]
        decoder_input = self.data['decoder_input'].iloc[chunk_idx]

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

        label_tokens = self.tokenizer(decoder_input)['input_ids'][1:] + [1]
        # labels = torch.tensor(
        #     label_tokens + [-100] * (self.max_length - len(label_tokens))
        # )
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
    

        

# data = pd.read_csv('/home/ailab/Desktop/NY/2023_ipactory/data/csv/valid.csv', chunksize=10000)