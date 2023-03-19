from json import encoder
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

class PretrainDataset(Dataset) :
    def __init__(self, csv_file : str, chunksize : int, num_data : int, args) :
        self.chunksize = chunksize
        self.csv_file = csv_file
        self.len = num_data // self.chunksize

        self.chunk_idx = 0
        self.data = next(
            pd.read_csv(
                self.csv_file,
                skiprows=1,
                chunksize=self.chunksize,
                names=['descriptions', 'claims']
            )
        )

        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.max_length = args.max_length

    
    def __getitem__(self, idx) :
        chunk_idx = idx % self.chunksize

        if chunk_idx == 0 :
            self.data = next(
                pd.read_csv(
                    self.csv_file,
                    skiprows=(idx//self.chunksize) * self.chunksize + 1,
                    chunksize=self.chunksize,
                    names=['description', 'claim']
                )
            )

        description = self.data['descriptions'].iloc[chunk_idx]
        claim = self.data['claims'].iloc[chunk_idx]

        encoder_inputs = self.tokenizer(
            description,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        decoder_inputs = self.tokenizer(
            claim,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        label_tokens = self.tokenizer.tokenize(claim)
        labels = torch.tensor(
            label_tokens + [-100] * (self.max_length - len(label_tokens))
        )

        return {
            'encoder_input_ids' : encoder_inputs['input_ids'][0],
            'encoder_attention_mask' : encoder_inputs['attention_mask'][0],
            'decoder_input_ids' : decoder_inputs['input_ids'][0],
            'decoder_attention_mask' : decoder_inputs['attention_mask'][0],
            'labels' : labels
        }
    
    def __len__(self) :
        return self.len
    

        

# data = pd.read_csv('/home/ailab/Desktop/NY/2023_ipactory/data/csv/valid.csv', chunksize=10000)