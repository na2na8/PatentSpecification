import os
import json
import numpy as np
import pandas as pd
import pickle
import re

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

class AIHUBDataset(Dataset) :
    def __init__(self, path, stage, tokenizer, args) :
        self.tokenizer = tokenizer
        self.max_len = args.max_len

        # make json dataset to csv
        csv_path = os.path.join(path, 'aihub_auto_cls_' + stage + '.csv')
        if not os.path.exists(csv_path) :
            self.make_dataset(path, stage)
        
        # DataFrame
        self.dataset = pd.read_csv(csv_path)
        self.dataset.drop_duplicates(subset=['claims'], inplace=True)
        self.dataset = self.dataset.dropna(axis=0)
        # self.dataset = self.dataset.iloc[:100]

        # task
        self.is_kobart = args.is_kobart
        self.task = args.task
        if self.task == 'cls' :
            self.cls = args.cls
            with open('/home/ailab/Desktop/NY/2023_ipactory/downstream/aihub_auto_cls/data/labels.pickle', 'rb') as p :
                self.labels = pickle.load(p)
                p.close()


    def __len__(self) :
        return len(self.dataset)

    def add_padding_data(self, inputs) :
        if len(inputs) < self.max_len :
            pad = np.array([self.tokenizer.pad_token_id] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else :
            inputs = inputs[:self.max_len]
            if self.task == 'cls' :
                inputs[-1] = self.tokenizer.eos_token_id
        return inputs
    
    def add_ignored_data(self, inputs) :
        if len(inputs) < self.max_len :
            pad = np.array([-100]*(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else :
            inputs = inputs[:self.max_len]
        return inputs

    def __getitem__(self, idx) :
        # =============================== SUMMARIZATION TASK ===============================
        # BART, BERTSUM(not yet, ELECTRA, KorPatELECTRA)
        if self.task == 'summary' :
            # encoder_inputs : <s> ~ </s>
            claims = self.tokenizer.bos_token + re.sub(r'\([^)]*\)', '', self.dataset['claims'].iloc[idx]) + self.tokenizer.eos_token
            encoder_input_ids = self.add_padding_data(self.tokenizer.encode(claims))
            # decoder_inputs : </s><s> ~ </s>, labels : <s> ~ </s>
            abstract = self.tokenizer.bos_token + re.sub(r'\([^)]*\)', '', self.dataset['abstract'].iloc[idx]) # <s> ~ 
            labels = self.tokenizer.encode(abstract)
            labels.append(self.tokenizer.eos_token_id) # <s> ~ </s>
            decoder_input_ids = [self.tokenizer.eos_token_id] # </s>
            decoder_input_ids += labels[:-1] # </s><s> ~
            decoder_input_ids = self.add_padding_data(decoder_input_ids)
            labels = self.add_ignored_data(labels)

            # make inputs to tensors
            encoder_input_ids = torch.from_numpy(np.array(encoder_input_ids, dtype=np.int_))
            encoder_attention_mask = encoder_input_ids.ne(self.tokenizer.pad_token_id).float()
            decoder_input_ids = torch.from_numpy(np.array(decoder_input_ids, dtype=np.int_))
            decoder_attention_mask = decoder_input_ids.ne(self.tokenizer.pad_token_id).float()
            labels = torch.from_numpy(np.array(labels, dtype=np.int_))
            
            return {
                'encoder_input_ids' : encoder_input_ids,
                'encoder_attention_mask' : encoder_attention_mask,
                'decoder_input_ids' : decoder_input_ids,
                'decoder_attention_mask' : decoder_attention_mask,
                'labels' : labels
            }

        # =============================== CLASSIFICATION TASK ===============================
        elif self.task == 'cls' :
            if self.is_kobart :
                # claims = self.tokenizer.bos_token + re.sub(r'\([^)]*\)', '', self.dataset['claims'].iloc[idx]) + self.tokenizer.eos_token
                claims = self.tokenizer.bos_token + self.dataset['claims'].iloc[idx] + self.tokenizer.eos_token
                input_ids = self.add_padding_data(self.tokenizer.encode(claims))
                input_ids = torch.from_numpy(np.array(input_ids, dtype=np.int_))
                attention_mask = input_ids.ne(self.tokenizer.pad_token_id).float()
            else :
                # claims = re.sub(r'\([^)]*\)', '', self.dataset['claims'].iloc[idx])
                claims = self.dataset['claims'].iloc[idx]
                inputs = self.tokenizer(
                    claims,
                    return_tensors='pt',
                    max_length=self.max_len,
                    padding='max_length',
                    truncation=True,
                    add_special_tokens=True
                )
                input_ids = inputs['input_ids'][0]
                attention_mask = inputs['attention_mask'][0]
            label_name = self.dataset[self.cls].iloc[idx]
            label = self.labels[self.cls][label_name]


            return {
                'input_ids' : input_ids,
                'attention_mask' : attention_mask,
                'labels' : label
            }

    def make_dataset(self, path, stage) -> None :
        # path : './data'
        # stage : [train, valid]
        csv_dataset = {
            'documentId' : [],
            'invention_title' : [],
            'claims' : [],
            'abstract' : [],
            'LLno' : [],
            'Lno' : [],
            'Mno' : [],
            'Sno' : [],
            'SSno' : []
        }
        # label
        label_jsons = self.get_jsons(os.path.join(path, stage), 'label')
        label_jsons.sort()
        
        # raw
        raw_jsons = self.get_jsons(os.path.join(path, stage), 'raw')
        raw_jsons.sort()

        for idx in range(len(label_jsons)) :
            with open(label_jsons[idx], 'r') as label_json :
                label_json_data = json.load(label_json)
            with open(raw_jsons[idx], 'r') as raw_json :
                raw_json_data = json.load(raw_json)
            
            for json_idx in range(len(label_json_data['dataset'])) :
                label = label_json_data['dataset'][json_idx]
                raw = raw_json_data['dataset'][json_idx]

                if label['documentId'] != raw['documentId'] :
                    print('Not same documentId')
                    exit()
                
                if 'claims' in raw.keys() :
                    csv_dataset['claims'].append(raw['claims'])
                else : csv_dataset['claims'].append(None)
                csv_dataset['documentId'].append(label['documentId'])
                csv_dataset['invention_title'].append(raw['invention_title'])
                if 'abstract' in raw.keys() :
                    csv_dataset['abstract'].append(raw['abstract'])
                else : csv_dataset['abstract'].append(None)
                csv_dataset['LLno'].append(label['LLno'])
                csv_dataset['Lno'].append(label['Lno'])
                csv_dataset['Mno'].append(label['Mno'])
                csv_dataset['Sno'].append(label['Sno'])
                csv_dataset['SSno'].append(label['SSno'])
        data = pd.DataFrame(csv_dataset)
        data.to_csv(os.path.join(path, 'aihub_auto_cls_' + stage + '.csv'), index=False)
        print(os.path.join(path, 'aihub_auto_cls_' + stage + '.csv'))
        

    # return jsons list
    def get_jsons(self, path, subfolder) -> list :
        jsons = []
        root_path = os.path.join(path, subfolder)
        first_dirs = list(filter(lambda directory : os.path.isdir(directory), [os.path.join(root_path, path) for path in os.listdir(root_path)]))
        for first_dir in first_dirs :
            second_dirs = [os.path.join(first_dir, directory) for directory in os.listdir(first_dir)]
            for second_dir in second_dirs :
                third_dirs = [os.path.join(second_dir, directory) for directory in os.listdir(second_dir)]
                for third_dir in third_dirs :
                    jsons += [os.path.join(third_dir, file) for file in os.listdir(third_dir)]

        return jsons

class AIHUBDataModule(pl.LightningDataModule) :
    def __init__(self, path, args, tokenizer) :
        '''
        paths(dict) : dictionary that has each paths of train, valid, test dataset
        '''
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer

        self.path = path

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.setup()

    def setup(self, stage=None) :
        self.set_train = AIHUBDataset(self.path, stage='train', tokenizer=self.tokenizer, args=self.args)
        self.set_valid = AIHUBDataset(self.path, stage='valid', tokenizer=self.tokenizer, args=self.args)
        # self.set_test = AIHUBDataset(self.path, stage='test', tokenizer=self.tokenizer, args=self.args)

    def train_dataloader(self) :
        train = DataLoader(self.set_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return train

    def val_dataloader(self) :
        valid = DataLoader(self.set_valid, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return valid
    
    # def test_dataloader(self):
    #     test = DataLoader(self.set_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    #     return test