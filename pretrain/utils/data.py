import chardet
import csv
import math
import os
import pickle
import random
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# get data files list
def get_datalists() -> list :
    path = '/home/ailab/Desktop/NY/2023_ipactory/data/pretrain'
    years = sorted(list([year for year in os.listdir(path)]))
    years = list(filter(lambda x : os.path.isdir(os.path.join(path,x)), os.listdir(path)))
    print(years)
    months = ['M0102', 'M0304', 'M0506', 'M0708', 'M0910', 'M1112']
    datalists = [] # datalists path
    for year in years :
        for month in months :
            comb = os.path.join(path, year, month)
            # datalists += sorted([os.path.join(comb, file) for file in os.listdir(comb)])
            datalists += sorted([os.path.join(comb, file) for file in os.listdir(comb)])
    
    return datalists

        
def masking(doc_tokens, tokenizer, mask_ratio=0.3, poisson_lambda=3.0) -> str :
    ####################################################################
    # - make noised data, encoder input ids, attention mask
    ####################################################################
    document = tokenizer.decode(doc_tokens[2:])
    # preprocess
    # document = document.replace('(', ' ( ')
    # document = document.replace(')', ' ) ')
    # document = re.sub(pattern=r'[ ]{2,}', repl=' ', string=document)
    tokens = document.split(' ')
    len_doc = len(tokens)
    to_mask = math.ceil(len(tokens) * mask_ratio) # num to mask
    num_masked = 0
    
    while num_masked < to_mask :
        len_span = np.minimum(np.random.poisson(lam=poisson_lambda), len_doc)
        mask_start_idx = int(np.random.uniform(0, len_doc - len_span))

        tokens = np.concatenate(
            [
                tokens[:mask_start_idx],
                ['<mask>'],
                tokens[mask_start_idx + len_span:]
            ],
            axis=None
        )

        len_doc -= len_span - 1
        num_masked += len_span
    noised_doc = '<s>' + ' '.join(tokens) + '</s>'

    return noised_doc

def truncate(path : str, max_length : int, rest : list, tokenizer) :
    ####################################################################
    # - make decoder input ids
    # - tokens.decode : '</s><s>~'
    # - truncated : list, [tokens, tokens, ... ]
    # - rest : list
    ####################################################################
    truncated = []
    f = open(path, 'r')
    lines = f.readlines()
    tokens = []
    # lines = lines.replace('\n', '')
    for line in lines :
        line = re.sub(r'\([\d]+\)', '', line)
        line = re.sub(r'\s{2,}', ' ', line)
        line = line.replace('(', ' ( ')
        line = line.replace(')', ' ) ')
        tokens += tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line))
        

    while len(tokens) >= max_length :
        trunc_size = max_length - len(rest) if not rest else max_length - len(rest) - 1
        trunc = tokens[:trunc_size]
        if rest :
            # concat
            trunc = rest + [1] + tokens[:trunc_size]
            rest = []
            
        tokens = tokens[trunc_size:]
        
        # add eos, bos tokens
        trunc = [1, 0] + trunc
        assert len(trunc) == max_length+2, f"length of truncation is {len(trunc)}, not {max_length+2}.\n"
        

        truncated.append(torch.tensor(trunc))
    rest = tokens
    return truncated, rest


def gen_data(max_length : int, tokenizer, save_path : str, init : bool, datalists : list, start_idx=0) :

    num_data = len(datalists)
    
    if init :
        with open(save_path, 'w', newline= '') as f :
            write = csv.writer(f)
            write.writerow(['idx', 'encoder_input', 'decoder_input'])
            f.close()
    
    rest = []
    idx = start_idx
    with open(save_path, 'a', newline='') as f :
        write = csv.writer(f)
        for idx in tqdm(range(start_idx, num_data)) :
            truncated, rest = truncate(datalists[idx], max_length - 2, rest, tokenizer)

            for trunc in truncated :
                encoder_input = masking(trunc, tokenizer)
                decoder_input = tokenizer.decode(trunc)
                row = [idx, encoder_input, decoder_input]
                write.writerow(row)
            
            with open('./data_gen.out', 'w') as log :
                line = f'last doc : {datalists[idx]}'
                log.write(line)
                log.close()

        # processing rest
        with_special_tokens = torch.tensor([1, 0] + rest)
        encoder_input = masking(with_special_tokens, tokenizer)
        decoder_input = tokenizer.decode(with_special_tokens)
        row = [idx, encoder_input, decoder_input]
        write.writerow(row)

if __name__ == "__main__" :
    datalists = get_datalists()
    random.shuffle(datalists)
    with open('./datalist.pkl', 'wb') as f :
        pickle.dump(datalists, f)
    
    # with open('/home/ailab/Desktop/NY/2023_ipactory/utils/datalist.pkl', 'rb') as f :
    #     datalists = pickle.load(f)
        
    split_idx = math.ceil(len(datalists)/10)
    valid = datalists[:split_idx]
    train = datalists[split_idx:]


    tokenizer = AutoTokenizer.from_pretrained('gogamza/kobart-base-v1')
    # valid
    gen_data(512, tokenizer, '/home/ailab/Desktop/NY/2023_ipactory/data/csv/valid_1.csv', True, valid, 0)
    
    # train
    gen_data(512, tokenizer, '/home/ailab/Desktop/NY/2023_ipactory/data/csv/train_1.csv', True, train, 0)


