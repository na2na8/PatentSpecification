import math
import os

import numpy as np
from transformers import AutoTokenizer

# get data files list
def get_datalists() -> list :
    path = '/home/ailab/Desktop/NY/2023_ipactory/data/pretrain'
    years = sorted(os.listdir(path))
    months = ['M0102', 'M0304', 'M0506', 'M0708', 'M0910', 'M1112']
    datalists = [] # datalists path
    for year in years :
        for month in months :
            comb = os.path.join(path, year, month)
            datalists += sorted([os.path.join(comb, file) for file in os.listdir(comb)])
    
    return datalists

# truncate documents to 512 tokens(with concat documents)
def truncate(datalists : list, max_length : int, tokenizer) -> list :
    rest = []
    data = []
    for file_path in datalists :
        f = open(file_path, 'r')
        lines = ''.join(f.readlines())
        lines = lines.replace('\n', '')
        tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(lines))

        while len(tokens) > max_length :
            trunc_size = max_length - len(rest) if not rest else max_length - len(rest) - 1

            trunc = tokens[:trunc_size]
            if rest :
                # concat
                trunc = rest + [1] + tokens[:trunc_size]
                rest = []
            
            tokens = tokens[trunc_size:]
            
            # add bos, eos tokens
            # trunc = [0] + trunc + [1]
            # assert len(trunc) == max_length + 2, f"length of truncation is {len(trunc)}, not {max_length+2}."
            assert len(trunc) == max_length, f"length of truncation is {len(trunc)}, not {max_length}."

            data.append(trunc)
        rest = tokens
    # data.append([0] + rest + [1])
    data.append(rest)
    
    return data
        
def masking(tokens, mask_id : int, mask_ratio=0.3, poisson_lambda=3.0) :
    to_mask = math.ceil(len(tokens) * mask_ratio) # num to mask
    num_masked = 0
    start_idx = 0
    while num_masked < to_mask :
        len_span = np.random.poisson(poisson_lambda)
        
        


    

print(len(get_datalists()))
    
# tokenizer = AutoTokenizer.from_pretrained('gogamza/kobart-base-v1')
# dl = ['/home/ailab/Desktop/NY/2023_ipactory/data/pretrain/Y2017/M0304/1020147028136', '/home/ailab/Desktop/NY/2023_ipactory/data/pretrain/Y2017/M0304/1020147034262']
# truncate(dl, 510, tokenizer)

