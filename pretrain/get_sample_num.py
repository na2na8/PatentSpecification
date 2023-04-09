from multiprocessing import Pool
import multiprocessing as mp
import concurrent.futures
from tqdm import tqdm
import time
import pandas as pd
import csv
import re

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gogamza/kobart-base-v1')
CHUNKSIZE = 10000



def func(dataframe) :
    total = 0
    under_4096 = 0
    for idx in tqdm(range(CHUNKSIZE)) :
        description = dataframe['descriptions'].iloc[idx]
        claim = dataframe['claims'].iloc[idx]
        total += 1
        if len(description.split(' ')) > 10000 :
            continue
        if len(claim.split(' ')) > 10000 :
            continue
        
        if len(tokenizer(description)['input_ids']) <= 4096 and len(tokenizer(claim)['input_ids']) <= 4096 :
            under_4096 += 1
        
    return (total, under_4096)

if __name__ == '__main__' :
    reader = pd.read_csv('/home/ailab/Desktop/NY/2023_ipactory/data/sample/train.csv', chunksize=CHUNKSIZE)
    count = (0, 0) # total, # under 4096 tokens
    futures = []
    for r in reader :
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=8)
        futures.append(pool.submit(func, r))
        
    for future in concurrent.futures.as_completed(futures) :
        count = tuple(sum(elem) for elem in zip(count, future.result()))
        # count += future.result()
        
    
    result = f'Total : {count[0]} | under 4096 : {count[1]}'
    print(result)
    f = open('./sample_num', 'w')
    f.write(result)
    f.close()