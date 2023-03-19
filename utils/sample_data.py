import chardet
import csv
import math
import os
import random
import re

from bs4 import BeautifulSoup

def get_datalists() :
    path = '/home/ailab/Desktop/NY/2023_ipactory/data/raw_data'
    years = ['Y' + str(year) for year in range(2013, 2022)]
    months = ['M0102', 'M0304', 'M0506', 'M0708', 'M0910', 'M1112']
    datalists = []
    for year in years :
        for month in months :
            comb = os.path.join(path, year, month)
            datalists += sorted([os.path.join(comb, file) for file in os.listdir(comb)])
    
    return datalists

def parse_xml(file : str) :
    raw = open(file, 'rb').read()
    result = chardet.detect(raw)
    enc = result['encoding']

    fp = open(file, 'r', encoding=enc)
    soup = BeautifulSoup(fp, 'html.parser')
    description = soup.find('description-of-embodiments').get_text()
    claim = soup.find('claims').get_text()
    claim = re.sub(r'\([\d]+\)', '', claim)

    return description, claim
    
def gen_csv(datalists : list, save_path : str) :
    with open(save_path, 'w', newline='') as f :
        write = csv.writer(f)
        write.writerow(['descriptions', 'claims'])
        f.close()
    
    with open(save_path, 'a', newline='') as f :
        write = csv.writer(f)
        
        for file in datalists :
            try :
                description, claim = parse_xml(file)
                write.writerow([description, claim])
            except Exception:
                continue
    
if __name__ == "__main__" :
    datalists = get_datalists()
    random.shuffle(datalists)
    split_idx = math.ceil(len(datalists)/10)
    valid = datalists[:split_idx]
    train = datalists[split_idx:]

    # valid
    valid_path = '/home/ailab/Desktop/NY/2023_ipactory/data/sample/valid.csv'
    gen_csv(valid, valid_path)

    # train
    train_path = '/home/ailab/Desktop/NY/2023_ipactory/data/sample/train.csv'
    gen_csv(train, train_path)