import os
import chardet
from itertools import product
import re

from kiwipiepy import Kiwi

def getProcessedXML(path, new_path, kiwi) :
    xmls = sorted([xml for xml in os.listdir(path) if xml.endswith('.XML') or xml.endswith('.xml') or xml.endswith('.sgm')])
    
    pattern = r'\<[^)]*\>'
    suc_newline = r'\n{2,}'
    for xml in xmls :
        xml_str = ''
        absolute_xml = os.path.join(path, xml)
        # find encoding
        raw = open(absolute_xml, 'rb').read()
        result = chardet.detect(raw)
        enc = result['encoding']

        # get orig xml and preprocess
        try :
            file = open(absolute_xml, 'r', encoding=enc)
            lines = file.readlines()  

            for line in lines :
                new_line = re.sub(pattern=pattern, repl='', string=line)
                analyzed = kiwi.analyze(new_line, top_n=1)
                if 'JK' in '|'.join(list(set([tok.tag for tok in analyzed[0][0]]))) :
                    xml_str += new_line
        except Exception :
            print(absolute_xml)
            continue

        xml_str = re.sub(pattern=suc_newline, repl='\n', string=xml_str)
        
        # save
        new_text_path = os.path.join(new_path, xml)[:-4]
        new_file = open(new_text_path, 'w', encoding='utf-8')
        new_file.write(xml_str)
        


def make_directory(path) :
    if not os.path.exists(path) :
        os.makedirs(path)

if __name__ == "__main__" :
    kiwi = Kiwi()
    kiwi.prepare()

    path = './data/raw_data/'
    new_path = './data/pretrain/'

    # orig folder lists
    folders = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    months = ['M0102', 'M0304', 'M0506', 'M0708', 'M0910', 'M1112']
    combs = list(product(folders, months))
    print(combs)
    start = 12
    print("start : ", combs[start])
    for comb in combs[start:31] :
        f_path = os.path.join(path, comb[0], comb[1])
        new_f_path = os.path.join(new_path, comb[0], comb[1])

        if not os.path.exists(new_f_path) :
            make_directory(new_f_path)
        getProcessedXML(f_path, new_f_path, kiwi)

    # 
    # for folder in folders :
    #     f_path = os.path.join(path, folder)
    #     new_f_path = os.path.join(new_path, folder)
        
    #     if not os.path.exists(new_f_path) :
    #         make_directory(new_f_path)

    #     for month in months :
    #         m_path = os.path.join(f_path, month)
    #         new_m_path = os.path.join(new_f_path, month)

    #         if not os.path.exists(new_m_path) :
    #             make_directory(new_m_path)
            
    #         # create preprocessed data
    #         getProcessedXML(m_path, new_m_path)

