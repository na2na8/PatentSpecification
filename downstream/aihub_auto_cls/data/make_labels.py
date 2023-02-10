import pandas as pd
import pickle
from pprint import pprint

# make labels dictionary pickle file
if __name__ == "__main__" :
    # train
    train_csv = pd.read_csv('/home/ailab/Desktop/NY/2023_ipactory/downstream/aihub_auto_cls/data/aihub_auto_cls_train.csv')

    # LLno,Lno,Mno,Sno,SSno
    LLno = {}
    Lno = {}
    Mno = {}
    Sno = {}
    SSno = {}

    label_ll = 0
    label_l = 0
    label_m = 0
    label_s = 0
    label_ss = 0

    for idx in range(len(train_csv)) :
        llno = train_csv['LLno'].iloc[idx]
        lno = train_csv['Lno'].iloc[idx]
        mno = train_csv['Mno'].iloc[idx]
        sno = train_csv['Sno'].iloc[idx]
        ssno = train_csv['SSno'].iloc[idx]

        if not llno in LLno :
            LLno[llno] = label_ll
            label_ll += 1
        
        if not lno in Lno :
            Lno[lno] = label_l
            label_l += 1
        
        if not mno in Mno :
            Mno[mno] = label_m
            label_m += 1
        
        if not sno in Sno :
            Sno[sno] = label_s
            label_s += 1
        
        if not ssno in SSno :
            SSno[ssno] = label_ss
            label_ss += 1
    # valid
    valid_csv = pd.read_csv('/home/ailab/Desktop/NY/2023_ipactory/downstream/aihub_auto_cls/data/aihub_auto_cls_valid.csv')
    valid_csv.drop_duplicates(subset=['documentId'], inplace=True)
    valid_csv = valid_csv.dropna(axis=0)

    val_LLno = {}
    val_Lno = {}
    val_Mno = {}
    val_Sno = {}
    val_SSno = {}

    val_label_ll = 0
    val_label_l = 0
    val_label_m = 0
    val_label_s = 0
    val_label_ss = 0

    for idx in range(len(valid_csv)) :
        val_llno = valid_csv['LLno'].iloc[idx]
        val_lno = valid_csv['Lno'].iloc[idx]
        val_mno = valid_csv['Mno'].iloc[idx]
        val_sno = valid_csv['Sno'].iloc[idx]
        val_ssno = valid_csv['SSno'].iloc[idx]

        if not val_lno in val_LLno :
            val_LLno[val_llno] = val_label_ll
            val_label_ll += 1
        
        if not val_lno in val_Lno :
            val_Lno[val_lno] = val_label_l
            val_label_l += 1
        
        if not val_mno in val_Mno :
            val_Mno[val_mno] = val_label_m
            val_label_m += 1
        
        if not val_sno in val_Sno :
            val_Sno[val_sno] = val_label_s
            val_label_s += 1
        
        if not val_ssno in val_SSno :
            val_SSno[val_ssno] = val_label_ss
            val_label_ss += 1

    labels = {}

    if sorted(list(LLno.keys())) == sorted(list(val_LLno.keys())) and sorted(list(Lno.keys())) == sorted(list(val_Lno.keys())) and \
        sorted(list(Mno.keys())) == sorted(list(val_Mno.keys())) and sorted(list(Sno.keys())) == sorted(list(val_Sno.keys())) and \
        sorted(list(SSno.keys())) == sorted(list(val_SSno.keys())) :
        labels = {
            'LLno' : LLno,
            'Lno' : Lno,
            'Mno' : Mno,
            'Sno' : Sno,
            'SSno' : SSno
        }
    pprint(labels)
    with open('./labels.pickle', 'wb') as wp :
        pickle.dump(labels, wp)