import argparse
import numpy as np
import os
import pickle
import random
import sys
# sys.path.append('/home/ailab/Desktop/NY/2023_ipactory/models/KIPI-KorPatELECTRA')

from kobart import get_kobart_tokenizer
from konlpy.tag import Mecab
from pytorch_lightning import loggers as pl_loggers
import torch
from transformers import PreTrainedTokenizerFast, AutoTokenizer

from AIHUBDataModule import *
from AIHUBSummarization import *
from AIHUBClassification import *

def boolean_string(s) :
    if s not in ['False', 'True'] :
        raise ValueError('Not a valid boolean string')
    return s == 'True'

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='monologg/koelectra-base-v3-discriminator')
    parser.add_argument('--tokenizer', type=str, default='monologg/koelectra-base-v3-discriminator')
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--gpu_list', type=str, default='0,1', help="string; make list by splitting by ','") # gpu list to be used
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--is_kobart', type=boolean_string, default=False, help='kobart tokenizer needs to add bos and eos token')
    parser.add_argument('--loss_func', type=str, default='CE', help="['CE', 'DICE]")
    # task
    parser.add_argument('--task', type=str, default='cls', help="select from ['summary', 'cls']")
    parser.add_argument('--cls', type=str, default='LLno', help="select from ['LLno', 'Lno', 'Mno', 'Sno', 'SSno']")

    # checkpoints
    parser.add_argument('--dir_path', type=str, default='./checkpoints')
    args = parser.parse_args()

    device = torch.device("cuda")

    with open('/home/ailab/Desktop/NY/2023_ipactory/downstream/aihub_auto_cls/data/labels.pickle', 'rb') as f :
        labels = pickle.load(f)

    ckpt_path = '/home/ailab/Desktop/NY/2023_ipactory/downstream/aihub_auto_cls/ckpt/ckpt_125500/epoch=02-VAL_LOSS=0.269-VAL_F1=0.904-VAL_ACC=0.905.ckpt'
    hparams_file = '/home/ailab/Desktop/NY/2023_ipactory/downstream/aihub_auto_cls/ckpt/ckpt_125500/lightning_logs/version_0/hparams.yamllogs/ver/sion_0/hparams.yaml'
    model = AIHUBClassification.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        # hparams_file=hparams_file,
        args=args, 
        device=device, 
        labels=labels,
        strict=False
    )
    tb_logger = pl_loggers.TensorBoardLogger(args.dir_path)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    dm = AIHUBDataModule('/home/ailab/Desktop/NY/2023_ipactory/downstream/aihub_auto_cls/data/metric', args, tokenizer)

    test = dm.test_dataloader()

    gpu_list = [int(gpu) for gpu in args.gpu_list.split(',')]
    trainer = pl.Trainer(
        gpus = gpu_list,
        logger = tb_logger
    )
    trainer.test(model, dataloaders=test)