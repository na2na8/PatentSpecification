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



def set_random_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)

    pl.seed_everything(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Set random seed : {random_seed}")

def boolean_string(s) :
    if s not in ['False', 'True'] :
        raise ValueError('Not a valid boolean string')
    return s == 'True'

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gogamza/kobart-base-v1')
    parser.add_argument('--tokenizer', type=str, default='gogamza/kobart-base-v1')
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--gpu_list', type=str, default='0', help="string; make list by splitting by ','") # gpu list to be used
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--is_kobart', type=boolean_string, default=True, help='kobart tokenizer needs to add bos and eos token')
    parser.add_argument('--loss_func', type=str, default='CE', help="['CE', 'DICE]")
    # task
    parser.add_argument('--task', type=str, default='summary', help="select from ['summary', 'cls']")
    parser.add_argument('--cls', type=str, default=None, help="select from ['LLno', 'Lno', 'Mno', 'Sno', 'SSno']")

    # checkpoints
    parser.add_argument('--dir_path', type=str, default='./checkpoints')
    args = parser.parse_args()

    set_random_seed(random_seed=42)

    if args.tokenizer == 'sk_kobart' :
        tokenizer = get_kobart_tokenizer()
    elif args.tokenizer == 'gogamza/kobart-base-v1':
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    else :
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.task == 'summary' :
        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='VAL_LOSS',
                                                    dirpath=args.dir_path,
                                                    filename='{epoch:02d}-{VAL_LOSS:.3f}-{VAL_ROUGE1:.3f}-{VAL_ROUGE2:.3f}-{VAL_ROUGEL:.3f}',
                                                    verbose=False,
                                                    save_last=True,
                                                    mode='min',
                                                    save_top_k=1,
                                                    )
    elif args.task == 'cls' :
        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='VAL_LOSS',
                                                    dirpath=args.dir_path,
                                                    filename='{epoch:02d}-{VAL_LOSS:.3f}-{VAL_F1:.3f}-{VAL_ACC:.3f}',
                                                    verbose=False,
                                                    save_last=True,
                                                    mode='min',
                                                    save_top_k=1,
                                                    )
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.dir_path, 'tb_logs'))
    lr_logger = pl.callbacks.LearningRateMonitor()

    gpu_list = [int(gpu) for gpu in args.gpu_list.split(',')]
    trainer = pl.Trainer(
        default_root_dir= os.path.join(args.dir_path, 'checkpoints'),
        logger = tb_logger,
        callbacks = [checkpoint_callback, lr_logger],
        max_epochs=args.epoch,
        gpus=gpu_list
    )

    device = torch.device("cuda")

    dm = AIHUBDataModule('/home/ailab/Desktop/NY/2023_ipactory/downstream/aihub_auto_cls/data', args, tokenizer)
    
    if args.task == 'summary' :
        model = AIHUBSummarization(args, tokenizer, device)
    elif args.task == 'cls' :
        with open('/home/ailab/Desktop/NY/2023_ipactory/downstream/aihub_auto_cls/data/labels.pickle', 'rb') as f :
            labels = pickle.load(f)
        model = AIHUBClassification(args, device, labels)
    
    trainer.fit(model, dm)