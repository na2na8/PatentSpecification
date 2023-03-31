import argparse
import math
import os
import random

from pytorch_lightning import loggers as pl_loggers
import torch
from transformers import AutoTokenizer

from PretrainDataModule import *
from PatentPretrain import *

def set_random_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)

    pl.seed_everything(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Set random seed : {random_seed}")

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    g1 = parser.add_argument_group("CommonArgument")
    g1.add_argument("--path", type=str, default='/home/ailab/Desktop/NY/2023_ipactory/data/csv/data.csv')
    g1.add_argument("--tokenizer", type=str, default="gogamza/kobart-base-v1")
    g1.add_argument("--model", type=str, default="gogamza/kobart-base-v1")
    g1.add_argument("--ckpt_path", type=str, default=None)
    g1.add_argument("--ckpt_last_step", type=int, default=None)
    g1.add_argument("--gpu_lists", type=str, default="0", help="string; make list by splitting by ','") # gpu list to be used

    g2 = parser.add_argument_group("TrainingArgument")
    g2.add_argument("--ckpt_dir", type=str, default='/home/ailab/Desktop/NY/2023_ipactory/ckpt/basic')
    g2.add_argument("--epochs", type=int, default=1)
    g2.add_argument("--num_warmup_steps", type=int, default=10000)
    g2.add_argument("--skiprows", type=int, default=3500000)
    g2.add_argument("--max_length", type=int, default=512)
    g2.add_argument("--batch_size", type=int, default=16)
    g2.add_argument("--chunksize", type=int, default=1000000)
    g2.add_argument("--num_data", type=int, default=35091902)
    g2.add_argument("--min_learning_rate", type=float, default=1e-5)
    g2.add_argument("--max_learning_rate", type=float, default=1e-4)
    g2.add_argument("--num_workers", type=int, default=20)
    g2.add_argument("--logging_dir", type=str, default='/home/ailab/Desktop/NY/2023_ipactory/log/basic')

    args = parser.parse_args()

    set_random_seed(random_seed=42)

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.ckpt_path :
        model = PatentPretrain.load_from_checkpoint(checkpoint_path=args.ckpt_path, args=args, device=device, tokenizer=tokenizer)
    else :
        model = PatentPretrain(args, device, tokenizer)

    dm = PretrainDataModule(args.path, args.skiprows, args.ckpt_last_step*args.batch_size , args.chunksize, args.num_data, tokenizer, args)

    gpu_list = [int(gpu) for gpu in args.gpu_lists.split(',')]

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
                                monitor='TRAIN_STEP_LOSS',
                                dirpath=args.ckpt_dir,
                                filename='{step:06d}-{TRAIN_STEP_LOSS:.3f}-{TRAIN_STEP_BLEU:.3f}',
                                verbose=False,
                                save_last=True,
                                every_n_train_steps=500,
                                mode='min',
                                save_top_k=2
                            )
    
    tb_logger = pl_loggers.TensorBoardLogger(args.logging_dir)
    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_logger],
        default_root_dir=args.ckpt_dir,
        log_every_n_steps=100,
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=gpu_list
    )

    trainer.fit(model, dm)