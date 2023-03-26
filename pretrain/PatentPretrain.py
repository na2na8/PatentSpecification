import argparse
import os
import sys

import evaluate
import numpy as np


from transformers import BartForConditionalGeneration, AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AdamW
from transformers.utils import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from PretrainDataLoader import PretrainDataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

device = torch.device("cuda")

metric = evaluate.load("sacrebleu")
tokenizer = AutoTokenizer.from_pretrained('gogamza/kobart-base-v1')

def compute_metrics(eval_preds) :
    preds, labels = eval_preds
    if isinstance(preds , tuple) :
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu" : result["score"]}

class PatentBART(nn.Module) :
    def __init__(self, args) :
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(args.model)
        
    def forward(self, inputs) :
        print(inputs)
        outputs = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            decoder_input_ids=inputs['decoder_input_ids'],
            decoder_attention_mask=inputs['decoder_attention_mask'],
            labels=inputs['labels']
        )

        return outputs


class PatentPretrain() :
    def __init__(self, args) :
        self.args = args
        # self.model = PatentBART(self.args)
        self.model = BartForConditionalGeneration.from_pretrained(args.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer)
        # self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        self.train_dataset = PretrainDataset(self.args.train_path, 10000, 25000, self.tokenizer, self.args)
        self.eval_dataset = PretrainDataset(self.args.valid_path, 10000, 5000, self.tokenizer, self.args)
    
    def main(self) :
        logging.set_verbosity_info()
        # logging.basicConfig(
        #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        #     datefmt="%m/%d/%Y %H:%M:%S",
        #     handlers=[logging.StreamHandler(sys.stdout)]
        # )
        logger = logging.get_logger("PatentBART")
        logger.debug("DEBUG")

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.args.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.args.epoch,
            per_device_train_batch_size=self.args.batch_size,
            save_strategy='steps',
            save_steps=10000,
            save_total_limit=1,
            # learning rate scheduler, optimizer
            learning_rate=self.args.learning_rate,
            weight_decay=0.01,
            lr_scheduler_type='linear',
            # warmup_steps=0,
            # warmup_ratio=0.0
            predict_with_generate=True,
            fp16=True,
            dataloader_num_workers=self.args.num_workers,
            # evaluation
            evaluation_strategy='no',
            # logging
            log_level='debug',
            logging_dir=self.args.logging_dir,
            logging_strategy='steps',
            logging_steps=500

        )

        log_level = training_args.get_process_log_level()
        
        logger.setLevel(log_level)
        # datasets.utils.logging.set_verbosity(log_level)
        logging.set_verbosity(log_level)

        # training_args._n_gpu = self.args.gpus

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            # data_collator=self.data_collator,
            compute_metrics=compute_metrics

        )

        trainer.get_train_dataloader(self.train_dataset)
        trainer.get_eval_dataloader(self.eval_dataset)

        trainer.train()
        # trainer.train(resume_from_checkpoint='path_to_ckpt')
        trainer.save_model(self.args.output_dir)
    
        trainer.evaluate(max_length=self.args.max_length)


        

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    g1 = parser.add_argument_group("CommonArgument")
    g1.add_argument("--train_path", type=str, default='/home/ailab/Desktop/NY/2023_ipactory/data/csv/train.csv')
    g1.add_argument("--valid_path", type=str, default='/home/ailab/Desktop/NY/2023_ipactory/data/csv/valid.csv')
    g1.add_argument("--max_length", type=int, default=512)
    g1.add_argument("--tokenizer", type=str, default="gogamza/kobart-base-v1")
    g1.add_argument("--model", type=str, default="gogamza/kobart-base-v1")

    g2 = parser.add_argument_group("TrainingArgument")
    g2.add_argument("--output_dir", type=str, default='/home/ailab/Desktop/NY/2023_ipactory/ckpt/test')
    g2.add_argument("--epoch", type=int, default=1)
    g2.add_argument("--batch_size", type=int, default=16)
    g2.add_argument("--learning_rate", type=float, default=1e-5)
    g2.add_argument("--num_workers", type=int, default=20)
    g2.add_argument("--logging_dir", type=str, default='/home/ailab/Desktop/NY/2023_ipactory/log/test')

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dataset = PretrainDataset(args.train_path, 10000, 25000, tokenizer, args)
    dm = DataLoader(dataset, batch_size=16, num_workers=16)
    t = next(iter(dm))
    # d = next(iter(dataset))
    print(t)
    # train = PatentPretrain(args)
    # train.main()
    # parser.add_argument("--gpus", )
