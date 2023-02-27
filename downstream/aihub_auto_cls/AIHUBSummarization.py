import re

import torch
import torch.nn as nn
# from torchmetrics.text.rouge import ROUGEScore
from rouge import Rouge
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_looggers
from kobart import get_pytorch_kobart_model
from transformers import BartForConditionalGeneration, AdamW

class AIHUBSummarization(pl.LightningModule) :
    def __init__(self, args, tokenizer, device) :
        super().__init__()
        self._device = device

        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        
        if args.model == 'sk_kobart' :
            self.model = BartForConditionalGeneration.from_pretrained(get_pytorch_kobart_model())
        else : self.model = BartForConditionalGeneration.from_pretrained(args.model)
        self.tokenizer = tokenizer
        # self.rouge = ROUGEScore()
        self.rouge = Rouge()

        self.save_hyperparameters()

    def forward(
        self,
        encoder_input_ids,
        encoder_attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        labels=None
    ) :
        outputs = self.model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

        return outputs
    
    def step(self, batch, batch_idx, state) :
        outputs = self(
            encoder_input_ids=batch['encoder_input_ids'].to(self._device),
            encoder_attention_mask=batch['encoder_attention_mask'].to(self._device),
            decoder_input_ids=batch['decoder_input_ids'].to(self._device),
            decoder_attention_mask=batch['decoder_attention_mask'].to(self._device),
            labels=batch['labels'].to(self._device)
        )

        loss = outputs.loss
        logits = outputs.logits # shape : (batch, len_decoder_inputs, vocab)
        
        preds = torch.argmax(logits, dim=2).to(self._device) # shape : (batch, len_decoder_inputs)
        targets = batch['decoder_input_ids'].to(self._device)

        self.log(f"[{state.upper()} LOSS]", loss, prog_bar=True)
        return {
            'loss' : loss,
            'preds' : preds,
            'targets' : targets
        }
    
    def training_step(self, batch, batch_idx) :
        return self.step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx) :
        return self.step(batch, batch_idx, 'valid')

    def training_epoch_end(self, outputs, state='train') :
        train_loss = torch.tensor(0, dtype=torch.float)
        train_rouge1 = torch.tensor(0, dtype=torch.float)
        train_rouge2 = torch.tensor(0, dtype=torch.float)
        train_rougeL = torch.tensor(0, dtype=torch.float)
        total = 0
        for idx in range(len(outputs)) :
            for batch_item in range(len(outputs[idx]['preds'])) :
                pred = self.tokenizer.decode(outputs[idx]['preds'][batch_item])
                pred = re.sub(r"</s>[\w\W]*", "", pred)
                target = self.tokenizer.decode(outputs[idx]['targets'][batch_item], skip_special_tokens=True)

                # torchmetrics
                # rouge_scores = self.rouge(pred, target)
                # valid_rouge1 += rouge_scores['rouge1_fmeasure']
                # valid_rouge2 += rouge_scores['rouge2_fmeasure']
                # valid_rougeL += rouge_scores['rougeL_fmeasure']
                
                if pred == '' or target == '' :
                    total += 1
                    train_loss += outputs[idx]['loss'].cpu().detach()
                    continue
                rouge_scores = self.rouge.get_scores(pred, target)[0]
                train_rouge1 += rouge_scores['rouge-1']['f']
                train_rouge2 += rouge_scores['rouge-2']['f']
                train_rougeL += rouge_scores['rouge-l']['f']
                
                total += 1
            train_loss += outputs[idx]['loss'].cpu().detach()

        train_loss = train_loss / len(outputs)
        train_rouge1 = train_rouge1 / total
        train_rouge2 = train_rouge2 / total
        train_rougeL = train_rougeL / total

        # print(f'[Epoch {self.trainer.current_epoch} {state.upper()}] Loss:{train_loss}, ROUGE1: {train_rouge1}, ROUGE2: {train_rouge2}, ROUGEL: {train_rougeL}')
        self.log('TRAIN_LOSS', train_loss, on_epoch=True, prog_bar=True)
        self.log('TRAIN_ROUGE1', train_rouge1, on_epoch=True, prog_bar=True)
        self.log('TRAIN_ROUGE2', train_rouge2, on_epoch=True, prog_bar=True)
        self.log('TRAIN_ROUGEL', train_rougeL, on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, outputs, state='valid') :
        valid_loss = torch.tensor(0, dtype=torch.float)
        valid_rouge1 = torch.tensor(0, dtype=torch.float)
        valid_rouge2 = torch.tensor(0, dtype=torch.float)
        valid_rougeL = torch.tensor(0, dtype=torch.float)
        total = 0
        for idx in range(len(outputs)) :
            for batch_item in range(len(outputs[idx]['preds'])) :
                pred = self.tokenizer.decode(outputs[idx]['preds'][batch_item])
                pred = re.sub(r"</s>[\w\W]*", "", pred)
                pred = re.sub(r"<s>", "", pred)
                target = self.tokenizer.decode(outputs[idx]['targets'][batch_item], skip_special_tokens=True)

                # torchmetrics
                # rouge_scores = self.rouge(pred, target)
                # valid_rouge1 += rouge_scores['rouge1_fmeasure']
                # valid_rouge2 += rouge_scores['rouge2_fmeasure']
                # valid_rougeL += rouge_scores['rougeL_fmeasure']
                
                if pred == '' or target == '' :
                    total += 1
                    valid_loss += outputs[idx]['loss'].cpu().detach()
                    continue
                rouge_scores = self.rouge.get_scores(pred, target)[0]
                valid_rouge1 += rouge_scores['rouge-1']['f']
                valid_rouge2 += rouge_scores['rouge-2']['f']
                valid_rougeL += rouge_scores['rouge-l']['f']

                total += 1
            valid_loss += outputs[idx]['loss'].cpu().detach()
        valid_loss = valid_loss / len(outputs)
        valid_rouge1 = valid_rouge1 / total
        valid_rouge2 = valid_rouge2 / total
        valid_rougeL = valid_rougeL / total

        # save best by VAL_LOSS
        self.log('VAL_LOSS', valid_loss, on_epoch=True, prog_bar=True)
        self.log('VAL_ROUGE1', valid_rouge1, on_epoch=True, prog_bar=True)
        self.log('VAL_ROUGE2', valid_rouge2, on_epoch=True, prog_bar=True)
        self.log('VAL_ROUGEL', valid_rougeL, on_epoch=True, prog_bar=True)
        
    def configure_optimizers(self) :
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150)
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [lr_scheduler]

        