import evaluate
import math
import numpy as np
import os
import re
import torch

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from transformers import BartForConditionalGeneration

from utils.scheduler import LinearWarmupLR

class PatentPretrain(pl.LightningModule) :
    def __init__(self, args, device, tokenizer) :
        super().__init__()
        self._device = device
        self.ckpt_dir = args.ckpt_dir
        self.num_warmup_steps = args.num_warmup_steps

        self.batch_size = args.batch_size

        self.min_learning_rate = args.min_learning_rate
        self.max_learning_rate = args.max_learning_rate
        self.num_training_steps = math.ceil((args.num_data - args.skiprows) / self.batch_size)
        
        self.tokenizer = tokenizer
        self.model = BartForConditionalGeneration.from_pretrained(args.model)

        self.metric = evaluate.load("sacrebleu")

        self.save_hyperparameters(
            {
                **self.model.config.to_dict(),
                "total_steps" : self.num_training_steps,
                "max_learning_rate": args.max_learning_rate,
                "min_learning_rate": args.min_learning_rate,
            }
        )

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
            encoder_input_ids=batch['input_ids'].to(self._device),
            encoder_attention_mask=batch['attention_mask'].to(self._device),
            decoder_input_ids=batch['decoder_input_ids'].to(self._device),
            decoder_attention_mask=batch['decoder_input_ids'].to(self._device),
            labels=batch['labels'].to(self._device)
        )

        loss = outputs.loss
        logits = outputs.logits

        preds = np.array(torch.argmax(logits.cpu(), dim=2))
        targets = np.array(batch['labels'].cpu())
        targets = np.where(targets != -100, targets, self.tokenizer.pad_token_id) # -100 to pad tokens

        decoded_preds = [re.sub(r"</s>[\w\W]*", "", pred) for pred in self.tokenizer.batch_decode(preds)]
        decoded_targets = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        bleu = self.metric.compute(predictions=decoded_preds, references=decoded_targets)['score']

        self.log(f"{state.upper()}_STEP_LOSS", loss, prog_bar=True)
        self.log(f"{state.upper()}_STEP_BLEU", bleu, prog_bar=True)

        return {
            'loss' : loss,
            'bleu' : torch.tensor(bleu)
        }
    
    def training_step(self, batch, batch_idx) :
        return self.step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx) :
        return self.step(batch, batch_idx, 'valid')
    
    def epoch_end(self, outputs, state) :
        loss = torch.stack([output['loss'] for output in outputs]).mean()
        bleu = torch.stack([output['bleu'] for output in outputs]).mean()

        self.log(f'{state.upper()}_LOSS', loss, on_epoch=True, prog_bar=True)
        self.log(f'{state.upper()}_BLEU', bleu, on_epoch=True, prog_bar=True)

        return (loss, bleu)
        
    
    def training_epoch_end(self, outputs, state='train') :
        self.epoch_end(outputs, state)

    def validation_epoch_end(self, outputs, state='valid') :
        loss, bleu = self.epoch_end(outputs, state)
        self.model.save_pretrained(
            os.path.join(
                self.ckpt_dir,
                f"model-{self.current_epoch:02d}epoch-{self.global_step}steps-{loss:.4f}loss-{bleu:.4f}bleu",
            ),
        )
    
    def configure_optimizers(self) :
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.max_learning_rate)
        lr_scheduler = LinearWarmupLR(
            optimizer, 
            self.num_warmup_steps, 
            self.num_training_steps, 
            self.min_learning_rate / self.max_learning_rate,
        )
        return {
            'optimizer' : optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step", "name": "Learning Rate"}
        }