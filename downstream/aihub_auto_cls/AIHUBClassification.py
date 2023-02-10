import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_looggers
from kobart import get_pytorch_kobart_model
from transformers import AutoModel, AutoModelForSequenceClassification, AdamW
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput, SequenceClassifierOutput

class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        input_dim = config.d_model
        inner_dim = config.d_model
        num_classes = config.num_labels,
        pooler_dropout = config.classifier_dropout
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class ElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.gelu = nn.GELU()

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.gelu(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class AIHUBClassification(pl.LightningModule) :
    def __init__(self, args, device, labels) :
        super().__init__()
        self._device = device

        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size

        num_classes = len(labels[args.cls].keys())
        self.model = AutoModel.from_pretrained(args.model)
        self.config = self.model.config
        self.config.num_labels = num_classes
        self.is_kobart = args.is_kobart
        if self.is_kobart :
            self.classifier = BartClassificationHead(self.config)
        else :
            self.classifier = ElectraClassificationHead(self.config)

        if args.loss_func == 'CE' :
            self.loss_func = nn.CrossEntropyLoss()
        elif args.loss_func == 'DICE' :
            pass

        self.softmax = nn.Softmax(dim=1)
        self.f1 = MulticlassF1Score(num_classes=num_classes, average='weighted')
        self.acc = MulticlassAccuracy(num_classes=num_classes, average='weighted')

        
    def forward(self, input_ids, attention_mask, labels=None) :
        if self.is_kobart :
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden_states = outputs[0]  # last hidden state

            eos_mask = input_ids.eq(self.config.eos_token_id).to(hidden_states.device)

            if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
                :, -1, :
            ]
            logits = self.classification_head(sentence_representation)

            loss = self.loss_func(logits.view(-1, self.config.num_labels), labels.view(-1))

            return Seq2SeqSequenceClassifierOutput(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )
        
        else : # electra
            discriminator_hidden_states = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            sequence_output = discriminator_hidden_states[0]
            logits = self.classifier(sequence_output)
            loss = self.loss_func(logits.view(-1, self.config.num_labels), labels.view(-1))

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=discriminator_hidden_states.hidden_states,
                attentions=discriminator_hidden_states.attentions,
            )
            
    def step(self, batch, batch_idx, state) :
        outputs = self(
            input_ids=batch['input_ids'].to(self._device),
            attention_mask=batch['attention_mask'].to(self._device),
            labels=batch['labels'].to(self._device)
        )

        loss = outputs.loss
        logits = outputs.logits

        self.log(f"[{state.upper()} LOSS]", loss, prog_bar=True)

        targets = batch['labels'].to(self._device)
        predicts = self.softmax(logits).argmax(dim=1).to(self._device)
        
        return {
            'loss' : loss,
            'preds' : predicts,
            'targets' : targets
        }

    def training_step(self, batch, batch_idx) :
        return self.step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx) :
        return self.step(batch, batch_idx, 'valid')

    def training_epoch_end(self, outputs, state='train') :
        train_loss = torch.tensor(0, dtype=torch.float)
        train_f1 = torch.tensor(0, dtype=torch.float)
        train_acc = torch.tensor(0, dtype=torch.float)
        for idx in range(len(outputs)) :
            train_loss += outputs[idx]['loss'].cpu().detach()
            train_preds = outputs[idx]['preds']
            train_targets = outputs[idx]['targets']

            train_f1 += self.f1(train_preds, train_targets).cpu().detach()
            train_acc += self.acc(train_preds, train_targets).cpu().detach()
        train_loss /= len(outputs)
        train_f1 /= len(outputs)
        train_acc /= len(outputs)

        print(f'[Epoch {self.trainer.current_epoch} {state.upper()}] Loss:{train_loss}, F1:{train_f1}, ACC:{train_acc}')
        self.log('TRAIN_LOSS', train_loss, on_epoch=True, prog_bar=True)
        self.log('TRAIN_F1', train_f1, on_epoch=True, prog_bar=True)
        self.log('TRAIN_ACC', train_acc, on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, outputs, state='valid') :
        valid_loss = torch.tensor(0, dtype=torch.float)
        valid_f1 = torch.tensor(0, dtype=torch.float)
        valid_acc = torch.tensor(0, dtype=torch.float)
        
        for idx in range(len(outputs)) :
            valid_loss += outputs[idx]['loss'].cpu().detach()
            valid_preds = outputs[idx]['preds']
            valid_targets = outputs[idx]['targets']

            valid_f1 += self.f1(valid_preds, valid_targets).cpu().detach()
            valid_acc += self.acc(valid_preds, valid_targets).cpu().detach()

        valid_loss /= len(outputs)
        valid_f1 /= len(outputs)
        valid_acc /= len(outputs)

        self.log('VAL_LOSS', valid_loss, on_epoch=True, prog_bar=True)
        self.log('VAL_F1', valid_f1, on_epoch=True, prog_bar=True)
        self.log('VAL_ACC', valid_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) :
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150)
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [lr_scheduler]