from unicodedata import bidirectional
import torch 
import torch.nn as nn 

from torchmetrics.functional import accuracy, precision, recall, f1_score

import pytorch_lightning as pl 

from allennlp.modules import ConditionalRandomField 
from allennlp.modules.conditional_random_field import allowed_transitions 

from transformers import AutoConfig, AutoModel, AdamW

from multiconer.metrics import SequenceAccuracy
from multiconer.config import (
    PATH_BASE_MODELS,
    WNUT_BIO_ID, 
    BIO_TAGS
)


class MultiCoNER(pl.LightningModule):

    def __init__(
        self, 
        model_name_or_path: str, 
        tag_to_id: dict = WNUT_BIO_ID, 
        lstm_hidden_size: int = 64, 
        learning_rate: float = 3e-5, 
        weight_decay: float = 0, 
        dropout_rate: float = 1e-3,
        max_seq_len: int = 32,
        padding: str = "max_length"
    ) -> None:

        super().__init__()
        self.model_name_or_path = model_name_or_path 
        self.learning_rate = learning_rate 
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.tag_to_id = tag_to_id 
        self.id_to_tag = {val:key for key, val in tag_to_id.items()}
        self.target_size = len(tag_to_id)
        self.lstm_hidden_size = lstm_hidden_size
        self.max_seq_length = max_seq_len
        self.padding = padding
        self.accuracy = SequenceAccuracy()

        self.config = AutoConfig.from_pretrained(
            self.model_name_or_path, 
        )

        self.bert_model = AutoModel.from_pretrained(
            self.model_name_or_path, 
            config=self.config, 
            cache_dir=PATH_BASE_MODELS
        )

        self.dropout = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.ReLU()
        )

        self.bi_lstm = nn.LSTM(
            input_size=self.bert_model.config.hidden_size,
            hidden_size=self.lstm_hidden_size,
            bidirectional=True
        )
        
        self.net = nn.Linear(self.lstm_hidden_size * 2, self.target_size)

        self.crf_layer = ConditionalRandomField(
            num_tags=self.target_size, 
            constraints=allowed_transitions(
                constraint_type='BIO', 
                labels=self.id_to_tag
            )
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        bert_outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        outs = self.dropout(bert_outputs.last_hidden_state)
        outs = self.bi_lstm(outs)

        outs = self.net(outs[0])

        return outs
    
    def training_step(self, batch, batch_idx):
        batch_size = batch['input_ids'].size(0)
        
        outs = self(batch['input_ids'], batch['attention_mask'])
        labels = batch['labels']

        stats = self._compute_token_tags(outs, labels, batch['attention_mask'], batch_size)
        self.log("loss/train", stats["loss"])
        self.log("acc/train", stats["acc"])
        self.log("prec/train", stats["prec"])
        self.log("rec/train", stats["rec"])
        self.log("f1_score/train", stats["f1_score"])

        # YET TO COMPLETE  

        return stats
    
    def validation_step(self, batch, batch_idx):
        batch_size = batch['input_ids'].size(0)
        
        outs = self(batch['input_ids'], batch['attention_mask'])
        labels = batch['labels']

        stats = self._compute_token_tags(outs, labels, batch['attention_mask'], batch_size)
        self.log("loss/val", stats["loss"])
        self.log("acc/val", stats["acc"])
        self.log("prec/val", stats["prec"])
        self.log("rec/val", stats["rec"])
        self.log("f1_score/val", stats["f1_score"])

        # YET TO COMPLETE  
        return stats
    
    def test_step(self, batch, batch_idx):
        pass 

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )

        return optimizer
    
    def _compute_token_tags(self, outs, tags, attention_mask, batch_size):

        loss = - self.crf_layer(outs, tags, attention_mask) / float(batch_size)
        best_path = self.crf_layer.viterbi_tags(outs, attention_mask)

        batch_tag_seq = best_path[0][0]
        final_labels = tags[0][:attention_mask[0].sum()]
        for i in range(1, len(best_path)):
            tag_seq = best_path[i][0] 
            batch_tag_seq.extend(tag_seq)

            labels = tags[i][:attention_mask[i].sum()]
            final_labels = torch.cat((final_labels, labels))

        batch_tag_seq = torch.Tensor(batch_tag_seq)
        batch_tag_seq = torch.Tensor(batch_tag_seq).to("cuda")
        
        # acc = self.accuracy(batch_tag_seq, final_labels)
        acc = accuracy(batch_tag_seq.int(), final_labels.int(), num_classes=self.target_size)
        prec = precision(batch_tag_seq.int(), final_labels.int(), average="macro", num_classes=self.target_size)
        rec = recall(batch_tag_seq.int(), final_labels.int(), average="macro", num_classes=self.target_size)
        f1_val = f1_score(batch_tag_seq.int(), final_labels.int(), average="macro", num_classes=self.target_size)

        return {'loss': loss, 'acc': acc, 'prec': prec, 'rec': rec, 'f1_score': f1_val} 
    
