from cProfile import label
from typing import Optional
import os
import pandas as pd

import torch 
from torch.utils.data import DataLoader

import pytorch_lightning as pl 

from transformers import AutoTokenizer
from datasets import load_dataset

from multiconer.config import (
    PATH_BASE_MODELS, 
    WNUT_BIO_ID
)


def get_file_paths(dataPath):
    files = os.listdir(dataPath)

    filePaths = []
    for file in files:
        if file.endswith(".conll"):
            filePaths.append(file)
    
    dev_data_file = f"{dataPath}/{filePaths[0]}"
    train_data_file = f"{dataPath}/{filePaths[1]}"
    
    return train_data_file, dev_data_file

def process_data_file(filePath):
    with open(filePath, 'r') as data_file:
        data = data_file.readlines()
        
        data_dict = {
            'sentence_id': [],
            'sentence': [],
            'bio_tags': []
        }

        sentence_id = ""
        
        token_list = []
        bio_tag_list = []
        flag = 0

        for line in data:
            if line == "\n":
                continue

            if line[0:4] == "# id":

                if flag == 1:
                    data_dict['sentence_id'].append(sentence_id)
                    data_dict['sentence'].append(" ".join(token_list))
                    data_dict['bio_tags'].append(",".join(bio_tag_list))

                sentence_id = line[5:41]
                flag = 1
                token_list = []
                bio_tag_list = []

            else:
                token_list.append(line.split(" ")[0])
                bio_tag_list.append(line.split(" ")[-1][:-1])
        
        data_dict['sentence_id'].append(sentence_id)
        data_dict['sentence'].append(" ".join(token_list))
        data_dict['bio_tags'].append(",".join(bio_tag_list))

        df = pd.DataFrame(data_dict)
        df.to_csv(f"{filePath}.csv", index=False)

def process_data(dataPath):
    train_data_file, dev_data_file = get_file_paths(dataPath)
    process_data_file(train_data_file)
    process_data_file(dev_data_file)



class CoNLLDataModule(pl.LightningDataModule):

    columns = [
        "input_ids", 
        "attention_mask", 
        "labels"
    ]

    def __init__(
        self, 
        model_name_or_path: str,
        path_dataset: str, 
        num_workers: int,
        dataset_name: str = None,
        tag_to_id: dict = WNUT_BIO_ID,
        max_seq_length: int = 32,
        padding: str = "max_length",
        batch_size: int = 32, 
        
    ) -> None:

        super().__init__()
        self.path_dataset = path_dataset
        self.model_name_or_path = model_name_or_path
        self.dataset_name = dataset_name
        self.tag_to_id = tag_to_id
        self.max_seq_length = max_seq_length
        self.padding = padding
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            use_fast=True,
            cache_dir=PATH_BASE_MODELS
        )
        if dataset_name == 'twitter':
            self.data_map = {
                'train': [f"{self.path_dataset}/twitter_train.csv"],
                'validation': [f"{self.path_dataset}/twitter_val.csv"]
            }
        else:
            train_file, val_file = get_file_paths(self.path_dataset)
            self.data_map = {
                'train': [f"{train_file}.csv"],
                'validation': [f"{val_file}.csv"]
            }
    
    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset_name == 'twitter':
            self.dataset = load_dataset("csv", data_files=self.data_map) 
        else:
            process_data(self.path_dataset)
            self.dataset = load_dataset("csv", data_files=self.data_map)

        self.dataset = self.dataset.map(
            self.convert_to_features,
            num_proc=self.num_workers
        )

        self.dataset.set_format(type="torch", columns=self.columns)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers, 
            drop_last=True 
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True
        )
    
    def convert_to_features(self, batch, indices=None):
        features = {}

        features = self.tokenizer(
            batch['sentence'],
            max_length=self.max_seq_length,
            padding=self.padding, 
            truncation=True 
        )

        features['tokens'] = batch['sentence'].split()

        labels = batch['bio_tags'].split(",")
        label_ids = [self.tag_to_id[label] for label in labels]

        if len(label_ids) <= self.max_seq_length:
            features['labels'] = label_ids + [0] * (self.max_seq_length - len(label_ids))
        else:
            features['labels'] = label_ids[:self.max_seq_length]
        
        return features

