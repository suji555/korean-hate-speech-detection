from score import BERTScore
import numpy as np
import os
import time
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.utils import clip_grad_norm_
from mixup_model import kcBERT_custom
from torch.nn import functional as F

import re
import emoji
from soynlp.normalizer import repeat_normalize

from sklearn.metrics import accuracy_score, f1_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:2" if USE_CUDA else "cpu")
# print("cpu와 cuda 중 다음 기기로 학습함:", DEVICE)    # cuda

class Arg:
    batch_size: int = 16  # Optional, Train/Eval Batch Size. Overrides `auto_batch_size` 
    max_length: int = 150  # Max Length input size
    train_data_path: str = "/HDD/dataset/korean-hate-speech-detection/train.hate.csv"  # Train Dataset file 
    val_data_path: str = "/HDD/dataset/korean-hate-speech-detection/dev.hate.csv"  # Validation Dataset file 
    test_data_path: str = "/HDD/dataset/korean-hate-speech-detection/test.hate.no_label.csv"
    cpu_workers: int = os.cpu_count()  # Multi cpu workers

args = Arg()

class make_dataloader:
    def __init__(self, options):
        super().__init__()
        self.args = options
        self.tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-large')

    def read_data(self, path):
        if path.endswith('xlsx'):
            return pd.read_excel(path)
        elif path.endswith('csv'):
            return pd.read_csv(path)
        elif path.endswith('tsv') or path.endswith('txt'):
            return pd.read_csv(path, sep='\t')
        else:
            raise NotImplementedError('Only Excel(xlsx)/Csv/Tsv(txt) are Supported')

    def preprocess_dataframe(self, df):
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

        def clean(x):
            x = pattern.sub(' ', x)
            x = url_pattern.sub('', x)
            x = x.strip()
            x = repeat_normalize(x, num_repeats=2)
            return x

        df['comments'] = df['comments'].map(lambda x: self.tokenizer.encode(
            clean(str(x)),
            pad_to_max_length=True,
            max_length=self.args.max_length,
            truncation=True,
        ))
        return df
    

    def mix_preprocess_dataframe(self, df):
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

        def clean(x):
            x = pattern.sub(' ', x)
            x = url_pattern.sub('', x)
            x = x.strip()
            x = repeat_normalize(x, num_repeats=2)
            return x

        df['comments'] = df['comments'].map(lambda x: clean(str(x)))

        label = df['label'].replace(['none', 'offensive', 'hate'],[0,1,2]).to_list()
        comments = df['comments'].to_list()
        candidates = df['comments'].to_list()
        mix_comments = []
        mix_label = []

        for i in range(len(comments)//2):
            sent1, sent2 = random.sample(candidates, 2)
            candidates.remove(sent1)
            candidates.remove(sent2)
            mix_comments.append(list(map(lambda x: self.tokenizer.encode(
                    x,
                    pad_to_max_length=True,
                    max_length=self.args.max_length,
                    truncation=True,
                ), [sent1, sent2])))
            label1 = label[comments.index(sent1)]
            label2 = label[comments.index(sent2)]
            mix_label.append([label1, label2])

        submission = pd.DataFrame({
                "comments": mix_comments,
                "label": mix_label
            })
        submission.to_csv('/HDD/jangsj/korean-hate-speech-detection/mix_random.csv',index=False)
        
        return df
    

    def train_dataloader(self):
        df = self.read_data(self.args.train_data_path)
        df = self.preprocess_dataframe(df)
        
        dataset = TensorDataset(
            torch.tensor(df['comments'].to_list(), dtype=torch.long),
            torch.tensor(df['label'].replace(['none', 'offensive', 'hate'],[0,1,2]).to_list(), dtype=torch.long),
            )

        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.cpu_workers,
        )
    
    def mix_train_dataloader(self):
        df = self.read_data(self.args.train_data_path)
        df = self.mix_preprocess_dataframe(df)
        df = self.read_data('/HDD/jangsj/korean-hate-speech-detection/mix_random.csv')
        p = re.compile('[0-9]+')
        comment = []
        for i in df['comments']:
            sents = p.findall(i)
            sents = [int(i) for i in sents]
            comment.append(sents)
        
        p = re.compile('[0-9]+')
        labels = []
        for i in df['label']:
            label = p.findall(i)
            label = [int(i) for i in label]
            labels.append(label)

        df['comments'] = comment
        df['label'] = labels

        dataset = TensorDataset(
            torch.tensor(df['comments'].to_list(), dtype=torch.long),
            torch.tensor(df['label'].to_list(), dtype=torch.long),
            )

        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.cpu_workers,
        )
    

    def val_dataloader(self):
        df = self.read_data(self.args.val_data_path)
        df = self.preprocess_dataframe(df)

        dataset = TensorDataset(
            torch.tensor(df['comments'].to_list(), dtype=torch.long),
            torch.tensor(df['label'].replace(['none', 'offensive', 'hate'],[0,1,2]).to_list(), dtype=torch.long),
        )

        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.cpu_workers,
        )
    
    def test_dataloader(self):
        df = self.read_data(self.args.test_data_path)
        df = self.preprocess_dataframe(df)

        dataset = TensorDataset(
            torch.tensor(df['comments'].to_list(), dtype=torch.long),
        )

        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.cpu_workers,
        )

train_loader = make_dataloader(args).train_dataloader()
mix_train_loader = make_dataloader(args).mix_train_dataloader()
valid_loader = make_dataloader(args).val_dataloader()
test_loader = make_dataloader(args).test_dataloader()

model = kcBERT_custom(num_labels = 3, device=DEVICE)
model = model.to(DEVICE)

lr = 5e-6
optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
scheduler = ExponentialLR(optimizer, gamma=0.5)


def train(model, scheduler, train_loader, mix_train_loader):
    model.train()
    total_loss = 0
    i = 0
    for x, y in train_loader:
        i += 1
        optimizer.zero_grad()
        comment, labels = x.to(DEVICE), y.to(DEVICE)
        pred, new_label = model(src_input_sentence=comment, src_label=labels)
        pred = pred.log_softmax(dim=-1)
        loss = torch.mean(torch.sum(-new_label.to(DEVICE) * pred, dim=-1))
        loss.backward()
        clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        scheduler.step()
        total_loss = loss.item()
        
        if i % 2 == 0:
            x, y = next(iter(mix_train_loader))
            optimizer.zero_grad()
            comment, labels = x.to(DEVICE), y.to(DEVICE)
            pred, new_label = model(src_input_sentence=comment, src_label=labels)
            pred = pred.log_softmax(dim=-1)
            loss = torch.mean(torch.sum(-new_label.to(DEVICE) * pred, dim=-1))
            loss.backward()
            clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            scheduler.step()
            total_loss = loss.item()

    return total_loss


def evaluate(model, valid_loader):
    model.eval()
    val_loss = 0
    val_f1 = 0
    i = 0
    for x, y in valid_loader:
        comment, labels = x.to(DEVICE), y.to(DEVICE)
        pred, _ = model(src_input_sentence=comment, src_label=labels)
        loss = F.cross_entropy(pred, labels)
        val_loss += loss.item()
        val_f1 += f1_score(pred.max(dim=1)[1].tolist(), labels.tolist(), average='macro')
        i += 1
    val_loss /= i
    val_f1 /= i
    return val_loss, val_f1

EPOCHS = 10
best_val_f1 = None
for e in range(1, EPOCHS+1):
    start_time_e = time.time()
    print(f'Model Fitting: [{e}/{EPOCHS}]')
    train_loss = train(model, optimizer, train_loader, mix_train_loader)
    val_loss, val_f1 = evaluate(model, valid_loader)

    print("[Epoch: %d] train loss : %5.2f | val loss : %5.2f | val f1 score : %5.2f" % (e, train_loss, val_loss, val_f1))
    print(f'Spend Time: [{(time.time() - start_time_e)/60}]')

    # 검증 오차가 가장 적은 최적의 모델을 저장
    if not best_val_f1 or val_f1 > best_val_f1:
        if not os.path.isdir("/home/jangsj/KoBERTScore/KoBERTScore/snapshot"):
            os.makedirs("/home/jangsj/KoBERTScore/KoBERTScore/snapshot")
        torch.save(model.state_dict(), '/home/jangsj/KoBERTScore/KoBERTScore/snapshot/txtclassificationrandom.pt')
        print('[save model]')
        best_val_f1 = val_f1
