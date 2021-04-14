import os
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.utils import clip_grad_norm_

import re
import emoji
from soynlp.normalizer import repeat_normalize

from sklearn.metrics import accuracy_score, f1_score


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
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
            padding='max_length',
            max_length=self.args.max_length,
            truncation=True,
        ))
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
valid_loader = make_dataloader(args).val_dataloader()
test_loader = make_dataloader(args).test_dataloader()

num_labels = 3
bert_config = BertConfig.from_pretrained('beomi/kcbert-base', num_labels=num_labels)
bert = BertForSequenceClassification.from_pretrained('beomi/kcbert-base', config=bert_config)

model = bert.to(DEVICE)

lr = 5e-6
optimizer = AdamW(model.parameters(), lr=lr)
scheduler = ExponentialLR(optimizer, gamma=0.5)

def train(model, scheduler, train_loader):
    model.train()
    total_loss = 0
    true = []
    pred = []
    for x, y in train_loader:
        optimizer.zero_grad()
        data, labels = x.to(DEVICE), y.to(DEVICE)
        output = model(input_ids=data, labels=labels)
        loss = output.loss
        logit = output.logits
        loss.backward()
        clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        _, predicted = logit.max(dim=1)
        pred.append(predicted.tolist())
        true.append(labels.tolist())
    size = len(train_loader.dataset)
    avg_loss = total_loss / size
    return avg_loss, true, pred

def evaluate(model, valid_loader):
    model.eval()
    total_loss = 0
    true = []
    pred = []
    for x, y in valid_loader:
        data, labels = x.to(DEVICE), y.to(DEVICE)
        output = model(input_ids=data, labels=labels)
        loss = output.loss
        logit = output.logits
        total_loss += loss.item()
        _, predicted = logit.max(dim=1)
        pred.append(predicted.tolist())
        true.append(labels.tolist())
    size = len(valid_loader.dataset)
    avg_loss = total_loss / size
    return avg_loss, true, pred

EPOCHS = 10
best_val_loss = None
for e in range(1, EPOCHS+1):
    start_time_e = time.time()
    print(f'Model Fitting: [{e}/{EPOCHS}]')
    train_loss, t_true, t_pred = train(model, optimizer, train_loader)
    val_loss, v_true, v_pred = evaluate(model, valid_loader)
    t_true = [j for i in t_true for j in i]
    t_pred = [j for i in t_pred for j in i]
    v_true = [j for i in v_true for j in i]
    v_pred = [j for i in v_pred for j in i]

    train_accuracy = accuracy_score(t_true, t_pred)
    val_accuracy = accuracy_score(v_true, v_pred)
    val_f1 = f1_score(v_true, v_pred, average='macro')

    print("[Epoch: %d] train loss : %5.2f | train accuracy : %5.2f | val loss : %5.2f | val accuracy : %5.2f | val f1 score : %5.2f" % (
        e, train_loss, train_accuracy, val_loss, val_accuracy, val_f1))
    print(f'Spend Time: [{(time.time() - start_time_e)/60}]')

    # 검증 오차가 가장 적은 최적의 모델을 저장
    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("/home/jangsj/korean-hate-speech-detection/KcBERT/snapshot"):
            os.makedirs("/home/jangsj/korean-hate-speech-detection/KcBERT/snapshot")
        torch.save(model.state_dict(), '/home/jangsj/korean-hate-speech-detection/KcBERT/snapshot/txtclassification.pt')
        print('[save model]')
        best_val_loss = val_loss


model = bert.to(DEVICE)
model.load_state_dict(torch.load('/home/jangsj/korean-hate-speech-detection/KcBERT/snapshot/txtclassification.pt'))
model.eval()
pred = []
for _, x in enumerate(test_loader):
    data = x[0].to(DEVICE)
    output = model(input_ids=data)
    logit = output.logits
    _, predicted = logit.max(dim=1)
    pred.append(predicted.tolist())

prediction = [j for i in pred for j in i]

test = pd.read_csv("/HDD/dataset/korean-hate-speech-detection/test.hate.no_label.csv", sep='\t')

submission = pd.DataFrame({
        "comments": test["comments"],
        "label": prediction
    })
submission.to_csv('/home/jangsj/korean-hate-speech-detection/KcBERT/submission.csv',index=False)
