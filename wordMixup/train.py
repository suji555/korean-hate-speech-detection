import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.legacy.data import Field
from torchtext.legacy.data import TabularDataset
from torchtext.legacy.data import Iterator
from konlpy.tag import Mecab
from models import Transformer
import torch.optim as optim
import os
from sklearn.metrics import f1_score


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
# print("cpu와 cuda 중 다음 기기로 학습함:", DEVICE)    # cuda

train_data = pd.read_csv('/HDD/jangsj/korean-hate-speech-detection/train.csv')
valid_data = pd.read_csv('/HDD/jangsj/korean-hate-speech-detection/valid.csv')

# print(train_data.head())
# print(valid_data.head())

# print('훈련 샘플의 개수 : {}'.format(len(train_data)))    # 7896
# print('테스트 샘플의 개수 : {}'.format(len(valid_data)))  # 471

tokenizer = Mecab()

TEXT = Field(sequential=True,
             use_vocab=True,
             tokenize=tokenizer.morphs,
             init_token = '<sos>',
             eos_token = '<eos>',
             lower=True,
             batch_first=True,
             fix_length=20)

LABEL = Field(sequential=False,
              use_vocab=False,
              is_target=True)

train_data, valid_data = TabularDataset.splits(path='/HDD/jangsj/korean-hate-speech-detection/', 
                            train='train.csv',
                            test='valid.csv',
                            format='csv', 
                            fields=[('comments', TEXT), ('label', LABEL)],
                            skip_header=True)

# print(vars(train_data[0]))
# {'comments': ['현재', '호텔', '주인', '심정', '아', '18', '난', '마른', '하늘', '에', '날벼락', '맞', '고', '호텔', '망하', '게', '생겼', '는데', '누군', '계속', '추모', '받', '네'], 'label': '2'}

TEXT.build_vocab(train_data, min_freq=1, max_size=10000)

# print('단어 집합의 크기 : {}'.format(len(TEXT.vocab)))  # 10004
# print(TEXT.vocab.stoi)

batch_size = 2
train_loader = Iterator(dataset=train_data, 
                        batch_size = batch_size,
                        device=DEVICE)
valid_loader = Iterator(dataset=valid_data, 
                        batch_size = batch_size,
                        device=DEVICE)

# print('훈련 데이터의 미니 배치 수 : {}'.format(len(train_loader)))      # 3948
# print('테스트 데이터의 미니 배치 수 : {}'.format(len(valid_loader)))    # 236

# for batch in train_loader:
#     print(batch.comments)
#     # tensor([[   2, 3376,  214,  151,   16, 2599,  332,  898,   16,    3,    1,    1,
#     #             1,    1,    1,    1,    1,    1,    1,    1],
#     #         [   2, 1950, 1055,  570,  118,    3,    1,    1,    1,    1,    1,    1,
#     #             1,    1,    1,    1,    1,    1,    1,    1]])
#     print(batch.label)  # tensor([0, 1])
#     break

n_vocab = len(TEXT.vocab)
n_trg = 3
pad_idx = 1
bos_idx = 2
eos_idx = 3
unk_idx = 0
alpha = 0.2

model = Transformer(
            n_vocab, n_trg, pad_idx, alpha,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=12, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            emb_prj_weight_sharing=True,
            scale_emb_or_prj='prj').to(DEVICE)

lr = 5e-4
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

def train(model, optimizer, train_loader):
    model.train()
    corrects, total_loss = 0, 0
    for b, batch in enumerate(train_loader):
        x, y = batch.comments.to(DEVICE), batch.label.to(DEVICE)
        optimizer.zero_grad()
        logit, beta = model(x)
        y_onehot = F.one_hot(y, num_classes=3).type_as(logit).to(DEVICE)
        if len(y_onehot)==2:
          y_onehot = torch.cat([y_onehot[0].unsqueeze(0), y_onehot[1].unsqueeze(0), (y_onehot[0].mul(beta) + y_onehot[1].mul(1-beta)).unsqueeze(0)], dim=0).to(DEVICE)
        loss = nn.BCEWithLogitsLoss()(logit, y_onehot)
        total_loss += loss.item()
        corrects += (logit.argmax(1).view(y_onehot.argmax(1).size()).data == y_onehot.argmax(1).data).sum()
        loss.backward()
        optimizer.step()
    size = len(train_loader.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy


def evaluate(model, valid_loader):
    """evaluate model"""
    model.eval()
    corrects, total_loss = 0, 0
    pred = []
    true = []
    for batch in valid_loader:
        x, y = batch.comments.to(DEVICE), batch.label.to(DEVICE)
        logit, beta = model(x)
        if logit.shape == (3, 3):
          logit = logit[:-1]
        y_onehot = F.one_hot(y, num_classes=3).type_as(logit).to(DEVICE)
        loss = nn.BCEWithLogitsLoss()(logit, y_onehot)
        total_loss += loss.item()
        _, predicted = logit.max(dim=1)
        pred.append(predicted.tolist())
        true.append(y.tolist())
        corrects += (logit.argmax(1).view(y.size()).data == y.data).sum()
    size = len(valid_loader.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy, pred, true

EPOCHS = 10
best_val_loss = None
for e in range(1, EPOCHS+1):
    train_loss, train_accuracy = train(model, optimizer, train_loader)
    val_loss, val_accuracy, pred, true = evaluate(model, valid_loader)
    y_pred = [j for i in pred for j in i]
    y_true = [j for i in true for j in i]
    f1 = f1_score(y_true, y_pred, average='macro')

    print("[Epoch: %d] train loss : %5.2f | train accuracy : %5.2f | val loss : %5.2f | val accuracy : %5.2f | f1 score : %5.2f" % (e, train_loss, train_accuracy, val_loss, val_accuracy, f1))

    # 검증 오차가 가장 적은 최적의 모델을 저장
    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("/home/jangsj/korean-hate-speech-detection/wordMixup/snapshot"):
            os.makedirs("/home/jangsj/korean-hate-speech-detection/wordMixup/snapshot")
        torch.save(model.state_dict(), '/home/jangsj/korean-hate-speech-detection/wordMixup/snapshot/txtclassification.pt')
        print('save model')
        best_val_loss = val_loss
