# Import modules
import random
import numpy as np
from itertools import combinations
# Import PyTorch
import torch
from torch import nn
from torch.nn import functional as F
# Import Huggingface
from transformers import BertForSequenceClassification, BertConfig

class kcBERT_custom(nn.Module):
    def __init__(self, num_labels=3, device=None):

        super(kcBERT_custom, self).__init__()

        # Hyper-parameter setting
        self.num_labels = num_labels
        self.device = device
        # Model Initiating
        bert_config = BertConfig.from_pretrained('beomi/kcbert-base', num_labels=num_labels)
        self.bert = BertForSequenceClassification.from_pretrained('beomi/kcbert-base', config=bert_config)
        # Split BERT embedding
        self.bert_embedding = self.bert.bert.embeddings
        for para in self.bert_embedding.parameters():
            para.requires_grad = False
        self.bert_embedding = self.bert_embedding.eval()
        # Split rest of BERT
        self.bert_encoder = self.bert.bert.encoder
        self.bert_pooler = self.bert.bert.pooler
        self.bert_dropout = self.bert.dropout
        self.bert_classifier = self.bert.classifier

    def forward(self, src_input_sentence, src_label=None):
        # Attention mask setting
        attention_mask = (src_input_sentence != 0)
        if not src_label==None:
            # Get BERT embedding values
            batch_size = src_input_sentence.shape[0]
            with torch.no_grad():
                if src_label.shape[-1]==2:
                    sen1, sen2 = torch.chunk(src_input_sentence, 2, dim=1)
                    embedding_pred_1 = self.bert_embedding(sen1)
                    embedding_pred_2 = self.bert_embedding(sen2)
                    label1, label2 = torch.chunk(src_label, 2, dim=-1)
                    label1 = label1.squeeze()
                    label2 = label2.squeeze()
                    processed_label_1 = torch.zeros(batch_size, 3).to(self.device)
                    processed_label_2 = torch.zeros(batch_size, 3).to(self.device)
                    processed_label_1[range(processed_label_1.shape[0]), label1]=1
                    processed_label_2[range(processed_label_2.shape[0]), label2]=1
                else:
                    embedding_pred = self.bert_embedding(src_input_sentence)
                    # Original label processing
                    processed_label = torch.zeros(batch_size, 3)
                    processed_label[range(processed_label.shape[0]), src_label]=1

            # Mixup label processing
            if src_label.shape[-1]==2:
                mix_lam = torch.tensor(np.random.beta(1, 1, size=(batch_size,1,1))).to(self.device)
                processed_label = mix_lam.mul(processed_label_1) + (1-mix_lam).mul(processed_label_2)
                # Attention mask setting
                mask1, mask2 = torch.chunk(attention_mask, 2, dim=1)
                attention_mask = torch.tensor([]).to(self.device)
                for i, j in zip(mask1, mask2):
                    if mask1.sum() > mask2.sum():
                        attention_mask = torch.cat([attention_mask, i.unsqueeze(0)])
                    else:
                        attention_mask = torch.cat([attention_mask, j.unsqueeze(0)])
                # Mixup word embedding
                with torch.no_grad():
                    emb1 = mix_lam.mul(embedding_pred_1)
                    emb2 = (1-mix_lam).mul(embedding_pred_2)
                    embedding_pred = emb1 + emb2
            # BERT process
            attention_mask = self.bert.get_extended_attention_mask(attention_mask, 
                                                                attention_mask.shape, self.device)
            out = self.bert_encoder(embedding_pred.float(), attention_mask=attention_mask)
            out = self.bert_classifier(self.bert_dropout(self.bert_pooler(out.last_hidden_state)))
            return out, processed_label
    
        else:
            with torch.no_grad():
                embedding_pred = self.bert_embedding(src_input_sentence)
            attention_mask = self.bert.get_extended_attention_mask(attention_mask, 
                                                                attention_mask.shape, self.device)
            out = self.bert_encoder(embedding_pred, attention_mask=attention_mask)
            out = self.bert_classifier(self.bert_dropout(self.bert_pooler(out.last_hidden_state)))
            return out
