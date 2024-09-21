import json
import torch.nn.functional as F
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup, AdamW, BertModel
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from losses_global import SupConLoss

plt.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def data_process():
    with open('./skeleton.json') as f:
        data = json.load(f)
    
    sql_label = {}
    num = 0
    for i in data:
        if i['sql_s'] not in sql_label.keys():
            sql_label[i['sql_s'].lower()] = num
            num += 1
    # print(sql_label)
    print(num)
    print("SQL 模板总长度", len(sql_label))
    texts = [i['q_s'].lower() for i in data]
    sqls = [i['sql_s'].lower() for i in data]
    labels = [sql_label[i['sql_s'].lower()] for i in data]
    return texts, sqls, labels

def data_loader(data, sql, label, tokenizer):
    data_token = [tokenizer.tokenize(i) for i in data]
    sql_token = [tokenizer.tokenize(i) for i in sql]
    max_len = 0
    for i in data_token:
        max_len = max(max_len, len(i))
    for i in sql_token:
        max_len = max(max_len, len(i))
    # print(max_len)
    # print(data_token[:10])
    label_type = []
    for i in label:
        if i not in label_type:
            label_type.append(i)
    label_len = len(label_type)
    print("label type", label_len)
    max_len = max_len+2
    data_len = len(data)
    text_input = torch.zeros((data_len, max_len)).long()
    sql_input = torch.zeros((data_len, max_len)).long()
    mask_input = torch.zeros((data_len, max_len), dtype=torch.uint8)
    sql_mask_input = torch.zeros((data_len, max_len), dtype=torch.uint8)
    data_label = torch.zeros(data_len).long()
    sql_label = torch.zeros(data_len).long()
    label_index = torch.zeros(label_len).long()
    label_martix = torch.zeros((label_len, max_len)).long()
    label_mask = torch.zeros((label_len, max_len), dtype=torch.uint8)
    ll = []
    
    for i in range(data_len):
        if label[i] not in ll:
            ll.append(label[i])
            label_index[label[i]] = label[i]
            sql = tokenizer.convert_tokens_to_ids(['[CLS]'] + list(sql_token[i][:max_len-2]) + ['[SEP]'])
            # ss = torch.zeros(max_len).long()
            label_martix[label[i]][:len(sql)] = torch.tensor(sql)
            label_mask[label[i]][:len(sql)] = 1
        data_label[i] = label[i]
        text = tokenizer.convert_tokens_to_ids(['[CLS]'] + list(data_token[i][:max_len-2]) + ['[SEP]'])
        text_input[i][:len(text)] = torch.tensor(text)
        mask_input[i][:len(text)] = 1

    print(text_input.size(), mask_input.size(), data_label.size(), sql_input.size(), sql_mask_input.size(), sql_label.size())

    return TensorDataset(text_input, mask_input, data_label), label_index, label_martix, label_mask

class Word_BERT(nn.Module):
    def __init__(self, label_num=20):
        super(Word_BERT, self).__init__()
        self.bert = BertModel.from_pretrained(r'/root/data/bert-base-uncased')

    def forward(self, word_input, masks):
        output = self.bert(word_input, attention_mask=masks)
        pool = output.pooler_output

        return pool


def train(train_data, label_index, label_martix, label_mask, epochs = 40):
    train_batch_size = 64
    test_batch_size = 300
    train_iter = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
    # test_iter = DataLoader(test_data, shuffle=False, batch_size=test_batch_size)

    model = Word_BERT()
    model.to(torch.device('cuda:1'))

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    no_bert = ['bert']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if
                    ((not any(nd in n for nd in no_decay)) and any(nd in n for nd in no_bert))], 'weight_decay': 0.01,
         'lr': 4e-5},
        {'params': [p for n, p in param_optimizer if
                    ((any(nd in n for nd in no_decay)) and any(nd in n for nd in no_bert))], 'weight_decay': 0.0,
         'lr': 4e-5},
        {'params': [p for n, p in param_optimizer if
                    ((not any(nd in n for nd in no_decay)) and (not any(nd in n for nd in no_bert)))],
         'weight_decay': 0.01, 'lr': 1e-3},
        {'params': [p for n, p in param_optimizer if
                    ((any(nd in n for nd in no_decay)) and (not any(nd in n for nd in no_bert)))], 'weight_decay': 0.0,
         'lr': 1e-3}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, correct_bias=True)
    warm_ratio = 0.05
    print("train_batch_size", train_batch_size)
    print("graident_steps", 1)
    # print(len(train_data))
    total_steps = (len(train_data) // train_batch_size + 1) * epochs

    print("total_steps", total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_ratio * total_steps,
                                                num_training_steps=total_steps)
    model.train()
    cir = SupConLoss(mask_aa=train_batch_size)
    label_index, label_martix, label_mask = label_index.to(torch.device('cuda:1')), label_martix.to(torch.device('cuda:1')), label_mask.to(torch.device('cuda:1'))
    # ll = []
    # for i in label_index:
    #     if i.item() not in ll:
    #         ll.append(i.item())
    # print(len(ll))
    for epoch in range(epochs):
        loss_n = []
        for step, batch in enumerate(train_iter):
            text_input, mask_input, data_label = batch
            text_input, mask_input, data_label = text_input.to(torch.device('cuda:1')), mask_input.to(torch.device('cuda:1')), data_label.to(torch.device('cuda:1'))
            out = model(text_input, mask_input)
            hidden_states = out.unsqueeze(1)
            hidden_states = F.normalize(hidden_states, dim=-1)

            # with torch.no_grad():
            sql_out = model(label_martix, label_mask)
            # print(text_input[0], label_martix[0])
            sql_hidden_states = sql_out.unsqueeze(1)
            sql_hidden_states = F.normalize(sql_hidden_states, dim=-1)
            hidden = torch.cat([hidden_states, sql_hidden_states], dim=0)
            # print(hidden.size())
            # print(data_label, label_index)
            loss = cir(features=hidden, labels=torch.cat([data_label, label_index], dim=0), min_len=text_input.size()[0])/2
            print(loss.item())
            loss_n.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        print("loss", np.average(loss_n))
        torch.save(model.state_dict(), "./cl_model/model_global.pth")




if __name__=='__main__':
    data, sqls, labels = data_process()
    print(data[0], sqls[0], labels[0])
    tokenizer = AutoTokenizer.from_pretrained(r'./bert-base-uncased', use_fast=True)
    config = AutoConfig.from_pretrained(r'./bert-base-uncased')
    train_tensor, label_index, label_martix, label_mask = data_loader(data, sqls, labels, tokenizer)
    train(train_tensor, label_index, label_martix, label_mask, epochs = 15)