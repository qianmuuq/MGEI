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
from sentence_transformers import SentenceTransformer, util

plt.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def data_process():
    with open('./docs_pred.json') as f:
        data = json.load(f)
    
    sql_label = {}
    num = 0
    for i in data:
        if i['sql_s'] not in sql_label.keys():
            sql_label[i['sql_s'].lower()] = num
            num += 1
    # print(sql_label)
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
    max_len = max_len+2
    data_len = len(data)
    text_input = torch.zeros((data_len, max_len)).long()
    sql_input = torch.zeros((data_len, max_len)).long()
    mask_input = torch.zeros((data_len, max_len), dtype=torch.uint8)
    sql_mask_input = torch.zeros((data_len, max_len), dtype=torch.uint8)
    data_label = torch.zeros(data_len).long()
    sql_label = torch.zeros(data_len).long()
    
    for i in range(data_len):
        # print(data[i][:254])
        data_label[i] = label[i]
        sql_label[i] = label[i]
        text = tokenizer.convert_tokens_to_ids(['[CLS]'] + list(data_token[i][:max_len-2]) + ['[SEP]'])
        sql = tokenizer.convert_tokens_to_ids(['[CLS]'] + list(sql_token[i][:max_len-2]) + ['[SEP]'])
        text_input[i][:len(text)] = torch.tensor(text)
        sql_input[i][:len(sql)] = torch.tensor(sql)
        mask_input[i][:len(text)] = 1
        sql_mask_input[i][:len(sql)] = 1
    # print(text_input[0])
    print(text_input.size(), mask_input.size(), data_label.size(), sql_input.size(), sql_mask_input.size(), sql_label.size())
    return TensorDataset(text_input, mask_input, data_label, sql_input, sql_mask_input, sql_label)

class Word_BERT(nn.Module):
    def __init__(self):
        super(Word_BERT, self).__init__()
        self.bert = BertModel.from_pretrained(r'./bert-base-uncased')

    def forward(self, word_input, masks):
        output = self.bert(word_input, attention_mask=masks)
        pool = output.pooler_output

        return pool
    
def predict(train_data, data, sqls):
    train_batch_size = 64
    train_iter = DataLoader(train_data, shuffle=False, batch_size=train_batch_size)

    model = Word_BERT()
    model.load_state_dict(torch.load(r"cl_model/model_global.pth"), strict=True)
    model.to(torch.device('cuda:1'))
    model.eval()
    total = []
    with torch.no_grad():
        for step, batch in enumerate(train_iter):
            text_input, mask_input, data_label, sql_input, sql_mask_input, sql_label = batch
            text_input, mask_input, data_label, sql_input, sql_mask_input, sql_label = text_input.to(torch.device('cuda:1')), mask_input.to(torch.device('cuda:1')), data_label.to(torch.device('cuda:1')), sql_input.to(torch.device('cuda:1')), sql_mask_input.to(torch.device('cuda:1')), sql_label.to(torch.device('cuda:1'))
            out = model(text_input, mask_input)
            sql_out = model(sql_input, sql_mask_input)

            cosine_scores = util.cos_sim(out.cpu(), sql_out.cpu())
            
            for i in range(out.size()[0]):
                total.append(cosine_scores[i][i].item())
    for i, j, k in zip(data, sqls, total):
        print(i, j, k)

    with open('./out/out_global.txt', mode='w') as f:
        for i in total:
            f.write(str(i)+'\n')
    
    
if __name__=='__main__':
    data, sqls, labels = data_process()
    print(data[0], sqls[0], labels[0])
    #加载bert token和config
    tokenizer = AutoTokenizer.from_pretrained(r'./bert-base-uncased', use_fast=True)
    config = AutoConfig.from_pretrained(r'./bert-base-uncased')
    # #处理成模型输入的dataloader
    train_tensor = data_loader(data, sqls, labels, tokenizer)
    predict(train_tensor, data, sqls)