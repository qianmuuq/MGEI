## 测试
import os
import re
import asyncio
import sqlite3
import threading
from typing import Tuple, Any, List, Set

# get the database cursor for a sqlite database path
def get_cursor_from_path(sqlite_path, query):
    try:
        if not os.path.exists(sqlite_path):
            print("Openning a new connection %s" % sqlite_path)
        connection = sqlite3.connect(sqlite_path)
    except Exception as e:
        print(sqlite_path)
        raise e
    connection.text_factory = lambda b: b.decode(errors="ignore")
    cursor = connection.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    return result


import json
with open('/root/spider_dataset/instruct_data/dev.json') as f:
    data = json.load(f)

with open('/root/spider_dataset/dev.json') as f:
    data_path = json.load(f)

preds = []
with open('/root/LLaMA-Efficient-Tuning/eval_test/model_saved_lora_codellama_13b_spider_insrtuct_epoch10_lora_target_all/sql_pred.txt') as f:
    lines = f.readlines()
    for line in lines:
        preds.append(line.strip())

datas = []
num = 0
index = []
for i, j, k in zip(data, data_path, preds):
    if "= '" in k:
        i["db_id"] = j["db_id"]
        i["pred"] = k
        datas.append(i)
        index.append(num)
    num += 1
print(datas[0])

data_total = []
for i, indexx in zip(datas, index):
    db_path = '/root/spider_dataset/database/' + i["db_id"] + '/' + i["db_id"] + '.sqlite'
    out = i["pred"].split()
    cols = []
    table = []
    for j, k in enumerate(out):
        if out[j-1]=='from':
            table.append(k)
        if out[j-1]=='=' and k[0]=="'":
            cols.append(out[j-2])
    # print(table, cols)
    if len(table)>1:
        continue
    cols_all = {}
    try:
        for j in cols: 
            if "." in j:
                t, c = j.split(".")
                g_str = "select distinct " + c + " from " + t
                # print(i["output"])
                # print(g_str)
                cc = c
                result = get_cursor_from_path(db_path, g_str)
            else:
                g_str = "select distinct " + j + " from " + table[0]
                # print(i["output"])
                # print(g_str)
                cc = j
                result = get_cursor_from_path(db_path, g_str)
            result_all = []
            for ii in result:
                if ii[0]==None:
                    continue
                result_all.append(str(ii[0]))
            # print(result_all)
            cols_all[cc] = ", ".join(result_all)
        
        i["cols"] = cols_all
        i["index"] = indexx
        data_total.append(i)
    except Exception as e:
            pass

data_a = []
for i in data_total:
    instruction = i["instruction"].split("\n\n### Input:\n")[0] + "\n### column value:\n"
    for j, k in i["cols"].items():
        instruction += j + ":[" + k + "]" + "\n"
    one = {
        "instruction": instruction + "select " + i["pred"],
        "input":"",
        "output":i["output"],
        "history":[],
        "index": i["index"]
    }
    data_a.append(one)
print(data_a[0])
print(len(data_a))
data_total = json.dumps(data_a, indent=4)
with open('data/pred.json', mode='w') as f:
    f.write(data_total)