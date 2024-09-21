import json
import re

def remove_str(text):
    pattern = r'"[^"]*"'
    text = re.sub(pattern, '_', text)
    pattern = r"'[^']*'"
    text = re.sub(pattern, '_', text)
    return text

with open('./instruct_data/dev.json') as f:
    data = json.load(f)

total = []

question = [i['instruction'].split('answer the question "')[1].split('"\n\n### Input')[0] for i in data]
schema = [i['instruction'].split('"\n\n### Input:\n')[1][:-1].replace("\n"," ").replace(",","").replace("("," ").replace(")"," ") for i in data]
sql = ["select " + i['output'] for i in data]

ques = []
for i, j, k in zip(question, sql, schema):
    ii = remove_str(i)
    q = ii.split(" ")
    jj = remove_str(j)
    s = jj.split(" ")
    kk = k.split(" ")
    k_s = []
    for schema_k in kk:
        k_s.append(schema_k)
        if '_' in schema_k:
            aaa = schema_k.split('_')
            for s_k in aaa:
                if s_k not in ['a','the','in','id','of','from','by','all','where','number','name']:
                    k_s.append(s_k)

    qq, ss = [], []
    for ii in q:
        
        if ii in k_s or ii[:-1] in k_s or ii+"s" in k_s:
            if (len(qq)!=0 and qq[-1]=='_'):
                continue
            qq.append("_")
        else:
            qq.append(ii)
    qq = ' '.join(qq)
    
    for indexx, ii in enumerate(s):
        if ii in ['count','min','max','mean','select','from','where','except','group','by','order','asc','desc','union','from','in','intersect']:
            ss.append(ii)
            continue
        if ii in ['join', 'on']:
            continue
        if ii == '=':
            if s[indexx-2] =='on':
                continue
        if ii in k_s or (len(ii)!=1 and ii[:-1] in k_s) or ii+"s" in k_s or "." in ii or ii==',' or (len(ss)!=0 and ss[-1]!='limit' and ii.isdigit()):
            if (len(ss)!=0 and ss[-1]=='_'):
                continue
            ss.append("_")
        else:
            ss.append(ii)
    ss = ' '.join(ss)

    # ques.append(qq + " | " + ss)
    one = {"q": i, "sql": j, "q_s": qq, "sql_s": ss}
    total.append(one)

data_total = json.dumps(total,indent=4,ensure_ascii=False)
with open('./skeleton.json',mode='w') as f:
    f.write(data_total)