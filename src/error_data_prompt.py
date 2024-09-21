# 纠错SQL 查看结果
#skeleton_error system_error
import json
#skeleton_error system_error
replace_sen = "You are a database expert. I will provide a question, the database schema, and incorrect SQL. The syntax of the incorrect SQL statements is correct, but the execution results do not match the answer to the question. Please correct it.\nQuestion: "
import json
with open('./check_skeleton_error.json') as f:
    skeleton_error = json.load(f)
skeleton_error = skeleton_error[0]
data_total = []
for _, j in skeleton_error.items():
    data_total += j
print("skeleton_error", len(data_total))

with open('./instruct_data/dev.json') as f:
    data_instruct = json.load(f)

sqls = []
with open('./sql_regen.txt', mode='r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        sqls.append(line.strip())

instruct_data = []
for i in data_total:
    sen = data_instruct[i]["instruction"].replace("Write a sql to answer the question ", replace_sen) + "Incorrect SQL:\n" + sqls[i]
    one = {
        "instruction": sen,
        "input": "",
        "output": data_instruct[i]["output"],
        "history": []
            }
    instruct_data.append(one)
 
data_total = json.dumps(instruct_data,indent=4,ensure_ascii=False)
with open('./skeleton_error.json',mode='w') as f:
    f.write(data_total)

replace_error = 'You are a database expert. I will provide a question, the database schema, and non-executable SQL. There are syntax errors in non-executable SQL, such as "syntax errors" and "no such column". Please correct it.\nQuestion: '
with open('./system_error.json') as f:
    system_error = json.load(f)
data_total = system_error[0]["err"]
print("system_error", len(data_total))

with open('./dev.json') as f:
    data_instruct = json.load(f)

sqls = []
with open('./sql_regen.txt', mode='r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        sqls.append(line.strip())

instruct_data = []
for i in data_total:
    sen = data_instruct[i]["instruction"].replace("Write a sql to answer the question ", replace_error) + "Incorrect SQL:\n" + sqls[i]
    one = {
        "instruction": sen,
        "input": "",
        "output": data_instruct[i]["output"],
        "history": []
            }
    instruct_data.append(one)
 
data_total = json.dumps(instruct_data,indent=4,ensure_ascii=False)
with open('./system_error.json',mode='w') as f:
    f.write(data_total)