# 纠错SQL 查看结果
#skeleton_error system_error
import json
#answer_error system_error
replace_sen = "You are a database expert. I will provide a question, the database schema, and incorrect SQL. The syntax of the incorrect SQL statements is correct, but the execution results do not match the answer to the question. Please correct it.\nQuestion: "
import json
with open('/root/error_corrected/output/5/check_answer_error.json') as f:
    answer_error = json.load(f)
answer_error = answer_error[0]
data_total = []
for _, j in answer_error.items():
    data_total += j
print("answer_error", len(data_total))

with open('/root/spider_dataset/instruct_data/dev.json') as f:
    data_instruct = json.load(f)

sqls = []
with open('/root/error_corrected/output/5/sql_regen.txt', mode='r', encoding='utf-8') as f:
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
with open('/root/error_corrected/output/5/answer_error.json',mode='w') as f:
    f.write(data_total)

replace_error = 'You are a database expert. I will provide a question, the database schema, and non-executable SQL. There are syntax errors in non-executable SQL, such as "syntax errors" and "no such column". Please correct it.\nQuestion: '
with open('/root/error_corrected/output/5/check_error.json') as f:
    answer_error = json.load(f)
data_total = answer_error[0]["err"]
print("system_error", len(data_total))

with open('/root/spider_dataset/instruct_data/dev.json') as f:
    data_instruct = json.load(f)

# 错误的SQL
# sqls = []
# with open('/root/error_corrected/output/2/sql_pred.txt', mode='r', encoding='utf-8') as f:
#     lines = f.readlines()
#     for line in lines:
#         sqls.append(line.strip())
sqls = []
with open('/root/error_corrected/output/5/sql_regen.txt', mode='r', encoding='utf-8') as f:
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
with open('/root/error_corrected/output/5/system_error.json',mode='w') as f:
    f.write(data_total)