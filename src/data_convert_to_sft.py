import pandas as pd
from transformers import AutoTokenizer
import json
import re

DATASET_SCHEMA = "./spider_dataset/tables.json"
DATASET = "./preprocessed_dev.json"

def load_data(DATASET):
    return pd.read_json(DATASET)

def find_foreign_keys_MYSQL_like(db_name):
  df = spider_foreign[spider_foreign['Database name'] == db_name]
  output = "["
  for index, row in df.iterrows():
    output += row['First Table Name'] + '.' + row['First Table Foreign Key'] + " = " + row['Second Table Name'] + '.' + row['Second Table Foreign Key'] + ', '
  output= output[:-2] + "]"
  return output
def find_fields_MYSQL_like(db_name):
  df = spider_schema[spider_schema['Database name'] == db_name]
  df = df.groupby(' Table Name')
  output = ""
  for name, group in df:
    output += "Table " +name+ ', columns = ['
    for index, row in group.iterrows():
      output += row[" Field Name"]+','
    output = output[:-1]
    output += "]\n"
  return output
def find_primary_keys_MYSQL_like(db_name):
  df = spider_primary[spider_primary['Database name'] == db_name]
  output = "["
  for index, row in df.iterrows():
    output += row['Table Name'] + '.' + row['Primary Key'] +','
  output = output[:-1]
  output += "]\n"
  return output
def creatiing_schema(DATASET_JSON):
    schema_df = pd.read_json(DATASET_JSON)
    schema_df = schema_df.drop(['column_names','table_names'], axis=1)
    schema = []
    f_keys = []
    p_keys = []
    for index, row in schema_df.iterrows():
        tables = row['table_names_original']
        col_names = row['column_names_original']
        col_types = row['column_types']
        foreign_keys = row['foreign_keys']
        primary_keys = row['primary_keys']
        for col, col_type in zip(col_names, col_types):
            index, col_name = col
            if index == -1:
                for table in tables:
                    schema.append([row['db_id'], table, '*', 'text'])
            else:
                schema.append([row['db_id'], tables[index], col_name, col_type])
        for primary_key in primary_keys:
            index, column = col_names[primary_key]
            p_keys.append([row['db_id'], tables[index], column])
        for foreign_key in foreign_keys:
            first, second = foreign_key
            first_index, first_column = col_names[first]
            second_index, second_column = col_names[second]
            f_keys.append([row['db_id'], tables[first_index], tables[second_index], first_column, second_column])
    spider_schema = pd.DataFrame(schema, columns=['Database name', ' Table Name', ' Field Name', ' Type'])
    spider_primary = pd.DataFrame(p_keys, columns=['Database name', 'Table Name', 'Primary Key'])
    spider_foreign = pd.DataFrame(f_keys,
                        columns=['Database name', 'First Table Name', 'Second Table Name', 'First Table Foreign Key',
                                 'Second Table Foreign Key'])
    return spider_schema,spider_primary,spider_foreign

def schema_linking_prompt_maker(test_sample_text,database):
  # instruction = "You are a data analyst who specializes in writing SQL query. Your task is to write the most efficient SQL based on the information provided about the table and the problem.\n"
  instruction = "I will provide information about tables and columns in the database, a natural language question, and the corresponding SQL query. Specifically, you should provide the incorect SQL, which can contain one or two erors. The database information is as follows: \n"
  fields = find_fields_MYSQL_like(database)
  foreign_keys = "Foreign_keys = " + find_foreign_keys_MYSQL_like(database) + '\n'
  prompt = instruction + fields +foreign_keys+ 'Question: ' + test_sample_text + "\nSQL query: "
  return prompt

def find_fields_MYSQL_like_2(db_name):
  df = spider_schema[spider_schema['Database name'] == db_name]
  df = df.groupby(' Table Name')
  output = ""
  for name, group in df:
    # print(name,group)
    output += name+ ' : '
    for index, row in group.iterrows():
      if row[" Field Name"]=='*':
        continue
      output += row[" Field Name"]+' , '
    output = output[:-3]
    output += " | "
  # print(output)
  return output

def normalize(query: str) -> str:
    def comma_fix(s):
        # Remove spaces in front of commas
        return s.replace(" , ", ", ")

    def white_space_fix(s):
        # Remove double and triple spaces
        return " ".join(s.split())

    def lower(s):
        # Convert everything except text between (single or double) quotation marks to lower case
        return re.sub(r"\b(?<!['\"])(\w+)(?!['\"])\b", lambda match: match.group(1).lower(), s)

    return comma_fix(white_space_fix(lower(query)))
  
def find_fields_MYSQL_like_instruction(db_name):
  df = spider_schema[spider_schema['Database name'] == db_name]
  df = df.groupby(' Table Name')
  output = ""
  for name, group in df:
    # print(name,group)
    output += name+ '('
    for index, row in group.iterrows():
      if row[" Field Name"]=='*':
        continue
      output += row[" Field Name"]+', '
    output = output[:-2]
    output += ")\n"
  # print(output)
  return output

if __name__ == '__main__':
    spider_schema,spider_primary,spider_foreign = creatiing_schema(DATASET_SCHEMA)
    # print(spider_schema)
    val_df = load_data(DATASET)
    print(f"Number of data samples {val_df.shape[0]}")
    
    data_total = []
    count = 0
          
    data_total = []
    for index, row in val_df.iterrows():
        sentence = find_fields_MYSQL_like_instruction(row['db_id'])

        foreign_keys = "Foreign_Keys = " + find_foreign_keys_MYSQL_like(row['db_id']) + '\n'

        instruct = 'Write a sql to answer the question "'
        sentence = instruct + row['question']+'"\n\n### Input:\n'+sentence.lower()
 
        out = row['norm_sql']

        one = {
            "instruction": sentence,
            "input": "",
            "output": out[7:],
            "history": []
        }

        data_total.append(one)

    data_total = json.dumps(data_total,indent=4,ensure_ascii=False)
    with open('./instruct_data/dev.json',mode='w') as f:
        f.write(data_total)
    