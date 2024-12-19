import torch
import re
import sqlite3
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import cuda  # noqa: F401
from transformers import StoppingCriteria
from tqdm import tqdm
from sql_metadata import Parser
import json
import numpy as np
import os
import gc
from accelerate.utils import release_memory

class SQLSchemaGenerator:
    def __init__(self, base_dataset_dir, base_databases_dir, output_dir,
                 schema_linker_adapter_path, sql_generation_adapter_path):
        self.BASE_DATASET_DIR = base_dataset_dir
        self.BASE_DATABASES_DIR = base_databases_dir
        self.OUTPUT_DIR = output_dir

        self.schema_linker_adapter_path = schema_linker_adapter_path
        self.sql_generation_adapter_path = sql_generation_adapter_path

        self.schema_model = AutoModelForCausalLM.from_pretrained(
            schema_linker_adapter_path, attn_implementation="eager",
            torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.sql_model = AutoModelForCausalLM.from_pretrained(
            sql_generation_adapter_path, attn_implementation="eager",
            torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(schema_linker_adapter_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    class EosListStoppingCriteria(StoppingCriteria):
        def __init__(self, eos_sequence):
            self.eos_sequence = eos_sequence

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
            return self.eos_sequence in last_ids

    def _append_item(self, query, db_id, counter):
        try:
            with open(self.OUTPUT_DIR, 'r') as json_file:
                data_dict = json.load(json_file)
        except FileNotFoundError:
            data_dict = {}

        item_value = query + "\t----- bird -----\t" + db_id
        data_dict[counter] = item_value

        with open(self.OUTPUT_DIR, 'w') as json_file:
            json.dump(data_dict, json_file, indent=4)

    @staticmethod
    def _flush():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def _remove_spaces(text):
        return re.sub(r'\s+', ' ', text)

    @staticmethod
    def _get_all_table_names(db_uri):
        import sys
        if not os.path.exists(db_uri):
            print(f"Error: The database file at {db_uri} does not exist.")
            return []
        
        conn = sqlite3.connect(db_uri)
        print(sys.path)

        cursor = conn.cursor()
        print(f"in get all table names,{db_uri}")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_names = cursor.fetchall()
        print(table_names)
        conn.close()
        return [table_name[0] for table_name in table_names]

    def _get_table_schema_with_samples(self, db_uri, table_name, sample_limit=0, columns_description={}):
        conn = sqlite3.connect(db_uri)
        cursor = conn.cursor()

        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        columns = cursor.fetchall()
        cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`);")
        foreign_keys = cursor.fetchall()
        cursor.execute(f"PRAGMA index_list(`{table_name}`);")
        primary_key_indices = cursor.fetchall()
        primary_key_columns = []

        for index_info in primary_key_indices:
            index_name = index_info[1]
            cursor.execute(f"PRAGMA index_info(`{index_name}`);")
            index_columns = cursor.fetchall()
            primary_key_columns.extend(column[2] for column in index_columns)

        schema_str = f"CREATE TABLE `{table_name}` (\n"
        for column in columns:
            column_name = column[1]
            data_type = column[2]
            schema_str += f"  {column_name} {data_type}"
            if column_name in primary_key_columns:
                schema_str += " PRIMARY KEY"
            for foreign_key in foreign_keys:
                if column_name == foreign_key[3]:
                    schema_str += f" REFERENCES {foreign_key[2]}({foreign_key[4]})"
            if column_name in columns_description:
                schema_str += f" -- '{columns_description[column_name]}'"
            schema_str += ",\n"
        schema_str = schema_str.rstrip(",\n")
        schema_str += "\n);\n"

        cursor.execute(f"SELECT * FROM `{table_name}` LIMIT {sample_limit};")
        sample_rows = cursor.fetchall()

        if len(sample_rows) > 0:
            schema_str += f"Sample rows from `{table_name}`:\n"
            for row in sample_rows:
                formatted_row = ", ".join(str(item) for item in row)
                schema_str += f"{formatted_row}\n"

        conn.close()
        return schema_str

    def _load_descriptions(self, db_path, table_name):
        description_file = f"{db_path}/database_description/{table_name}.csv"
        if not os.path.exists(description_file):
            return {}
        try:
            df = pd.read_csv(description_file)
        except Exception:
            return {}
        if "column_description" not in df.columns or "value_description" not in df.columns:
            return {}
        columns_description = {}
        for index, row in df.iterrows():
            if pd.notna(row["column_description"]):
                desc = self._remove_spaces(row["column_description"])
                if pd.notna(row["value_description"]):
                    desc += f" has values: ({self._remove_spaces(row['value_description'])})"
                columns_description[row["original_column_name"]] = desc
        return columns_description

    def _generate_sql(self, inputs):
        output_tokens = self.sql_model.generate(
            inputs, max_new_tokens=300, do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=[self.EosListStoppingCriteria([6204, 185, 10897])]
        )
        return self.tokenizer.decode(output_tokens[0][len(inputs[0]):], skip_special_tokens=True)

    def _generate_schema(self, inputs):
        output_tokens = self.schema_model.generate(
            inputs, max_new_tokens=250, do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=[self.EosListStoppingCriteria([6204, 185, 10897])]
        )
        return self.tokenizer.decode(output_tokens[0][len(inputs[0]):], skip_special_tokens=True)

    def generate_question(self, question):
        db_id = question["db_id"]
        query = question["SQL"]
        question_text = question["question"]
        db_uri = f"{self.BASE_DATABASES_DIR}/{db_id}/{db_id}.sqlite"
        db_path = f"{self.BASE_DATABASES_DIR}/{db_id}"
        table_names = self._get_all_table_names(db_uri)
        print(table_names)
        database_schema = ""

        for table_name in table_names:
            columns_description = self._load_descriptions(db_path, table_name)
            schema = self._get_table_schema_with_samples(db_uri, table_name, 0, columns_description)
            database_schema += schema + "\n"

        user_message = f"""Given the following SQL tables, your job is to determine the columns and tables that the question is referring to.
        {database_schema}
        ####
        Question: {question_text}
        """

        messages = [
            {"role": "user", "content": user_message.strip()}
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, tokenize=True
        ).to(self.schema_model.device)

        response = self._generate_schema(inputs)

        if "Tables: " in response:
            response = response.split("Tables: ")[1]
        if ";" in response:
            response = response.split(";")[0]
        schema_linking_tables = re.sub(r'\s+', ' ', response).strip().split(", ")

        return {
            "question": question_text,
            "db_id": db_id,
            "query": query,
            "database_schema": database_schema,
        }, schema_linking_tables

    def test_question(self, questions, schema_linking_tables):
        results = []
        for idx, row in tqdm(enumerate(questions), total=len(questions), desc="Generating SQL"):
            query = row["query"]
            db_id = row["db_id"]
            question_text = row["question"]
            database_schema = row["database_schema"]

            user_message = f"""Given the following SQL tables, your job is to generate the Sqlite SQL query given the user's question.
            Put your answer inside the ```sql and ``` tags.
            {database_schema}
            ####
            Question: {question_text}
            """

            messages = [
                {"role": "user", "content": user_message.strip()}
            ]

            inputs = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True, tokenize=True
            ).to(self.sql_model.device)

            response = self._generate_sql(inputs)

            if ";" in response:
                response = response.split(";")[0]
            if "```sql" in response:
                response = response.split("```sql")[1]
                response = response.split("```")[0]

            response = re.sub(r'\s+', ' ', response).strip()

            if "SELECT" not in response:
                response = "SELECT * FROM " + schema_linking_tables[0]

            results.append(response)

            self._append_item(response, db_id, idx)

        return results
  
