# DTS-SQL


## Overview
This repository hosts the code for the DTS-SQL paper, featuring state-of-the-art text-to-SQL models with 7B parameters. Our models demonstrate exceptional performance on the Spider and Spider-SYN datasets, setting new benchmarks in the field.

## Repository Contents

### Finetuning Scripts
- `schema_linking_generation_finetuning.ipynb`: Contains code for finetuning DeepSeek and Mistral models for text-to-SQL tasks.
- `sql_generation_finetuning.ipynb`: Dedicated to finetuning processes, specifically focusing on SQL generation.

### Inference Scripts
- `schema_linking_generation_inference.ipynb`: Script for schema linking generation using the finetuned models.
- `sql_generation_inference.ipynb`: Script for SQL generation at inference time using the finetuned models.

### Dataset Preparation
- `finetuning_dataset_creator.py`: Code for creating the finetuning dataset from the Spider dataset.

### Utilities
- `Utils` directory: Contains all necessary helper functions for formatting tables and creating the dataset.

### Results
- All of the results are stored in the results directory.

### Citation
@article{pourreza2024dts,
  title={DTS-SQL: Decomposed Text-to-SQL with Small Large Language Models},
  author={Pourreza, Mohammadreza and Rafiei, Davood},
  journal={arXiv preprint arXiv:2402.01117},
  year={2024}
}


### Dev.Json Format
[
  {
    "db_id": "string",          // Unique identifier for the database
    "SQL": "string",            // The SQL query corresponding to the question
    "question": "string",       // Natural language question
    "evidence": "string"        // (Optional) Additional context or hint for the question
  },
  ...
]

### BASE_DATABASES_DIR
- Path:
  ./dev_databases/
- Structure:
  ./dev_databases/
  └── {db_id}/
      ├── {db_id}.sqlite           // SQLite database file
      └── database_description/   // (Optional) Folder for table descriptions
          ├── {table_name}.csv   // CSV file describing table columns and values
- database_description
  - Columns:
  original_column_name: Name of the column in the database table
  column_description: Description of the column
  value_description: Description of possible values in the column


### Format for finetuning_dataset_creator
- train_others.json and train_spider.json
  [
    {
      "db_id": "string",           // Database identifier
      "question": "string",        // Natural language question
      "query": "string"            // Corresponding SQL query
    },
    ...
  ]
- dev.json
  [
    {
      "db_id": "string",                  // Database identifier
      "SpiderSynQuestion": "string",     // Natural language question in Spider Syn format
      "query": "string"                  // Corresponding SQL query
    },
    ...
  ]

