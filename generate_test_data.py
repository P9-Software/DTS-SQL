import os
import json
import pandas as pd
import sqlite3

base_dataset_dir = "dev.json"
base_databases_dir = "./dev_databases/"
output_dir = "predict_dev.json"

os.makedirs(base_databases_dir, exist_ok=True)

dev_dataset = [
    {
        "db_id": "employees",
        "SQL": "SELECT * FROM employee WHERE age > 30;",
        "question": "Which employees are older than 30?",
        "evidence": "age column might be useful"
    },
    {
        "db_id": "products",
        "SQL": "SELECT name, price FROM products WHERE category = 'electronics';",
        "question": "What are the names and prices of electronic products?",
        "evidence": ""
    }
]

with open(base_dataset_dir, 'w') as f:
    json.dump(dev_dataset, f, indent=4)

sqlite_db_paths = {
    "employees": os.path.join(base_databases_dir, "employees/employees.sqlite"),
    "products": os.path.join(base_databases_dir, "products/products.sqlite")
}

for db_id, db_path in sqlite_db_paths.items():
    db_dir = os.path.dirname(db_path)
    os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if db_id == "employees":
        cursor.execute("""
            CREATE TABLE employee (
                id INTEGER PRIMARY KEY,
                name TEXT,
                age INTEGER,
                department TEXT
            );
        """)
        cursor.executemany("INSERT INTO employee (name, age, department) VALUES (?, ?, ?);", [
            ("Alice", 35, "HR"),
            ("Bob", 40, "IT"),
            ("Charlie", 28, "Sales")
        ])
    elif db_id == "products":
        cursor.execute("""
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT,
                price REAL,
                category TEXT
            );
        """)
        cursor.executemany("INSERT INTO products (name, price, category) VALUES (?, ?, ?);", [
            ("Laptop", 1200.00, "electronics"),
            ("Headphones", 200.00, "electronics"),
            ("Book", 20.00, "books")
        ])
    conn.commit()
    conn.close()

description_dir = os.path.join(base_databases_dir, "employees/database_description/")
os.makedirs(description_dir, exist_ok=True)
employee_description = pd.DataFrame({
    "original_column_name": ["id", "name", "age", "department"],
    "column_description": ["Employee ID", "Employee Name", "Employee Age", "Employee Department"],
    "value_description": ["Unique identifier", "Full name", "Age in years", "Department name"]
})
employee_description.to_csv(os.path.join(description_dir, "employee.csv"), index=False)

description_dir = os.path.join(base_databases_dir, "products/database_description/")
os.makedirs(description_dir, exist_ok=True)
products_description = pd.DataFrame({
    "original_column_name": ["id", "name", "price", "category"],
    "column_description": ["Product ID", "Product Name", "Product Price", "Product Category"],
    "value_description": ["Unique identifier", "Name of the product", "Price in USD", "Category of product"]
})
products_description.to_csv(os.path.join(description_dir, "products.csv"), index=False)

with open(output_dir, 'w') as f:
    json.dump({}, f, indent=4)

