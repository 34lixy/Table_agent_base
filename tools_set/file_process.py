import json

import pandas as pd
from langchain.schema import Document
from langchain_community.document_loaders import CSVLoader


def get_row_data_types(df, row_number, start_col):
    """ 获取某一行中各个单元格的非空数据类型 """
    types = []
    row_values = df.iloc[row_number, start_col - 1:]

    for value in row_values:
        if pd.notnull(value):
            types.append(type(value))

    return types


def structure_headers(df, start_row=1, start_col=1):
    start_row -= 1  # 调整为 0-based 索引
    start_col -= 1  # 调整为 0-based 索引

    header_end_row = None
    data_start_row = None

    for row_number in range(start_row, len(df)):
        current_row_types = get_row_data_types(df, row_number, start_col)
        next_row_types = get_row_data_types(df, row_number + 1, start_col)

        if current_row_types and next_row_types and current_row_types == next_row_types:
            header_end_row = row_number - 1
            data_start_row = row_number
            break

    if header_end_row is None:
        return [], None

    column_headers = [[] for _ in range(len(df.columns))]
    for row in range(start_row, data_start_row):
        for col in range(start_col, len(df.columns)):
            value = str(df.iloc[row, col])
            if value != 'nan':  # pandas 会将缺失值显示为 'nan'
                column_headers[col].append(value)

    flat_headers = ['-'.join([i for i in item if i.strip()]) for item in column_headers]

    return flat_headers, header_end_row


def load_xlsx_file(path, text_splitter, num_part=None):
    """处理 xlsx 文件，将其转换为 csv 后处理"""
    df = pd.read_excel(path, header=None)
    headers, end_row = structure_headers(df)
    new_df = df.iloc[end_row + 1:, :]
    new_df.columns = headers
    csv_path = path.replace('xlsx', 'csv')
    new_df.to_csv(csv_path, index=False)
    return load_csv_file(path, text_splitter, num_part)


def load_csv_file(path, text_splitter, num_part=None):
    loader = CSVLoader(file_path=path, encoding='utf-8')
    tables = loader.load()

    if not num_part:
        return text_splitter.split_documents(tables)
    else:
        total_length = len(tables)

        part_size = total_length // num_part
        parts = []

        start = 0
        for i in range(num_part):
            end = start + part_size if i < num_part - 1 else total_length
            parts.append(tables[start:end])
            start = end

        split_docs = []
        for part in parts:
            part_split = text_splitter.split_documents(part)
            split_docs.append(part_split)

        return split_docs


def load_txt_file(path, text_splitter, num_part=None):
    """处理 txt 文件"""

    with open(path, 'r', encoding='utf-8') as file:
        text = file.read()
    text_document = Document(page_content=text, metadata={"source": path})
    if not num_part:
        return text_splitter.split_documents([text_document])
    else:
        total_length = len([text_document])

        part_size = total_length // num_part
        parts = []

        start = 0
        for i in range(num_part):
            end = start + part_size if i < num_part - 1 else total_length
            parts.append([text_document][start:end])
            start = end

        split_docs = []
        for part in parts:
            part_split = text_splitter.split_documents(part)
            split_docs.append(part_split)

        return split_docs


def load_json_file(path, text_splitter, num_part=None):
    """处理 json 文件"""

    documents = []
    with open(path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    for item in json_data:
        json_document = Document(page_content=json.dumps(item), metadata={"source": path})
        documents.extend(text_splitter.split_documents([json_document]))
    if not num_part:
        return documents
    else:
        total_length = len(documents)

        part_size = total_length // num_part
        parts = []

        start = 0
        for i in range(num_part):
            end = start + part_size if i < num_part - 1 else total_length
            parts.append(documents[start:end])
            start = end

        split_docs = []
        for part in parts:
            part_split = text_splitter.split_documents(part)
            split_docs.append(part_split)

        return split_docs


def process_path(path, text_splitter, num_part):
    if path.endswith('.csv'):
        return load_csv_file(path, text_splitter, num_part)
    elif path.endswith('.xlsx'):
        return load_xlsx_file(path, text_splitter, num_part)
    elif path.endswith('.txt'):
        return load_txt_file(path, text_splitter, num_part)
    elif path.endswith('.json'):
        return load_json_file(path, text_splitter, num_part)
