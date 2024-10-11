import os

import pandas as pd


def get_row_data_types(df, row_number, start_col):
    """ 获取某一行中各个单元格的非空数据类型 """
    types = []
    row_values = df.iloc[row_number, start_col:]

    for value in row_values:
        if pd.notnull(value):
            types.append(type(value))
        else:
            types.append(None)

    return types


def structure_headers(df, start_row=1, start_col=1):
    start_row -= 1  # 调整为 0-based 索引
    start_col -= 1  # 调整为 0-based 索引

    header_end_row = None

    for row_number in range(start_row, len(df) - 1):
        current_row_types = get_row_data_types(df, row_number, start_col)
        next_row_types = get_row_data_types(df, row_number + 1, start_col)
        if current_row_types and next_row_types and current_row_types == next_row_types:
            header_end_row = max(row_number - 1, 0)
            break

    if header_end_row is None:
        header_end_row = row_number  # 如果没有找到合适的结束行，则默认为最后一行是内容，倒是第二行是表头末尾

    column_headers = [df.iloc[list(range(header_end_row + 1)), i].tolist() for i in range(len(df.columns))]
    new_column_headers = []
    for ix_col, col_header in enumerate(column_headers):
        new_col_header = []
        cur_cell = ''
        for ix_row, cell in enumerate(col_header):
            if pd.isna(cell) or cell.strip() == '':
                if cur_cell != '':
                    col_header[ix_row] = cur_cell
                else:
                    ix_col_tmp = ix_col - 1
                    while ix_col_tmp >= 0:
                        if pd.isna(column_headers[ix_col_tmp][ix_row]) or column_headers[ix_col_tmp][
                            ix_row].strip() == '':
                            ix_col -= 1
                        else:
                            col_header[ix_row] = column_headers[ix_col_tmp][ix_row]
                            new_col_header.append(column_headers[ix_col_tmp][ix_row])
                            break
            else:
                cur_cell = cell
                new_col_header.append(cell)
        new_column_headers.append(new_col_header)

    flat_headers = ['-'.join([i.strip() for i in item if i.strip()]) for item in new_column_headers]

    return flat_headers, header_end_row


def structure_indexes(df, col_header_ix_list=[0, 1]):
    col_header = df.columns[col_header_ix_list]
    # 将多个列合并为一个列， A-B-C形式
    col_header_data = df[col_header].apply(lambda x: '-'.join(x), axis=1).tolist()
    # 将合并后的列替代前面的列
    df.drop(columns=col_header, inplace=True)
    df.insert(0, '-'.join(col_header), col_header_data)
    return df


def update_new_headers_csv(input_file, output_file, col_header_ix_list=[0, 1], start_row=1, start_col=1):
    # 多行头合并为A-B-C形式
    # 多列头col_header_ix_list合并为X-Y形式
    df = pd.read_excel(input_file, header=None)
    headers, header_end_row = structure_headers(df, start_row, start_col)
    if headers:
        df_new = df[header_end_row + 1:]
        df_new.columns = headers
        df_new = structure_indexes(df_new, col_header_ix_list=col_header_ix_list)
        df_new.to_csv(output_file, index=False)
    else:
        print("未找到有效的表头信息。")
    return headers, header_end_row


def get_all_file_paths(directory):
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths


def preprocess_table(input_dir):
    """
    对文件夹下的数据文件预处理，并返回关键字段（文件名、表头和关键列）
    example:
    input_dir: 36大中城市居民消费价格分类指数(上年同月＝100)(2016-)
    output:表格名称：2023年第二季度累计各地区建筑业总产值和竣工产值.csv
          表头：时间-地区； 建筑业总产值(亿元)； 建筑业总产值(亿元)-装饰装修； 建筑业总产值(亿元)-在外省完成的产值； 竣工产值(亿元)
          关键列：
          时间-地区: 2023年第二季度-广西壮族自治区;2023年第二季度-浙江省;2023年第二季度-河南省
    """
    for filepath, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('xlsx'):
                # print(os.path.join(filepath, filename))
                update_new_headers_csv(os.path.join(filepath, filename),
                                       os.path.join(filepath, filename.split('.')[0] + '.csv'),
                                       col_header_ix_list=[0, 1])
    header_list = []
    title_list = []
    index_list = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(input_dir, filename))
            header = '； '.join([col.replace(' ', '') for col in list(df.columns)])
            header_list.append(header)
            title_list.append(filename)
            key_header = df.columns[:1]
            index_data = [key_header[0], [s for s in set(list(df[key_header[0]])) if isinstance(s, str)]]
            index = index_data[0] + ': ' + ';'.join(index_data[1])
            index_list.append(index)
    data_str = '\n'.join(
        [f'表格名称：{filename}\n表头：{header}\n关键列：\n{index}\n' for header, filename, index in
         zip(header_list, title_list, index_list)])
    return data_str
