import json
import random


def split_train_validation(query_train_path, validation_path, train_path, validation_ratio=0.2):
    """
    将训练数据划分成训练集和验证集，并保存到相应文件，同时从原训练数据中移除划分到验证集的数据。

    参数:
    query_train_path (str): 原始训练数据文件（包含18000条数据的json文件）的路径，例如 "query_train_18000.json"
    validation_path (str): 验证集保存的文件路径，例如 "you_can_make_it.json"
    train_path (str): 剩余训练集保存的文件路径
    validation_ratio (float): 验证集所占的比例，默认0.2（即20%）
    """
    # 读取原始训练数据
    with open(query_train_path, 'r') as f:
        all_data = json.load(f)

    # 获取数据总数
    num_data = len(all_data)
    # 计算验证集的数量
    num_validation = int(num_data * validation_ratio)

    # 随机打乱数据顺序
    random.shuffle(all_data)

    # 划分验证集
    validation_data = all_data[:num_validation]
    train_data = all_data[num_validation:]

    # 将验证集数据保存到指定文件
    with open(validation_path, 'w') as f:
        json.dump(validation_data, f, indent=4)

    # 将剩余的训练集数据保存到指定文件（原训练数据中去掉了验证集对应的数据）
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=4)


# 使用示例，根据你的实际文件路径进行调整
query_train_path = "data/query_train_18000.json"
validation_path = "data/you_can_make_it.json"
train_path = "data/new_train_data.json"
split_train_validation(query_train_path, validation_path, train_path)