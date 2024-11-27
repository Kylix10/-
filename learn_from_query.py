from torch import optim
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import BaggingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
import evaluation_utils as eval_utils
import matplotlib.pyplot as plt
import numpy as np
import range_query as rq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import statistics as stats
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from collections import namedtuple
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

def min_max_normalize(v, min_v, max_v):
    # The function may be useful when dealing with lower/upper bounds of columns.
    assert max_v > min_v
    return (v-min_v)/(max_v-min_v)


def extract_features_from_query(range_query, table_stats, considered_cols):
    # feat:     [c1_begin, c1_end, c2_begin, c2_end, ... cn_begin, cn_end, AVI_sel, EBO_sel, Min_sel]
    #           <-                   range features                    ->, <-     est features     ->
    feature = []
    # YOUR CODE HERE: extract features from query
    # 提取范围特征（range features）
    for col in considered_cols:
        min_val = table_stats.columns[col].min_val()
        max_val = table_stats.columns[col].max_val()
        (left, right) = range_query.column_range(col, min_val, max_val)
        # 对范围的起始值和结束值进行归一化处理
        norm_left = min_max_normalize(left, min_val, max_val)
        norm_right = min_max_normalize(right, min_val, max_val)
        feature.extend([norm_left, norm_right])

    # 使用不同的估计器来计算估计特征（est features）
    avi_estimator = stats.AVIEstimator()
    avi_sel = avi_estimator.estimate(range_query, table_stats)
    feature.append(avi_sel)

    ebo_estimator = stats.ExpBackoffEstimator()
    ebo_sel = ebo_estimator.estimate(range_query, table_stats)
    feature.append(ebo_sel)

    min_sel_estimator = stats.MinSelEstimator()
    min_sel = min_sel_estimator.estimate(range_query, table_stats)
    feature.append(min_sel)

    return feature


def preprocess_queries(queries, table_stats, columns):
    """
    preprocess_queries turn queries into features and labels, which are used for regression model.
    """
    features, labels = [], []
    for item in queries:
        query, act_rows = item['query'], item['act_rows']
        # YOUR CODE HERE: transform (query, act_rows) to (feature, label)
        # Some functions like rq.ParsedRangeQuery.parse_range_query and extract_features_from_query may be helpful.
        # 使用rq.ParsedRangeQuery.parse_range_query将原始查询字符串解析为ParsedRangeQuery对象
        parsed_query = rq.ParsedRangeQuery.parse_range_query(query)
        # 调用extract_features_from_query提取特征，传入解析后的查询对象、数据表统计信息以及指定列信息
        feature = extract_features_from_query(parsed_query, table_stats, columns)
        label = act_rows
        features.append(feature)
        labels.append(label)
    return features, labels

class QueryDataset(torch.utils.data.Dataset):
    def __init__(self, queries, table_stats, columns):
        super().__init__()
        self.query_data = list(zip(*preprocess_queries(queries, table_stats, columns)))

    def __getitem__(self, index):

        return self.query_data[index]

    def __len__(self):
        return len(self.query_data)
#
# import random
#
# # 检查数据分布的函数
# def check_data_distribution(data):
#     print("数据的均值：", np.mean(data))
#     print("数据的方差：", np.var(data))
#     print("数据的最小值：", np.min(data))
#     print("数据的最大值：", np.max(data))
# import torch.nn.functional as F
# # 定义神经网络模型类，可配置网络结构
# def create_model(input_size):
#     class Net(nn.Module):
#         def __init__(self):
#             super(Net, self).__init__()
#             self.fc1 = nn.Linear(input_size, 64)
#             self.fc2 = nn.Linear(64, 32)
#             self.fc3 = nn.Linear(32, 1)
#
#         def forward(self, x):
#             x = F.relu(self.fc1(x))
#             x = F.relu(self.fc2(x))
#             x = self.fc3(x)
#             return x
#
#     return Net()
# def est_AI1(train_data, test_data, table_stats, columns, optimizer_type='Adam', lr=0.0025):
#     """
#     produce estimated rows for train_data and test_data
#     """
#     # 划分训练集和验证集，这里按照8:2的比例划分，可根据实际情况调整
#     train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=42)
#
#     train_dataset = QueryDataset(train_data, table_stats, columns)
#     train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1)
#     valid_dataset = QueryDataset(valid_data, table_stats, columns)
#     valid_loader = DataLoader(valid_dataset, batch_size=10, shuffle=True, num_workers=1)
#     train_est_rows, train_act_rows = [], []
#     valid_est_rows, valid_act_rows = [], []
#
#     # 创建神经网络模型实例，根据特征维度确定输入大小
#     model = create_model(input_size=15)
#     # 定义损失函数
#     criterion = nn.MSELoss(reduction='sum')
#
#     # 选择优化器
#     if optimizer_type == 'Adam':
#         optimizer = optim.Adam(model.parameters(), lr=lr)  # 简化，先不添加L2正则化项，可后续按需添加
#     elif optimizer_type == 'SGD':
#         optimizer = optim.SGD(model.parameters(), lr=lr)  # 简化，先不添加复杂参数，可后续按需添加
#     else:
#         raise ValueError("不支持的优化器类型")
#
#     # 训练过程
#     for epoch in range(100):  # 适当增加训练轮数，可根据实际情况调整
#         train_loss = 0
#         model.train()
#         for batch in train_loader:
#             features, labels = batch
#             features = [feature.clone().detach().float() for feature in
#                         features]  # 使用clone().detach()方式创建新张量，并转换为单精度浮点数类型
#             features = torch.stack(features)
#             features = features.transpose(0, 1)
#             labels = labels.clone().detach().float().view(-1, 1)
#             optimizer.zero_grad()
#             outputs = model(features)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#         train_loss /= len(train_loader)
#
#         valid_loss = 0
#         model.eval()
#         with torch.no_grad():
#             for batch in valid_loader:
#                 features, labels = batch
#                 features = [feature.clone().detach().float() for feature in
#                             features]  # 使用clone().detach()方式创建新张量，并转换为单精度浮点数类型
#                 features = torch.stack(features)
#                 features = features.transpose(0, 1)
#                 labels = labels.clone().detach().float().view(-1, 1)
#                 outputs = model(features)
#                 loss = criterion(outputs, labels)
#                 valid_loss += loss.item()
#             valid_loss /= len(valid_loader)
#
#         print(f'Epoch {epoch}: Train Loss = {train_loss}, Valid Loss = {valid_loss}')
#
#         # 保存验证集损失最小的模型参数
#         if valid_loss < train_loss:  # 简单比较，可根据实际情况优化判断条件
#             torch.save(model.state_dict(), 'best_model.pth')
#
#     # 加载验证集损失最小的模型参数用于测试
#     model.load_state_dict(torch.load('best_model.pth'))
#     model.eval()
#
#     # 测试过程
#     test_dataset = QueryDataset(test_data, table_stats, columns)
#     test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=1)
#     test_est_rows, test_act_rows = [], []
#     with torch.no_grad():
#         for batch in test_loader:
#             features, _ = batch
#             features = [feature.clone().detach().float() for feature in
#                         features]  # 使用clone().detach()方式创建新张量，并转换为单精度浮点数类型
#             features = torch.stack(features)
#             features = features.transpose(0, 1)
#             est_rows = model(features).squeeze().tolist()
#             test_est_rows.extend(est_rows)
#             test_act_rows.extend([element[-1] for element in batch])
#
#     return train_est_rows, train_act_rows, test_est_rows, test_act_rows
#
#
# def random_search(train_data, test_data, table_stats, columns, n_trials=20):  # 增加搜索次数
#     best_loss = float('inf')
#     best_params = None
#     for _ in range(n_trials):
#         lr = random.uniform(0.0001, 0.01)
#         try:
#             _, _, _, _, test_est_rows, test_act_rows = est_AI1(train_data, test_data, table_stats, columns, 'Adam', lr)
#             loss = np.sum((np.array(test_est_rows) - np.array(test_act_rows)) ** 2)
#             if loss < best_loss:
#                 best_loss = loss
#                 best_params = ('Adam', lr)
#         except Exception as e:
#             print(e)
#             continue
#     return best_params

def est_AI1(train_data, test_data, table_stats, columns):
    """
    produce estimated rows for train_data and test_data
    """
    train_dataset = QueryDataset(train_data, table_stats, columns)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1)
    train_est_rows, train_act_rows = [], []

    # 划分验证集（这里按照8:2的比例划分训练集和验证集，可根据实际情况调整）
    train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=42)
    valid_dataset = QueryDataset(valid_data, table_stats, columns)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=10, shuffle=True, num_workers=1)

    # 定义神经网络模型类
    class Net(nn.Module):
        def __init__(self, input_size):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 1)

            self.shortcut1 = nn.Linear(input_size, 64)  # 用于第一层的残差连接
            self.shortcut2 = nn.Linear(64, 64)  # 用于第二层的残差连接
            self.shortcut3 = nn.Linear(64, 32)  # 用于第三层的残差连接

        def forward(self, x):
            residual = self.shortcut1(x)
            x = torch.relu(self.fc1(x))
            x = x + residual

            residual = self.shortcut2(x)
            x = torch.relu(self.fc2(x))
            x = x + residual

            residual = self.shortcut3(x)
            x = torch.relu(self.fc3(x))
            x = x + residual

            x = self.fc4(x)
            return x

    # 创建神经网络模型实例
    model = Net(input_size=15)
    # 定义损失函数和优化器
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # 早停机制相关变量初始化
    patience = 5  # 连续多少次验证集损失没有下降就触发早停
    epochs_no_improve = 0
    min_valid_loss = float('inf')

    # 训练过程
    for epoch in range(200):  # 可以调整训练的轮数
        train_loss = 0
        model.train()
        for batch in train_loader:
            features, labels = batch
            features = [feature.clone().detach().float() for feature in
                        features]  # 使用clone().detach()方式创建新张量，并转换为单精度浮点数类型
            features = torch.stack(features)
            features = features.transpose(0, 1)
            labels = labels.clone().detach().float().view(-1, 1)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        valid_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                features, labels = batch
                features = [feature.clone().detach().float() for feature in
                            features]  # 使用clone().detach()方式创建新张量，并转换为单精度浮点数类型
                features = torch.stack(features)
                features = features.transpose(0, 1)
                labels = labels.clone().detach().float().view(-1, 1)
                outputs = model(features)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
            valid_loss /= len(valid_loader)

        print(f'Epoch {epoch}: Train Loss = {train_loss}, Valid Loss = {valid_loss}')

        # 检查验证集损失是否下降，更新相关变量
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1

        # 判断是否触发早停机制
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    # 加载验证集损失最小的模型参数用于后续测试和预测
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # 在训练集上进行预测并收集结果（可用于查看训练过程中的表现）
    with torch.no_grad():
        for batch in train_loader:
            features, _ = batch
            features = [feature.clone().detach().float() for feature in
                        features]  # 使用clone().detach()方式创建新张量，并转换为单精度浮点数类型
            features = torch.stack(features)
            features = features.transpose(0, 1)
            est_rows = model(features).squeeze().tolist()
            train_est_rows.extend(est_rows)
            train_act_rows.extend([element[-1] for element in batch])

    # 测试过程
    test_dataset = QueryDataset(test_data, table_stats, columns)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=1)
    test_est_rows, test_act_rows = [], []
    with torch.no_grad():
        for batch in test_loader:
            features, _ = batch
            features = [feature.clone().detach().float() for feature in
                        features]  # 使用clone().detach()方式创建新张量，并转换为单精度浮点数类型
            features = torch.stack(features)
            features = features.transpose(0, 1)
            est_rows = model(features).squeeze().tolist()
            test_est_rows.extend(est_rows)
            test_act_rows.extend([element[-1] for element in batch])

    return train_est_rows, train_act_rows, test_est_rows, test_act_rows

def est_AI2(train_data, test_data, table_stats, columns):
    """
    使用局部加权回归（通过KernelRidge模拟）为训练数据和测试数据生成估计的行数。
    """
    # 预处理查询数据
    train_x, train_y = preprocess_queries(train_data, table_stats, columns)
    test_x, test_y = preprocess_queries(test_data, table_stats, columns)

    # 将数据转换为NumPy数组
    train_x = np.array(train_x)
    train_y = np.array(train_y).reshape(-1, 1)
    test_x = np.array(test_x)
    if test_y is not None:
        test_y = np.array(test_y).reshape(-1, 1)  # 将test_y转换为numpy数组后再操作

    # 标准化特征（对于KernelRidge很重要）
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)

    # 特征选择，选择与目标相关性较高的特征
    k = 10
    selector = SelectKBest(f_regression, k=k)
    train_y_1d = train_y.ravel()
    train_x_selected = selector.fit_transform(train_x_scaled, train_y_1d)
    if test_y is not None:
        test_y_1d = test_y.ravel()
        test_x_selected = selector.transform(test_x_scaled)
    else:
        test_x_selected = selector.transform(test_x_scaled)

    # 超参数调优 - 网格搜索不同的gamma和alpha组合，找到最佳的KernelRidge模型
    param_grid = {'gamma': [0.1, 1, 10], 'alpha': [0.01, 0.1, 1]}
    grid_search = GridSearchCV(KernelRidge(kernel='rbf'), param_grid, cv=5)  # 5折交叉验证
    grid_search.fit(train_x_selected, train_y_1d)
    best_kr_rbf = grid_search.best_estimator_

    # 尝试线性核的KernelRidge并训练
    kr_linear = KernelRidge(kernel='linear', alpha=1.0)
    kr_linear.fit(train_x_selected, train_y_1d)

    # 尝试多项式核的KernelRidge（degree设为2，可调整）并训练
    kr_poly = KernelRidge(kernel='poly', degree=2, alpha=1.0)
    kr_poly.fit(train_x_selected, train_y_1d)

    # 模型融合 - 使用Bagging对不同核的KernelRidge模型进行集成（集成数量可调整）
    n_estimators = 5
    # 创建一个列表来存储不同的KernelRidge模型实例
    estimators_list = []
    estimators_list.append(('rbf_kr', best_kr_rbf))
    estimators_list.append(('linear_kr', kr_linear))
    estimators_list.append(('poly_kr', kr_poly))

    bagging_kr = BaggingRegressor(
        estimator=estimators_list[0][1],  # 先使用其中一个模型作为初始估计器，后续会进行替换
        n_estimators=n_estimators
    )

    # 循环替换BaggingRegressor中的估计器为不同的KernelRidge模型
    for name, estimator in estimators_list:
        bagging_kr.estimators_ = [(name, estimator)]

    bagging_kr.fit(train_x_selected, train_y_1d)

    # 预测训练集和测试集
    train_est_rows_bagging = bagging_kr.predict(train_x_selected).flatten()
    if test_y is not None:
        test_est_rows_bagging = bagging_kr.predict(test_x_selected).flatten()
    else:
        test_est_rows_bagging = []

    # 获取实际的行数（用于评估）
    train_act_rows = train_y_1d
    # 如果test_y为None，将test_act_rows设置为空列表，确保返回值可解包
    test_act_rows = test_y_1d.tolist() if test_y is not None else []

    return train_est_rows_bagging, train_act_rows, test_est_rows_bagging, test_act_rows


def eval_model(model, train_data, test_data, table_stats, columns):
    if model == 'ai1':
        est_fn = est_AI1
    else:
        est_fn = est_AI2

    train_est_rows, train_act_rows, test_est_rows, test_act_rows = est_fn(train_data, test_data, table_stats, columns)

    name = f'{model}_train_{len(train_data)}'
    eval_utils.draw_act_est_figure(name, train_act_rows, train_est_rows)
    p50, p80, p90, p99 = eval_utils.cal_p_error_distribution(train_act_rows, train_est_rows)
    print(f'{name}, p50:{p50}, p80:{p80}, p90:{p90}, p99:{p99}')

    name = f'{model}_test_{len(test_data)}'
    eval_utils.draw_act_est_figure(name, test_act_rows, test_est_rows)
    p50, p80, p90, p99 = eval_utils.cal_p_error_distribution(test_act_rows, test_est_rows)
    print(f'{name}, p50:{p50}, p80:{p80}, p90:{p90}, p99:{p99}')


if __name__ == '__main__':
    stats_json_file = 'data/title_stats.json'
    train_json_file = 'data/query_train_18000.json'
    test_json_file = 'data/validation_2000.json'
    columns = ['kind_id', 'production_year', 'imdb_id', 'episode_of_id', 'season_nr', 'episode_nr']
    table_stats = stats.TableStats.load_from_json_file(stats_json_file, columns)
    with open(train_json_file, 'r') as f:
        train_data = json.load(f)
    with open(test_json_file, 'r') as f:
        test_data = json.load(f)

    eval_model('your_ai_model1', train_data, test_data, table_stats, columns)
    eval_model('your_ai_model2', train_data, test_data, table_stats, columns)