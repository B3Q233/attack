'''
创建于2020年3月1日
Xiangnan He等人提出的LightGCN的PyTorch实现
论文：LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

作者: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import world  # 项目全局配置
import torch
from torch import nn, optim  # 神经网络与优化器
import numpy as np  # 数值计算库
from torch import log  # PyTorch的log函数
from dataloader import BasicDataset  # 基础数据集类
from time import time  # 时间统计
from model import LightGCN  # LightGCN模型
from model import PairWiseModel  #  pairwise损失模型基类
from sklearn.metrics import roc_auc_score  # AUC计算工具
import random  # 随机数工具
import os  # 文件操作

# 尝试导入C++扩展用于加速负采样（若失败则使用Python实现）
try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    # 加载C++采样模块
    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(world.seed)  # 设置随机种子
    sample_ext = True  # 标记使用C++扩展
except:
    world.cprint("未加载C++扩展，将使用Python采样")
    sample_ext = False  # 标记使用Python实现


class BPRLoss:
    """BPR损失函数管理器，负责模型训练中的损失计算与参数优化"""
    def __init__(self,
                 recmodel: PairWiseModel,  # 推荐模型（需支持bpr_loss方法）
                 config: dict):  # 配置字典（包含学习率、权重衰减等）
        self.model = recmodel  # 推荐模型实例
        self.weight_decay = config['decay']  # 权重衰减系数（L2正则化）
        self.lr = config['lr']  # 学习率
        # 初始化Adam优化器
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        """
        执行单步训练（BPR损失计算 + 反向传播 + 参数更新）
        Args:
            users: 用户ID列表
            pos: 正样本物品ID列表（用户已交互）
            neg: 负样本物品ID列表（用户未交互）
        Returns:
            损失值（CPU上的标量）
        """
        # 计算BPR损失和正则化损失
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        # 应用权重衰减（正则化项乘以系数）
        reg_loss = reg_loss * self.weight_decay
        # 总损失 = BPR损失 + 正则化损失
        loss = loss + reg_loss

        # 清空梯度、反向传播、更新参数
        self.opt.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播计算梯度
        self.opt.step()  # 优化器更新参数

        # 返回损失值（转换为CPU上的Python数值）
        return loss.cpu().item()


def UniformSample_original(dataset, neg_ratio=1):
    """
    生成BPR训练所需的（用户，正样本，负样本）三元组
    优先使用C++扩展加速，失败则使用Python实现
    Args:
        dataset: 数据集实例（BasicDataset）
        neg_ratio: 负样本比例（默认为1，即1个正样本配1个负样本）
    Returns:
        三元组数组，形状为[样本数, 3]
    """
    dataset: BasicDataset
    allPos = dataset.allPos  # 每个用户的正样本列表
    start = time()
    if sample_ext:
        # 使用C++扩展进行负采样（高效）
        S = sampling.sample_negative(
            dataset.n_users, dataset.m_items,
            dataset.trainDataSize, allPos, neg_ratio
        )
    else:
        # 使用Python实现的负采样（较慢）
        S = UniformSample_original_python(dataset)
    return S


def UniformSample_original_python(dataset):
    """
    BPR采样的Python实现（LightGCN原论文中的采样逻辑）
    Returns:
        np.array: 三元组数组 [user, positem, negitem]
    """
    total_start = time()
    dataset: BasicDataset
    user_num = dataset.trainDataSize  # 训练样本数量（与用户数相关）
    # 随机选择用户（可能重复，模拟多轮采样）
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos  # 每个用户的正样本列表
    S = []  # 存储采样结果
    sample_time1 = 0.  # 采样耗时统计
    sample_time2 = 0.

    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]  # 当前用户的正样本列表
        if len(posForUser) == 0:  # 无正样本的用户跳过
            continue
        sample_time2 += time() - start  # 统计正样本获取耗时

        # 随机选择一个正样本
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]

        # 随机选择负样本（需确保不在正样本列表中）
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue  # 若负样本在正样本中，重新采样
            else:
                break
        S.append([user, positem, negitem])  # 添加三元组

        end = time()
        sample_time1 += end - start  # 统计单次采样总耗时

    total = time() - total_start  # 总耗时
    return np.array(S)


# ===================采样器结束==========================
# =====================工具函数===============================

def set_seed(seed):
    """设置随机种子，确保实验可复现"""
    np.random.seed(seed)  # NumPy随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 单GPU种子
        torch.cuda.manual_seed_all(seed)  # 多GPU种子
    torch.manual_seed(seed)  # PyTorch随机种子


def getFileName():
    """生成模型权重保存的文件名（包含模型类型、数据集、超参数）"""
    if world.model_name == 'mf':
        # 矩阵分解模型：mf-数据集-嵌入维度.pth.tar
        file = f"mf-{world.dataset}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == 'lgn':
        # LightGCN模型：lgn-数据集-层数-嵌入维度.pth.tar
        file = f"lgn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}.pth.tar"
    # 返回完整路径（保存目录+文件名）
    return os.path.join(world.FILE_PATH, file)


def minibatch(*tensors, **kwargs):
    """
    生成批处理数据迭代器
    Args:
        *tensors: 输入的张量或数组（需长度一致）
        **kwargs: 可选参数，batch_size指定批大小
    Returns:
        迭代器，每次返回一批数据
    """
    # 从配置获取批大小，默认为bpr_batch_size
    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        # 单输入数据的情况
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]  # 返回当前批次
    else:
        # 多输入数据的情况（需保证长度一致）
        for i in range(0, len(tensors[0]), batch_size):
            # 按批次截取所有输入数据并返回元组
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    """
    打乱输入数组（保持各数组间的对应关系）
    Args:
        *arrays: 需打乱的数组（长度需一致）
        **kwargs: indices=True时返回打乱的索引
    Returns:
        打乱后的数组（+ 索引，若指定）
    """
    require_indices = kwargs.get('indices', False)  # 是否返回打乱的索引

    # 检查所有数组长度是否一致
    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('所有输入数组必须具有相同的长度')

    # 生成打乱的索引
    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        # 单个数组的情况
        result = arrays[0][shuffle_indices]
    else:
        # 多个数组的情况（按相同索引打乱）
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices  # 返回结果和索引
    else:
        return result  # 仅返回结果


class timer:
    """
    代码块耗时统计的上下文管理器
    用法:
        with timer():
            执行某些操作
        print(timer.get())  # 获取耗时
    """
    from time import time
    TAPE = [-1]  # 全局时间记录列表
    NAMED_TAPE = {}  # 命名时间记录字典（用于多任务统计）

    @staticmethod
    def get():
        """获取最近一次未获取的耗时"""
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        """返回命名时间记录的字符串（用于日志）"""
        hint = "|"
        if select_keys is None:
            # 输出所有命名时间
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            # 仅输出指定的命名时间
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        """重置命名时间记录"""
        if select_keys is None:
            # 重置所有
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            # 仅重置指定的
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        """
        初始化计时器
        Args:
            tape: 时间记录列表（默认使用全局TAPE）
            **kwargs: name指定命名（用于NAMED_TAPE）
        """
        if kwargs.get('name'):
            # 命名计时器（加入NAMED_TAPE）
            self.named = kwargs['name']
            # 初始化该名称的时间记录（若不存在）
            timer.NAMED_TAPE[self.named] = timer.NAMED_TAPE.get(self.named, 0.)
        else:
            self.named = False
            self.tape = tape or timer.TAPE  # 使用指定列表或全局TAPE

    def __enter__(self):
        """进入上下文时记录开始时间"""
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时计算耗时并记录"""
        if self.named:
            # 更新命名时间记录
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            # 记录到时间列表
            self.tape.append(timer.time() - self.start)


# ====================评估指标==============================
# =========================================================

def RecallPrecision_ATk(test_data, r, k):
    """
    计算Recall@k和Precision@k
    Args:
        test_data: 测试集的真实交互（每个用户的正样本列表）
        r: 预测结果的二值矩阵（r[i][j]=1表示第i个用户的第j个预测是正样本）
        k: 前k个推荐结果
    Returns:
        字典：{'recall': 平均召回率, 'precision': 平均精确率}
    """
    # 计算每个用户前k个推荐中命中的正样本数
    right_pred = r[:, :k].sum(1)
    precis_n = k  # 每个用户的推荐总数（k）
    # 每个用户的真实正样本数（用于召回率分母）
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    
    # 平均召回率 = 总命中数 / 总真实正样本数
    recall = np.sum(right_pred / recall_n)
    # 平均精确率 = 总命中数 / 总推荐数（用户数 * k）
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    计算MRR@k（Mean Reciprocal Rank）
    Args:
        r: 预测结果的二值矩阵
        k: 前k个推荐结果
    Returns:
        总MRR值（需除以用户数得到平均值）
    """
    pred_data = r[:, :k]  # 取前k个推荐结果
    # 计算折扣因子：1/log2(位置+1)，位置从1开始
    scores = np.log2(1. / np.arange(1, k + 1))
    # 命中项的倒数排名之和（乘以折扣因子）
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)  # 每个用户的MRR
    return np.sum(pred_data)  # 总MRR


def NDCGatK_r(test_data, r, k):
    """
    计算NDCG@k（Normalized Discounted Cumulative Gain）
    Args:
        test_data: 测试集的真实交互
        r: 预测结果的二值矩阵
        k: 前k个推荐结果
    Returns:
        总NDCG值（需除以用户数得到平均值）
    """
    assert len(r) == len(test_data)  # 确保预测与真实数据长度一致
    pred_data = r[:, :k]  # 取前k个推荐

    # 构建理想情况下的二值矩阵（前len(items)个为1）
    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1  # 理想情况下的命中
    max_r = test_matrix  # 理想DCG的基础

    # 计算IDCG（理想DCG）
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    # 计算DCG（实际DCG）
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)

    # 避免除零错误（IDCG为0时设为1）
    idcg[idcg == 0.] = 1.
    # 计算NDCG = DCG / IDCG
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.  # 处理NaN（如0/0）
    return np.sum(ndcg)  # 总NDCG


def AUC(all_item_scores, dataset, test_data):
    """
    计算AUC（Area Under ROC Curve）
    Args:
        all_item_scores: 所有物品的预测分数（对单个用户）
        dataset: 数据集实例
        test_data: 该用户的测试集正样本
    Returns:
        AUC值
    """
    dataset: BasicDataset
    # 构建真实标签（测试集中的物品为1，其余为0）
    r_all = np.zeros((dataset.m_items,))
    r_all[test_data] = 1
    # 过滤无效分数（仅保留有分数的物品）
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    # 计算AUC
    return roc_auc_score(r, test_item_scores)


def getLabel(test_data, pred_data):
    """
    将预测结果转换为二值标签矩阵（1表示命中真实正样本，0表示未命中）
    Args:
        test_data: 测试集真实交互（每个用户的正样本列表）
        pred_data: 预测的Top-K物品列表（每个用户的推荐结果）
    Returns:
        二值矩阵 r[i][j] = 1 表示用户i的第j个推荐是真实正样本
    """
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]  # 用户i的真实正样本
        predictTopK = pred_data[i]  # 用户i的Top-K推荐
        # 判断每个推荐物品是否在真实正样本中
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")  # 转换为0/1浮点数
        r.append(pred)
    return np.array(r).astype('float')  # 返回矩阵


# ====================评估指标结束=============================
# =========================================================