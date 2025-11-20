"""
创建于2020年3月1日
Xiangnan He等人提出的LightGCN的PyTorch实现
论文：LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

作者: Shuxian Bi (stanbi@mail.ustc.edu.cn), Jianbai Ye (gusye@mail.ustc.edu.cn)
此处设计数据集类
每个数据集的索引必须从0开始
"""
import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader  # PyTorch数据集工具
from scipy.sparse import csr_matrix  # 稀疏矩阵工具
import scipy.sparse as sp
import world  # 全局配置
from world import cprint  # 带颜色的打印函数
from time import time  # 时间统计


class BasicDataset(Dataset):
    """数据集基类，定义所有数据集必须实现的接口"""
    def __init__(self):
        print("初始化数据集")
    
    @property
    def n_users(self):
        """返回用户数量（抽象属性，子类必须实现）"""
        raise NotImplementedError
    
    @property
    def m_items(self):
        """返回物品数量（抽象属性，子类必须实现）"""
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        """返回训练集交互数量（抽象属性，子类必须实现）"""
        raise NotImplementedError
    
    @property
    def testDict(self):
        """返回测试集字典 {用户: [物品列表]}（抽象属性，子类必须实现）"""
        raise NotImplementedError
    
    @property
    def allPos(self):
        """返回每个用户的正样本列表（抽象属性，子类必须实现）"""
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        """
        获取用户对物品的交互反馈（0或1）
        Args:
            users: 用户ID列表
            items: 物品ID列表
        Returns:
            反馈列表（1表示有交互，0表示无）
        """
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        """获取指定用户的正样本物品列表"""
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        获取指定用户的负样本物品列表
        注意：对于大型数据集可能不适用（返回所有未交互物品会占用大量内存）
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        构建稀疏图的torch.sparse.FloatTensor表示
        结构参考NGCF中的矩阵形式：
        A = 
            |I,   R|  （I为单位矩阵，R为用户-物品交互矩阵）
            |R^T, I|  （R^T为R的转置）
        """
        raise NotImplementedError


class ML100k(BasicDataset):
    """
    ML100K数据集的实现类
    继承自BasicDataset，适配LightGCN的图结构需求
    """
    def __init__(self, path="../data/ML100k/real",fake_path = '../data/ML100k/fake'):
        print("初始化ML100k数据集")
        cprint("加载 [ML100K] 数据集")
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']  # 默认训练模式
        
        # 读取训练集和测试集（格式：user_id item_id）
        train_path = path + '/train.txt'
        test_path = path + '/test.txt'
        fake_data_path = 'data\\ML100k\\fake\\fake_data.txt'
        if os.path.exists(fake_data_path):
            print(114514)
            fakeData = pd.read_table(fake_data_path, header=None, sep=' ')
            trainData = pd.read_table(train_path, header=None, sep=' ')
            testData = pd.read_table(test_path, header=None, sep=' ')
            trainData = pd.concat([trainData, fakeData], ignore_index=True)
        else:
            trainData = pd.read_table(train_path, header=None, sep=' ')
            testData = pd.read_table(test_path, header=None, sep=' ')
        print(f"train_data_size{len(trainData)}")
        # 存储原始数据（ML100K无社交网络，故无trustNet）
        self.trainData = trainData
        self.testData = testData
        
        # 提取训练集的用户和物品ID（确保为整数类型）
        self.trainUser = np.array(trainData[0], dtype=np.int64)
        self.trainUniqueUsers = np.unique(self.trainUser)  # 去重的训练用户
        self.trainItem = np.array(trainData[1], dtype=np.int64)
        
        # 提取测试集的用户和物品ID
        self.testUser = np.array(testData[0], dtype=np.int64)
        self.testUniqueUsers = np.unique(self.testUser)  # 去重的测试用户
        self.testItem = np.array(testData[1], dtype=np.int64)
        
        self.Graph = None  # 稀疏图（延迟初始化）
        
        # 计算并打印数据集稀疏度
        sparsity = (len(self.trainUser) + len(self.testUser)) / (self.n_users * self.m_items)
        print(f"ML100K 稀疏度 : {sparsity:.6f}")
        
        # 构建用户-物品交互矩阵（二部图）
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
            shape=(self.n_users, self.m_items)
        )
        
        # 预计算所有用户的正样本和负样本
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos  # 负样本为未交互物品
            self.allNeg.append(np.array(list(neg)))
        
        # 构建测试集字典
        self.__testDict = self.__build_test()

    @property
    def n_users(self):
        """ML100K的用户数量（基于训练集和测试集的最大用户ID+1）"""
        return max(np.max(self.trainUser), np.max(self.testUser)) + 1
    
    @property
    def m_items(self):
        """ML100K的物品数量（基于训练集和测试集的最大物品ID+1）"""
        return max(np.max(self.trainItem), np.max(self.testItem)) + 1
    
    @property
    def trainDataSize(self):
        """训练集交互数量"""
        return len(self.trainUser)
    
    @property
    def testDict(self):
        """测试集字典 {用户: [物品列表]}"""
        return self.__testDict

    @property
    def allPos(self):
        """所有用户的正样本列表"""
        return self._allPos

    def getSparseGraph(self):
        """构建并返回用户-物品二部图的稀疏矩阵（带对称归一化）"""
        if self.Graph is None:
            # 转换用户和物品ID为LongTensor
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)
        
            # 构建邻接矩阵索引（用户-物品 和 物品-用户 边）
            # 物品ID偏移：物品索引 = 物品ID + 用户数量（避免与用户ID冲突）
            first_sub = torch.stack([user_dim, item_dim + self.n_users])  # 用户->物品
            second_sub = torch.stack([item_dim + self.n_users, user_dim])  # 物品->用户
            index = torch.cat([first_sub, second_sub], dim=1)  # 形状：[2, E]，E为边数
            data = torch.ones(index.size(-1), dtype=torch.int)  # 边权重为1，数量与边数一致
        
            # 初始稀疏邻接矩阵（改用sparse_coo_tensor）
            total_nodes = self.n_users + self.m_items
            # 注意：sparse_coo_tensor的indices需为[2, N]形状，data为[N]形状
            self.Graph = torch.sparse_coo_tensor(
                index, data, 
                torch.Size([total_nodes, total_nodes]),
                dtype=torch.int
            )
        
            # 对称归一化：D^-0.5 * A * D^-0.5
            dense = self.Graph.to_dense().float()  # 转为稠密矩阵
            D = torch.sum(dense, dim=1)  # 节点度（行和）
            D[D == 0] = 1.0  # 避免除零
            D_sqrt = torch.sqrt(D).unsqueeze(0)  # 度的平方根（行向量）
            dense = dense / D_sqrt  # 左乘D^-0.5
            dense = dense / D_sqrt.t()  # 右乘D^-0.5（列向量）
        
            # 提取非零元素的索引和值（关键修复）
            # 注意：dense.nonzero()返回的是[N, 2]形状，需转置为[2, N]
            index = dense.nonzero().t()  # 转置后形状：[2, N]
            data = dense[dense.nonzero(as_tuple=True)]  # 按非零索引提取值，形状：[N]
        
            # 验证索引和值的数量是否一致
            assert index.size(1) == data.size(0), f"索引与值数量不匹配：{index.size(1)} vs {data.size(0)}"
        
            # 构建归一化后的稀疏矩阵（改用sparse_coo_tensor）
            self.Graph = torch.sparse_coo_tensor(
                index, data, 
                torch.Size([total_nodes, total_nodes]),
                dtype=torch.float
            )
            self.Graph = self.Graph.coalesce().to(world.device)  # 合并重复索引并移至设备
        return self.Graph

    def __build_test(self):
        """构建测试集字典 {用户: [测试物品列表]}"""
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if user in test_data:
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def getUserItemFeedback(self, users, items):
        """获取用户对物品的交互反馈（1表示有交互，0表示无）"""
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape(-1, )
    
    def getUserPosItems(self, users):
        """获取指定用户的正样本物品列表"""
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def getUserNegItems(self, users):
        """获取指定用户的负样本物品列表（未交互物品）"""
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems
            
    def __getitem__(self, index):
        """按索引返回训练集中的用户ID（供DataLoader迭代）"""
        user = self.trainUniqueUsers[index]
        return user
    
    def switch2test(self):
        """切换为测试模式"""
        self.mode = self.mode_dict['test']
    
    def __len__(self):
        """返回训练集中的用户数量（迭代长度）"""
        return len(self.trainUniqueUsers)
class LastFM(BasicDataset):
    """
    LastFM数据集的实现类
    继承自PyTorch的Dataset，包含图结构信息
    """
    def __init__(self, path="../data/lastfm"):
        # 加载数据时的模式（训练或测试）
        cprint("加载 [last fm] 数据集")
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']  # 默认训练模式
        
        # 读取训练集、测试集和社交网络数据
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        testData = pd.read_table(join(path, 'test1.txt'), header=None)
        # 用户之间的信任关系，作用是 在协同过滤中融入信任网络
        # 利用用户的信任关系提升推荐准确性（比如信任的人喜欢的物品更可能被推荐
        trustNet = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        
        # 数据索引从0开始（原始数据可能从1开始，需减1）
        trustNet -= 1
        trainData -= 1
        testData -= 1
        
        # 存储原始数据
        self.trustNet = trustNet  # 社交网络（用户-用户信任关系）
        self.trainData = trainData
        self.testData = testData
        
        # 提取训练集的用户和物品ID
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)  # 去重的训练用户
        self.trainItem = np.array(trainData[:][1])
        
        # 提取测试集的用户和物品ID
        self.testUser = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)  # 去重的测试用户
        self.testItem = np.array(testData[:][1])
        
        self.Graph = None  # 存储稀疏图（延迟初始化）
        
        # 打印数据集稀疏度（交互数 / 总可能数）
        print(f"LastFm 稀疏度 : {(len(self.trainUser) + len(self.testUser)) / self.n_users / self.m_items}")
        
        # 构建稀疏矩阵
        # 社交网络矩阵（用户-用户，信任关系）
        self.socialNet = csr_matrix(
            (np.ones(len(trustNet)), (trustNet[:, 0], trustNet[:, 1])),
            shape=(self.n_users, self.n_users)
        )
        # 用户-物品交互矩阵（二部图）
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
            shape=(self.n_users, self.m_items)
        )
        
        # 预计算每个用户的正样本和负样本
        self._allPos = self.getUserPosItems(list(range(self.n_users)))  # 所有用户的正样本
        self.allNeg = []  # 所有用户的负样本
        allItems = set(range(self.m_items))  # 所有物品的集合
        for i in range(self.n_users):
            pos = set(self._allPos[i])  # 当前用户的正样本
            neg = allItems - pos  # 负样本 = 所有物品 - 正样本
            self.allNeg.append(np.array(list(neg)))
        
        # 构建测试集字典
        self.__testDict = self.__build_test()

    @property
    def n_users(self):
        """LastFM数据集的用户数量（固定值）"""
        return 1892
    
    @property
    def m_items(self):
        """LastFM数据集的物品数量（固定值）"""
        return 4489
    
    @property
    def trainDataSize(self):
        """训练集交互数量"""
        return len(self.trainUser)
    
    @property
    def testDict(self):
        """测试集字典 {用户: [物品列表]}"""
        return self.__testDict

    @property
    def allPos(self):
        """所有用户的正样本列表"""
        return self._allPos

    def getSparseGraph(self):
        """构建并返回用户-物品二部图的稀疏矩阵（带归一化）"""
        if self.Graph is None:  # 延迟初始化，首次调用时构建
            # 转换用户和物品ID为LongTensor
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)
            
            # 构建邻接矩阵的索引（用户-物品 和 物品-用户 边）
            # 物品ID偏移：物品索引 = 物品ID + 用户数量（避免与用户ID冲突）
            first_sub = torch.stack([user_dim, item_dim + self.n_users])  # 用户->物品边
            second_sub = torch.stack([item_dim + self.n_users, user_dim])  # 物品->用户边
            index = torch.cat([first_sub, second_sub], dim=1)  # 合并所有边
            data = torch.ones(index.size(-1)).int()  # 边的权重初始化为1
            
            # 构建初始稀疏邻接矩阵（未归一化）
            self.Graph = torch.sparse.IntTensor(
                index, data, 
                torch.Size([self.n_users + self.m_items, self.n_users + self.m_items])
            )
            
            # 归一化邻接矩阵（对称归一化：D^-0.5 * A * D^-0.5）
            dense = self.Graph.to_dense()  # 转为稠密矩阵计算度
            D = torch.sum(dense, dim=1).float()  # 计算每个节点的度（行和）
            D[D == 0.] = 1.  # 避免除零错误
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)  # 度的平方根（行向量）
            dense = dense / D_sqrt  # 左乘D^-0.5
            dense = dense / D_sqrt.t()  # 右乘D^-0.5（转置为列向量）
            
            # 重新构建稀疏矩阵（只保留非零元素）
            index = dense.nonzero()  # 非零元素的索引
            data = dense[dense >= 1e-9]  # 非零元素的值（过滤微小值）
            assert len(index) == len(data)  # 索引和值的数量必须一致
            
            # 构建归一化后的稀疏矩阵并移动到指定设备
            self.Graph = torch.sparse.FloatTensor(
                index.t(), data, 
                torch.Size([self.n_users + self.m_items, self.n_users + self.m_items])
            )
            self.Graph = self.Graph.coalesce().to(world.device)  # 合并重复索引并移至设备
        return self.Graph

    def __build_test(self):
        """
        构建测试集字典
        Returns:
            dict: {用户ID: [该用户在测试集的物品列表]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def getUserItemFeedback(self, users, items):
        """
        获取用户对物品的交互反馈（1表示有交互，0表示无）
        Args:
            users: 用户ID列表
            items: 物品ID列表
        Returns:
            反馈列表（numpy数组）
        """
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))
    
    def getUserPosItems(self, users):
        """
        获取指定用户的正样本物品列表
        Args:
            users: 用户ID列表
        Returns:
            列表的列表，每个子列表为对应用户的正样本物品
        """
        posItems = []
        for user in users:
            # 从稀疏矩阵中提取该用户有交互的物品ID
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def getUserNegItems(self, users):
        """
        获取指定用户的负样本物品列表
        Args:
            users: 用户ID列表
        Returns:
            列表的列表，每个子列表为对应用户的负样本物品
        """
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems
            
    def __getitem__(self, index):
        """
        实现Dataset的核心方法，支持按索引获取样本
        Args:
            index: 索引
        Returns:
            训练集中的用户ID（用于迭代训练用户）
        """
        user = self.trainUniqueUsers[index]
        return user  # 返回用户ID（后续可根据用户ID获取正负样本）
    
    def switch2test(self):
        """切换数据集模式为测试模式（供DataLoader使用）"""
        self.mode = self.mode_dict['test']
    
    def __len__(self):
        """返回训练集中的用户数量（用于迭代长度）"""
        return len(self.trainUniqueUsers)


class Loader(BasicDataset):
    """
    通用数据集加载器（适用于Gowalla、Yelp2018、Amazon-Book等）
    继承自BasicDataset，支持大规模数据集
    """

    def __init__(self, config=world.config, path="../data/gowalla"):
        cprint(f'加载 [{path}] 数据集')
        self.split = config['A_split']  # 是否拆分邻接矩阵（用于大规模数据）
        self.folds = config['A_n_fold']  # 邻接矩阵拆分的份数
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']  # 默认训练模式
        self.n_user = 0  # 用户数量（动态计算）
        self.m_item = 0  # 物品数量（动态计算）
        train_file = path + '/train.txt'  # 训练集路径
        test_file = path + '/test.txt'  # 测试集路径
        self.path = path  # 数据集根路径
        
        # 初始化训练集和测试集的列表
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0  # 训练集交互数量
        self.testDataSize = 0  # 测试集交互数量

        # 读取训练集（格式：每行 "用户ID 物品ID1 物品ID2 ..."）
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')  # 分割行
                    items = [int(i) for i in l[1:]]  # 物品列表
                    uid = int(l[0])  # 用户ID
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))  # 重复用户ID（与物品数量匹配）
                    trainItem.extend(items)  # 物品ID列表
                    # 更新最大物品ID和用户ID
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)  # 累加交互数量
        
        # 转换为numpy数组
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        # 读取测试集（格式同训练集）
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        
        # 确保索引从0开始（若原始ID从1开始，+1避免溢出）
        self.m_item += 1
        self.n_user += 1
        
        # 转换为numpy数组
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        
        self.Graph = None  # 稀疏图（延迟初始化）
        
        # 打印数据集基本信息
        print(f"{self.trainDataSize} 条训练交互")
        print(f"{self.testDataSize} 条测试交互")
        print(f"{world.dataset} 稀疏度 : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # 构建用户-物品交互稀疏矩阵（二部图）
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item)
        )
        
        # 计算用户和物品的度（交互数量），用于归一化
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()  # 用户的度（每行和）
        self.users_D[self.users_D == 0.] = 1  # 避免除零
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()  # 物品的度（每列和）
        self.items_D[self.items_D == 0.] = 1.
        
        # 预计算正样本和测试集字典
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{world.dataset} 加载完成")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        """
        将归一化的邻接矩阵拆分为多个子矩阵（用于大规模数据并行处理）
        Args:
            A: 归一化后的邻接矩阵（scipy稀疏矩阵）
        Returns:
            拆分后的PyTorch稀疏张量列表
        """
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds  # 每份的长度
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            # 最后一份可能更长（处理整除余数）
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            # 转换为PyTorch稀疏张量并移至设备
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        """
        将scipy稀疏矩阵转换为PyTorch稀疏张量
        Args:
            X: scipy稀疏矩阵
        Returns:
            torch.sparse.FloatTensor
        """
        coo = X.tocoo().astype(np.float32)  # 转换为COO格式（便于提取索引和值）
        row = torch.Tensor(coo.row).long()  # 行索引
        col = torch.Tensor(coo.col).long()  # 列索引
        index = torch.stack([row, col])  # 合并索引
        data = torch.FloatTensor(coo.data)  # 数据值
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        """
        构建并返回归一化的稀疏邻接矩阵（支持缓存，避免重复计算）
        若已存在缓存文件（s_pre_adj_mat.npz），则直接加载；否则重新计算并保存
        """
        print("加载邻接矩阵")
        if self.Graph is None:
            try:
                # 尝试加载预计算的归一化邻接矩阵
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("成功加载预计算矩阵...")
                norm_adj = pre_adj_mat
            except:
                # 重新计算归一化邻接矩阵
                print("生成邻接矩阵中...")
                s = time()
                # 初始化邻接矩阵（大小：(用户数+物品数) x (用户数+物品数)）
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()  # 转换为LIL格式便于修改
                
                # 构建二部图邻接矩阵
                R = self.UserItemNet.tolil()  # 用户-物品交互矩阵（LIL格式）
                adj_mat[:self.n_users, self.n_users:] = R  # 上右块：用户->物品
                adj_mat[self.n_users:, :self.n_users] = R.T  # 下左块：物品->用户（转置）
                adj_mat = adj_mat.todok()  # 转换为DOK格式
                
                # 计算归一化系数（D^-0.5，D为度矩阵）
                rowsum = np.array(adj_mat.sum(axis=1))  # 每行的和（节点的度）
                d_inv = np.power(rowsum, -0.5).flatten()  # 度的倒数平方根
                d_inv[np.isinf(d_inv)] = 0.  # 处理无穷大（度为0的节点）
                d_mat = sp.diags(d_inv)  # 度的对角矩阵
                
                # 对称归一化：D^-0.5 * A * D^-0.5
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()  # 转换为CSR格式便于存储
                
                # 计时并保存结果
                end = time()
                print(f"计算耗时 {end - s}s，已保存归一化矩阵...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            # 根据配置决定是否拆分邻接矩阵
            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("已拆分矩阵")
            else:
                # 转换为PyTorch稀疏张量并移至设备
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("未拆分矩阵")
        return self.Graph

    def __build_test(self):
        """构建测试集字典 {用户: [物品列表]}"""
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """获取用户对物品的交互反馈（1有交互，0无）"""
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        """获取指定用户的正样本物品列表"""
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems