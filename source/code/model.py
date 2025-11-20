"""
创建于2020年3月1日
Xiangnan He等人提出的LightGCN的PyTorch实现
论文：LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

作者: Jianbai Ye (gusye@mail.ustc.edu.cn)

在此定义模型
"""
import world  # 自定义配置模块
import torch
from dataloader import BasicDataset  # 基础数据集类
from torch import nn
import numpy as np


class BasicModel(nn.Module):    
    """基础模型类，所有推荐模型的父类"""
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        """获取用户对所有物品的评分预测
        Args:
            users: 用户ID列表
        Returns:
            评分预测矩阵
        """
        raise NotImplementedError  # 子类必须实现此方法
    
class PairWiseModel(BasicModel):
    """ pairwise 损失函数的模型基类，用于排序任务"""
    def __init__(self):
        super(PairWiseModel, self).__init__()
    
    def bpr_loss(self, users, pos, neg):
        """
        计算BPR损失（贝叶斯个性化排序损失）
        Parameters:
            users: 用户ID列表
            pos: 每个用户对应的正样本物品ID（已交互）
            neg: 每个用户对应的负样本物品ID（未交互）
        Return:
            (log损失, L2正则化损失)
        """
        raise NotImplementedError  # 子类必须实现此方法
    
class PureMF(BasicModel):
    """纯矩阵分解模型（Matrix Factorization），作为对比基准"""
    def __init__(self, 
                 config:dict,  # 模型配置字典
                 dataset:BasicDataset):  # 数据集对象
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users  # 用户数量
        self.num_items  = dataset.m_items  # 物品数量
        self.latent_dim = config['latent_dim_rec']  # 嵌入维度
        self.f = nn.Sigmoid()  # 激活函数，将评分映射到[0,1]
        self.__init_weight()  # 初始化权重
        
    def __init_weight(self):
        """初始化用户和物品的嵌入矩阵"""
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)  # 用户嵌入
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)  # 物品嵌入
        print("使用标准正态分布N(0,1)初始化PureMF")
        
    def getUsersRating(self, users):
        """获取用户对所有物品的评分预测"""
        users = users.long()  # 转换为长整型
        users_emb = self.embedding_user(users)  # 获取用户嵌入
        items_emb = self.embedding_item.weight  # 获取所有物品嵌入
        scores = torch.matmul(users_emb, items_emb.t())  # 计算用户-物品评分（内积）
        return self.f(scores)  # 通过sigmoid激活
    
    def bpr_loss(self, users, pos, neg):
        """计算BPR损失和正则化损失"""
        users_emb = self.embedding_user(users.long())  # 用户嵌入
        pos_emb   = self.embedding_item(pos.long())    # 正样本物品嵌入
        neg_emb   = self.embedding_item(neg.long())    # 负样本物品嵌入
        
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)  # 正样本评分（内积求和）
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)  # 负样本评分
        
        # BPR损失：希望正样本评分 > 负样本评分，使用softplus实现
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        
        # L2正则化损失：约束嵌入向量的模长，防止过拟合
        reg_loss = (1/2) * (users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        """前向传播：计算用户对特定物品的评分"""
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb * items_emb, dim=1)  # 内积计算评分
        return self.f(scores)  # 激活后输出


class LightGCN(BasicModel):
    """LightGCN模型实现：简化的图卷积网络用于推荐"""
    def __init__(self, 
                 config:dict,  # 模型配置
                 dataset:BasicDataset):  # 数据集
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset  # 数据集对象（包含交互图）
        self.__init_weight()  # 初始化权重

    def __init_weight(self):
        """初始化模型参数和嵌入矩阵"""
        print("嵌入生成")
        self.num_users  = self.dataset.n_users  # 用户数量
        self.num_items  = self.dataset.m_items  # 物品数量
        print(f"用户数量: {self.num_users}, 物品数量: {self.num_items}")
        self.latent_dim = self.config['latent_dim_rec']  # 嵌入维度
        self.n_layers = self.config['lightGCN_n_layers']  # GCN层数
        self.keep_prob = self.config['keep_prob']  # dropout保留概率
        self.A_split = self.config['A_split']  # 是否拆分邻接矩阵（用于大规模数据）
        
        # 初始化用户和物品嵌入
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        
        # 权重初始化方式
        if self.config['pretrain'] == 0:
            # 随机正态分布初始化（论文中指出在无激活函数时更优）
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('使用正态分布初始化')
        else:
            # 使用预训练嵌入
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('使用预训练数据初始化')
        
        self.f = nn.Sigmoid()  # 评分预测激活函数
        self.Graph = self.dataset.getSparseGraph()  # 获取稀疏交互图（邻接矩阵）
        print(f"LightGCN初始化完成（dropout:{self.config['dropout']}）")

    def __dropout_x(self, x, keep_prob):
        """对稀疏矩阵执行dropout（仅保留部分边）
        Args:
            x: 稀疏邻接矩阵
            keep_prob: 保留概率
        Returns:
            经过dropout的稀疏矩阵
        """
        size = x.size()
        index = x.indices().t()  # 边的索引
        values = x.values()      # 边的权重
        
        # 随机保留部分边
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()  # 布尔掩码筛选保留的边
        
        index = index[random_index]
        values = values[random_index] / keep_prob  # 缩放保留边的权重
        g = torch.sparse.FloatTensor(index.t(), values, size)  # 重构稀疏矩阵
        return g
    
    def __dropout(self, keep_prob):
        """对图执行dropout（支持拆分的邻接矩阵）"""
        if self.A_split:
            # 若邻接矩阵被拆分，则对每个子图执行dropout
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            # 对整体邻接矩阵执行dropout
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        LightGCN的核心传播函数：通过多层图卷积计算最终嵌入
        Returns:
            users: 最终用户嵌入矩阵
            items: 最终物品嵌入矩阵
        """       
        # 获取初始嵌入（第0层）
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])  # 合并用户和物品嵌入（便于统一计算）[m+n,d]
        embs = [all_emb]  # 存储各层嵌入
        
        # 处理dropout（仅训练时生效）
        if self.config['dropout']:
            if self.training:
                print("执行dropout")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph  # 测试时不dropout
        else:
            g_droped = self.Graph  # 不启用dropout时直接使用原图
        
        # 多层图卷积传播
        for layer in range(self.n_layers):
            if self.A_split:
                # 若邻接矩阵拆分，则逐层聚合各子图的信息
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))  # 稀疏矩阵乘法
                side_emb = torch.cat(temp_emb, dim=0)  # 合并子图聚合结果
                all_emb = side_emb
            else:
                # 整体邻接矩阵的聚合（核心操作：A * E） [m+n,m+n] * [m+n,d] = [m+n,d]
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)  # 保存当前层嵌入
        
        # 层组合：对所有层嵌入取平均（论文中的均匀权重策略）
        embs = torch.stack(embs, dim=1)  # 按层维度堆叠
        light_out = torch.mean(embs, dim=1)  # 平均各层嵌入
        
        # 拆分用户和物品嵌入
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        """获取用户对所有物品的评分预测"""
        all_users, all_items = self.computer()  # 计算最终嵌入
        users_emb = all_users[users.long()]  # 获取目标用户嵌入
        items_emb = all_items  # 获取所有物品嵌入
        rating = self.f(torch.matmul(users_emb, items_emb.t()))  # 内积计算评分并激活
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        """获取用户、正样本、负样本的嵌入（用于计算损失）
        Returns:
            多层聚合后的嵌入 + 初始嵌入（用于正则化）
        """
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        
        # 初始嵌入（用于正则化损失）
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        """计算BPR损失和正则化损失"""
        # 获取嵌入（聚合后嵌入 + 初始嵌入）
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        
        # 正则化损失：仅约束初始嵌入（论文设计）
        reg_loss = (1/2) * (userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2) +
                         negEmb0.norm(2).pow(2)) / float(len(users))
        
        # 计算正/负样本评分
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        
        # BPR损失：softplus确保损失非负，且当正分>负分时损失接近0
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        """前向传播：计算用户对特定物品的预测评分"""
        all_users, all_items = self.computer()  # 计算最终嵌入
        users_emb = all_users[users]  # 目标用户嵌入
        items_emb = all_items[items]  # 目标物品嵌入
        inner_pro = torch.mul(users_emb, items_emb)  # 元素乘
        gamma = torch.sum(inner_pro, dim=1)  # 内积求和作为评分
        return gamma